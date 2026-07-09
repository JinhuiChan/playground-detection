"""
Step 5: 训练主程序 — 完整的训练+验证循环, 支持 Dice+BCE 组合损失
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
    TILE_SIZE, NUM_BANDS, NUM_CLASSES, MEAN, STD,
    BATCH_SIZE, VAL_BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, LR_PATIENCE, LR_FACTOR, EARLY_STOP_PATIENCE,
    NUM_WORKERS, DICE_WEIGHT, BCE_WEIGHT,
    CHECKPOINT_DIR, LOG_DIR, MODEL_NAME, ENCODER_NAME,
    ENCODER_WEIGHTS, IN_CHANNELS, ensure_dirs
)
from dataset import create_dataloaders
from model import create_model
from metrics import SegmentationMetrics


# ==================== 损失函数 ====================

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) sigmoid输出
            target: (B, 1, H, W) 二值掩膜
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1. - dice


class CombinedLoss(nn.Module):
    """Dice Loss + BCE Loss 组合"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# ==================== 训练函数 ====================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0.0
    metrics = SegmentationMetrics()

    pbar = tqdm(dataloader, desc=f"  Epoch {epoch} [Train]")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        # 记录
        epoch_loss += loss.item()
        metrics.update(
            outputs.detach().cpu().numpy(),
            masks.detach().cpu().numpy()
        )

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'IoU': f'{metrics.get_metrics()["IoU"]:.3f}'
        })

        # TensorBoard (每50个batch)
        if writer and batch_idx % 50 == 0:
            step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss_step', loss.item(), step)

    avg_loss = epoch_loss / len(dataloader)
    train_metrics = metrics.get_metrics()

    if writer:
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        for k, v in train_metrics.items():
            writer.add_scalar(f'Train/{k}', v, epoch)

    return avg_loss, train_metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer=None):
    """验证"""
    model.eval()
    epoch_loss = 0.0
    metrics = SegmentationMetrics()

    pbar = tqdm(dataloader, desc=f"  Epoch {epoch} [Val]  ")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        epoch_loss += loss.item()
        metrics.update(
            outputs.detach().cpu().numpy(),
            masks.detach().cpu().numpy()
        )

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'IoU': f'{metrics.get_metrics()["IoU"]:.3f}'
        })

    avg_loss = epoch_loss / len(dataloader)
    val_metrics = metrics.get_metrics()

    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)

    return avg_loss, val_metrics


# ==================== 主训练流程 ====================

def train(args):
    """主训练函数"""
    ensure_dirs()

    # 创建数据加载器
    print(f"\n{'='*50}")
    print(f"  加载数据集...")
    print(f"{'='*50}")

    train_loader, val_loader = create_dataloaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir, args.val_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mean=MEAN, std=STD
    )

    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")

    # 创建模型
    print(f"\n{'='*50}")
    print(f"  创建模型...")
    print(f"{'='*50}")

    model, device = create_model(
        model_type=args.model_type,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights if args.pretrained else None,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
    )

    # 损失函数
    criterion = CombinedLoss(
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight
    )

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_factor,
        patience=args.lr_patience, min_lr=1e-7
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir) if args.use_tensorboard else None

    # 训练状态
    best_iou = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': []}

    print(f"\n{'='*50}")
    print(f"  开始训练 ({args.epochs} epochs)")
    print(f"{'='*50}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # 验证
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        # 学习率调度
        scheduler.step(val_metrics['IoU'])

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_metrics['IoU'])
        history['val_f1'].append(val_metrics['F1'])

        elapsed = time.time() - epoch_start

        # 打印结果
        print(f"\n  Epoch {epoch} 完成 ({elapsed:.1f}s)")
        print(f"    Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        print(f"    Val IoU: {val_metrics['IoU']:.4f}  |  Val F1: {val_metrics['F1']:.4f}  |  Val Prec: {val_metrics['Precision']:.4f}  |  Val Rec: {val_metrics['Recall']:.4f}")
        print(f"    LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'val_metrics': val_metrics,
                'args': vars(args),
                'history': history,
            }
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"    [BEST] 保存最佳模型 (IoU: {best_iou:.4f}) -> {checkpoint_path}")

            # 同时保存为jit可加载的格式
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, 'best_model_weights.pth'))
        else:
            patience_counter += 1
            print(f"    Patience: {patience_counter}/{args.early_stop_patience} (best IoU: {best_iou:.4f} @ epoch {best_epoch})")

        # 早停检查
        if patience_counter >= args.early_stop_patience:
            print(f"\n  ⏹ 早停触发! 最佳IoU: {best_iou:.4f} (epoch {best_epoch})")
            break

    # 保存训练历史
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    if writer:
        writer.close()

    print(f"\n{'='*50}")
    print(f"  训练完成!")
    print(f"  最佳 IoU: {best_iou:.4f} (epoch {best_epoch})")
    print(f"  模型保存在: {args.checkpoint_dir}")
    print(f"{'='*50}")

    return best_iou


# ==================== 命令行入口 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="操场检测语义分割训练")

    # 数据参数
    parser.add_argument('--train_img_dir', type=str, default=TRAIN_IMG_DIR)
    parser.add_argument('--train_mask_dir', type=str, default=TRAIN_MASK_DIR)
    parser.add_argument('--val_img_dir', type=str, default=VAL_IMG_DIR)
    parser.add_argument('--val_mask_dir', type=str, default=VAL_MASK_DIR)

    # 模型参数
    parser.add_argument('--model_type', type=str, default='smp',
                        choices=['smp', 'lightweight'],
                        help='模型类型: smp=预训练U-Net, lightweight=手写轻量U-Net')
    parser.add_argument('--encoder_name', type=str, default=ENCODER_NAME)
    parser.add_argument('--encoder_weights', type=str, default=ENCODER_WEIGHTS)
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用ImageNet预训练权重')
    parser.add_argument('--in_channels', type=int, default=IN_CHANNELS)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--lr_patience', type=int, default=LR_PATIENCE)
    parser.add_argument('--lr_factor', type=float, default=LR_FACTOR)
    parser.add_argument('--early_stop_patience', type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--dice_weight', type=float, default=DICE_WEIGHT)
    parser.add_argument('--bce_weight', type=float, default=BCE_WEIGHT)

    # 其他
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--log_dir', type=str, default=LOG_DIR)
    parser.add_argument('--use_tensorboard', action='store_true', default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # CPU 时自动降低 batch size
    if not torch.cuda.is_available():
        try:
            import torch_npu
            if not torch.npu.is_available():
                if args.batch_size > 4:
                    print(f"  [WARN] CPU模式, batch_size {args.batch_size} -> 4")
                    args.batch_size = 4
        except ImportError:
            if args.batch_size > 4:
                print(f"  [WARN] CPU模式, batch_size {args.batch_size} -> 4")
                args.batch_size = 4

    print(f"\n{'='*50}")
    print(f"  操场检测 — 语义分割训练")
    print(f"{'='*50}")
    print(f"  模型类型: {args.model_type}")
    print(f"  编码器: {args.encoder_name}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")

    # NPU设备信息
    try:
        import torch_npu
        if torch.npu.is_available():
            print(f"  NPU: 昇腾 ({torch.npu.device_count()} 设备)")
    except ImportError:
        pass
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}")

    train(args)
