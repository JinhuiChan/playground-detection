"""
Step 6: 推理与可视化 — 滑动窗口推理 + 结果对比
"""
import os
import sys
import argparse
import numpy as np
import rasterio
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    TIF_PATH, MASK_PATH, TILE_SIZE, TILE_STRIDE,
    CHECKPOINT_DIR, MEAN, STD, NUM_BANDS, IN_CHANNELS, NUM_CLASSES,
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
    OUTPUT_DIR
)
from model import create_model
from metrics import SegmentationMetrics


@torch.no_grad()
def predict_tiles(model, image_tiles, device):
    """
    对一批图像瓦片进行推理

    Args:
        model: 训练好的模型
        image_tiles: list of (C, H, W) numpy arrays
        device: torch device

    Returns:
        list of (H, W) binary masks
    """
    model.eval()
    results = []

    for img in image_tiles:
        # 归一化
        img = img.astype(np.float32) / 255.0
        mean_arr = np.array(MEAN, dtype=np.float32).reshape(4, 1, 1)
        std_arr = np.array(STD, dtype=np.float32).reshape(4, 1, 1)
        img = (img - mean_arr) / std_arr

        # 推理
        tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)  # (1, 4, H, W)
        pred = model(tensor)  # (1, 1, H, W)
        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        results.append(pred_mask)

    return results


def sliding_window_inference(model, image_path, tile_size=256, stride=256, device='cpu'):
    """
    对大图进行滑动窗口推理

    Returns:
        full_prediction: (H, W) 完整预测掩膜
    """
    with rasterio.open(image_path) as src:
        img_array = src.read()  # (4, H, W)
        H, W = src.height, src.width
        transform = src.transform
        profile = src.profile

    # 预测累加器和计数器 (用于重叠区域取平均)
    pred_accum = np.zeros((H, W), dtype=np.float32)
    count_accum = np.zeros((H, W), dtype=np.float32)

    n_rows = (H - tile_size) // stride + 1
    n_cols = (W - tile_size) // stride + 1

    print(f"  滑动窗口推理: {n_rows}x{n_cols} 个窗口...")
    model.eval()

    for r in tqdm(range(n_rows), desc="  推理"):
        for c in range(n_cols):
            row_start = r * stride
            col_start = c * stride

            tile = img_array[:, row_start:row_start + tile_size,
                             col_start:col_start + tile_size].astype(np.float32)

            if tile.shape[1] != tile_size or tile.shape[2] != tile_size:
                continue

            # 归一化
            tile = tile / 255.0
            mean_arr = np.array(MEAN, dtype=np.float32).reshape(4, 1, 1)
            std_arr = np.array(STD, dtype=np.float32).reshape(4, 1, 1)
            tile = (tile - mean_arr) / std_arr

            tensor = torch.from_numpy(tile).float().unsqueeze(0).to(device)
            pred = model(tensor)
            pred_prob = pred.squeeze().cpu().numpy()

            pred_accum[row_start:row_start + tile_size,
                       col_start:col_start + tile_size] += pred_prob
            count_accum[row_start:row_start + tile_size,
                        col_start:col_start + tile_size] += 1.0

    # 平均
    full_pred = (pred_accum / count_accum) > 0.5
    return full_pred.astype(np.uint8), profile


def visualize_prediction(image_tile, true_mask, pred_mask, save_path=None, idx=0):
    """
    可视化单张瓦片的 原图 + 真值 + 预测

    Args:
        image_tile: (C, H, W) 或 (H, W, C) numpy array
        true_mask: (H, W) numpy array
        pred_mask: (H, W) numpy array
        save_path: 保存路径
        idx: 样本索引
    """
    # 转换为(H, W, C) - 显示RGB (波段0,1,2)
    if image_tile.shape[0] <= 4:
        rgb = image_tile[:3, :, :].transpose(1, 2, 0)
    else:
        rgb = image_tile

    # 归一化到[0,1]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 原图 (RGB)
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    # NIR波段
    if image_tile.shape[0] >= 4:
        nir = image_tile[3, :, :]
        axes[1].imshow(nir, cmap='gray')
        axes[1].set_title('NIR Band')
    else:
        axes[1].axis('off')
    axes[1].axis('off')

    # 真值
    axes[2].imshow(true_mask, cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    # 预测
    axes[3].imshow(pred_mask, cmap='Blues', vmin=0, vmax=1)
    axes[3].set_title('Prediction')
    axes[3].axis('off')

    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='True'),
        Patch(facecolor='blue', alpha=0.6, label='Predicted'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2)

    plt.suptitle(f'Playground Detection — Sample {idx}', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存可视化: {save_path}")

    plt.close()


def evaluate_on_valset(model, device, num_samples=5):
    """
    在验证集上评估并生成可视化
    """
    print(f"\n{'='*50}")
    print(f"  验证集评估...")
    print(f"{'='*50}")

    # 加载所有验证数据
    val_files = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.endswith('_img.npy')])

    metrics = SegmentationMetrics()
    all_images, all_masks, all_preds = [], [], []

    for fname in tqdm(val_files, desc="  评估"):
        basename = fname.replace('_img.npy', '')
        img = np.load(os.path.join(VAL_IMG_DIR, fname))  # (4, 256, 256)
        true_mask = np.load(os.path.join(VAL_MASK_DIR, f"{basename}_mask.npy"))  # (256, 256)

        # 推理
        img_norm = img.astype(np.float32) / 255.0
        mean_arr = np.array(MEAN, dtype=np.float32).reshape(4, 1, 1)
        std_arr = np.array(STD, dtype=np.float32).reshape(4, 1, 1)
        img_norm = (img_norm - mean_arr) / std_arr

        tensor = torch.from_numpy(img_norm).float().unsqueeze(0).to(device)
        pred = model(tensor)
        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        metrics.update(pred_mask, true_mask)

        if len(all_images) < num_samples * 3:
            all_images.append(img)
            all_masks.append(true_mask)
            all_preds.append(pred_mask)

    # 打印报告
    print(f"\n{metrics.get_report()}")

    # 可视化样本
    vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"\n  生成可视化样本...")
    # 选择正样本和负样本
    positive_samples = [(i, img, msk, prd) for i, (img, msk, prd) in
                        enumerate(zip(all_images, all_masks, all_preds))
                        if msk.sum() > 0]
    negative_samples = [(i, img, msk, prd) for i, (img, msk, prd) in
                        enumerate(zip(all_images, all_masks, all_preds))
                        if msk.sum() == 0]

    # 显示3个正样本 + 2个负样本
    selected = positive_samples[:3] + negative_samples[:2]

    for i, (orig_idx, img, msk, prd) in enumerate(selected):
        save_path = os.path.join(vis_dir, f"sample_{i+1:02d}.png")
        visualize_prediction(img, msk, prd, save_path, i + 1)

    return metrics.get_metrics()


def main():
    parser = argparse.ArgumentParser(description="操场检测推理与可视化")
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(CHECKPOINT_DIR, 'best_model.pth'),
                        help='模型checkpoint路径')
    parser.add_argument('--mode', type=str, default='eval',
                        choices=['eval', 'full_image', 'vis'],
                        help='推理模式: eval=验证集评估, full_image=大图推理, vis=可视化')
    parser.add_argument('--model_type', type=str, default='smp')
    parser.add_argument('--image_path', type=str, default=TIF_PATH,
                        help='大图推理用的TIF路径')
    parser.add_argument('--output_path', type=str,
                        default=os.path.join(OUTPUT_DIR, 'prediction_full.tif'),
                        help='大图推理输出路径')
    parser.add_argument('--num_samples', type=int, default=5)

    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  操场检测 — 推理")
    print(f"{'='*50}")

    # 自动检测设备 (NPU > CUDA > CPU)
    try:
        import torch_npu
        if torch.npu.is_available():
            device = torch.device('npu:0')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")

    # 加载模型
    print(f"\n  加载模型: {args.checkpoint}")
    model, _ = create_model(
        model_type=args.model_type,
        encoder_weights=None,  # 不使用预训练权重
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  [OK] 加载成功 (epoch {checkpoint.get('epoch', '?')}, "
          f"best IoU: {checkpoint.get('best_iou', '?')})")

    if args.mode == 'eval':
        evaluate_on_valset(model, device, args.num_samples)

    elif args.mode == 'full_image':
        print(f"\n  大图推理: {args.image_path}")
        full_pred, profile = sliding_window_inference(
            model, args.image_path, TILE_SIZE, TILE_STRIDE, device
        )
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
        with rasterio.open(args.output_path, 'w', **profile) as dst:
            dst.write(full_pred, 1)
        print(f"  [OK] 保存预测大图: {args.output_path}")

    elif args.mode == 'vis':
        # 仅可视化模式
        val_files = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.endswith('_img.npy')])[:6]
        vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        for i, fname in enumerate(val_files):
            basename = fname.replace('_img.npy', '')
            img = np.load(os.path.join(VAL_IMG_DIR, fname))
            true_mask = np.load(os.path.join(VAL_MASK_DIR, f"{basename}_mask.npy"))

            img_norm = img.astype(np.float32) / 255.0
            mean_arr = np.array(MEAN, dtype=np.float32).reshape(4, 1, 1)
            std_arr = np.array(STD, dtype=np.float32).reshape(4, 1, 1)
            img_norm = (img_norm - mean_arr) / std_arr

            tensor = torch.from_numpy(img_norm).float().unsqueeze(0).to(device)
            pred = model(tensor)
            pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            visualize_prediction(
                img, true_mask, pred_mask,
                os.path.join(vis_dir, f"vis_{i+1:02d}.png"), i + 1
            )

    print(f"\n  [OK] 推理完成!")


if __name__ == "__main__":
    main()
