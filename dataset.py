"""
Step 3: PyTorch Dataset — 数据加载与增强
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


class PlaygroundDataset(Dataset):
    """
    操场检测语义分割数据集
    加载.npy瓦片，应用标准化和数据增强
    """

    def __init__(self, img_dir, mask_dir, mean=None, std=None, augment=False):
        """
        Args:
            img_dir: 图像瓦片目录
            mask_dir: 掩膜瓦片目录
            mean: 各波段均值 (list of 4 floats)
            std: 各波段标准差 (list of 4 floats)
            augment: 是否启用数据增强 (训练=True, 验证=False)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment

        # 默认ImageNet统计值 + NIR
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406, 0.350]
        self.std = std if std is not None else [0.229, 0.224, 0.225, 0.150]

        # 获取所有文件列表 (按图像文件)
        self.filenames = sorted([
            f for f in os.listdir(img_dir) if f.endswith('_img.npy')
        ])

        if len(self.filenames) == 0:
            raise RuntimeError(f"在 {img_dir} 中未找到瓦片文件！请先运行 tile_dataset.py")

        print(f"  加载数据集: {len(self.filenames)} 个样本 (augment={augment})")

        # 定义数据增强pipeline
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(p=0.2),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        basename = self.filenames[idx]
        prefix = basename.replace('_img.npy', '')

        # 加载 .npy 文件
        img_path = os.path.join(self.img_dir, basename)
        mask_path = os.path.join(self.mask_dir, f"{prefix}_mask.npy")

        image = np.load(img_path)   # (4, 256, 256)
        mask = np.load(mask_path)   # (256, 256)

        # 转换为 (H, W, C) 格式供albumentations使用
        image = image.transpose(1, 2, 0).astype(np.float32)  # (256, 256, 4)
        mask = mask.astype(np.float32)  # (256, 256)

        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 标准化: (image - mean) / std (逐波段)
        image = image / 255.0  # 先归一化到[0,1]
        mean_arr = np.array(self.mean, dtype=np.float32).reshape(1, 1, 4)
        std_arr = np.array(self.std, dtype=np.float32).reshape(1, 1, 4)
        image = (image - mean_arr) / std_arr

        # 转换回 PyTorch 格式: (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

        return image, mask


def create_dataloaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                       batch_size=16, num_workers=4, mean=None, std=None):
    """
    创建训练和验证的DataLoader

    Returns:
        train_loader, val_loader
    """
    train_dataset = PlaygroundDataset(
        train_img_dir, train_mask_dir,
        mean=mean, std=std, augment=True
    )
    val_dataset = PlaygroundDataset(
        val_img_dir, val_mask_dir,
        mean=mean, std=std, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
