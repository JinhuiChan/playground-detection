"""
Step 2: 瓦片切分 — 将大图+掩膜切为256x256训练瓦片，处理正负样本平衡
"""
import os
import sys
import rasterio
import numpy as np
from tqdm import tqdm
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    TIF_PATH, MASK_PATH, TILE_SIZE, TILE_STRIDE,
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
    TRAIN_RATIO, MAX_NEGATIVE_RATIO, POSITIVE_CLASS, ensure_dirs
)


def extract_tiles(image_path, mask_path, tile_size=256, stride=256):
    """
    从大图和掩膜中提取瓦片

    Args:
        image_path: TIF影像路径
        mask_path: 掩膜路径
        tile_size: 瓦片大小
        stride: 滑动步长

    Returns:
        positive_tiles: [(img_tile, mask_tile, row, col), ...] 含操场的瓦片
        negative_tiles: [(img_tile, mask_tile, row, col), ...] 纯背景瓦片
    """
    print(f"\n[1/3] 读取影像和掩膜...")
    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as msk_src:
        img_array = img_src.read()  # (4, H, W)
        mask_array = msk_src.read(1)  # (H, W)
        H, W = mask_array.shape

        # 计算瓦片网格
        n_rows = (H - tile_size) // stride + 1
        n_cols = (W - tile_size) // stride + 1
        total_tiles = n_rows * n_cols
        print(f"  影像尺寸: {W}x{H}")
        print(f"  瓦片网格: {n_rows}行 x {n_cols}列 = {total_tiles} 个瓦片")

        positive_tiles = []
        negative_tiles = []

        print(f"\n[2/3] 提取瓦片中...")
        # 使用滑动窗口
        with tqdm(total=total_tiles, desc="  扫描瓦片") as pbar:
            for r in range(n_rows):
                for c in range(n_cols):
                    row_start = r * stride
                    col_start = c * stride

                    img_tile = img_array[:, row_start:row_start + tile_size,
                                          col_start:col_start + tile_size]
                    mask_tile = mask_array[row_start:row_start + tile_size,
                                           col_start:col_start + tile_size]

                    # 跳过边界不完整的瓦片
                    if img_tile.shape[1] != tile_size or img_tile.shape[2] != tile_size:
                        pbar.update(1)
                        continue

                    # 判断是否含操场像素
                    has_positive = np.any(mask_tile == POSITIVE_CLASS)
                    tile_info = (img_tile.copy(), mask_tile.copy(), r, c)

                    if has_positive:
                        positive_tiles.append(tile_info)
                    else:
                        negative_tiles.append(tile_info)

                    pbar.update(1)

    print(f"\n[3/3] 瓦片统计:")
    print(f"  正样本(含操场): {len(positive_tiles)}")
    print(f"  负样本(纯背景): {len(negative_tiles)}")
    return positive_tiles, negative_tiles


def split_and_save(positive_tiles, negative_tiles):
    """
    划分训练/验证集并保存为.npy文件

    策略:
    1. 保留所有正样本
    2. 负样本按比例采样
    3. 按TRAIN_RATIO划分
    """
    # 负样本采样
    num_pos = len(positive_tiles)
    max_neg = int(num_pos * MAX_NEGATIVE_RATIO)
    if len(negative_tiles) > max_neg:
        random.seed(42)
        negative_tiles = random.sample(negative_tiles, max_neg)
        print(f"\n  负样本采样至: {len(negative_tiles)} (比例 {MAX_NEGATIVE_RATIO}:1)")

    # 打乱
    random.seed(42)
    random.shuffle(positive_tiles)
    random.shuffle(negative_tiles)

    # 划分
    n_pos_train = int(len(positive_tiles) * TRAIN_RATIO)
    n_neg_train = int(len(negative_tiles) * TRAIN_RATIO)

    train_pos = positive_tiles[:n_pos_train]
    val_pos = positive_tiles[n_pos_train:]
    train_neg = negative_tiles[:n_neg_train]
    val_neg = negative_tiles[n_neg_train:]

    train_tiles = train_pos + train_neg
    val_tiles = val_pos + val_neg

    # 再次打乱
    random.shuffle(train_tiles)
    random.shuffle(val_tiles)

    print(f"\n  数据集划分:")
    print(f"    训练集: {len(train_tiles)} ({len(train_pos)}正 + {len(train_neg)}负)")
    print(f"    验证集: {len(val_tiles)} ({len(val_pos)}正 + {len(val_neg)}负)")

    # 保存
    print(f"\n  保存训练集瓦片...")
    for i, (img, msk, r, c) in enumerate(tqdm(train_tiles, desc="    训练集")):
        np.save(os.path.join(TRAIN_IMG_DIR, f"tile_{i:05d}_img.npy"), img)
        np.save(os.path.join(TRAIN_MASK_DIR, f"tile_{i:05d}_mask.npy"), msk)

    print(f"  保存验证集瓦片...")
    for i, (img, msk, r, c) in enumerate(tqdm(val_tiles, desc="    验证集")):
        np.save(os.path.join(VAL_IMG_DIR, f"tile_{i:05d}_img.npy"), img)
        np.save(os.path.join(VAL_MASK_DIR, f"tile_{i:05d}_mask.npy"), msk)

    return len(train_tiles), len(val_tiles)


def main():
    ensure_dirs()

    if not os.path.exists(TIF_PATH):
        print(f"[ERROR] 找不到TIF: {TIF_PATH}")
        print("  请确保已运行 rasterize_labels.py 或将TIF放在正确路径")
        sys.exit(1)

    if not os.path.exists(MASK_PATH):
        print(f"[ERROR] 找不到掩膜: {MASK_PATH}")
        print("  请先运行 rasterize_labels.py 生成掩膜")
        sys.exit(1)

    try:
        positive_tiles, negative_tiles = extract_tiles(
            TIF_PATH, MASK_PATH, TILE_SIZE, TILE_STRIDE
        )

        if len(positive_tiles) == 0:
            print("\n[ERROR] 没有找到任何含操场的瓦片！请检查标注数据。")
            sys.exit(1)

        n_train, n_val = split_and_save(positive_tiles, negative_tiles)

        print(f"\n{'='*50}")
        print(f"[SUCCESS] 瓦片切分完成！")
        print(f"  训练集: {n_train} 个瓦片 -> {TRAIN_IMG_DIR}")
        print(f"  验证集: {n_val} 个瓦片 -> {VAL_IMG_DIR}")
        print(f"{'='*50}")

    except Exception as e:
        print(f"[ERROR] 瓦片切分失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
