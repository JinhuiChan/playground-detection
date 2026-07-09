"""
Step 1: 矢量栅格化 — 将操场矢量多边形转为与TIF对齐的二值掩膜
"""
import os
import sys
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from tqdm import tqdm

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import TIF_PATH, GDB_PATH, GDB_LAYER, MASK_PATH, OUTPUT_DIR, ensure_dirs


def rasterize_labels(tif_path, gdb_path, gdb_layer, output_path):
    """
    将GDB中的矢量多边形栅格化为与TIF完全对齐的二值掩膜

    Args:
        tif_path: 参考TIF路径
        gdb_path: GDB路径
        gdb_layer: GDB图层名
        output_path: 输出掩膜路径
    """
    print(f"[1/4] 读取参考TIF: {tif_path}")
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        out_shape = (src.height, src.width)
        print(f"  TIF尺寸: {src.width} x {src.height}")
        print(f"  CRS: {crs}")
        print(f"  分辨率: {src.res}")

    print(f"\n[2/4] 读取矢量数据: {gdb_path} -> {gdb_layer}")
    gdf = gpd.read_file(gdb_path, layer=gdb_layer)
    print(f"  要素数量: {len(gdf)}")
    print(f"  矢量CRS: {gdf.crs}")
    print(f"  几何类型: {gdf.geometry.type.unique()}")

    # 确保CRS一致
    if gdf.crs != crs:
        print(f"  ⚠ CRS不一致，转换中...")
        gdf = gdf.to_crs(crs)

    # 检查数据有效性
    gdf = gdf[gdf.geometry.is_valid].copy()
    print(f"  有效要素数: {len(gdf)}")

    if len(gdf) == 0:
        raise ValueError("没有有效的矢量要素！")

    print(f"\n[3/4] 栅格化中... (输出尺寸: {out_shape[1]}x{out_shape[0]})")

    # 构建 (geometry, value) 对
    # 所有操场多边形赋值为1
    shapes = [(geom, 1) for geom in gdf.geometry]

    # 执行栅格化
    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,             # 背景=0
        dtype=np.uint8,
        all_touched=True    # 被多边形触碰的像素都设为1
    )

    positive_pixels = np.sum(mask == 1)
    total_pixels = mask.size
    print(f"  操场像素数: {positive_pixels:,}")
    print(f"  背景像素数: {total_pixels - positive_pixels:,}")
    print(f"  操场占比: {100 * positive_pixels / total_pixels:.4f}%")

    # 更新profile为单波段uint8
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress='lzw',      # LZW压缩减少文件体积
        tiled=True,
        blockxsize=256,
        blockysize=256
    )

    print(f"\n[4/4] 保存掩膜到: {output_path}")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mask, 1)

    # 快速验证
    print(f"\n  [OK] 掩膜尺寸: {mask.shape[1]}x{mask.shape[0]}")
    print(f"  [OK] 唯一值: {np.unique(mask)}")
    print(f"  [OK] 操场面积: {positive_pixels} 像素")

    return mask


def main():
    ensure_dirs()

    if not os.path.exists(TIF_PATH):
        print(f"[ERROR] 找不到TIF文件: {TIF_PATH}")
        sys.exit(1)

    if not os.path.exists(GDB_PATH):
        print(f"[ERROR] 找不到GDB文件: {GDB_PATH}")
        sys.exit(1)

    try:
        mask = rasterize_labels(TIF_PATH, GDB_PATH, GDB_LAYER, MASK_PATH)
        print(f"\n{'='*50}")
        print(f"[SUCCESS] 栅格化完成！掩膜已保存到: {MASK_PATH}")
        print(f"{'='*50}")
    except Exception as e:
        print(f"[ERROR] 栅格化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
