# 操场检测 — 遥感影像语义分割

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 数据预处理 (本地Windows)
```bash
# Step 1: 矢量转栅格掩膜
python rasterize_labels.py

# Step 2: 切分训练瓦片
python tile_dataset.py
```

### 3. 训练 (本地CPU或远程GPU服务器)
```bash
# 使用预训练U-Net (推荐, 需要GPU)
python train.py --model_type smp --batch_size 16 --epochs 100

# 使用手写轻量U-Net (CPU友好)
python train.py --model_type lightweight --batch_size 4 --epochs 100

# 完整参数
python train.py --help
```

### 4. 评估
```bash
# 验证集评估 + 可视化
python predict.py --mode eval

# 大图滑动窗口推理
python predict.py --mode full_image

# 仅可视化
python predict.py --mode vis
```

## 输出文件结构
```
output/
├── label_mask.tif               # 栅格化掩膜
├── data/
│   ├── train/images/            # 训练瓦片 (*.npy)
│   ├── train/masks/
│   ├── val/images/              # 验证瓦片
│   └── val/masks/
├── checkpoints/
│   ├── best_model.pth           # 最佳模型
│   └── training_history.json   # 训练历史
├── logs/                        # TensorBoard日志
├── visualizations/              # 预测可视化
└── prediction_full.tif          # 大图预测结果
```
