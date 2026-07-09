# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

遥感影像操场检测 — 基于珞珈三号II MSS卫星影像（4波段: R/G/B/NIR, 0.5m分辨率, ~29000×29000像素）的语义分割任务。75个人工标注的矢量多边形作为真值标签，输出二值掩膜区分操场与背景。

## 架构

```
矢量多边形 (GDB) ──► rasterize_labels.py ──► label_mask.tif (与TIF对齐)
                                                                     │
TIF + 掩膜 ──► tile_dataset.py ──► 256×256 .npy瓦片 ──► dataset.py ──► train.py
                                                                              │
                                                              predict.py ◄────┘
```

**数据流**: `rasterize_labels.py` → `tile_dataset.py` → `dataset.py` → `train.py` → `predict.py`

**关键设计决策**:
- 两种模型：`"smp"`（U-Net+ResNet34，依赖segmentation_models_pytorch/torchvision，NPU不兼容）和 `"lightweight"`（手写U-Net，纯PyTorch，NPU可用）
- `model.py` 中 `get_device()` 自动检测设备：昇腾NPU → NVIDIA CUDA → CPU
- 损失函数：Dice Loss + BCE Loss 组合（50:50权重），`config.py` 中可调
- 评价指标：`SegmentationMetrics` 类累积 TP/FP/FN/TN 计算 IoU/Precision/Recall/F1
- 类别不平衡：`tile_dataset.py` 保留全部正样本瓦片，负样本按 `MAX_NEGATIVE_RATIO` 随机采样

## 双环境

| 用途 | 系统 | 路径 |
|------|------|------|
| 数据预处理 | Windows本地 | `C:\Users\Lenovo\Desktop\ssh\Final assignment\` |
| 训练/评估 | Linux服务器(昇腾910ProB NPU) | `/home/u2024302131048/myWork/final_assignment/` |

**服务器连接**: `ssh u2024302131048@10.105.103.136`

## 常用命令

### 数据预处理（Windows本地）
```bash
cd "C:\Users\Lenovo\Desktop\ssh\Final assignment\final_assignment"
python rasterize_labels.py    # 矢量→栅格掩膜（输出: ../output/label_mask.tif）
python tile_dataset.py        # 切256×256瓦片 → ../output/data/{train,val}/
```

### 训练（Linux服务器 — NPU）
```bash
ssh u2024302131048@10.105.103.136
cd /home/u2024302131048/myWork/final_assignment/
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 必须执行！否则NPU不可用

# 守护进程训练（断网不中断）:
python3 launcher.py

# 或直接运行:
python3 train.py --model_type lightweight --batch_size 16 --epochs 100 \
  --train_img_dir ./data/train/images --train_mask_dir ./data/train/masks \
  --val_img_dir ./data/val/images --val_mask_dir ./data/val/masks

# 查看进度: tail -f train.log  或  npu-smi info
```

### 评估（Linux服务器）
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 predict.py --mode eval --model_type lightweight --checkpoint ./checkpoints/best_model.pth
```

### 上传代码/数据到服务器
```bash
scp "C:\Users\Lenovo\Desktop\ssh\Final assignment\final_assignment\"*.py u2024302131048@10.105.103.136:/home/u2024302131048/myWork/final_assignment/
scp -r "C:\Users\Lenovo\Desktop\ssh\Final assignment\output\data" u2024302131048@10.105.103.136:/home/u2024302131048/myWork/final_assignment/
```

## 关键注意事项

1. **昇腾NPU环境必须source**: 任何使用 `torch_npu` 的Python命令前必须先执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，否则 `libhccl.so` 加载失败。

2. **`segmentation_models_pytorch` / `torchvision` 与NPU不兼容**: `torchvision::nms` 算子在NPU版PyTorch中不存在。在昇腾服务器上必须使用 `--model_type lightweight`。`smp` 模式仅适用于CUDA或CPU环境。

3. **GBK终端编码问题**: Windows中文终端使用GBK编码。`print` 语句中避免使用 ✓、❌、⚠ 等Unicode字符，改用 `[OK]`、`[ERROR]`、`[WARN]`。

4. **SSH训练必须分离进程**: 使用 `launcher.py`（内部用 `subprocess.Popen` + `start_new_session=True` 创建脱离终端的子进程）。直接在SSH命令中使用 `&` 后台符号会被工具拦截。

5. **4波段输入适配**: 使用ImageNet预训练编码器（3通道）时，`_adapt_input_channels()` 复制RGB权重并将三个通道的均值赋给NIR通道。仅 `smp` 模式需要此处理。

6. **`pin_memory=True` 在NPU上报警告**: DataLoader 会警告 pinned memory 在非CUDA加速器上不可用，无害可忽略。

7. **训练最佳结果**: 学习率 3e-5 + 梯度裁剪(max_norm=1.0)，最佳IoU 0.4237（epoch 86/100），模型保存在 `checkpoints/best_model.pth`。

## 文件功能

| 文件 | 功能 |
|------|------|
| `config.py` | 全局路径、超参数、瓦片大小、波段归一化常量 |
| `rasterize_labels.py` | GDAL/rasterio: 矢量GDB多边形 → 单波段栅格掩膜 |
| `tile_dataset.py` | 滑动窗口提取瓦片，正负样本平衡 |
| `dataset.py` | `PlaygroundDataset` 类，albumentations数据增强 |
| `model.py` | `LightweightUNet`、`create_unet_smp()`、`get_device()`、`create_model()` |
| `train.py` | `DiceLoss`、`CombinedLoss`，训练循环含早停和checkpoint |
| `metrics.py` | `SegmentationMetrics` — 累积TP/FP/FN/TN → IoU/Precision/Recall/F1 |
| `predict.py` | 滑动窗口推理、验证集评估、结果可视化 |
| `launcher.py` | 写bash脚本并以脱离终端的子进程启动，用于SSH安全训练 |
