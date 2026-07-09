# 实验报告：基于深度学习的遥感影像操场检测

> **武汉大学遥感信息工程学院** | 遥感深度学习实践 | 2026年

---

## 摘要

本实验基于珞珈三号II MSS遥感影像（4波段：Red/Green/Blue/NIR，空间分辨率约0.5m），采用U-Net语义分割模型，对武汉市区域的操场（运动场）进行自动化提取。标注数据包含75个矢量多边形面要素，经栅格化、瓦片切分后构建二分类语义分割数据集（392训练/100验证，256×256像素）。使用手写LightweightUNet模型在昇腾910ProB NPU上训练100轮，最终验证集**IoU=0.4237，F1=0.5952，Precision=0.7473，Recall=0.4945**。实验表明深度学习在操场检测任务上具有可行性，但受限于标注数量，召回率仍有较大提升空间。

---

## 一、任务概述

### 1.1 任务目标

基于珞珈三号II MSS多光谱遥感影像和人工标注的操场矢量数据，设计并训练一个深度学习语义分割模型，实现影像中操场的自动化检测。

### 1.2 数据概况

| 项目 | 详情 |
|------|------|
| 遥感影像 | 珞珈三号II MSS，28936×29425像素，4波段(R/G/B/NIR)，uint8 |
| 空间分辨率 | ~0.5m/像素 |
| 坐标系 | EPSG:4326 |
| 覆盖范围 | 武汉市，约14.7km×16.7km |
| 标注数据 | 75个MultiPolygon面要素，标注类别为"操场" |
| 标注格式 | ESRI Geodatabase (.gdb) |

---

## 二、数据预处理

### 2.1 矢量栅格化

使用`rasterio.features.rasterize`将75个矢量多边形转为与TIF影像完全对齐的二值掩膜。操场像素值=1，背景=0。掩膜尺寸与原始影像一致（29425×28936像素），操场像素共2,922,817个，占总像素的0.34%。

```
矢量多边形 (75个) → rasterio.rasterize() → label_mask.tif
  output_shape: (28936, 29425)  ← 与TIF完全对齐
  fill=0, 操场=1,  dtype=uint8
```

### 2.2 瓦片切分与数据集划分

使用256×256滑动窗口（stride=256，无重叠）从大图和掩膜中切分瓦片。共生成12,882个瓦片，其中仅246个（1.9%）含操场像素。

**正负样本平衡策略**：保留全部246个含操场的正样本，从12,636个背景瓦片中随机采样等量（246个）作为负样本，形成1:1平衡数据集。

| 数据集 | 正样本 | 负样本 | 总计 |
|--------|--------|--------|------|
| 训练集 | 196 | 196 | **392** |
| 验证集 | 50 | 50 | **100** |

数据以`.npy`格式保存：训练图像为`(4, 256, 256)` float32数组，掩膜为`(256, 256)` uint8二值数组。

---

## 三、模型设计

### 3.1 模型架构

采用**LightweightUNet**模型，经典的编码器-解码器结构配合跳跃连接：

```
输入 (4, 256, 256)
  │
  ├─ Encoder: DoubleConv(4→64) → MaxPool → DoubleConv(64→128) → MaxPool
  │           → DoubleConv(128→256) → MaxPool → DoubleConv(256→512) → MaxPool
  │
  ├─ Bottleneck: DoubleConv(512→1024)
  │
  └─ Decoder: ConvTranspose2d + DoubleConv (×4层)
     跳跃连接 ← 编码器对应层特征图拼接
  │
  └─ Conv2d(64→1) → Sigmoid
  │
输出 (1, 256, 256) 概率图 [0,1]
```

| 参数 | 值 |
|------|-----|
| 总参数量 | 31,038,209 |
| 输入通道 | 4 (R/G/B/NIR) |
| 输出 | 1 (Sigmoid概率) |
| 基础特征数 | [64, 128, 256, 512] |
| DoubleConv | (Conv3×3→BN→ReLU)×2 |

### 3.2 为什么不用预训练模型

尝试了`segmentation_models_pytorch`的U-Net+ResNet34（ImageNet预训练），但`torchvision::nms`算子在昇腾NPU版PyTorch中不存在，导致无法加载。因此采用纯PyTorch手写U-Net，在NPU上可直接运行。

### 3.3 4波段输入处理

归一化策略：各波段分别标准化

```python
mean = [0.485, 0.456, 0.406, 0.350]  # RGB(ImageNet) + NIR(均值)
std  = [0.229, 0.224, 0.225, 0.150]
image = (image / 255.0 - mean) / std
```

---

## 四、训练配置

### 4.1 超参数

| 超参数 | 值 | 说明 |
|--------|-----|------|
| Batch Size | 16 | NPU单卡可承受 |
| Epochs | 100 | 实际86轮达到最优 |
| Learning Rate | 3e-5 | 低于默认1e-4，防collapse |
| Optimizer | AdamW | weight_decay=1e-5 |
| LR Scheduler | ReduceLROnPlateau | mode=max, factor=0.5, patience=10 |
| Early Stopping | patience=20 | 未触发(最佳在epoch 86) |
| Gradient Clipping | max_norm=1.0 | 防止NPU数值不稳定导致梯度爆炸 |
| 损失函数 | 0.5×Dice + 0.5×BCE | 组合损失处理正负样本不平衡 |

### 4.2 数据增强

训练时使用`albumentations`进行在线增强：

| 增强方式 | 概率 |
|----------|------|
| 水平翻转 | 0.5 |
| 垂直翻转 | 0.5 |
| 随机90°旋转 | 0.5 |
| 亮度/对比度调整 | 0.3 |
| 高斯模糊 | 0.2 |
| 高斯噪声 | 0.2 |

验证集仅做归一化，不做增强。

### 4.3 训练环境

| 项目 | 配置 |
|------|------|
| 服务器 | 华为昇腾910ProB NPU ×1 |
| CPU | 16核 / 64GB内存 |
| PyTorch | 2.4.0 (NPU适配版) |
| torch_npu | 2.4.0 |
| CANN | 8.3.RC1 |

---

## 五、实验结果

### 5.1 验证集评估

| 指标 | 值 | 说明 |
|------|-----|------|
| **IoU** | **0.4237** | 预测与真值重叠42.4% |
| **Precision** | **0.7473** | 预测为操场的像素中74.7%正确 |
| **Recall** | **0.4945** | 真实操场像素中找回49.5% |
| **F1-Score** | **0.5952** | 综合衡量精确率和召回率 |
| Pixel Accuracy | 0.9549 | 整体像素准确率（背景多，虚高） |

混淆矩阵统计：

|  | 预测=正 | 预测=负 |
|--|---------|---------|
| **真值=正** | TP: 217,501 | FN: 222,328 |
| **真值=负** | FP: 73,541 | TN: 6,040,230 |

### 5.2 训练曲线

![训练曲线](output/visualizations/training_curves.png)

**Loss曲线（左）**：训练损失和验证损失均稳定下降。前30轮收敛较快，之后趋于平缓。验证损失在epoch 86附近达到最低点后略有回升，表明模型开始过拟合。

**IoU/F1曲线（右）**：两个指标整体呈上升趋势，从epoch 1的IoU≈0.07逐步提升至epoch 86的IoU≈0.42。前期（epoch 1-30）增长缓慢（0.07→0.19），中期（epoch 30-60）加速提升（0.19→0.33），后期（epoch 60-86）继续上升至峰值（0.42），之后小幅回落至0.35左右。

![训练历史](output/visualizations/training_history.png)

关键节点数据：

| 阶段 | Epoch | Val IoU | Val F1 |
|------|-------|---------|--------|
| 初始学习 | 1 | 0.074 | 0.137 |
| 缓慢提升 | 14 | 0.107 | 0.193 |
| 突破 | 29 | 0.194 | 0.324 |
| 加速提升 | 58 | 0.325 | 0.491 |
| **最佳** | **86** | **0.424** | **0.595** |
| 过拟合回落 | 100 | 0.350 | 0.519 |

训练100轮总耗时约8分钟（NPU加速），每轮约4.5秒。

### 5.3 结果可视化

### 5.3 数据预览

![正样本预览](output/visualizations/raw_positive_samples.png)

![正样本分布](output/visualizations/positive_distribution.png)

---

## 六、结果分析

### 6.1 优点

- **精确率74.7%**：模型预测为操场的区域中，四分之三确实为操场，误检较少。
- **训练稳定性**：梯度裁剪和低学习率策略有效避免了NPU上的训练崩溃（初次尝试时IoU从0.08 collapse到0）。
- **NPU适配**：成功在昇腾NPU上完成训练，验证了国产AI芯片在遥感语义分割任务上的可行性。

### 6.2 不足与原因分析

- **召回率仅49.5%**：约一半操场像素被漏检。可能原因：
  - 标注样本有限（仅75个多边形），不足以覆盖操场形状/颜色的多样性
  - 操场边缘像素模糊，模型倾向于保守预测（高精确率、低召回率）
  - 正负样本1:1平衡策略可能丢弃了过多有用负样本的上下文信息
- **IoU 0.42仅为中等水平**：距离实用化（IoU>0.7）仍有差距
- **模型未使用预训练权重**：受NPU兼容性限制，31M参数从零开始训练

### 6.3 改进方向

1. **增加标注数据**：补充小型操场、不规则操场的标注
2. **模型优化**：解决torchvision/NPU兼容问题后使用预训练U-Net+ResNet34
3. **损失函数调整**：增大Dice Loss权重（当前5:5），提升召回率
4. **后处理**：使用CRF或形态学操作优化预测边界
5. **更大输入尺寸**：256×256可能不足以捕获操场完整上下文，尝试512×512

---

## 七、代码结构与使用说明

### 7.1 项目文件

| 文件 | 功能 |
|------|------|
| `config.py` | 全局配置，路径和超参数 |
| `rasterize_labels.py` | 矢量GDB多边形→栅格掩膜 |
| `tile_dataset.py` | 大图切分256×256瓦片 |
| `dataset.py` | PyTorch Dataset + albumentations增强 |
| `model.py` | LightweightUNet + 设备自动检测(NPU/CUDA/CPU) |
| `train.py` | Dice+BCE组合损失，训练循环含早停 |
| `metrics.py` | SegmentationMetrics (IoU/Precision/Recall/F1) |
| `predict.py` | 验证集评估与大图滑动窗口推理 |

### 7.2 快速复现

```bash
git clone https://github.com/JinhuiChan/playground-detection.git
cd playground-detection
pip install -r requirements.txt
python train.py --model_type lightweight --batch_size 16 --epochs 100
python predict.py --mode eval --model_type lightweight --checkpoint ./checkpoints/best_model.pth
```

### 7.3 设备支持

模型自动检测运行设备：**昇腾NPU > NVIDIA CUDA > CPU**。在NPU环境下需先`source` CANN环境变量。

---

## 八、结论

本实验完成了基于深度学习的遥感影像操场检测全流程，从矢量标注栅格化、瓦片数据集构建到U-Net模型训练与评估。最终IoU达到0.42，验证了语义分割方法在操场提取任务上的可行性。受限于标注数量和NPU环境下的模型选择，结果有较大提升空间。后续工作重点在于扩充标注数据和解决预训练模型的NPU适配问题。

---

*实验代码及数据集开源在：https://github.com/JinhuiChan/playground-detection*
