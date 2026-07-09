"""
全局配置文件 — 遥感影像操场检测语义分割
"""
import os

# ==================== 路径配置 ====================
# 原始数据路径 (Windows本地处理用)
TIF_PATH = r"C:\Users\Lenovo\Desktop\ssh\Final assignment\LJ3II_MSS_E114.30_N30.53_20240916_L1A_020_GS.tif"
GDB_PATH = r"C:\Users\Lenovo\Desktop\ssh\Final assignment\label\label.gdb"
GDB_LAYER = "label_ExportFeatures"

# 中间输出路径
OUTPUT_DIR = r"C:\Users\Lenovo\Desktop\ssh\Final assignment\output"
MASK_PATH = os.path.join(OUTPUT_DIR, "label_mask.tif")

# 瓦片数据集路径
TILE_DIR = os.path.join(OUTPUT_DIR, "data")
TRAIN_IMG_DIR = os.path.join(TILE_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(TILE_DIR, "train", "masks")
VAL_IMG_DIR = os.path.join(TILE_DIR, "val", "images")
VAL_MASK_DIR = os.path.join(TILE_DIR, "val", "masks")

# 模型与日志
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ==================== 服务器路径 (训练用) ====================
SERVER_BASE = "/home/u2024302131048/myWork/final assignment"
SERVER_DATA = os.path.join(SERVER_BASE, "data")
SERVER_CHECKPOINT = os.path.join(SERVER_BASE, "checkpoints")
SERVER_LOG = os.path.join(SERVER_BASE, "logs")

# ==================== 数据处理参数 ====================
TILE_SIZE = 256           # 瓦片大小 (像素)
TILE_STRIDE = 256         # 滑动步长 (=TILE_SIZE 则为无重叠)
POSITIVE_CLASS = 1        # 操场像素值
BACKGROUND_CLASS = 0      # 背景像素值
TRAIN_RATIO = 0.8         # 训练集比例
MAX_NEGATIVE_RATIO = 1.0  # 负样本与正样本的最大比例

# ==================== 影像波段信息 ====================
# 珞珈三号II MSS: Band 1=Red, Band 2=Green, Band 3=Blue, Band 4=NIR
BAND_ORDER = ["Red", "Green", "Blue", "NIR"]
NUM_BANDS = 4
# ImageNet 均值/标准差 (用于RGB三通道, NIR用均值)
MEAN = [0.485, 0.456, 0.406, 0.350]
STD  = [0.229, 0.224, 0.225, 0.150]

# ==================== 模型参数 ====================
MODEL_NAME = "unet"                      # 模型架构
ENCODER_NAME = "resnet34"                # 编码器
ENCODER_WEIGHTS = "imagenet"             # 预训练权重
IN_CHANNELS = 4                          # 输入通道 (R/G/B/NIR)
NUM_CLASSES = 1                          # 输出类别 (1 = 二分类sigmoid输出)
ACTIVATION = "sigmoid"                   # 输出激活函数

# ==================== 训练参数 ====================
BATCH_SIZE = 16                          # 训练批次大小
VAL_BATCH_SIZE = 16                      # 验证批次大小
NUM_EPOCHS = 100                         # 最大训练轮次
LEARNING_RATE = 1e-4                     # 初始学习率
WEIGHT_DECAY = 1e-5                      # 权重衰减
LR_PATIENCE = 10                         # 学习率下降耐心值
LR_FACTOR = 0.5                          # 学习率下降因子
EARLY_STOP_PATIENCE = 20                 # 早停耐心值
NUM_WORKERS = 4                          # 数据加载线程数

# ==================== 损失函数权重 ====================
DICE_WEIGHT = 0.5                        # Dice Loss 权重
BCE_WEIGHT = 0.5                         # BCE Loss 权重

# ==================== 自动创建目录 ====================
def ensure_dirs():
    """创建所有输出目录"""
    for d in [OUTPUT_DIR, TILE_DIR, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
              VAL_IMG_DIR, VAL_MASK_DIR, CHECKPOINT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
