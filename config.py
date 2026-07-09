"""
全局配置文件 — 遥感影像操场检测语义分割

所有路径默认使用相对路径，克隆仓库后开箱即用。
Windows预处理路径仅在本地存在时自动使用。
"""
import os

# ==================== 项目根目录 ====================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 原始数据路径 (仅Windows预处理用) ====================
_TIF_WIN = r"C:\Users\Lenovo\Desktop\ssh\Final assignment\LJ3II_MSS_E114.30_N30.53_20240916_L1A_020_GS.tif"
_GDB_WIN = r"C:\Users\Lenovo\Desktop\ssh\Final assignment\label\label.gdb"
_GDB_LAYER = "label_ExportFeatures"

# 自动检测：如果Windows路径存在则使用，否则回退到相对路径
if os.path.exists(_TIF_WIN):
    TIF_PATH = _TIF_WIN
    GDB_PATH = _GDB_WIN
    GDB_LAYER = _GDB_LAYER
    OUTPUT_DIR = r"C:\Users\Lenovo\Desktop\ssh\Final assignment\output"
else:
    TIF_PATH = None
    GDB_PATH = None
    GDB_LAYER = None
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    # 如果克隆了仓库但没有TIF/GDB，这些脚本会报友好提示
    print("[INFO] 未检测到原始TIF/GDB文件。如需重新生成数据集，请修改 config.py 中的路径。")

# ==================== 数据集路径 (相对路径，开箱即用) ====================
TILE_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_IMG_DIR = os.path.join(TILE_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(TILE_DIR, "train", "masks")
VAL_IMG_DIR = os.path.join(TILE_DIR, "val", "images")
VAL_MASK_DIR = os.path.join(TILE_DIR, "val", "masks")

# ==================== 输出路径 ====================
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

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
# ImageNet 均值/标准差 (RGB前三通道 + NIR均值)
MEAN = [0.485, 0.456, 0.406, 0.350]
STD  = [0.229, 0.224, 0.225, 0.150]

# ==================== 模型参数 ====================
MODEL_NAME = "unet"
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 4
NUM_CLASSES = 1
ACTIVATION = "sigmoid"

# ==================== 训练参数 ====================
BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LR_PATIENCE = 10
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 20
NUM_WORKERS = 4

# ==================== 损失函数权重 ====================
DICE_WEIGHT = 0.5
BCE_WEIGHT = 0.5

# ==================== 自动创建目录 ====================
def ensure_dirs():
    """创建所有输出目录"""
    for d in [CHECKPOINT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
