"""
Step 4: 语义分割模型 — U-Net (支持 ResNet/EfficientNet 编码器 和 手写轻量U-Net)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 方案A: segmentation_models_pytorch 封装 ====================

def create_unet_smp(encoder_name="resnet34", encoder_weights="imagenet",
                    in_channels=4, num_classes=1):
    """
    使用 segmentation_models_pytorch 创建U-Net模型

    Args:
        encoder_name: 编码器名称 (resnet34, efficientnet-b0, etc.)
        encoder_weights: 预训练权重
        in_channels: 输入通道数 (4 = R/G/B/NIR)
        num_classes: 输出类别数 (1 = 二分类)

    Returns:
        model
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "请安装 segmentation_models_pytorch: "
            "pip install segmentation-models-pytorch"
        )

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation='sigmoid' if num_classes == 1 else None,
    )

    # 如果使用预训练权重且in_channels != 3, 需要适配第一层卷积
    if encoder_weights is not None and in_channels != 3:
        _adapt_input_channels(model, encoder_name, in_channels)

    return model


def _adapt_input_channels(model, encoder_name, in_channels):
    """
    将预训练模型的3通道输入层适配为N通道
    策略: 复制RGB通道均值到新通道 (如NIR)
    """
    # 获取第一层卷积
    if hasattr(model.encoder, 'conv1'):
        old_conv = model.encoder.conv1
    elif hasattr(model.encoder, '_in_channels'):
        # 某些encoder的结构
        old_conv = None
    else:
        # 尝试通用方法
        for name, module in model.encoder.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                old_conv = module
                break
        else:
            print("  [警告] 无法找到第一层卷积，跳过输入通道适配")
            return

    if old_conv is None:
        print("  [警告] 无法找到第一层卷积，跳过输入通道适配")
        return

    old_weight = old_conv.weight.data  # (out_channels, 3, k, k)

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    # 前3通道复制预训练权重, 后续通道复制RGB均值或随机初始化
    new_weight = torch.zeros(new_conv.weight.data.shape)
    new_weight[:, :3, :, :] = old_weight

    # 对额外通道 (如NIR), 使用RGB通道权重的均值
    for c in range(3, in_channels):
        new_weight[:, c, :, :] = old_weight.mean(dim=1)

    new_conv.weight.data = new_weight
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data

    # 替换conv层
    if hasattr(model.encoder, 'conv1'):
        model.encoder.conv1 = new_conv
    else:
        # 找到并替换
        for name, module in model.encoder.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                parent = model.encoder
                for part in name.split('.')[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, name.split('.')[-1], new_conv)
                break

    print(f"  [适配] 输入通道: 3 -> {in_channels}")


# ==================== 方案B: 手写轻量U-Net (CPU友好) ====================

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LightweightUNet(nn.Module):
    """
    手写轻量U-Net (~7M参数, CPU可训练)
    4层编码器-解码器 + 跳跃连接
    """

    def __init__(self, in_channels=4, num_classes=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # 编码器 (下采样路径)
        in_ch = in_channels
        for feat in features:
            self.encoder.append(DoubleConv(in_ch, feat))
            in_ch = feat

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 解码器 (上采样路径)
        for feat in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feat * 2, feat))

        # 输出层
        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # 编码 + 保存跳跃连接
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈
        x = self.bottleneck(x)

        # 解码 + 跳跃连接
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # ConvTranspose2d 上采样
            skip = skip_connections[i // 2]

            # 处理尺寸不匹配
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat([skip, x], dim=1)  # 跳跃连接
            x = self.decoder[i + 1](x)  # DoubleConv

        return torch.sigmoid(self.out_conv(x))


# ==================== 设备自动检测 ====================

def get_device():
    """
    自动检测最佳可用设备: 昇腾NPU > NVIDIA CUDA > CPU
    Returns: torch.device
    """
    # 1. 检测昇腾NPU
    try:
        import torch_npu
        if torch.npu.is_available():
            device = torch.device('npu:0')
            print(f"  [NPU] 检测到昇腾NPU, 设备数: {torch.npu.device_count()}")
            return device
    except ImportError:
        pass
    # 2. 检测NVIDIA CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"  [CUDA] 检测到NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return device
    # 3. 回退到CPU
    device = torch.device('cpu')
    print(f"  [CPU] 未检测到加速器, 使用CPU")
    return device


# ==================== 模型创建工厂 ====================

def create_model(model_type="smp", **kwargs):
    """
    模型工厂函数

    Args:
        model_type: "smp" (推荐), "lightweight" (CPU友好)
        **kwargs: 传递给具体创建函数的参数

    Returns:
        model, device
    """
    device = get_device()
    print(f"  使用设备: {device}")

    if model_type == "smp":
        model = create_unet_smp(
            encoder_name=kwargs.get('encoder_name', 'resnet34'),
            encoder_weights=kwargs.get('encoder_weights', 'imagenet'),
            in_channels=kwargs.get('in_channels', 4),
            num_classes=kwargs.get('num_classes', 1),
        )
    elif model_type == "lightweight":
        model = LightweightUNet(
            in_channels=kwargs.get('in_channels', 4),
            num_classes=kwargs.get('num_classes', 1),
            features=kwargs.get('features', [64, 128, 256, 512]),
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}. 可选: 'smp', 'lightweight'")

    model = model.to(device)

    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型类型: {model_type}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    return model, device
