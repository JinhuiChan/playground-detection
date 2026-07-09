#!/bin/bash
# ============================================================
# 昇腾910ProB NPU 环境安装脚本
# 服务器: 10.105.103.136
# ============================================================

echo "============================================"
echo "  操场检测 — 服务器NPU环境安装"
echo "============================================"

# 1. 检查NPU驱动和CANN
echo "[1/5] 检查昇腾环境..."
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "  CANN toolkit 已找到"
else
    echo "  [WARN] 未找到CANN toolkit, 请确认昇腾环境已安装"
fi

npu-smi info 2>/dev/null && echo "  NPU驱动正常" || echo "  [WARN] npu-smi不可用"

# 2. 安装/检查 torch-npu
echo "[2/5] 检查 PyTorch NPU..."
python -c "import torch; import torch_npu; print('  torch:', torch.__version__); print('  torch_npu:', torch_npu.__version__); print('  NPU available:', torch.npu.is_available())" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "  torch_npu 不可用，尝试安装..."
    pip install torch-npu 2>/dev/null || {
        echo "  [ERROR] torch-npu 安装失败"
        echo "  请手动安装: pip install torch-npu"
        echo "  或参考: https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/softwareinst/instg/instg_0003.html"
        exit 1
    }
fi

# 3. 安装Python依赖
echo "[3/5] 安装Python依赖..."
pip install -r requirements_npu.txt -q 2>&1 | tail -3

# 4. 验证安装
echo "[4/5] 验证环境..."
python -c "
import torch
import torch_npu
print('  PyTorch:', torch.__version__)
print('  NPU设备数:', torch.npu.device_count())
print('  NPU可用:', torch.npu.is_available())
if torch.npu.is_available():
    print('  NPU设备名:', torch.npu.get_device_name(0))
    # 测试tensor在NPU上
    x = torch.ones(2, 3).npu()
    print('  NPU tensor test: OK')
print('  环境就绪!')
" || {
    echo "  [ERROR] 环境验证失败"
    exit 1
}

# 5. 创建必要目录
echo "[5/5] 创建目录..."
mkdir -p checkpoints logs

echo ""
echo "============================================"
echo "  环境安装完成!"
echo "  运行训练: python train.py --model_type smp --batch_size 16 --epochs 100"
echo "============================================"
