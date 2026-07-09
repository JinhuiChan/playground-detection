"""
语义分割评价指标: IoU, Precision, Recall, F1-score
"""
import numpy as np


class SegmentationMetrics:
    """累积计算语义分割评价指标"""

    def __init__(self, threshold=0.5, num_classes=1, smooth=1e-6):
        """
        Args:
            threshold: 二分类概率阈值
            num_classes: 类别数 (1表示二分类sigmoid, >1表示多分类softmax)
            smooth: 平滑因子防止除零
        """
        self.threshold = threshold
        self.num_classes = num_classes
        self.smooth = smooth
        self.reset()

    def reset(self):
        """重置累积计数器"""
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred, target):
        """
        更新累积统计量
        Args:
            pred: 模型预测 (B, C, H, W) 或 (B, H, W), 概率值 [0,1]
            target: 真值 (B, H, W), 二值 [0,1]
        """
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)  # (B, 1, H, W) -> (B, H, W)

        # 二值化预测
        pred_bin = (pred > self.threshold).astype(np.uint8)
        target = target.astype(np.uint8)

        # 展平计算
        pred_flat = pred_bin.flatten()
        target_flat = target.flatten()

        self.tp += np.sum((pred_flat == 1) & (target_flat == 1))
        self.fp += np.sum((pred_flat == 1) & (target_flat == 0))
        self.fn += np.sum((pred_flat == 0) & (target_flat == 1))
        self.tn += np.sum((pred_flat == 0) & (target_flat == 0))

    def get_metrics(self):
        """计算当前指标"""
        tp, fp, fn = self.tp, self.fp, self.fn
        smooth = self.smooth

        iou = (tp + smooth) / (tp + fp + fn + smooth)
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        f1 = 2 * precision * recall / (precision + recall + smooth)
        accuracy = (tp + self.tn + smooth) / (tp + fp + fn + self.tn + smooth)

        return {
            "IoU": round(iou, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Accuracy": round(accuracy, 4),
        }

    def get_report(self):
        """生成文本报告"""
        m = self.get_metrics()
        report = (
            f"{'='*45}\n"
            f"  Semantic Segmentation Evaluation Report\n"
            f"{'='*45}\n"
            f"  IoU (Jaccard):       {m['IoU']:.4f}\n"
            f"  Precision:           {m['Precision']:.4f}\n"
            f"  Recall:              {m['Recall']:.4f}\n"
            f"  F1-Score:            {m['F1']:.4f}\n"
            f"  Pixel Accuracy:      {m['Accuracy']:.4f}\n"
            f"{'='*45}\n"
            f"  TP: {self.tp}  FP: {self.fp}  FN: {self.fn}  TN: {self.tn}\n"
            f"{'='*45}"
        )
        return report


def compute_iou(pred, target, smooth=1e-6):
    """快速计算单个batch的IoU"""
    pred_bin = (pred > 0.5).astype(np.uint8).flatten()
    target_flat = target.astype(np.uint8).flatten()
    intersection = np.sum(pred_bin & target_flat)
    union = np.sum(pred_bin | target_flat)
    return (intersection + smooth) / (union + smooth)
