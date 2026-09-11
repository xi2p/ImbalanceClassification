# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for 长尾分布分类
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: 类别权重。可以是 float（对所有类别相同）、list/Tensor（每类不同），
               传 None 则不使用类别加权。
        gamma: 聚焦参数，越大越关注难分样本（常用 1.0~3.0，长尾任务建议 2.0）
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 逐样本 CE（如果 alpha 是 Tensor 会作为 weight 参与）
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            reduction='none'
        )
        # p_t = exp(-CE)，注意：这里用 exp(-CE) 表示预测正确类的概率
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(labels, num_classes, mode='inv_sqrt'):
    """
    根据训练集标签分布计算类别权重，用于 Focal Loss 的 alpha。

    mode:
        'inv'      : w_c = N / (C * n_c)          —— 完全反比，最激进
        'inv_sqrt' : w_c = 1 / sqrt(n_c)，再归一化 —— 长尾常用，较温和（推荐）
        'none'     : 返回 None
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # 防止除零

    if mode == 'none':
        return None
    elif mode == 'inv':
        weights = counts.sum() / (num_classes * counts)
    elif mode == 'inv_sqrt':
        weights = 1.0 / np.sqrt(counts)
        weights = weights / weights.mean()   # 归一化，均值=1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.tensor(weights, dtype=torch.float32)
