# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_cb_weights(labels, num_classes, beta=0.9999, normalize=True):
    """
    根据训练集标签分布计算 Class-Balanced 权重：
        w_i = (1 - beta) / (1 - beta^{n_i})

    Args:
        labels: 训练集标签列表 / np.array
        num_classes: 类别数
        beta: CB Loss 的 beta 参数（越大越强调尾部类）
              长尾任务常用 0.99 / 0.999 / 0.9999
        normalize: 是否把权重归一化到均值 = 1（数值更稳定，推荐 True）

    Returns:
        torch.FloatTensor, shape (num_classes,)
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)          # 防止某个类计数为 0 导致除零

    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num    # w_i = (1-beta)/(1-beta^{n_i})

    if normalize:
        weights = weights / weights.sum() * num_classes   # 均值 = 1

    return torch.tensor(weights, dtype=torch.float32)


class CBLoss(nn.Module):
    """
    Class-Balanced Loss (Cui et al., CVPR 2019)

        CB(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    其中 alpha_t 就是 CB 权重 w_y。
    当 gamma = 0 时退化为 CB Softmax Loss；
    当 gamma > 0 时退化为 CB Focal Loss。

    Args:
        samples_per_cls: list / np.array，每个类的样本数（长度 = num_classes）
                        若不传，则必须传 weights
        num_classes: 类别数
        beta: CB 的 beta 参数，默认 0.9999
        gamma: focal 的聚焦参数，0 表示纯 CB Softmax
        reduction: 'mean' | 'sum' | 'none'
        weights: 直接传入预先算好的类别权重（与 samples_per_cls 二选一）
    """

    def __init__(self,
                 samples_per_cls=None,
                 num_classes=None,
                 beta=0.9999,
                 gamma=2.0,
                 reduction='mean',
                 weights=None):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

        if weights is not None:
            cb_weights = weights
        else:
            assert samples_per_cls is not None, \
                "必须提供 samples_per_cls 或 weights 之一"
            if num_classes is None:
                num_classes = len(samples_per_cls)
            counts = np.asarray(samples_per_cls, dtype=np.float64)
            counts = np.maximum(counts, 1.0)
            effective_num = 1.0 - np.power(beta, counts)
            cb_weights = (1.0 - beta) / effective_num
            # 归一化到均值 = 1，训练更稳定
            cb_weights = cb_weights / cb_weights.sum() * num_classes
            cb_weights = torch.tensor(cb_weights, dtype=torch.float32)

        # 注册为 buffer，随模型一起 to(device)
        self.register_buffer('cb_weights', cb_weights)

    def forward(self, inputs, targets):
        # 逐样本 CE，用 CB 权重作为 class weight
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.cb_weights,
            reduction='none'
        )

        if self.gamma > 0:
            pt = torch.exp(-ce_loss)           # 预测正确类的概率
            loss = ((1.0 - pt) ** self.gamma) * ce_loss
        else:
            loss = ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss