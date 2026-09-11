# ftl_utils.py
# FTL 论文里特征迁移相关的工具函数
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score


@torch.no_grad()
def compute_class_centers(model, loader, device, num_classes, feat_dim):
    """
    计算每个类的 rich feature 中心 c_i（论文 Eqn.5 的简化版）
    简化点：不做 pose filtering（没有 pose 估计器），只用原图特征
    """
    model.eval()
    feat_sum = torch.zeros(num_classes, feat_dim, device=device)
    count = torch.zeros(num_classes, device=device)

    for inputs, labels in tqdm(loader, desc='  centers', leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        g = model.extract_rich_features(inputs)
        feat_sum.index_add_(0, labels, g)
        count.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))

    count = count.clamp(min=1.0)
    centers = feat_sum / count.unsqueeze(1)
    return centers, count.cpu().numpy()


@torch.no_grad()
def compute_pca_basis(model, loader, centers, regular_classes, device,
                      feat_dim, n_components):
    """
    计算常规类的类内协方差 V（论文 Eqn.6），做特征分解取前 n_components 个基 Q
    V = Σ_{i ∈ D_reg} Σ_k (g_ik - c_i)^T (g_ik - c_i)
    """
    model.eval()
    V = torch.zeros(feat_dim, feat_dim, device=device)
    reg_tensor = torch.tensor(regular_classes, dtype=torch.long, device=device)

    for inputs, labels in tqdm(loader, desc='  PCA   ', leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        mask = torch.isin(labels, reg_tensor)
        if not mask.any():
            continue
        inputs_r, labels_r = inputs[mask], labels[mask]
        g = model.extract_rich_features(inputs_r)
        c = centers[labels_r]
        diff = g - c                       # (B, D)
        V += diff.t() @ diff

    # 特征分解（V 对称，用 eigh）
    _, eigvecs = torch.linalg.eigh(V)
    n_components = min(n_components, feat_dim)
    Q = eigvecs[:, -n_components:]        # eigh 升序，取最后 n 个（最大）
    return Q


def transfer_features(g_regular, centers, Q, y_regular, y_ur):
    """
    论文 Eqn.7 的特征迁移：
        ĝ_ik = c_i + Q Q^T (g_jk - c_j)
    其中 j 是常规类，i 是 UR 类。
    结果：保持 i 的身份，但借用了常规类样本 k 的类内变化（姿态/光照/表情等）
    """
    c_j = centers[y_regular]              # (B, D)
    c_i = centers[y_ur]                   # (B, D)
    diff = g_regular - c_j                # (B, D)
    proj = (diff @ Q) @ Q.t()             # (B, D)
    return c_i + proj


def compute_metrics(all_preds, all_labels, total_loss, total, num_classes):
    """macro F1 + acc + loss"""
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, labels=list(range(num_classes)),
                  average='macro', zero_division=0)
    return {'loss': total_loss / max(total, 1), 'acc': float(acc), 'f1': float(f1)}
