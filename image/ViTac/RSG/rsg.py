# rsg.py
# RSG: Rare-class Sample Generator (Wang et al., 2021)
# 严格按照论文，在空间特征图 [B, C, H, W] 上操作
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1) 中心估计模块（Eqn.1）
#    对全局平均池化后的特征算 K 个中心的分配概率 γ，
#    再用 γ 加权得到中心，并沿 H、W 上采样回原特征图大小
# =========================================================
class CenterEstimationModule(nn.Module):
    def __init__(self, num_classes, feat_dim, num_centers=15):
        super().__init__()
        self.num_centers = num_centers
        self.feat_dim = feat_dim
        # 论文每类一个 A^l；这里用共享线性层 + 每类独立中心，参数量更小
        self.center_linear = nn.Linear(feat_dim, num_centers)
        # 每类 K 个中心向量，形状 [Nc, K, D]
        self.centers = nn.Parameter(
            torch.randn(num_classes, num_centers, feat_dim) * 0.01
        )

    def forward(self, featmap, labels):
        """
        featmap: [B, C, H, W]
        labels:  [B]
        返回：
            gamma:       [B, K]          中心分配概率
            center:      [B, C]          加权后的中心向量
            center_up:   [B, C, H, W]    上采样回原尺寸的中心
        """
        B, C, H, W = featmap.shape
        pooled = featmap.mean(dim=(2, 3))                          # [B, C]
        gamma = F.softmax(self.center_linear(pooled), dim=1)       # [B, K]
        class_centers = self.centers[labels]                       # [B, K, C]
        center = (gamma.unsqueeze(-1) * class_centers).sum(dim=1)  # [B, C]
        center_up = center.view(B, C, 1, 1).expand(B, C, H, W)     # [B, C, H, W]
        return gamma, center, center_up


# =========================================================
# 2) 对比模块（Eqn.2）
#    输入两张特征图，沿通道拼接后用 2 层 3×3 Conv
# =========================================================
class ContrastiveModule(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(2 * feat_dim, hidden_dim, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                               stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x1, x2):
        """x1, x2: [B, C, H, W] → [B, 2]  logits（0=不同类，1=同类）"""
        x = torch.cat([x1, x2], dim=1)             # [B, 2C, H, W]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.mean(dim=(2, 3))                     # GAP → [B, hidden]
        return self.fc(x)


# =========================================================
# 3) 向量变换模块（Eqn.4 中的 T）
#    论文：kernel=3, stride=1, padding=1 的单层 Conv
# =========================================================
class VectorTransformationModule(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3,
                              stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


# =========================================================
# 4) RSG 主模块
# =========================================================
class RSG(nn.Module):
    def __init__(self, num_classes, feat_dim, num_centers=15, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.num_centers = num_centers

        self.center_est = CenterEstimationModule(num_classes, feat_dim, num_centers)
        self.contrastive = ContrastiveModule(feat_dim, hidden_dim)
        self.vt = VectorTransformationModule(feat_dim)

    # -------- 特征位移（Eqn.3）--------
    def compute_displacement(self, featmap, labels):
        """x_df = x - up(C_K^l)"""
        _, _, center_up = self.center_est(featmap, labels)
        disp = featmap - center_up                              # [B, C, H, W]
        return disp, center_up

    # -------- 生成新样本（Eqn.4）--------
    def generate_new_samples(self, fm_freq, lbl_freq, fm_rare, lbl_rare):
        """
        从 frequent 位移为 rare 生成新样本：
            x_new = T(x_df-freq) + x_rare
        返回：
            feat_new:     [B_rare, C, H, W]
            disp_freq:    [B_rare, C, H, W]   使用的 frequent 位移（未变换）
            disp_freq_t:  [B_rare, C, H, W]   经过 T 变换后的位移
        """
        disp_freq_all, _ = self.compute_displacement(fm_freq, lbl_freq)
        disp_freq_t_all = self.vt(disp_freq_all)

        B_f = disp_freq_all.size(0)
        B_r = fm_rare.size(0)

        if B_f >= B_r:
            idx = torch.randperm(B_f, device=fm_freq.device)[:B_r]
            disp_freq = disp_freq_all[idx]
            disp_freq_t = disp_freq_t_all[idx]
        else:
            repeat = (B_r + B_f - 1) // B_f
            disp_freq = disp_freq_all.repeat(repeat, 1, 1, 1)[:B_r]
            disp_freq_t = disp_freq_t_all.repeat(repeat, 1, 1, 1)[:B_r]

        feat_new = fm_rare + disp_freq_t                        # Eqn.4
        return feat_new, disp_freq, disp_freq_t

    # -------- CESC 损失（Eqn.5）--------
    def cesc_loss(self, featmap, labels):
        """
        L_CESC = <Σ_k γ_k ||x - up(C_k)||²> + CE(对比模块)
        调用方必须传入 detach 后的 featmap。
        """
        B = featmap.size(0)
        gamma, _, _ = self.center_est(featmap, labels)          # [B, K]
        class_centers = self.center_est.centers[labels]         # [B, K, C]

        # 因为 up 只是空间复制，(x - up(C_k))² 的均值 =
        #   ||pooled(x) - C_k||² + Var(x)（Var 项与 k 无关，对优化无影响）
        pooled = featmap.mean(dim=(2, 3))                       # [B, C]
        dist_sq = ((pooled.unsqueeze(1) - class_centers) ** 2).sum(-1)  # [B, K]
        term_dist = (gamma * dist_sq).sum(1).mean()

        # 对比模块：随机配对
        num_pairs = max(B // 2, 1)
        idx1 = torch.randperm(B, device=featmap.device)[:num_pairs]
        idx2 = torch.randperm(B, device=featmap.device)[:num_pairs]
        y_pair = (labels[idx1] == labels[idx2]).long()
        logits = self.contrastive(featmap[idx1], featmap[idx2])
        term_contrast = F.cross_entropy(logits, y_pair)

        return term_dist + term_contrast

    # -------- MV 损失（Eqn.6）--------
    def mv_loss(self, disp_freq_t, disp_freq, disp_rare):
        """
        L_MV = <|cos(T(x_df), x_df_rare) - 1|>
               + <||T(x_df)|| - ||x_df||>
               - <log γ*>
        逐空间位置 (j, k) 计算。
        """
        B, C, H, W = disp_freq_t.shape
        a = disp_freq_t.view(B, C, -1)                          # [B, C, HW]
        b = disp_rare.view(B, C, -1)                            # [B, C, HW]
        o = disp_freq.view(B, C, -1)                            # [B, C, HW]

        # ① 逐位置 cos → 1
        cos = F.cosine_similarity(a, b, dim=1)                  # [B, HW]
        term1 = (cos - 1.0).abs().mean()

        # ② 逐位置长度保持一致
        len_t = a.norm(p=2, dim=1)                              # [B, HW]
        len_o = o.norm(p=2, dim=1)
        term2 = (len_t - len_o).abs().mean()

        # ③ 对比模块：T(x_df) 与 x_df 应不属于同类
        logits = self.contrastive(disp_freq_t, disp_freq)
        zero = torch.zeros(logits.size(0), dtype=torch.long,
                           device=logits.device)
        term3 = F.cross_entropy(logits, zero)

        return term1 + term2 + term3
