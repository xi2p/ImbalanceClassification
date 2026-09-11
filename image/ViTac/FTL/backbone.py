import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
from torchvision import models


class FTLResNet50(nn.Module):
    """
    FTL 架构：ResNet50 拆分为 Encoder + Filter + Classifier
        对应论文:
            g      = Enc(x)          # rich feature  (2048 维)
            f      = R(g)            # identity feature
            logits = FC(f)           # 分类输出
    论文中的 Decoder 省略（预训练 ResNet50 的 g 已具备丰富的语义表示）
    """

    def __init__(self, num_classes=100, use_filter=True, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)

        # Encoder: 除 fc 外的所有层（含 avgpool），输出 (B, 2048, 1, 1)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        feat_dim = base.fc.in_features   # 2048

        # Filter R: 把 rich feature 映射到 identity feature
        # 论文 R 里没有 BN，这里也保持一致（BN 会因迁移样本而偏移 running stats）
        if use_filter:
            self.filter = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.filter = nn.Identity()

        # 分类器 FC
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.feat_dim = feat_dim

    def forward(self, x, return_features=False):
        g = self.encoder(x)
        g = torch.flatten(g, 1)          # (B, 2048)  rich feature
        f = self.filter(g)               # (B, 2048)  identity feature
        logits = self.classifier(f)      # (B, num_classes)
        if return_features:
            return g, f, logits
        return logits

    @torch.no_grad()
    def extract_rich_features(self, x):
        """只提取 rich feature g，用于计算类中心、PCA、迁移"""
        g = self.encoder(x)
        return torch.flatten(g, 1)