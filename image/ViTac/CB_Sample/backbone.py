import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
from torchvision import models


class AlexNet(nn.Module):
    """
    基于AlexNet的双模态特征提取器
    对应论文图3的网络结构：卷积层 + 全连接层提取4096维特征，附带分类头
    """

    def __init__(self, num_classes=100, pretrained=True):
        super(AlexNet, self).__init__()
        # 加载ImageNet预训练的AlexNet

        if pretrained:
            weights = models.AlexNet_Weights.IMAGENET1K_V1
        else:
            weights = None
        base_alexnet = models.alexnet(weights=weights)

        # 卷积特征提取部分（对应图3的Conv1~Conv5 + ReLU）
        self.features = base_alexnet.features
        self.avgpool = base_alexnet.avgpool

        # 全连接特征提取部分：保留到fc7层，输出4096维特征
        # 对应论文中FC8层的隐藏表示输出，维度D=4096
        self.feature_extractor = nn.Sequential(
            *list(base_alexnet.classifier.children())[:-1]  # 去掉原始1000类分类层
        )

        # 单模态分类头：用于单模态预训练的100类分类输出
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x, return_feature=True):
        """
        前向传播
        Args:
            x: 输入图像，shape (batch_size, 3, 227, 227)
            return_feature: 是否返回4096维特征
        Returns:
            feat: 4096维深度特征，shape (batch_size, 4096)
            out: 分类logits，shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        feat = self.feature_extractor(x)  # 提取共享特征
        out = self.classifier(feat)  # 分类预测

        if return_feature:
            return feat, out
        else:
            return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(ResNet50, self).__init__()

        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.resnet = models.resnet50(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)

        return x
