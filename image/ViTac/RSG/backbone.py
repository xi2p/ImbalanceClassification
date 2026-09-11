import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
from torchvision import models


class RSGResNet50(nn.Module):
    """
    RSG 专用 ResNet50：
        - forward_stem(x)  返回 layer3 输出特征图 [B, 1024, 14, 14]，给 RSG 用
        - forward_head(f)  从特征图算 logits
    state_dict 键名与 ResNet50 完全一致（resnet.xxx），eval.py 可直接加载。
    """

    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.feat_dim = self.resnet.fc.in_features   # 2048
        self.rsg_channels = 1024                     # layer3 输出通道

    def forward_stem(self, x):
        r = self.resnet
        x = r.conv1(x); x = r.bn1(x); x = r.relu(x); x = r.maxpool(x)
        x = r.layer1(x); x = r.layer2(x); x = r.layer3(x)
        return x                                     # [B, 1024, 14, 14]

    def forward_head(self, featmap):
        r = self.resnet
        x = r.layer4(featmap)
        x = r.avgpool(x)
        x = torch.flatten(x, 1)
        return r.fc(x)

    def forward(self, x, return_featmap=False):
        featmap = self.forward_stem(x)
        logits = self.forward_head(featmap)
        if return_featmap:
            return featmap, logits
        return logits