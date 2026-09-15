"""
用于测试数据变换功能的模块，包含各种数据预处理和增强方法的测试用例。
"""
import torch
from torch.utils.data import DataLoader
from dataset import ViTacDataset
from matplotlib import pyplot as plt
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    min_area_ratio = (224 / 362) ** 2
    max_area_ratio = (336 / 362) ** 2

    augment_transform = transforms.Compose([
        transforms.CenterCrop(720),                          # 960x720 -> 720x720
        transforms.RandomRotation(degrees=(0, 360)),         # 随机旋转 0~360°
        transforms.CenterCrop(720),                          # 旋转后中心裁剪 509x509
        transforms.Resize((224, 224)),                       # 缩放到 224x224
        transforms.ToTensor()
    ])

    to_tensor_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 创建数据集和数据加载器
    visual_train_dataset = ViTacDataset(root_dir=r"E:\PythonProjects\SlipDetection\ViTac\vitac_dataset",
                                        txt_dir=r"..\_Splits\C30-IM50-SEQ",
                                        split="train",
                                        transform=augment_transform)

    # 获取一个批次的数据
    LABEL_TO_SAMPLE = [0, 200, 400, 600]
    images, labels = [visual_train_dataset.get_pic(i)[0] for i in LABEL_TO_SAMPLE], [visual_train_dataset.get_pic(i)[1] for i in LABEL_TO_SAMPLE]

    # 可视化原始图像
    fig, axes = plt.subplots(3, 4, figsize=(12, 6))
    for i in range(4):
        axes[0, i].imshow(to_tensor_transform(images[i]).permute(1, 2, 0))
        axes[0, i].set_title(f"Original Label: {labels[i]}")
        axes[0, i].axis('off')

    # 应用变换并可视化增强后的图像
    for i in range(4):
        img_aug = augment_transform(images[i])
        axes[1, i].imshow(img_aug.permute(1, 2, 0))
        axes[1, i].set_title(f"Augmented 1 Label: {labels[i]}")
        axes[1, i].axis('off')

    for i in range(4):
        img_aug = augment_transform(images[i])
        axes[2, i].imshow(img_aug.permute(1, 2, 0))
        axes[2, i].set_title(f"Augmented 2 Label: {labels[i]}")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()
