import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



def parse_txt_file(txt_path, root_dir):
    """
    解析txt标注文件，提取所有样本的完整路径和对应标签
    Args:
        txt_path: 标注txt文件路径
        root_dir: 数据集根目录，所有相对路径基于此目录
    Returns:
        image_paths: 所有样本的绝对路径列表
        labels: 所有样本的标签列表（0~99）
    """
    image_paths = []
    labels = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 去掉路径开头的 "./"，拼接绝对路径
            rel_path = line.lstrip('./')
            full_path = os.path.join(root_dir, rel_path)
            image_paths.append(full_path)

            # 按路径分段提取类别文件夹名
            path_parts = rel_path.replace('\\', '/').split('/')
            class_folder = path_parts[0]  # 视觉: vi0011；触觉: CD0011-a2

            # CD0011-a2 → 提取第2-5位数字 0011
            label_str = class_folder[2:6]

            # 标签转换：原始数字减11，得到0~99的类别编号
            label = int(label_str) - 11
            labels.append(label)

    return image_paths, labels


class ViTacDataset(Dataset):
    def __init__(self, root_dir, txt_dir='./', split='train', transform=None):
        super().__init__()
        self.split = split
        self.data_root = root_dir

        # 读取txt标注
        if split == 'train':
            txt_name = 'train.txt'
        elif split == 'test':
            txt_name = 'test.txt'
        elif split == 'val':
            txt_name = 'val.txt'
        else:
            raise ValueError(f"Unsupported split: {split}")
        txt_path = os.path.join(txt_dir, txt_name)
        self.image_paths, self.labels = parse_txt_file(txt_path, root_dir)

        # 默认 transform（不再 resize/crop，只做 ToTensor 和 Normalize）
        if transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(720),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]

        # 加载原图
        img_pil = Image.open(img_path).convert('RGB')

        # 应用 transform（ToTensor + 归一化）
        img_tensor = self.transform(img_pil)

        return img_tensor, label

    def __len__(self):
        return len(self.image_paths)


# ===================== 使用示例 =====================
if __name__ == '__main__':
    # 数据集根目录，按你实际路径修改
    ROOT = r"E:\PythonProjects\SlipDetection\ViTac\vitac_dataset"
    # txt文件所在目录，按你实际存放位置修改
    TXT_DIR = r"../"

    # 测试视觉训练集
    tac_train = ViTacDataset(
        root_dir=ROOT,
        txt_dir=TXT_DIR,
        split='train',
    )
    print(f"训练集样本数: {len(tac_train)}")
    img, label = tac_train[0]
    print(f"样本0 - 图像shape: {img.shape}, 标签: {label}")

    # 测试触觉测试集
    tac_test = ViTacDataset(
        root_dir=ROOT,
        txt_dir=TXT_DIR,
        split='test',
    )
    print(f"测试集样本数: {len(tac_test)}")
    img, label = tac_test[0]
    print(f"样本0 - 图像shape: {img.shape}, 标签: {label}")
