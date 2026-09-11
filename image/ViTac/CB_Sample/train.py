# train.py
# 分类任务训练脚本（基于 AlexNet，使用 F1 Score 应对样本不平衡）
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import ViTacDataset
from backbone import AlexNet, ResNet50

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification Training')
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=100, help='类别数')
    parser.add_argument('--num_workers', type=int, default=6, help='数据加载器的 worker 数量')
    # 类别平衡采样
    parser.add_argument('--sampler_power', type=float, default=1.0,
                        help='采样权重指数。1.0=完全平衡(1/n)；'
                             '0.5=温和平衡(1/sqrt(n))；0.0=等价于普通随机采样')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    # CB Loss 参数
    parser.add_argument('--beta', type=float, default=0.9999,
                        help='CB Loss 的 beta 参数，越大越强调尾部类（推荐 0.99~0.9999）')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='CB Focal 的 gamma 参数；gamma=0 退化为 CB Softmax')
    # 保存参数
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log_file', type=str, default='./train.log')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def build_balanced_sampler(labels, num_classes, power=1.0):
    """
    构造 WeightedRandomSampler 实现类别平衡采样。

    Args:
        labels: 训练集所有样本的标签列表（list / np.array）
        num_classes: 类别总数
        power: 权重指数。
               1.0 -> w_i = 1 / n_c        完全类别平衡（尾部类被采样的次数明显增多）
               0.5 -> w_i = 1 / sqrt(n_c)  温和平衡，效果通常更稳，推荐
               0.0 -> w_i = 1              等价于普通随机采样

    Returns:
        sampler: WeightedRandomSampler
        class_counts: 每个类别的样本数（用于打印）
    """
    labels = np.asarray(labels)
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    class_counts_safe = np.maximum(class_counts, 1.0)   # 防止除零

    # 每个类别的采样权重
    class_weights = 1.0 / np.power(class_counts_safe, power)

    # 映射回每个样本
    sample_weights = class_weights[labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),   # 每个 epoch 采样数 = 训练集大小
        replacement=True                   # 尾部类样本需重复采样，必须为 True
    )
    return sampler, class_counts


def train_epoch(model, loader, optimizer, device, epoch, criterion):
    model.train()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 累积损失与预测
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        # 进度条即时显示 batch 损失
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    # 计算 epoch 整体指标
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / total

    # 在进度条关闭前更新最终指标（可选）
    pbar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{acc:.4f}', f1=f'{f1:.4f}')

    return {'loss': avg_loss, 'acc': acc, 'f1': f1}


@torch.no_grad()
def val_epoch(model, loader, device, epoch, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        pbar.set_postfix(loss=f'{loss.item():.4f}')

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / total

    pbar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{acc:.4f}', f1=f'{f1:.4f}')
    return avg_loss, acc, f1


def save_ckpt(state, is_best, epoch, interval):
    if is_best:
        torch.save(state, './best_model.pth')
    if epoch % interval == 0:
        torch.save(state, f'./checkpoint_epoch_{epoch}.pth')


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger()
    logger.info(f'Args: {args}')
    logger.info(f'Device: {device}')

    # 数据集
    train_set = ViTacDataset(args.data_root, '../', split='train')
    val_set = ViTacDataset(args.data_root, '../', split='val')

    sampler, class_counts = build_balanced_sampler(
        train_set.labels, args.num_classes, power=args.sampler_power
    )
    logger.info(
        f'BalancedSampler ON | power={args.sampler_power} | '
        f'imbalance ratio={class_counts.max() / max(class_counts.min(), 1):.1f} | '
        f'min/max counts={int(class_counts.min())}/{int(class_counts.max())}'
    )
    # 注意：使用 sampler 时不能再传 shuffle=True
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    logger.info(f'Train: {len(train_set)}, Val: {len(val_set)}')

    # 模型
    model = ResNet50().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 断点续训
    start_epoch = 1
    best_f1 = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        start_epoch = ckpt['epoch'] + 1
        best_f1 = ckpt.get('best_f1', 0.0)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(
            f'Resumed from epoch {ckpt["epoch"]}, best F1: {best_f1:.4f}'
        )

    logger.info('Start training...')
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, criterion
        )
        val_loss, val_acc, val_f1 = val_epoch(
            model, val_loader, device, epoch, criterion
        )
        scheduler.step()

        logger.info(
            f'Epoch {epoch:02d} | '
            f'Train Loss {train_metrics["loss"]:.4f} '
            f'Train Acc {train_metrics["acc"]:.4f} '
            f'Train F1 {train_metrics["f1"]:.4f} | '
            f'Val Loss {val_loss:.4f} '
            f'Val Acc {val_acc:.4f} '
            f'Val F1 {val_f1:.4f}'
        )

        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            logger.info(f'New best val F1: {best_f1:.4f}')

        save_ckpt(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
                'args': args
            },
            is_best, epoch, args.save_interval
        )

    logger.info(f'Training finished. Best val F1: {best_f1:.4f}')


if __name__ == '__main__':
    main()