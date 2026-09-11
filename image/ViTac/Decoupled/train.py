# train.py
# 解耦训练脚本（Decoupled Training for Long-tailed Recognition）
#   Stage 1: 长尾数据集上普通训练整个网络（shuffle=True）
#   Stage 2: 冻结编码器，用类别平衡采样重训最终分类头
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataset import ViTacDataset
from backbone import AlexNet, ResNet50


def parse_args():
    parser = argparse.ArgumentParser(description='Decoupled Training for Long-tailed Classification')
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=100, help='类别数')
    parser.add_argument('--num_workers', type=int, default=6, help='数据加载器的 worker 数量')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)

    # ===== Stage 1：普通训练整网 =====
    parser.add_argument('--stage1_epochs', type=int, default=20,
                        help='Stage1 训练整网的 epoch 数')
    parser.add_argument('--stage1_lr', type=float, default=1e-4,
                        help='Stage1 学习率')
    parser.add_argument('--stage1_weight_decay', type=float, default=1e-4,
                        help='Stage1 weight decay')

    # ===== Stage 2：冻结编码器 + 类别平衡采样重训分类头 =====
    parser.add_argument('--stage2_epochs', type=int, default=10,
                        help='Stage2 重训分类头的 epoch 数')
    parser.add_argument('--stage2_lr', type=float, default=1e-3,
                        help='Stage2 分类头学习率（通常比 Stage1 大 5~10 倍）')
    parser.add_argument('--stage2_weight_decay', type=float, default=1e-4,
                        help='Stage2 weight decay')
    parser.add_argument('--stage2_sampler_power', type=float, default=1.0,
                        help='Stage2 类别平衡采样权重指数：'
                             '1.0=完全平衡(1/n)，0.5=温和平衡(1/sqrt(n))')

    # 保存参数
    parser.add_argument('--save_interval', type=int, default=5)
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


# ============ 类别平衡采样工具 ============
def build_balanced_sampler(labels, num_classes, power=1.0):
    """
    构造 WeightedRandomSampler 实现类别平衡采样。
    power=1.0 -> w_i = 1/n_c        （完全平衡）
    power=0.5 -> w_i = 1/sqrt(n_c)  （温和平衡，推荐）
    """
    labels = np.asarray(labels)
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    class_counts_safe = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / np.power(class_counts_safe, power)
    sample_weights = class_weights[labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, class_counts


# ============ 冻结编码器 ============
def freeze_encoder(model, arch='resnet50'):
    """冻结编码器，只保留分类头可训练。返回可训练参数名列表。"""
    if arch == 'resnet50':
        for name, p in model.named_parameters():
            if not name.startswith('resnet.fc'):
                p.requires_grad = False
    elif arch == 'alexnet':
        for name, p in model.named_parameters():
            if not name.startswith('classifier'):
                p.requires_grad = False
    else:
        raise ValueError(f'Unknown arch: {arch}')
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    return trainable


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


def save_ckpt(state, is_best, epoch, interval, prefix=''):
    if is_best:
        torch.save(state, f'./best_model{prefix}.pth')
    if epoch % interval == 0:
        torch.save(state, f'./checkpoint{prefix}_epoch_{epoch}.pth')


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

    # 验证集 loader（两阶段共用，务必 shuffle=False 保持真实分布评估）
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    logger.info(f'Train: {len(train_set)}, Val: {len(val_set)}')

    # 模型
    model = ResNet50(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # ==========================================================
    # Stage 1：普通训练整个网络（普通 shuffle 采样）
    # ==========================================================
    logger.info('=' * 60)
    logger.info('Stage 1: 普通训练整个网络（shuffle=True）')
    logger.info('=' * 60)

    train_loader_s1 = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    optimizer = optim.Adam(
        model.parameters(), lr=args.stage1_lr, weight_decay=args.stage1_weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    best_f1_s1 = 0.0
    logger.info('Start Stage-1 training...')
    for epoch in range(1, args.stage1_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader_s1, optimizer, device, epoch, criterion
        )
        val_loss, val_acc, val_f1 = val_epoch(
            model, val_loader, device, epoch, criterion
        )
        scheduler.step()

        logger.info(
            f'[S1] Epoch {epoch:02d} | '
            f'Train Loss {train_metrics["loss"]:.4f} '
            f'Train Acc {train_metrics["acc"]:.4f} '
            f'Train F1 {train_metrics["f1"]:.4f} | '
            f'Val Loss {val_loss:.4f} '
            f'Val Acc {val_acc:.4f} '
            f'Val F1 {val_f1:.4f}'
        )

        is_best = val_f1 > best_f1_s1
        if is_best:
            best_f1_s1 = val_f1
            logger.info(f'[S1] New best val F1: {best_f1_s1:.4f}')

        save_ckpt(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1_s1,
                'args': args,
            },
            is_best, epoch, args.save_interval, prefix='_stage1'
        )

    logger.info(f'Stage 1 finished. Best val F1: {best_f1_s1:.4f}')

    # ==========================================================
    # Stage 2：冻结编码器，类别平衡采样重训分类头
    # ==========================================================
    logger.info('=' * 60)
    logger.info('Stage 2: 冻结编码器 + 类别平衡采样重训分类头')
    logger.info('=' * 60)

    # 2.1 加载 Stage 1 保存的最佳模型
    ckpt_path = './best_model_stage1.pth'
    assert os.path.isfile(ckpt_path), f'找不到 Stage1 最佳模型: {ckpt_path}'
    s1_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(s1_ckpt['model_state_dict'])
    logger.info(f'Loaded Stage-1 best model (epoch={s1_ckpt["epoch"]}, '
                f'F1={s1_ckpt.get("best_f1", 0.0):.4f})')

    # 2.2 冻结编码器
    trainable = freeze_encoder(model, arch='resnet50')
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Encoder frozen. Trainable params: {n_train}/{n_total} '
                f'({100.0 * n_train / n_total:.2f}%)')
    logger.info(f'Trainable modules: {trainable}')

    # 2.3 Stage 2 使用类别平衡采样
    sampler, class_counts = build_balanced_sampler(
        train_set.labels, args.num_classes, power=args.stage2_sampler_power
    )
    logger.info(
        f'BalancedSampler ON | power={args.stage2_sampler_power} | '
        f'imbalance ratio={class_counts.max() / max(class_counts.min(), 1):.1f} | '
        f'min/max counts={int(class_counts.min())}/{int(class_counts.max())}'
    )
    train_loader_s2 = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    # 2.4 只优化可训练参数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        params, lr=args.stage2_lr, weight_decay=args.stage2_weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 2.5 Stage 2 训练循环
    best_f1_s2 = best_f1_s1
    for epoch in range(1, args.stage2_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader_s2, optimizer, device, epoch, criterion
        )
        val_loss, val_acc, val_f1 = val_epoch(
            model, val_loader, device, epoch, criterion
        )
        scheduler.step()

        logger.info(
            f'[S2] Epoch {epoch:02d} | '
            f'Train Loss {train_metrics["loss"]:.4f} '
            f'Train Acc {train_metrics["acc"]:.4f} '
            f'Train F1 {train_metrics["f1"]:.4f} | '
            f'Val Loss {val_loss:.4f} '
            f'Val Acc {val_acc:.4f} '
            f'Val F1 {val_f1:.4f}'
        )

        is_best = val_f1 > best_f1_s2
        if is_best:
            best_f1_s2 = val_f1
            logger.info(f'[S2] New best val F1: {best_f1_s2:.4f}')

        save_ckpt(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1_s2,
                'args': args,
            },
            is_best, epoch, args.save_interval, prefix='_stage2'
        )

    logger.info(f'Stage 2 finished. Best val F1: {best_f1_s2:.4f}')
    logger.info(f'Overall best F1: Stage1={best_f1_s1:.4f} -> Stage2={best_f1_s2:.4f}')


if __name__ == '__main__':
    main()
