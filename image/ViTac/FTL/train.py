# train_ftl.py
# FTL 长尾分类训练脚本，按论文 Algorithm 1 实现三阶段：
#   Stage 0: 预训练（普通训练 + m-L2 正则化）
#   交替循环:
#       Stage 1: 冻结 Encoder，用特征迁移样本矫正分类器
#       Stage 2: 冻结 Classifier，学习更紧凑的特征
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataset import ViTacDataset
from backbone import FTLResNet50
from ftl_utils import (compute_class_centers, compute_pca_basis,
                       transfer_features, compute_metrics)


def parse_args():
    parser = argparse.ArgumentParser(description='FTL for Long-tailed Classification')
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=6)
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    # === FTL 参数 ===
    parser.add_argument('--ur_threshold', type=int, default=20,
                        help='样本数 <= 该阈值的类视为 UR 类（论文对 MS1M 用 20）')
    parser.add_argument('--alpha_reg', type=float, default=0.001,
                        help='m-L2 正则化系数（论文人脸任务 0.25，分类建议 1e-3 量级）')
    parser.add_argument('--n_pca_components', type=int, default=300,
                        help='PCA 保留的主成分数，论文 320 维特征取 150 保留 95% 能量；'
                             '对 2048 维特征可适当增大')

    # 训练阶段参数
    parser.add_argument('--pretrain_epochs', type=int, default=20,
                        help='Stage 0 预训练 epoch 数')
    parser.add_argument('--alt_rounds', type=int, default=5,
                        help='Stage1/Stage2 交替总轮数')
    parser.add_argument('--stage1_epochs', type=int, default=3,
                        help='每轮 Stage1 的 epoch 数')
    parser.add_argument('--stage2_epochs', type=int, default=3,
                        help='每轮 Stage2 的 epoch 数')

    # 保存参数
    parser.add_argument('--log_file', type=str, default='./train_ftl.log')
    return parser.parse_args()


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad


# =====================================================================
# 普通训练 epoch：用于 Stage 0 和 Stage 2
# 损失 = L_sfmx + alpha_reg * ||logits||^2   （论文 Eqn.3 的 m-L2）
# =====================================================================
def train_normal_epoch(model, loader, optimizer, device, epoch,
                       criterion, alpha_reg, num_classes):
    model.train()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        _, _, logits = model(inputs, return_features=True)
        loss = criterion(logits, labels) + \
               alpha_reg * (logits ** 2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total += bs
        all_preds.append(logits.argmax(1).detach().cpu())
        all_labels.append(labels.cpu())
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return compute_metrics(all_preds, all_labels, total_loss, total, num_classes)


# =====================================================================
# Stage 1: Decision boundary reshape
# 冻结 Encoder，训练 Filter + Classifier
# 每次迭代使用 3 个批次：
#   ① 常规类批次 (regular)
#   ② UR 类批次 (ur)
#   ③ 迁移批次：从常规批次借类内方差，注入到 UR 类中心
# =====================================================================
def stage1_epoch(model, regular_loader, ur_loader, optimizer, device, epoch,
                 centers, Q, criterion, alpha_reg, num_classes):
    model.train()
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.filter, True)
    set_requires_grad(model.classifier, True)

    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    ur_iter = iter(ur_loader)
    pbar = tqdm(regular_loader, desc=f'Epoch {epoch} [S1]')
    for x_r, y_r in pbar:
        # 取一个 UR 批次（循环使用）
        try:
            x_u, y_u = next(ur_iter)
        except StopIteration:
            ur_iter = iter(ur_loader)
            x_u, y_u = next(ur_iter)

        x_r, y_r = x_r.to(device), y_r.to(device)
        x_u, y_u = x_u.to(device), y_u.to(device)
        bs = min(x_r.size(0), x_u.size(0))
        x_r, y_r = x_r[:bs], y_r[:bs]
        x_u, y_u = x_u[:bs], y_u[:bs]

        optimizer.zero_grad()

        # ----- ① 常规 batch -----
        _, _, logits_r = model(x_r, return_features=True)
        loss_r = criterion(logits_r, y_r) + \
                 alpha_reg * (logits_r ** 2).sum(dim=1).mean()

        # ----- ② UR batch -----
        _, _, logits_u = model(x_u, return_features=True)
        loss_u = criterion(logits_u, y_u) + \
                 alpha_reg * (logits_u ** 2).sum(dim=1).mean()

        # ----- ③ 迁移 batch -----
        with torch.no_grad():
            g_r = model.extract_rich_features(x_r)
            g_tilde_u = transfer_features(g_r, centers, Q, y_r, y_u)
        f_tilde = model.filter(g_tilde_u)
        logits_t = model.classifier(f_tilde)
        loss_t = criterion(logits_t, y_u) + \
                 alpha_reg * (logits_t ** 2).sum(dim=1).mean()

        loss = (loss_r + loss_u + loss_t) / 3.0
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * bs
        total += bs
        all_preds.append(logits_u.argmax(1).detach().cpu())
        all_labels.append(y_u.cpu())
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return compute_metrics(all_preds, all_labels, total_loss, total, num_classes)


# =====================================================================
# 验证 epoch
# =====================================================================
@torch.no_grad()
def val_epoch(model, loader, device, epoch, criterion, num_classes):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        _, _, logits = model(inputs, return_features=True)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total += bs
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels.cpu())
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    m = compute_metrics(all_preds, all_labels, total_loss, total, num_classes)
    return m['loss'], m['acc'], m['f1']


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger()
    logger.info(f'Args: {args}')
    logger.info(f'Device: {device}')

    # ---------------- 数据 ----------------
    train_set = ViTacDataset(args.data_root, '../', split='train')
    val_set = ViTacDataset(args.data_root, '../', split='val')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=2)
    logger.info(f'Train: {len(train_set)}, Val: {len(val_set)}')

    # ---------------- 划分 regular / UR ----------------
    counts = np.bincount(train_set.labels, minlength=args.num_classes)
    regular_classes = [c for c in range(args.num_classes) if counts[c] > args.ur_threshold]
    ur_classes = [c for c in range(args.num_classes) if counts[c] <= args.ur_threshold]
    logger.info(f'Regular: {len(regular_classes)} | UR: {len(ur_classes)} | '
                f'threshold={args.ur_threshold}')
    logger.info(f'Class count: min={counts.min()}, max={counts.max()}, '
                f'imbalance ratio={counts.max() / max(counts.min(), 1):.1f}')
    if len(ur_classes) == 0:
        logger.warning('没有检测到 UR 类，请降低 --ur_threshold')
    if len(regular_classes) == 0:
        logger.warning('没有检测到常规类，请提高 --ur_threshold')

    # 为 Stage 1 准备的子集 loader
    regular_idx = [i for i, l in enumerate(train_set.labels) if l in regular_classes]
    ur_idx = [i for i, l in enumerate(train_set.labels) if l in ur_classes]
    bs_half = max(args.batch_size // 2, 1)

    regular_loader = DataLoader(
        Subset(train_set, regular_idx), batch_size=bs_half, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    ur_loader = DataLoader(
        Subset(train_set, ur_idx), batch_size=bs_half, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    # ---------------- 模型 ----------------
    model = FTLResNet50(num_classes=args.num_classes, use_filter=True,
                        pretrained=True).to(device)
    feat_dim = model.feat_dim
    criterion = nn.CrossEntropyLoss()
    logger.info(f'FTL ResNet50 built. feat_dim={feat_dim}, '
                f'num_classes={args.num_classes}')

    best_f1 = 0.0
    ckpt_path = './best_model_ftl.pth'

    # =========================================================
    # Stage 0: 预训练（普通训练 + m-L2）
    # =========================================================
    logger.info('=' * 60)
    logger.info('Stage 0: 预训练 (L_sfmx + alpha_reg * m-L2)')
    logger.info('=' * 60)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in range(1, args.pretrain_epochs + 1):
        tr = train_normal_epoch(model, train_loader, optimizer, device,
                                epoch, criterion, args.alpha_reg, args.num_classes)
        val_loss, val_acc, val_f1 = val_epoch(model, val_loader, device,
                                              epoch, criterion, args.num_classes)
        scheduler.step()
        logger.info(
            f'[S0] Epoch {epoch:02d} | '
            f'Train Loss {tr["loss"]:.4f} Acc {tr["acc"]:.4f} F1 {tr["f1"]:.4f} | '
            f'Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}'
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch, 'stage': 'pretrain',
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
                'num_classes': args.num_classes,
                'feat_dim': feat_dim,
            }, ckpt_path)
            logger.info(f'[S0] New best F1: {best_f1:.4f}')

    # =========================================================
    # 交替训练：Stage 1 ↔ Stage 2
    # =========================================================
    logger.info('=' * 60)
    logger.info('交替训练: Stage 1 (决策边界矫正) ↔ Stage 2 (紧凑特征)')
    logger.info('=' * 60)

    for rnd in range(1, args.alt_rounds + 1):
        logger.info(f'--- Alternate Round {rnd}/{args.alt_rounds} ---')

        # 每次进入新一轮，都重新估计 centers 和 PCA basis（因为 encoder 会更新）
        centers, _ = compute_class_centers(model, train_loader, device,
                                           args.num_classes, feat_dim)
        Q = compute_pca_basis(model, train_loader, centers, regular_classes,
                              device, feat_dim, args.n_pca_components)
        logger.info(f'  Centers shape={tuple(centers.shape)}, '
                    f'Q shape={tuple(Q.shape)}')

        # ---------- Stage 1 ----------
        logger.info(f'  [Round {rnd}] Stage 1: 决策边界矫正（冻结 Encoder）')
        s1_params = list(model.filter.parameters()) + list(model.classifier.parameters())
        s1_optimizer = optim.Adam(s1_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
        for epoch in range(1, args.stage1_epochs + 1):
            tr = stage1_epoch(model, regular_loader, ur_loader, s1_optimizer,
                              device, epoch, centers, Q, criterion,
                              args.alpha_reg, args.num_classes)
            val_loss, val_acc, val_f1 = val_epoch(model, val_loader, device,
                                                  epoch, criterion, args.num_classes)
            logger.info(
                f'[S1 R{rnd}] Epoch {epoch:02d} | '
                f'Train Loss {tr["loss"]:.4f} Acc {tr["acc"]:.4f} F1 {tr["f1"]:.4f} | '
                f'Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}'
            )
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'epoch': epoch, 'stage': f's1_r{rnd}',
                    'model_state_dict': model.state_dict(),
                    'best_f1': best_f1,
                    'num_classes': args.num_classes,
                    'feat_dim': feat_dim,
                }, ckpt_path)
                logger.info(f'[S1 R{rnd}] New best F1: {best_f1:.4f}')

        # ---------- Stage 2 ----------
        logger.info(f'  [Round {rnd}] Stage 2: 紧凑特征学习（冻结 Classifier）')
        set_requires_grad(model.classifier, False)
        set_requires_grad(model.encoder, True)
        set_requires_grad(model.filter, True)

        s2_params = list(model.encoder.parameters()) + list(model.filter.parameters())
        s2_optimizer = optim.Adam(s2_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
        for epoch in range(1, args.stage2_epochs + 1):
            tr = train_normal_epoch(model, train_loader, s2_optimizer, device,
                                    epoch, criterion, args.alpha_reg, args.num_classes)
            val_loss, val_acc, val_f1 = val_epoch(model, val_loader, device,
                                                  epoch, criterion, args.num_classes)
            logger.info(
                f'[S2 R{rnd}] Epoch {epoch:02d} | '
                f'Train Loss {tr["loss"]:.4f} Acc {tr["acc"]:.4f} F1 {tr["f1"]:.4f} | '
                f'Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}'
            )
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'epoch': epoch, 'stage': f's2_r{rnd}',
                    'model_state_dict': model.state_dict(),
                    'best_f1': best_f1,
                    'num_classes': args.num_classes,
                    'feat_dim': feat_dim,
                }, ckpt_path)
                logger.info(f'[S2 R{rnd}] New best F1: {best_f1:.4f}')

        # 下一轮 Stage1 需要 classifier 可训练
        set_requires_grad(model.classifier, True)

    logger.info(f'FTL training finished. Best val F1: {best_f1:.4f}')


if __name__ == '__main__':
    main()