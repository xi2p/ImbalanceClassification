# train_rsg.py
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataset import ViTacDataset
from backbone import RSGResNet50
from rsg import RSG


def parse_args():
    parser = argparse.ArgumentParser(description='RSG for Long-tailed Classification')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    # RSG 参数
    parser.add_argument('--num_centers', type=int, default=15)
    parser.add_argument('--freq_ratio', type=float, default=0.2)
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=0.01)
    parser.add_argument('--rsg_start_ratio', type=float, default=0.4)

    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--log_file', type=str, default='./train_rsg.log')
    return parser.parse_args()


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


# =========================================================
# 训练 epoch
# =========================================================
def train_epoch(model, rsg, loader, optimizer, device, epoch,
                criterion, args, freq_set, generate):
    model.train()
    rsg.train()

    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        B = labels.size(0)

        # ---- 前向：拿到 layer3 特征图 ----
        featmap = model.forward_stem(inputs)                    # [B, 1024, 14, 14]

        # ---- CESC 损失（对 detach 特征图） ----
        cesc = rsg.cesc_loss(featmap.detach(), labels)

        # ---- 拆分 frequent / rare ----
        is_freq = torch.tensor([int(l.item()) in freq_set for l in labels],
                               device=device, dtype=torch.bool)
        is_rare = ~is_freq

        if generate and is_freq.any() and is_rare.any():
            fm_freq = featmap[is_freq].detach()
            lbl_freq = labels[is_freq]
            fm_rare = featmap[is_rare]
            lbl_rare = labels[is_rare]

            # 生成新样本
            feat_new, disp_freq, disp_freq_t = rsg.generate_new_samples(
                fm_freq, lbl_freq, fm_rare, lbl_rare
            )
            # 稀有类位移（不需要梯度）
            with torch.no_grad():
                disp_rare, _ = rsg.compute_displacement(
                    fm_rare.detach(), lbl_rare
                )
            mv = rsg.mv_loss(disp_freq_t, disp_freq, disp_rare)

            # 拼接特征图：原始 + 生成
            featmap_aug = torch.cat([featmap, feat_new], dim=0)
            labels_aug = torch.cat([labels, lbl_rare], dim=0)
        else:
            mv = torch.tensor(0.0, device=device)
            featmap_aug = featmap
            labels_aug = labels

        # ---- 分类损失 ----
        logits_aug = model.forward_head(featmap_aug)
        cls_loss = criterion(logits_aug, labels_aug)

        # ---- 总损失 ----
        loss = cls_loss + args.lambda1 * cesc + args.lambda2 * mv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- 统计（只用原始 logits） ----
        with torch.no_grad():
            logits_orig = model.forward_head(featmap)
            preds = logits_orig.argmax(1)
        total_loss += cls_loss.item() * B
        total += B
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        pbar.set_postfix(loss=f'{cls_loss.item():.4f}',
                         cesc=f'{cesc.item():.3f}',
                         mv=f'{mv.item():.3f}')

    preds = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    acc = (preds == labels_np).mean()
    f1 = f1_score(labels_np, preds,
                  labels=list(range(args.num_classes)),
                  average='macro', zero_division=0)
    return {'loss': total_loss / max(total, 1),
            'acc': float(acc), 'f1': float(f1)}


# =========================================================
# 验证 epoch
# =========================================================
@torch.no_grad()
def val_epoch(model, loader, device, epoch, criterion, num_classes):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels.cpu())
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    preds = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    acc = (preds == labels_np).mean()
    f1 = f1_score(labels_np, preds,
                  labels=list(range(num_classes)),
                  average='macro', zero_division=0)
    return total_loss / max(total, 1), float(acc), float(f1)


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

    # ---------------- 划分 frequent / rare ----------------
    counts = np.bincount(train_set.labels, minlength=args.num_classes)
    sorted_cls = np.argsort(-counts)
    n_freq = max(1, int(args.num_classes * args.freq_ratio))
    freq_set = set(sorted_cls[:n_freq].tolist())
    rare_set = set(range(args.num_classes)) - freq_set
    logger.info(f'Freq: {n_freq} | Rare: {len(rare_set)} | '
                f'imbalance ratio={counts.max() / max(counts.min(), 1):.1f}')

    # ---------------- 模型 ----------------
    model = RSGResNet50(num_classes=args.num_classes, pretrained=True).to(device)
    rsg = RSG(num_classes=args.num_classes,
              feat_dim=model.rsg_channels,               # ★ 用 layer3 的通道数 1024
              num_centers=args.num_centers).to(device)
    logger.info(f'Model built. RSG feature dim={model.rsg_channels}, '
                f'num_centers={args.num_centers}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(model.parameters()) + list(rsg.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    rsg_start_epoch = max(1, int(args.epochs * args.rsg_start_ratio))
    logger.info(f'RSG generation starts at epoch > {rsg_start_epoch}')

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        generate = epoch > rsg_start_epoch
        # T_th 之后冻结对比模块参数（但梯度仍可穿过它传给 vt）
        for p in rsg.contrastive.parameters():
            p.requires_grad = not generate

        tr = train_epoch(model, rsg, train_loader, optimizer, device, epoch,
                         criterion, args, freq_set, generate)
        val_loss, val_acc, val_f1 = val_epoch(model, val_loader, device,
                                              epoch, criterion, args.num_classes)
        scheduler.step()

        logger.info(
            f'Epoch {epoch:02d} | '
            f'Train Loss {tr["loss"]:.4f} Acc {tr["acc"]:.4f} F1 {tr["f1"]:.4f} | '
            f'Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f} '
            f'{"[RSG ON]" if generate else "[RSG OFF]"}'
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
                'num_classes': args.num_classes,
            }, './best_model_rsg.pth')
            logger.info(f'New best F1: {best_f1:.4f}')

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'rsg_state_dict': rsg.state_dict(),
                'best_f1': best_f1,
                'num_classes': args.num_classes,
            }, f'./checkpoint_rsg_epoch_{epoch}.pth')

    logger.info(f'Training finished. Best val F1: {best_f1:.4f}')


if __name__ == '__main__':
    main()
