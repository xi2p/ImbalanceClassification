# eval.py
# 通用模型评估脚本（兼容所有训练版本）
#   支持普通训练 / Focal Loss / CB Loss / 类别平衡采样 / 解耦训练的 checkpoint
import os
import argparse
import logging
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, balanced_accuracy_score
)

from dataset import ViTacDataset
from backbone import AlexNet, ResNet50


# =========================================================
# 参数
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation for Long-tailed Classification')

    # 数据参数（必须与训练时一致）
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--txt_dir', type=str, default='../',
                        help='train.txt / val.txt / test.txt 所在目录')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='要评估的数据划分')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='类别数；不填则从 checkpoint 的 args 里读取，'
                             '读不到时默认 100')
    parser.add_argument('--num_workers', type=int, default=4)

    # 模型参数
    parser.add_argument('--arch', type=str, default='resnet50',
                        choices=['resnet50', 'alexnet'],
                        help='backbone 类型')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='checkpoint 路径，例如 ./best_model_stage2.pth')

    # 评估参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_report', type=str, default=None,
                        help='可选：将详细报告保存为 json 文件')
    parser.add_argument('--confusion_matrix_csv', type=str, default=None,
                        help='可选：把混淆矩阵保存为 csv')

    return parser.parse_args()


# =========================================================
# 构建模型（不加载预训练权重，因为马上要 load_state_dict）
# =========================================================
def build_model(arch, num_classes):
    if arch == 'resnet50':
        return ResNet50(num_classes=num_classes, pretrained=False)
    elif arch == 'alexnet':
        return AlexNet(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f'Unknown arch: {arch}')


# =========================================================
# 评估主流程
# =========================================================
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc='Evaluating')
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.append(preds.cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()

    # ---------- 指标计算 ----------
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)

    # 每类 F1 / precision / recall
    per_class_f1 = f1_score(
        labels, preds, average=None,
        labels=list(range(num_classes)), zero_division=0
    )

    # 每个类的样本数
    class_counts = np.bincount(labels, minlength=num_classes)

    # 混淆矩阵
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    report = {
        'num_samples': int(len(labels)),
        'acc': float(acc),
        'balanced_acc': float(bal_acc),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'f1_weighted': float(f1_weighted),
    }

    return {
        'metrics': report,
        'per_class_f1': per_class_f1,
        'class_counts': class_counts,
        'confusion_matrix': cm,
        'labels': labels,
        'preds': preds,
        'probs': probs,
    }


# =========================================================
# 输出：控制台打印 + 可选落盘
# =========================================================
def print_report(result, num_classes, logger, class_names=None):
    m = result['metrics']
    logger.info('=' * 60)
    logger.info('评估结果')
    logger.info('=' * 60)
    logger.info(f'样本数           : {m["num_samples"]}')
    logger.info(f'Accuracy         : {m["acc"]:.4f}')
    logger.info(f'Balanced Acc     : {m["balanced_acc"]:.4f}')
    logger.info(f'F1 (macro)       : {m["f1_macro"]:.4f}')
    logger.info(f'F1 (micro)       : {m["f1_micro"]:.4f}')
    logger.info(f'F1 (weighted)    : {m["f1_weighted"]:.4f}')

    # 每个类的 F1 + 样本数
    logger.info('-' * 60)
    logger.info('每类详细指标 (class_id | 样本数 | F1)')
    logger.info('-' * 60)
    counts = result['class_counts']
    per_f1 = result['per_class_f1']
    for c in range(num_classes):
        name = class_names[c] if class_names else f'class_{c:03d}'
        logger.info(f'{name:>12s} | n={int(counts[c]):>6d} | F1={per_f1[c]:.4f}')

    # 头部 / 尾部对比（按样本数把类别分成 3 组，看长尾效果）
    if counts.sum() > 0:
        logger.info('-' * 60)
        logger.info('长尾分组对比（按类别样本数排序）')
        logger.info('-' * 60)
        order = np.argsort(-counts)          # 从多到少排序
        n = num_classes
        head = order[: max(1, n // 3)]
        mid = order[max(1, n // 3): max(2, 2 * n // 3)]
        tail = order[max(2, 2 * n // 3):]
        for tag, idx in [('Head', head), ('Mid ', mid), ('Tail', tail)]:
            logger.info(
                f'{tag} | 类数={len(idx):>3d} | '
                f'样本数={int(counts[idx].sum()):>6d} | '
                f'平均 F1={per_f1[idx].mean():.4f}'
            )


def dump_json(path, result, num_classes):
    obj = {
        'metrics': result['metrics'],
        'per_class_f1': result['per_class_f1'].tolist(),
        'class_counts': result['class_counts'].tolist(),
        'confusion_matrix': result['confusion_matrix'].tolist(),
        'num_classes': num_classes,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def dump_confusion_matrix_csv(path, cm):
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true\\pred'] + [f'p{i}' for i in range(cm.shape[0])])
        for i in range(cm.shape[0]):
            writer.writerow([f't{i}'] + cm[i].tolist())


# =========================================================
# main
# =========================================================
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger()

    # ---------- 1. 读取 checkpoint，确定 num_classes ----------
    assert os.path.isfile(args.ckpt), f'找不到 checkpoint: {args.ckpt}'
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    num_classes = args.num_classes
    if num_classes is None:
        ckpt_args = ckpt.get('args', None)
        if ckpt_args is not None and hasattr(ckpt_args, 'num_classes'):
            num_classes = ckpt_args.num_classes
        else:
            num_classes = 100
    logger.info(f'Checkpoint : {args.ckpt}')
    logger.info(f'Arch       : {args.arch}')
    logger.info(f'Num classes: {num_classes}')
    if 'epoch' in ckpt:
        logger.info(f'Best epoch : {ckpt.get("epoch")}')
    if 'best_f1' in ckpt:
        logger.info(f'Train-time best F1: {ckpt.get("best_f1"):.4f}')

    # ---------- 2. 构建模型并加载权重 ----------
    model = build_model(args.arch, num_classes).to(device)

    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    # 兼容 DataParallel / DDP 保存时带 "module." 前缀的情况
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f'Missing keys ({len(missing)}): {missing[:5]} ...')
    if unexpected:
        logger.warning(f'Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...')

    # ---------- 3. 数据集 ----------
    dataset = ViTacDataset(args.data_root, args.txt_dir, split=args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    logger.info(f'Eval split : {args.split}, samples: {len(dataset)}')

    # 尝试从 dataset 里抽取类别名（可选，取不到就跳过）
    class_names = None

    # ---------- 4. 评估 ----------
    result = evaluate(model, loader, device, num_classes)

    # ---------- 5. 输出 ----------
    print_report(result, num_classes, logger, class_names)

    if args.save_report:
        dump_json(args.save_report, result, num_classes)
        logger.info(f'详细报告已保存: {args.save_report}')
    if args.confusion_matrix_csv:
        dump_confusion_matrix_csv(args.confusion_matrix_csv, result['confusion_matrix'])
        logger.info(f'混淆矩阵已保存: {args.confusion_matrix_csv}')


if __name__ == '__main__':
    main()