import os
import re
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================== 参数 ====================
ALL_ITEM = "__all_item.txt"
OUT_DIR = "."

SEED = 42
NUM_CLASSES = 30        # 所保留的类别数量。例如30 -> 保留0~29类
C_EXPECTED = NUM_CLASSES

# 长尾不平衡因子：训练集最大类保留 seq 数 / 最小类保留 seq 数
IM = 50

# 每个保留的 seq 中随机保留的样本数
M = 10

# 先按 seq 做 7:2:1 划分
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
# =============================================

rng = random.Random(SEED)
np.random.seed(SEED)


def parse_line(line):
    """解析一行，返回 (原始行, 真实类别id, seq_key)"""
    line = line.rstrip("\n").rstrip("\r")
    if not line:
        return None

    parts = line.split("\\")
    if len(parts) < 4:
        return None

    # 例如 CD0011-a1 -> 0011 -> 11 -> 真实类别 0
    m = re.match(r"CD(\d+)-", parts[0])
    if not m:
        return None

    class_id = int(m.group(1)) - 11

    # seq_key 到 seq_id 为止，例如 CD0011-a1\press\10
    seq_key = "\\".join(parts[:-1])
    return line, class_id, seq_key


# 读取 __all_item.txt
records = []
with open(os.path.join(OUT_DIR, ALL_ITEM), "r", encoding="utf-8") as f:
    for line in f:
        parsed = parse_line(line)
        if parsed is None:
            continue

        _, cid, _ = parsed

        # 只保留前 NUM_CLASSES 类：0, 1, ..., NUM_CLASSES-1
        if 0 <= cid < NUM_CLASSES:
            records.append(parsed)

print(f"读取样本数: {len(records)}")

# 按类别 -> seq -> 样本行 分组
class_to_seq = defaultdict(lambda: defaultdict(list))
for line, cid, seq in records:
    class_to_seq[cid][seq].append(line)

all_classes = sorted(class_to_seq.keys())
C = len(all_classes)
print(f"类别数: {C}")

if C_EXPECTED is not None and C != C_EXPECTED:
    print(f"警告: 实际类别数为 {C}，与预期 {C_EXPECTED} 不一致。")

# 随机打乱类别顺序，决定哪些类成为多数类、哪些成为少数类
class_order = all_classes.copy()
rng.shuffle(class_order)
class_rank = {cid: i for i, cid in enumerate(class_order)}

# ==================== 第一步：按 seq 做 7:2:1 划分 ====================
train_seq_keys = set()
val_seq_keys = set()
test_seq_keys = set()

# 记录每个类别的 train/val/test seq
class_train_seqs = defaultdict(list)
class_val_seqs = defaultdict(list)
class_test_seqs = defaultdict(list)

for cid in all_classes:
    seqs = list(class_to_seq[cid].keys())
    rng.shuffle(seqs)
    N = len(seqs)

    if N == 0:
        continue

    # 按 seq 数量比例划分，尽量保证 7:2:1
    train_n = int(round(N * TRAIN_RATIO))
    val_n = int(round(N * VAL_RATIO))
    test_n = N - train_n - val_n

    # 鲁棒性处理：至少保证 train 有 seq；如果数量足够，val/test 也有
    if train_n < 1:
        train_n = 1
    if N >= 3:
        if val_n < 1:
            val_n = 1
        if test_n < 1:
            test_n = 1

    # 重新调整，保证总数不超过 N
    while train_n + val_n + test_n > N:
        if train_n > 1:
            train_n -= 1
        elif val_n > 0:
            val_n -= 1
        elif test_n > 0:
            test_n -= 1
        else:
            break

    if train_n + val_n + test_n > N:
        train_n = N - val_n - test_n
        if train_n < 0:
            train_n = 0

    train_seqs = seqs[:train_n]
    val_seqs = seqs[train_n:train_n + val_n]
    test_seqs = seqs[train_n + val_n:train_n + val_n + test_n]

    class_train_seqs[cid] = train_seqs
    class_val_seqs[cid] = val_seqs
    class_test_seqs[cid] = test_seqs

    for s in train_seqs:
        train_seq_keys.add(s)
    for s in val_seqs:
        val_seq_keys.add(s)
    for s in test_seqs:
        test_seq_keys.add(s)

print(f"划分完成：train seq={len(train_seq_keys)}, val seq={len(val_seq_keys)}, test seq={len(test_seq_keys)}")

# ==================== 第二步：训练集 seq 层面长尾改造 ====================
# 统计每个类训练集可用 seq 数
avail_seq_counts = {cid: len(class_train_seqs[cid]) for cid in all_classes}
min_avail_seq = min(avail_seq_counts.values())
max_avail_seq = max(avail_seq_counts.values())

print(f"训练集每类可用 seq 数：min={min_avail_seq}, max={max_avail_seq}")

# 自动确定 N_MAX / N_MIN，使 N_MAX / N_MIN ≈ IM
# 这里 N_MAX / N_MIN 是训练集最大类保留 seq 数 / 最小类保留 seq 数
N_MIN = max(1, min_avail_seq // IM)
N_MAX = N_MIN * IM

if N_MAX > min_avail_seq:
    N_MAX = min_avail_seq
    N_MIN = max(1, N_MAX // IM)
    N_MAX = N_MIN * IM

print(f"长尾目标（seq 粒度）：N_MAX={N_MAX}, N_MIN={N_MIN}, IM≈{N_MAX / max(1, N_MIN):.2f}")

# 按随机类别顺序做指数衰减，得到每个类的目标保留 seq 数
class_target_seqs = {}
for cid in all_classes:
    rank = class_rank[cid]
    if C > 1:
        target = round(N_MAX * (N_MIN / N_MAX) ** (rank / (C - 1)))
    else:
        target = N_MAX
    target = max(1, int(target))
    # 不超过可用 seq 数
    target = min(target, len(class_train_seqs[cid]))
    class_target_seqs[cid] = target

# 在 seq 层面选择，并在每个选中的 seq 中随机保留 M 个样本
train_selected_seqs = set()
train_selected_lines = set()
train_actual_seqs = defaultdict(set)
train_actual_samples = {}

for cid in all_classes:
    available_seqs = class_train_seqs[cid]
    target = class_target_seqs[cid]

    if target >= len(available_seqs):
        selected_seqs = available_seqs[:]
    else:
        selected_seqs = rng.sample(available_seqs, target)

    sample_count = 0
    for seq in selected_seqs:
        train_selected_seqs.add(seq)
        samples = class_to_seq[cid][seq]

        if len(samples) <= M:
            selected_samples = samples[:]
        else:
            selected_samples = rng.sample(samples, M)

        for line in selected_samples:
            train_selected_lines.add(line)

        sample_count += len(selected_samples)
        train_actual_seqs[cid].add(seq)

    train_actual_samples[cid] = sample_count

# ==================== 第三步：写出 train / val / test ====================
def write_split(filename, selected_lines=None, selected_seqs=None):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line, cid, seq in records:
            if selected_lines is not None and line in selected_lines:
                f.write(line + "\n")
            elif selected_seqs is not None and seq in selected_seqs:
                f.write(line + "\n")
    print(f"已写出 {filename}")


write_split("train.txt", selected_lines=train_selected_lines)
write_split("val.txt", selected_seqs=val_seq_keys)
write_split("test.txt", selected_seqs=test_seq_keys)

# ==================== 第四步：统计 ====================
def compute_split_stats(selected_lines=None, selected_seqs=None):
    stats = defaultdict(lambda: {"seqs": set(), "samples": 0})
    for line, cid, seq in records:
        if selected_lines is not None and line in selected_lines:
            stats[cid]["seqs"].add(seq)
            stats[cid]["samples"] += 1
        elif selected_seqs is not None and seq in selected_seqs:
            stats[cid]["seqs"].add(seq)
            stats[cid]["samples"] += 1
    return stats


train_stats = compute_split_stats(selected_lines=train_selected_lines)
val_stats = compute_split_stats(selected_seqs=val_seq_keys)
test_stats = compute_split_stats(selected_seqs=test_seq_keys)

rows = []
for cid in all_classes:
    rows.append({
        "class_id": cid,
        "class_rank": class_rank[cid],
        "train_available_seqs": len(class_train_seqs[cid]),
        "train_target_seqs": class_target_seqs[cid],
        "train_actual_seqs": len(train_stats[cid]["seqs"]),
        "train_actual_samples": train_stats[cid]["samples"],
        "val_seqs": len(val_stats[cid]["seqs"]),
        "val_samples": val_stats[cid]["samples"],
        "test_seqs": len(test_stats[cid]["seqs"]),
        "test_samples": test_stats[cid]["samples"],
    })

df = pd.DataFrame(rows)
df.to_csv(
    os.path.join(OUT_DIR, "dataset_split_statistics.csv"),
    index=False,
    encoding="utf-8-sig"
)

df_train = df[
    [
        "class_id",
        "class_rank",
        "train_available_seqs",
        "train_target_seqs",
        "train_actual_seqs",
        "train_actual_samples",
    ]
].sort_values("train_actual_samples", ascending=False).reset_index(drop=True)

df_train.to_csv(
    os.path.join(OUT_DIR, "longtail_train_statistics.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 保存随机类别顺序
pd.DataFrame({
    "class_id": class_order,
    "rank": range(len(class_order)),
}).to_csv(
    os.path.join(OUT_DIR, "class_order.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ==================== 第五步：绘图 ====================
# 训练集长尾分布：seq 粒度 + 样本粒度
df_plot = df.sort_values("train_actual_samples", ascending=False).reset_index(drop=True)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

axes[0].bar(df_plot.index, df_plot["train_actual_seqs"], color="darkorange")
axes[0].set_xlabel("Class (sorted by train samples)")
axes[0].set_ylabel("Number of sequences")
axes[0].set_title("Training Set Long-tailed Distribution (Seq-level)")

axes[1].bar(df_plot.index, df_plot["train_actual_samples"], color="steelblue")
axes[1].set_xlabel("Class (sorted by train samples)")
axes[1].set_ylabel("Number of samples")
axes[1].set_title("Training Set Long-tailed Distribution (Sample-level)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "train_longtail_distribution.png"), dpi=300)
plt.close()

# 验证集 / 测试集样本分布
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

axes[0].bar(df["class_id"], df["val_samples"], color="green")
axes[0].set_xlabel("Class ID")
axes[0].set_ylabel("Number of samples")
axes[0].set_title("Validation Set Distribution (Sample-level)")

axes[1].bar(df["class_id"], df["test_samples"], color="purple")
axes[1].set_xlabel("Class ID")
axes[1].set_ylabel("Number of samples")
axes[1].set_title("Test Set Distribution (Sample-level)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "val_test_distribution.png"), dpi=300)
plt.close()

print("完成。生成文件：")
print("  train.txt")
print("  val.txt")
print("  test.txt")
print("  dataset_split_statistics.csv")
print("  longtail_train_statistics.csv")
print("  class_order.csv")
print("  train_longtail_distribution.png")
print("  val_test_distribution.png")