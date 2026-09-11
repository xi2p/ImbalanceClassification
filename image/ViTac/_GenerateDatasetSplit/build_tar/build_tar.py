import os
import tarfile

# ==================== 配置 ====================
DATASET_ROOT = r"E:\PythonProjects\SlipDetection\ViTac\vitac_dataset"
TXT_FILES = ["../../train.txt", "../../val.txt", "../../test.txt"]
OUTPUT_TAR = "vitac-LT.tar"
# ==============================================

def main():
    # 收集三个 txt 中所有相对路径，并去重
    rel_paths = set()

    for txt in TXT_FILES:
        if not os.path.exists(txt):
            print(f"警告: {txt} 不存在，跳过")
            continue

        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 统一转成正斜杠，便于去重和写入 tar
                rel = line.replace("\\", "/")
                rel_paths.add(rel)

    print(f"三个 txt 合计去重后文件数: {len(rel_paths)}")

    missing = []
    added = 0
    total_bytes = 0

    # 开始打包
    with tarfile.open(OUTPUT_TAR, "w") as tar:
        for rel in sorted(rel_paths):
            # 在 Windows 上，实际文件路径用 os.sep 拼接
            src_path = os.path.join(DATASET_ROOT, rel.replace("/", os.sep))

            if not os.path.isfile(src_path):
                missing.append(rel)
                continue

            # arcname 使用正斜杠，解压后目录结构与原始数据集一致
            tar.add(src_path, arcname=rel)

            added += 1
            total_bytes += os.path.getsize(src_path)

            if added % 10000 == 0:
                print(f"已添加 {added} 个文件...")

    print("=" * 60)
    print(f"打包完成: {OUTPUT_TAR}")
    print(f"成功添加文件数: {added}")
    print(f"总大小: {total_bytes / 1024**3:.2f} GB")

    if missing:
        print(f"缺失文件数: {len(missing)}")
        print("前 10 个缺失文件示例:")
        for m in missing[:10]:
            print("  ", m)

        with open("missing_files.txt", "w", encoding="utf-8") as f:
            for m in missing:
                f.write(m + "\n")

        print("完整缺失列表已保存到 missing_files.txt")
    else:
        print("没有缺失文件。")


if __name__ == "__main__":
    main()