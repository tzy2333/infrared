import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# ==================== 1. 配置参数 (请仔细核对) ====================
# 输入路径
img_dir = Path(r"D:\yanhuo\reality")
label_dir = Path(r"D:\yanhuo\fasdd\annotations\YOLO\labels")

# 输出路径 (会自动创建)
output_root = img_dir.parent / "my_dataset"

# 类别名称 (请按 classes.txt 的顺序填写，用于显示统计结果)
# 如果你不知道顺序，就填 ['class_0', 'class_1', ...]
class_names = ['fire', 'smoke']  # 假设 0是烟, 1是火，请根据你实际情况修改！

# 划分比例
ratios = [0.8, 0.1, 0.1]

# 支持的图片后缀
img_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# ================================================================

def analyze_and_split():
    print("🚀 正在启动：数据清洗、类别统计与划分程序...\n")

    # --- 步骤 1: 建立索引与清洗 ---
    print("1️⃣ 正在建立索引...")
    images_map = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in img_formats}

    # 建立标注映射（处理文件名空格和后缀问题）
    labels_map = {}
    for p in label_dir.glob('*.txt'):
        if p.name == "classes.txt": continue  # 跳过类别文件
        clean_name = p.stem.replace('.jpg', '').replace('.png', '').strip()
        labels_map[clean_name] = p

    # --- 步骤 2: 智能配对 ---
    print("2️⃣ 正在配对并剔除无效数据...")
    valid_pairs = []  # 存放 (img_path, label_path, stem)

    for stem, img_path in images_map.items():
        # 优先精确匹配
        if stem in labels_map:
            valid_pairs.append((img_path, labels_map[stem], stem))
        else:
            # 模糊匹配 (忽略大小写)
            for l_stem, l_path in labels_map.items():
                if l_stem.lower() == stem.lower():
                    valid_pairs.append((img_path, l_path, stem))
                    break

    skipped_count = len(images_map) - len(valid_pairs)
    print(f"   - 原始图片总数: {len(images_map)}")
    print(f"   - 有效配对数量: {len(valid_pairs)}")
    print(f"   - 🗑️ 已剔除无标注图片: {skipped_count} 张 (避免误导模型)")

    if len(valid_pairs) == 0:
        print("❌ 错误：未找到任何有效配对！请检查路径。")
        return

    # --- 步骤 3: 类别统计 (核心新增功能) ---
    print("\n3️⃣ 正在统计有效数据中的类别分布...")
    class_counter = Counter()

    # 读取每一个有效的 txt 文件
    for _, label_path, _ in tqdm(valid_pairs, desc="分析类别"):
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])  # 获取每一行的第一个数字
                        class_counter[class_id] += 1
        except Exception as e:
            print(f"⚠️ 读取文件 {label_path.name} 出错: {e}")

    print("\n📊 类别统计报告 (基于有效数据):")
    print("-" * 40)
    sorted_ids = sorted(class_counter.keys())
    for cid in sorted_ids:
        name = class_names[cid] if cid < len(class_names) else f"未知类别_{cid}"
        count = class_counter[cid]
        print(f"   类别 ID [{cid}] ({name}): \t{count} 个实例")
    print("-" * 40)
    print("💡 请确认：如果有某个类别数量极少(例如<100)，建议去检查丢失的那300个文件。")
    print("   如果数量都比较健康，可以直接继续。\n")

    # --- 步骤 4: 划分与输出 ---
    confirm = input("按回车键开始划分数据集，或输入 'q' 退出: ")
    if confirm.lower() == 'q': return

    print("\n4️⃣ 正在划分并复制文件...")
    random.shuffle(valid_pairs)

    total = len(valid_pairs)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])

    datasets = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }

    for split, pairs in datasets.items():
        save_dir = output_root / split
        (save_dir / 'images').mkdir(parents=True, exist_ok=True)
        (save_dir / 'labels').mkdir(parents=True, exist_ok=True)

        for img_path, label_path, stem in tqdm(pairs, desc=f"生成 {split}"):
            shutil.copy(img_path, save_dir / 'images' / img_path.name)
            # 强制重命名为标准格式，防止后续训练报错
            shutil.copy(label_path, save_dir / 'labels' / f"{stem}.txt")

    print("\n✅ 全部完成！")
    print(f"📂 数据集保存在: {output_root}")


if __name__ == "__main__":
    analyze_and_split()