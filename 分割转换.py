from pathlib import Path
import shutil

# 原始 YOLO 分割数据集
src_root = Path(r"C:/Users/TZY/Desktop/ultralytics/ultralytics/cfg/datasets/my_dataset")

# 输出 YOLO 检测数据集
out_root = Path(r"C:/Users/TZY/Desktop/ultralytics/ultralytics/cfg/datasets/my_dataset_det")

image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def convert_label_file(src_txt: Path, dst_txt: Path):
    det_lines = []

    with open(src_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            # 分割格式至少需要 class + 3个点 = 1 + 6个坐标
            continue

        cls_id = parts[0]
        coords = list(map(float, parts[1:]))

        if len(coords) % 2 != 0:
            print(f"坐标数量不是偶数，跳过: {src_txt.name}")
            continue

        xs = coords[0::2]
        ys = coords[1::2]

        x_min = max(0.0, min(xs))
        x_max = min(1.0, max(xs))
        y_min = max(0.0, min(ys))
        y_max = min(1.0, max(ys))

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        if width <= 0 or height <= 0:
            print(f"无效框，跳过: {src_txt.name}")
            continue

        det_lines.append(
            f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(det_lines))


for split in ["train", "val"]:
    src_img_dir = src_root / split / "images"
    src_label_dir = src_root / split / "labels"

    out_img_dir = out_root / split / "images"
    out_label_dir = out_root / split / "labels"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    # 复制图片
    for img_path in src_img_dir.iterdir():
        if img_path.suffix.lower() in image_exts:
            shutil.copy2(img_path, out_img_dir / img_path.name)

    # 转换标签
    for txt_path in src_label_dir.glob("*.txt"):
        convert_label_file(txt_path, out_label_dir / txt_path.name)

print("分割标签已转换为检测标签！")
print(f"输出目录: {out_root}")