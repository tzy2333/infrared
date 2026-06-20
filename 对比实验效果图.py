import os
import re
from PIL import Image

# =========================
# 配置区：改这里就可以
# =========================
ROOT_DIR = r"C:\Users\TZY\Desktop\文件\对比"   # 总文件夹路径
OUTPUT_PATH = r"C:\Users\TZY\Desktop\文件\对比\compare_result.jpg"  # 输出图片路径

# 支持的图片格式
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 是否把同一行的图片统一高度
RESIZE_TO_SAME_HEIGHT = True

# 统一高度时使用的目标高度
TARGET_HEIGHT = 320

# 图片之间的间距
H_GAP = 10   # 横向拼接时每张图之间的间距
V_GAP = 10   # 纵向拼接时每一行之间的间距

# 背景颜色（白色）
BG_COLOR = (255, 255, 255)


def natural_key(s: str):
    """
    自然排序：1, 2, 10，而不是 1, 10, 2
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def resize_keep_ratio(img: Image.Image, target_height: int) -> Image.Image:
    """
    按目标高度等比例缩放
    """
    w, h = img.size
    if h == target_height:
        return img
    new_w = int(w * target_height / h)
    return img.resize((new_w, target_height), Image.LANCZOS)


def load_images_from_folder(folder_path: str):
    """
    读取文件夹内所有图片，按自然顺序排序
    """
    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(IMG_EXTS)]
    files.sort(key=natural_key)

    images = []
    for f in files:
        img_path = os.path.join(folder_path, f)
        img = Image.open(img_path).convert("RGB")
        images.append((f, img))
    return images


def stitch_horizontal(images, gap=10, bg_color=(255, 255, 255), resize_to_same_height=True, target_height=320):
    """
    把一组图片横向拼接成一行
    images: [(filename, PIL.Image), ...]
    """
    if not images:
        return None

    pil_images = [img for _, img in images]

    if resize_to_same_height:
        pil_images = [resize_keep_ratio(img, target_height) for img in pil_images]

    heights = [img.size[1] for img in pil_images]
    widths = [img.size[0] for img in pil_images]

    row_height = max(heights)
    row_width = sum(widths) + gap * (len(pil_images) - 1)

    row_canvas = Image.new("RGB", (row_width, row_height), bg_color)

    x = 0
    for img in pil_images:
        y = (row_height - img.size[1]) // 2
        row_canvas.paste(img, (x, y))
        x += img.size[0] + gap

    return row_canvas


def stitch_vertical(rows, gap=10, bg_color=(255, 255, 255)):
    """
    把多行图片纵向拼接
    rows: [PIL.Image, PIL.Image, ...]
    """
    if not rows:
        return None

    max_width = max(row.size[0] for row in rows)
    total_height = sum(row.size[1] for row in rows) + gap * (len(rows) - 1)

    final_canvas = Image.new("RGB", (max_width, total_height), bg_color)

    y = 0
    for row in rows:
        x = (max_width - row.size[0]) // 2
        final_canvas.paste(row, (x, y))
        y += row.size[1] + gap

    return final_canvas


def main():
    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(f"总文件夹不存在: {ROOT_DIR}")

    subfolders = [d for d in os.listdir(ROOT_DIR)
                  if os.path.isdir(os.path.join(ROOT_DIR, d))]
    subfolders.sort(key=natural_key)

    if not subfolders:
        raise ValueError(f"总文件夹下没有找到子文件夹: {ROOT_DIR}")

    row_images = []

    for folder_name in subfolders:
        folder_path = os.path.join(ROOT_DIR, folder_name)
        images = load_images_from_folder(folder_path)

        if not images:
            print(f"跳过空文件夹: {folder_path}")
            continue

        row = stitch_horizontal(
            images,
            gap=H_GAP,
            bg_color=BG_COLOR,
            resize_to_same_height=RESIZE_TO_SAME_HEIGHT,
            target_height=TARGET_HEIGHT
        )

        row_images.append(row)
        print(f"已处理: {folder_name}，共 {len(images)} 张")

    if not row_images:
        raise ValueError("所有子文件夹都没有可用图片。")

    final_img = stitch_vertical(
        row_images,
        gap=V_GAP,
        bg_color=BG_COLOR
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_img.save(OUTPUT_PATH, quality=95)
    print(f"完成，输出路径: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()