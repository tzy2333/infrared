import os
import cv2
import numpy as np
from pathlib import Path


def imread_chinese(path):
    """读取包含中文路径的图片。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_chinese(path, img):
    """保存到包含中文路径的位置。"""
    ext = Path(path).suffix
    success, encoded_img = cv2.imencode(ext, img)

    if success:
        encoded_img.tofile(str(path))
        return True

    return False


def resize_with_padding(img, target_width, target_height, background_color=(255, 255, 255)):
    """
    保持图片比例，将图片缩放到指定单元格内。
    空白区域使用背景色填充。
    """
    height, width = img.shape[:2]

    scale = min(target_width / width, target_height / height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized_img = cv2.resize(
        img,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.full(
        (target_height, target_width, 3),
        background_color,
        dtype=np.uint8
    )

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized_img

    return canvas


def combine_multiple_images(
    folders_list,
    output_folder,
    rows,
    cols,
    gap=10,
    background_color=(255, 255, 255)
):
    """
    将多个文件夹中同名的图片按照指定行列数拼接。

    参数：
        folders_list:
            图片文件夹列表。每个文件夹代表网格中的一个位置。

        output_folder:
            拼接结果保存位置。

        rows:
            网格行数。

        cols:
            网格列数。

        gap:
            图片之间的间距，单位为像素。

        background_color:
            背景颜色，OpenCV 使用 BGR 顺序。
            白色：(255, 255, 255)
            黑色：(0, 0, 0)
    """
    if not folders_list:
        raise ValueError("folders_list 不能为空。")

    if rows <= 0 or cols <= 0:
        raise ValueError("rows 和 cols 必须大于 0。")

    if rows * cols < len(folders_list):
        raise ValueError(
            f"网格容量不足：当前有 {len(folders_list)} 个文件夹，"
            f"但 {rows} 行 × {cols} 列只能放置 {rows * cols} 张图片。"
        )

    os.makedirs(output_folder, exist_ok=True)

    first_folder = Path(folders_list[0])

    valid_exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff",
        ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF"
    }

    image_files = [
        p for p in first_folder.iterdir()
        if p.is_file() and p.suffix in valid_exts
    ]

    image_files.sort()

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for img_path in image_files:
        img_paths = []
        missing_files = []

        # 检查所有文件夹中是否存在完全同名的图片
        for folder in folders_list:
            current_path = Path(folder) / img_path.name

            if not current_path.exists():
                missing_files.append(str(current_path))
            else:
                img_paths.append(current_path)

        if missing_files:
            skipped_count += 1
            print(f"\n跳过 {img_path.name}：以下文件不存在")

            for p in missing_files:
                print("   ", p)

            continue

        try:
            images = []

            for path in img_paths:
                img = imread_chinese(path)

                if img is None:
                    raise ValueError(f"无法读取图片：{path}")

                images.append(img)

            # 使用第一张图片的尺寸作为每个网格单元的尺寸
            cell_height, cell_width = images[0].shape[:2]

            grid_images = []

            for img in images:
                resized_img = resize_with_padding(
                    img=img,
                    target_width=cell_width,
                    target_height=cell_height,
                    background_color=background_color
                )

                grid_images.append(resized_img)

            # 如果网格位置数量多于图片数量，补充空白图片
            blank_img = np.full(
                (cell_height, cell_width, 3),
                background_color,
                dtype=np.uint8
            )

            while len(grid_images) < rows * cols:
                grid_images.append(blank_img.copy())

            # 创建图片间距
            vertical_gap = np.full(
                (cell_height, gap, 3),
                background_color,
                dtype=np.uint8
            )

            horizontal_gap = np.full(
                (gap, cols * cell_width + (cols - 1) * gap, 3),
                background_color,
                dtype=np.uint8
            )

            combined_rows = []

            for row_index in range(rows):
                row_images = grid_images[
                    row_index * cols:(row_index + 1) * cols
                ]

                row_parts = []

                for col_index, image in enumerate(row_images):
                    row_parts.append(image)

                    if col_index < cols - 1:
                        row_parts.append(vertical_gap)

                combined_row = np.hstack(row_parts)
                combined_rows.append(combined_row)

            final_parts = []

            for row_index, combined_row in enumerate(combined_rows):
                final_parts.append(combined_row)

                if row_index < rows - 1:
                    final_parts.append(horizontal_gap)

            combined = np.vstack(final_parts)

            output_path = Path(output_folder) / img_path.name

            ok = imwrite_chinese(output_path, combined)

            if not ok:
                raise ValueError(f"保存失败：{output_path}")

            processed_count += 1
            print(f"已处理：{img_path.name}")

        except Exception as e:
            error_count += 1
            print(f"处理图片 {img_path.name} 时出错：{e}")

    print("\n处理完成！")
    print(f"成功：{processed_count} 张")
    print(f"跳过：{skipped_count} 张")
    print(f"报错：{error_count} 张")


if __name__ == "__main__":
    folders = [
        r"C:\Users\TZY\Desktop\ultralytics\runs\detect\test yolo11n-mslkaconv-vam-fpn4.0-dpgm 200",
        r"C:\Users\TZY\Desktop\ultralytics\runs\detect\test yolo11n-vam-fpn4.0-dpgm",
        r"C:\Users\TZY\Desktop\ultralytics\runs\detect\test yolo11n-mslkaconv-fpn4.0-dpgm",
        r"C:\Users\TZY\Desktop\ultralytics\runs\detect\test yolo11n-mslkaconv-vam-fpn4.0",
        r"C:\Users\TZY\Desktop\ultralytics\runs\detect\test yolo11n",
        # r"C:\Users\TZY\Desktop\ultralytics\runs\detect\test yolo11n-fpn4.0-dpgm",
        # r"C:\Users\TZY\Desktop\ultralytics\runs\detect\test yolo11n-vam-fpn4.0",
        # r"C:\Users\TZY\Desktop\ultralytics\runs\detect\test yolo11n-mslkaconv-fpn4.0",
    ]

    output_folder = r"C:\Users\TZY\Desktop\ultralytics\runs\detect\combined2"

    combine_multiple_images(
        folders_list=folders,
        output_folder=output_folder,
        rows=2,                       # 设置行数
        cols=3,                       # 设置列数
        gap=10,                       # 图片间距，可改为 0
        background_color=(255, 255, 255)  # 白色背景
    )