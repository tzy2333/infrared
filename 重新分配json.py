import os
import json
import shutil

# ========== 配置（根据你的实际路径修改） ==========
JSON_DIR = r"D:\红外图像\data"
# ================================================

json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]

for json_name in json_files:
    json_path = os.path.join(JSON_DIR, json_name)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_path = data.get("imagePath", "")

    # 计算图片的绝对路径（基于 data 文件夹和 imagePath 中的相对路径）
    absolute_image_path = os.path.normpath(os.path.join(JSON_DIR, image_path))

    if not os.path.exists(absolute_image_path):
        print(f"[跳过] {json_name}: 图片不存在 -> {absolute_image_path}")
        continue

    # 修改 imagePath 为纯文件名
    image_filename = os.path.basename(absolute_image_path)
    data["imagePath"] = image_filename

    # 写回 JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 获取图片所在的文件夹
    image_dir = os.path.dirname(absolute_image_path)

    # 如果 JSON 已经在正确的文件夹中，则跳过移动
    if image_dir == JSON_DIR:
        print(f"[修正] {json_name}: 路径已修正为 {image_filename}")
        continue

    # 移动 JSON 到图片文件夹
    dest_path = os.path.join(image_dir, json_name)
    shutil.move(json_path, dest_path)
    print(f"[完成] {json_name} -> {dest_path}")

print("\n全部处理完毕。")