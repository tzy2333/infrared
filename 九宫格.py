import os
from PIL import Image

def make_grid_collage_to_folder(
    folder_path: str,
    rows: int = 3,
    cols: int = 3,
    out_name: str = "grid.jpg",
    cell_size: int = 512,     # 每格正方形边长
    margin: int = 12,
    bg_color=(255, 255, 255)
):
    if rows <= 0 or cols <= 0:
        raise ValueError("rows 和 cols 必须是正整数")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in exts
    ]
    files.sort(key=lambda p: os.path.basename(p).lower())

    need = rows * cols
    if len(files) < need:
        raise ValueError(f"需要 {need} 张图片（{rows}×{cols}），但文件夹里只有 {len(files)} 张。")

    files = files[:need]

    W = cols * cell_size + (cols - 1) * margin
    H = rows * cell_size + (rows - 1) * margin
    canvas = Image.new("RGB", (W, H), bg_color)

    def fit_to_square(img: Image.Image, size: int) -> Image.Image:
        """等比缩放+居中裁剪，得到 size×size 的正方形"""
        img = img.convert("RGB")
        w, h = img.size
        scale = max(size / w, size / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        left = (nw - size) // 2
        top = (nh - size) // 2
        return img.crop((left, top, left + size, top + size))

    for idx, path in enumerate(files):
        img = Image.open(path)
        tile = fit_to_square(img, cell_size)

        r = idx // cols
        c = idx % cols
        x = c * (cell_size + margin)
        y = r * (cell_size + margin)
        canvas.paste(tile, (x, y))

    # 输出到同一个文件夹
    out_path = os.path.join(folder_path, out_name)

    # 同名避免覆盖：自动加序号
    if os.path.exists(out_path):
        base, ext = os.path.splitext(out_name)
        k = 1
        while True:
            candidate = os.path.join(folder_path, f"{base}_{k}{ext}")
            if not os.path.exists(candidate):
                out_path = candidate
                break
            k += 1

    canvas.save(out_path, quality=95)
    print("已输出到：", out_path)


if __name__ == "__main__":
    folder = r"C:\Users\TZY\Desktop\222"  # ✅ 改成你的路径
    make_grid_collage_to_folder(
        folder_path=folder,
        rows=3,
        cols=3,
        out_name="nine_grid.jpg",
        cell_size=512,
        margin=12
    )
