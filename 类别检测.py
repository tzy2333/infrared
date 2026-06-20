from pathlib import Path
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ✅ 直接写你的数据集根目录（包含 train/val/test）
ROOT = Path(r"C:\Users\TZY\Desktop\ultralytics\ultralytics\cfg\datasets\my_dataset")

def count_split(split: str, nc=2):
    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"

    images = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

    obj = Counter()
    img_has = Counter()
    missing_labels = 0
    empty_labels = 0

    for im in images:
        lp = lbl_dir / f"{im.stem}.txt"
        if not lp.exists():
            missing_labels += 1
            continue
        txt = lp.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            empty_labels += 1
            continue

        classes_in_img = set()
        for line in txt.splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            obj[cls] += 1
            classes_in_img.add(cls)
        for c in classes_in_img:
            img_has[c] += 1

    # ensure keys
    for k in range(nc):
        obj[k] += 0
        img_has[k] += 0

    return {
        "split": split,
        "images": len(images),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "obj": obj,
        "img_has": img_has,
        "total_obj": sum(obj.values()),
    }

def show(rep):
    print(f"\n=== {rep['split']} ===")
    print(f"Images: {rep['images']}")
    print(f"Missing labels: {rep['missing_labels']}  Empty labels: {rep['empty_labels']}")
    print(f"Total objects: {rep['total_obj']}")
    print(f"Objects:  smoke(0)={rep['obj'][0]}  fire(1)={rep['obj'][1]}")
    print(f"Imgs w/:  smoke(0)={rep['img_has'][0]}  fire(1)={rep['img_has'][1]}")
    if rep["obj"][0] == 0:
        print("Obj ratio fire/smoke: inf" if rep["obj"][1] > 0 else "Obj ratio fire/smoke: 0")
    else:
        print(f"Obj ratio fire/smoke: {rep['obj'][1]/rep['obj'][0]:.3f}")

def main():
    splits = ["train", "val", "test"]
    total_obj = Counter()
    total_img = Counter()
    totals = Counter()

    for sp in splits:
        rep = count_split(sp, nc=2)
        show(rep)
        total_obj.update(rep["obj"])
        total_img.update(rep["img_has"])
        totals["images"] += rep["images"]
        totals["missing"] += rep["missing_labels"]
        totals["empty"] += rep["empty_labels"]

    print("\n=== overall ===")
    print(f"Images: {totals['images']}")
    print(f"Missing labels: {totals['missing']}  Empty labels: {totals['empty']}")
    print(f"Objects:  smoke(0)={total_obj[0]}  fire(1)={total_obj[1]}")
    print(f"Imgs w/:  smoke(0)={total_img[0]}  fire(1)={total_img[1]}")
    if total_obj[0] == 0:
        print("Obj ratio fire/smoke: inf" if total_obj[1] > 0 else "Obj ratio fire/smoke: 0")
    else:
        print(f"Obj ratio fire/smoke: {total_obj[1]/total_obj[0]:.3f}")

if __name__ == "__main__":
    main()
