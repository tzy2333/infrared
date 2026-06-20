import os
import matplotlib
matplotlib.use("Agg")  # 强制无GUI后端，避免 backend_interagg 报错

import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 只需要改这里两行
# =========================
RUN_DIR  = Path(r"C:\Users\TZY\Desktop\ultralytics\runs\detect\yolov8s+fep+lk+c2fg")   # 你的训练输出目录（里面有 results.csv）
OUT_FILE = Path(r"C:\Users\TZY\Desktop\ultralytics\runs\detect\yolov8s+fep+lk+c2fg")        # 你要保存到哪（png 或 pdf）
# 如果你更想直接写 results.csv 的完整路径，也可以：
# CSV_PATH = Path(r"D:\ultralytics\runs\detect\train23\results.csv")
# =========================

PLOTS = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]

def find_col(columns, name):
    if name in columns:
        return name
    target = name.lower().replace(" ", "")
    for c in columns:
        cc = c.lower().replace(" ", "")
        if cc == target:
            return c
    for c in columns:
        if target in c.lower().replace(" ", ""):
            return c
    return None

def smooth(y, win=11):
    if win <= 1:
        return y
    return pd.Series(y).rolling(win, center=True, min_periods=1).mean().to_numpy()

def main():
    # 1) 定位 results.csv
    csv_path = RUN_DIR / "results.csv"
    # 如果你用了 CSV_PATH 方式，就把上面一行换成：csv_path = CSV_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 results.csv：{csv_path}")

    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    # x轴（epoch）
    if "epoch" in cols:
        x = df["epoch"].to_numpy()
    else:
        x = np.arange(len(df))

    # 2×5 拼图（图8）
    dpi = 300
    fig, axes = plt.subplots(2, 5, figsize=(12, 5), dpi=dpi)
    axes = axes.flatten()

    missing = []
    for i, key in enumerate(PLOTS):
        ax = axes[i]
        real_col = find_col(cols, key)
        if real_col is None:
            missing.append(key)
            ax.set_title(key, fontsize=9)
            ax.text(0.5, 0.5, "MISSING", ha="center", va="center")
            ax.set_axis_off()
            continue

        y = df[real_col].to_numpy(dtype=float)
        ys = smooth(y, win=11)

        ax.plot(x, y, label="results")
        ax.plot(x, ys, label="smooth")
        ax.set_title(key, fontsize=9)
        ax.tick_params(labelsize=7)

        if i == 1:
            ax.legend(fontsize=7)
        else:
            ax.legend().remove()

    plt.tight_layout()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if str(OUT_FILE).lower().endswith(".pdf"):
        plt.savefig(OUT_FILE, bbox_inches="tight")
    else:
        plt.savefig(OUT_FILE, bbox_inches="tight", dpi=dpi)
    print(f"Saved: {OUT_FILE}")

    if missing:
        print("\n警告：results.csv 没找到这些列（可能是你ultralytics版本列名不同）：")
        for m in missing:
            print("  -", m)

if __name__ == "__main__":
    main()
