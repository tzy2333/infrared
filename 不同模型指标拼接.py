import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 不弹窗，直接保存，避免 backend 报错

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1) 在这里填写你的模型名称和对应 results.csv 路径
#    想加几个就加几个
# =========================================================
MODEL_CSVS = {
    "yolov5s": r"C:\Users\TZY\Desktop\ultralytics\runs\detect\yolov5s\results.csv",
    "yolov8s": r"C:\Users\TZY\Desktop\ultralytics\runs\detect\yolov8s\results.csv",
    "v8-GCP":  r"C:\Users\TZY\Desktop\ultralytics\runs\detect\yolov8s+p6+lk+gc2f2+bifpn\results.csv",
}

SAVE_PATH = r"C:\Users\TZY\Desktop\ultralytics\multi_models_2x4_curves.png"


# =========================================================
# 2) 固定画这 8 个指标，和你发的那种图一致
# =========================================================
METRICS = [
    ("train/box_loss", ["train/box_loss", "box_loss"]),
    ("train/dfl_loss", ["train/dfl_loss", "dfl_loss"]),
    ("precision", ["metrics/precision(B)", "metrics/precision", "precision"]),
    ("recall", ["metrics/recall(B)", "metrics/recall", "recall"]),

    ("val/box_loss", ["val/box_loss"]),
    ("val/dfl_loss", ["val/dfl_loss"]),
    ("mAP_0.5", ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"]),
    ("mAP_0.5:0.95", ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"]),
]


# =========================================================
# 3) 找真实列名
# =========================================================
def find_real_column(df, aliases):
    for col in aliases:
        if col in df.columns:
            return col
    return None


# =========================================================
# 4) 读取 results.csv
# =========================================================
def load_results(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "epoch" in df.columns:
        x = df["epoch"].values
    else:
        x = list(range(len(df)))

    return df, x


# =========================================================
# 5) 主函数：多模型叠加，固定 2×4
# =========================================================
def plot_multi_models_2x4(model_csvs, save_path, dpi=600):
    model_data = {}

    for model_name, csv_path in model_csvs.items():
        csv_path = str(Path(csv_path))
        if not Path(csv_path).exists():
            print("[跳过] 文件不存在: {} -> {}".format(model_name, csv_path))
            continue

        try:
            df, x = load_results(csv_path)
            model_data[model_name] = (df, x)
            print("[已读取] {} -> {}".format(model_name, csv_path))
        except Exception as e:
            print("[跳过] 读取失败: {} -> {}\n原因: {}".format(model_name, csv_path, e))

    if not model_data:
        raise ValueError("没有成功读取任何 results.csv，请检查路径。")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for i, (title_name, aliases) in enumerate(METRICS):
        ax = axes[i]
        plotted = False

        for model_name, (df, x) in model_data.items():
            real_col = find_real_column(df, aliases)
            if real_col is None:
                continue

            y = df[real_col].values
            ax.plot(x, y, linewidth=1.5, label=model_name)
            plotted = True

        ax.set_title(title_name, fontsize=11)
        ax.set_xlabel("epoch", fontsize=9)
        ax.tick_params(labelsize=8)

        if plotted:
            ax.legend(fontsize=8, frameon=True)

    plt.tight_layout()

    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print("\n已保存: {}".format(save_path))


# =========================================================
# 6) 运行
# =========================================================
if __name__ == "__main__":
    plot_multi_models_2x4(
        model_csvs=MODEL_CSVS,
        save_path=SAVE_PATH,
        dpi=600
    )