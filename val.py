# val_model.py
from ultralytics import YOLO
from pathlib import Path
import yaml

import hashlib
from pathlib import Path
# 1. 模型路径
model_path = r"C:\Users\TZY\Desktop\ultralytics\runs\detect\yolo11n-vam-fpn2.0\weights\best.pt"

# 2. 验证集图片路径
val_images = r"C:\Users\TZY\Desktop\ultralytics\ultralytics\cfg\datasets\my_dataset_det\val\images"

# 3. 自动推断数据集根目录
dataset_root = Path(val_images).parents[1]

# 4. 创建临时 data.yaml
data_yaml = {
    "path": str(dataset_root),
    "train": "train/images",
    "val": "val/images",
    "test": "val/images",
    "nc": 2,
    "names": ["t", "a"]
}

yaml_path = dataset_root / "temp_val.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

# 5. 加载模型
model = YOLO(model_path)

# 6. 验证模型：只输出数据，不画图
metrics = model.val(
    data=str(yaml_path),
    split="val",
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    save_json=False,
    plots=False,
    project=r"C:\Users\TZY\Desktop\ultralytics\runs\val",
    name="yolo11n_vam_fpn_val_metrics"
)

# 7. 输出主要指标
print("\n========== 验证完成 ==========")
print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")
print(f"mAP50:       {metrics.box.map50:.4f}")
print(f"mAP50-95:    {metrics.box.map:.4f}")
print("================================")

# 8. 输出每个类别的 AP50-95
print("\n========== 每个类别 AP50-95 ==========")
for i, name in model.names.items():
    print(f"{i}: {name} -> AP50-95: {metrics.box.maps[i]:.4f}")
print("======================================")