from ultralytics import YOLO
import os

# 1. 加载训练好的模型
# 注意修改路径为你最新的训练结果，例如 runs/detect/train5/weights/best.pt
model = YOLO(r'C:\Users\TZY\Desktop\ultralytics\runs\detect\yolo11n\weights\best.pt')

# 2. 指定验证集图片路径
# 修改为你数据集的实际路径
source_path = r'C:\Users\TZY\Desktop\ultralytics\ultralytics\cfg\datasets\my_dataset\val\images'

# 关键在这里：
# project='runs/detect' -> 指定根目录
# name='predict'        -> 指定子文件夹的基础名称
# exist_ok=False        -> (默认就是False) 如果文件夹已存在，就新建 predict2, predict3...
results = model.predict(
    source=source_path,
    save=True,
    conf=0.2,
    iou=0.2,
    project='runs/detect',  # 显式指定结果保存在 runs/detect 下
    name='predict',         # 基础名字叫 predict
    exist_ok=False          # 设为 False (默认)，它就会自动变 predict2, predict3
)

print(f"检测完成！结果保存在: {results[0].save_dir}")