from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载模型
    model = YOLO('./ultralytics/cfg/models/11/yolo11-irmslkaconv-vam-fpn4.0-dpgm.yaml')

    # 2. 开始训练 (在括号里直接添加增强参数)
    model.train(
        data='./ultralytics/cfg/datasets/infrared.yaml',
        epochs=150,
        patience=80,
        batch=16,
        workers=8,
        amp=True,
        device=0,

        # 红外图像：弱化颜色扰动
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,

        # 柱状目标：几何增强要保守
        degrees=5.0,
        translate=0.06,
        scale=0.3,
        shear=1.0,
        perspective=0.0,

        # 翻转
        fliplr=0.5,
        flipud=0.0,

        # 拼图增强
        mosaic=0.5,
        mixup=0.0,
        close_mosaic=20,

        # 不建议擦除细长目标
        erasing=0.0,
    )
