from multiprocessing import freeze_support
from ultralytics import YOLO
import torch

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def fmt(v, nd=4):
    return "n/a" if v is None else f"{float(v):.{nd}f}"

def main():
    model_path = r"C:\Users\TZY\Desktop\ultralytics\runs\detect\yolo-firi\weights\best.pt"
    data_yaml  = r"C:\Users\TZY\Desktop\ultralytics\ultralytics\cfg\datasets\infrared.yaml"

    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"

    model = YOLO(model_path)

    # Params
    total_params, trainable_params = count_params(model.model)
    params_m = total_params / 1e6

    # Val (PyTorch only)
    results = model.val(data=data_yaml, imgsz=640, device=0, plots=False)

    # Speed/FPS
    sp = results.speed  # ms/img
    pre  = float(sp.get("preprocess", 0.0))
    inf  = float(sp.get("inference", 0.0))
    post = float(sp.get("postprocess", 0.0))
    total_ms = pre + inf + post
    fps_e2e = 1000.0 / total_ms if total_ms > 0 else 0.0
    fps_inf = 1000.0 / inf if inf > 0 else 0.0  # 纯推理FPS

    # Metrics
    map50 = getattr(results.box, "map50", None)
    map5095 = getattr(results.box, "map", None)

    # ===== 一行汇总（保留）=====
    print(
        f"[PyTorch best.pt @640 | {gpu_name}] "
        f"Params={total_params:,}({params_m:.2f}M) "
        f"FPS(E2E)={fps_e2e:.2f} FPS(Inf)={fps_inf:.2f} "
        f"(pre={pre:.3f}ms inf={inf:.3f}ms post={post:.3f}ms) "
        f"mAP50={fmt(map50)} mAP50-95={fmt(map5095)}"
    )

    # ===== 表格输出（更清楚）=====
    rows = [
        ("Backend", "PyTorch (.pt)"),
        ("Device", gpu_name),
        ("Input", "640"),
        ("Params", f"{total_params:,} ({params_m:.2f}M)"),
        ("Trainable Params", f"{trainable_params:,} ({trainable_params/1e6:.2f}M)"),
        ("Preprocess (ms/img)", f"{pre:.3f}"),
        ("Inference (ms/img)", f"{inf:.3f}"),
        ("Postprocess (ms/img)", f"{post:.3f}"),
        ("Total E2E (ms/img)", f"{total_ms:.3f}"),
        ("FPS (End-to-End)", f"{fps_e2e:.2f}"),
        ("FPS (Inference only)", f"{fps_inf:.2f}"),
        ("mAP50", fmt(map50)),
        ("mAP50-95", fmt(map5095)),
    ]

    # 计算列宽并打印ASCII表格
    col1_w = max(len(r[0]) for r in rows)
    col2_w = max(len(r[1]) for r in rows)
    sep = "+" + "-"*(col1_w+2) + "+" + "-"*(col2_w+2) + "+"

    print("\n" + sep)
    print(f"| {'Metric'.ljust(col1_w)} | {'Value'.ljust(col2_w)} |")
    print(sep)
    for k, v in rows:
        print(f"| {k.ljust(col1_w)} | {v.ljust(col2_w)} |")
    print(sep)

if __name__ == "__main__":
    freeze_support()
    main()
