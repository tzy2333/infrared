import yaml
import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks
from ultralytics.nn.tasks import DetectionModel


class BiFPN_Concat(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x):
        if len(x) == 2:
            w = self.w1_weight
            weight = w / (torch.sum(w) + self.epsilon)
            return weight[0] * x[0] + weight[1] * x[1]
        elif len(x) == 3:
            w = self.w2_weight
            weight = w / (torch.sum(w) + self.epsilon)
            return weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]
        else:
            raise ValueError(f"BiFPN_Concat expects 2 or 3 inputs, but got {len(x)}")


tasks.BiFPN_Concat = BiFPN_Concat

yaml_path = r"C:\Users\TZY\Desktop\ultralytics\ultralytics\cfg\models\11\yolo11-vam-fpn4.0-dpgm.yaml"

with open(yaml_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if "scales" in cfg and "scale" not in cfg:
    if isinstance(cfg["scales"], dict) and len(cfg["scales"]) == 1:
        cfg["scale"] = next(iter(cfg["scales"].keys()))

cfg["nc"] = 2  # 你的类别数

model = DetectionModel(cfg=cfg, ch=3, nc=cfg["nc"], verbose=False)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")