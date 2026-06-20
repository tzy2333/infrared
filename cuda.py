import torch
import torch.nn as nn
# 检查是否支持CUDA（即是否为GPU版本）
print("CUDA是否可用：", torch.cuda.is_available())

# 查看PyTorch绑定的CUDA版本（若为CPU版本则返回None）
print("绑定的CUDA版本：", torch.version.cuda)

# 查看当前可用的GPU数量（CPU版本返回0）
print("可用GPU数量：", torch.cuda.device_count())


