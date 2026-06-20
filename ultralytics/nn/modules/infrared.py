# Ultralytics YOLO 🚀
# Custom modules for infrared column-like object detection

import torch
import torch.nn as nn
import math
from .conv import Conv
from .block import C2f

def autopad(k, d=1):
    """Auto padding to keep feature map size unchanged."""
    if isinstance(k, tuple):
        if isinstance(d, int):
            d = (d, d)
        return tuple(((ki - 1) * di) // 2 for ki, di in zip(k, d))
    return ((k - 1) * d) // 2


class ColumnConv(nn.Module):
    """Conv + BN + SiLU."""

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ColumnDWConv(nn.Module):
    """Depthwise Conv + Pointwise Conv."""

    def __init__(self, c1, c2, k=3, s=1, d=1):
        super().__init__()
        self.dw = ColumnConv(c1, c1, k=k, s=s, g=c1, d=d)
        self.pw = ColumnConv(c1, c2, k=1, s=1)

    def forward(self, x):
        return self.pw(self.dw(x))


class RectangularAttention(nn.Module):
    """
    Rectangular-aware Attention, RCA.

    Designed for infrared chimney-like or column-like targets.

    改进点：
    1. 不再直接 x * a_h * a_w * s，避免弱红外目标被压掉。
    2. 改成残差增强式注意力。
    """

    def __init__(self, channels, reduction=32, k_h=9, k_w=5):
        super().__init__()

        hidden = max(8, channels // reduction)

        self.reduce_h = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        self.reduce_w = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        self.attn_h = nn.Conv2d(
            hidden,
            channels,
            kernel_size=(k_h, 1),
            padding=(k_h // 2, 0),
            bias=True,
        )

        self.attn_w = nn.Conv2d(
            hidden,
            channels,
            kernel_size=(1, k_w),
            padding=(0, k_w // 2),
            bias=True,
        )

        self.strip_attn = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=(k_h, 1),
                padding=(k_h // 2, 0),
                bias=False,
            ),
            nn.BatchNorm2d(1),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                1,
                1,
                kernel_size=(1, k_w),
                padding=(0, k_w // 2),
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True)

        h_feat = self.reduce_h(x_h)
        w_feat = self.reduce_w(x_w)

        a_h = torch.sigmoid(self.attn_h(h_feat))
        a_w = torch.sigmoid(self.attn_w(w_feat))

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        s = self.strip_attn(torch.cat([avg_out, max_out], dim=1))

        attn = a_h * a_w * s

        # 残差增强，不直接压制原特征
        return x * (1.0 + 0.5 * attn)


class VAM(nn.Module):
    """
    Vertical-Aware Feature Enhancement Module.

    Args:
        c1: input channels
        c2: output channels
        s: stride, used to change feature map size
           s=1: keep H/W unchanged
           s=2: downsample H/W by 2
        e: expansion ratio
        k_h: vertical kernel size
        k_w: horizontal kernel size
        reduction: reduction ratio of RCA
    """

    def __init__(self, c1, c2, s=1, e=0.5, k_h=9, k_w=5, reduction=32):
        super().__init__()

        c_ = max(16, int(c2 * e))

        # Branch 1: detail-preserving branch
        # 用 s 控制是否下采样
        self.b1 = ColumnConv(c1, c_, k=1, s=s)

        # Branch 2: local texture branch
        self.b2 = nn.Sequential(
            ColumnConv(c1, c_, k=1, s=s),
            ColumnDWConv(c_, c_, k=3),
        )

        # Branch 3: vertical structure branch
        self.b3 = nn.Sequential(
            ColumnConv(c1, c_, k=1, s=s),
            ColumnConv(c_, c_, k=(k_h, 1), g=c_),
            ColumnConv(c_, c_, k=1),
        )

        # Branch 4: horizontal boundary branch
        self.b4 = nn.Sequential(
            ColumnConv(c1, c_, k=1, s=s),
            ColumnConv(c_, c_, k=(1, k_w), g=c_),
            ColumnConv(c_, c_, k=1),
        )

        # Adaptive branch weighting
        hidden = max(8, c1 // 16)
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=1, bias=True),
        )

        # Multi-branch fusion
        self.fuse = ColumnConv(c_ * 4, c2, k=1)

        # Rectangular-aware attention
        self.ra = RectangularAttention(
            channels=c2,
            reduction=reduction,
            k_h=k_h,
            k_w=k_w,
        )

        # 只有通道数和尺寸都不变时，才允许残差相加
        self.add = (c1 == c2 and s == 1)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)

        # [B, 4, 1, 1]
        w = self.branch_gate(x)
        w = torch.softmax(w, dim=1)

        y1 = y1 * w[:, 0:1]
        y2 = y2 * w[:, 1:2]
        y3 = y3 * w[:, 2:3]
        y4 = y4 * w[:, 3:4]

        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.fuse(y)
        y = self.ra(y)

        return x + y if self.add else y

class DPLConv(nn.Module):
    """
    Directional Partial Lightweight Convolution.

    改进版：
    1. 使用部分通道做方向卷积，降低计算量；
    2. 输入前先 channel shuffle，避免固定前半通道长期参与方向卷积；
    3. concat 后再次 channel shuffle，增强方向分支和保留分支的信息混合；
    4. 保留残差连接。
    """

    def __init__(self, c1, c2, k=7, ratio=0.5, s=1, act=True):
        super().__init__()

        assert 0 < ratio <= 1.0, "ratio must be in (0, 1]."
        assert k % 2 == 1, "k should be an odd number, such as 5, 7, 9."

        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.ratio = ratio
        self.s = s

        # 参与方向卷积的通道数
        c_part = max(8, int(c1 * ratio))
        c_part = min(c_part, c1)
        c_keep = c1 - c_part

        self.c_part = c_part
        self.c_keep = c_keep

        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        # 方向感知轻量分支
        self.dir_conv = nn.Sequential(
            nn.Conv2d(
                c_part,
                c_part,
                kernel_size=(k, 1),
                stride=(s, 1),
                padding=(k // 2, 0),
                groups=c_part,
                bias=False
            ),
            nn.BatchNorm2d(c_part),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                c_part,
                c_part,
                kernel_size=(1, k),
                stride=(1, s),
                padding=(0, k // 2),
                groups=c_part,
                bias=False
            ),
            nn.BatchNorm2d(c_part),
            nn.SiLU(inplace=True)
        )

        # 保留分支同步下采样
        if c_keep > 0:
            if s == 2:
                self.keep_down = nn.AvgPool2d(kernel_size=2, stride=2)
            else:
                self.keep_down = nn.Identity()
        else:
            self.keep_down = None

        # 1×1 融合方向增强特征和保留特征
        self.fuse = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            self.act
        )

        # 残差连接
        self.use_shortcut = (c1 == c2 and s == 1)

    @staticmethod
    def channel_shuffle(x, groups=2):
        """
        Channel shuffle for partial convolution.
        避免固定通道长期进入同一个分支。
        """
        b, c, h, w = x.size()

        if c % groups != 0:
            return x

        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        identity = x

        # 先打乱输入通道，再划分方向分支和保留分支
        x_mix = self.channel_shuffle(x, groups=2)

        x_part = x_mix[:, :self.c_part, :, :]
        y_part = self.dir_conv(x_part)

        if self.c_keep > 0:
            x_keep = x_mix[:, self.c_part:, :, :]
            y_keep = self.keep_down(x_keep)

            y = torch.cat((y_part, y_keep), dim=1)

            # concat 后再 shuffle，增强两类特征混合
            y = self.channel_shuffle(y, groups=2)
        else:
            y = y_part

        y = self.fuse(y)

        if self.use_shortcut:
            y = y + identity

        return y


class LightGuide(nn.Module):
    """
    Lightweight context guide branch.
    用于从保留分支中提取轻量上下文信息。
    """

    def __init__(self, c):
        super().__init__()

        self.guide = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),

            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.guide(x)

class DirectionGate(nn.Module):
    """
    Directional gated fusion.
    使用 Guide 特征生成门控权重，温和调制方向特征。
    """

    def __init__(self, c):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, fd, fg):
        """
        fd: directional feature
        fg: guide feature
        """
        g = self.gate(fg)

        # 原来是 fd * (1.0 + g)，增强范围是 1~2 倍
        # 改成 1~1.5 倍，更稳
        return fd * (1.0 + 0.5 * g)

class AdaptiveFusion(nn.Module):
    """
    Lightweight adaptive channel fusion.
    """

    def __init__(self, c):
        super().__init__()

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.channel_gate(x)

        # 原来是 x * w + x，增强范围 1~2 倍
        # 改成 1~1.5 倍，减少过强通道放大
        return x * (1.0 + 0.5 * w)

class DGFBlock(nn.Module):
    """
    Directional-Gated Feature Block.

    结构：
    Input
      ├─ DPLConv directional branch
      ├─ LightGuide guide branch
      ├─ DirectionGate
      └─ Fusion
    """

    def __init__(self, c, k=7, ratio=0.5, shortcut=True):
        super().__init__()

        self.dpl = DPLConv(c, c, k=k, ratio=ratio, s=1)

        self.guide = LightGuide(c)

        self.gate = DirectionGate(c)

        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        self.add = shortcut

    def forward(self, x):
        fd = self.dpl(x)
        fg = self.guide(x)

        fd = self.gate(fd, fg)

        y = self.fuse(torch.cat((fd, fg), dim=1))

        if self.add:
            y = y + x

        return y

class DPGM(nn.Module):
    """
    Directional-Gated Feature Fusion C3k2.

    Compared with original C3k2:
    1. Uses DGFBlock instead of normal bottleneck.
    2. Adds lightweight guide branch.
    3. Adds directional gated fusion.
    4. Adds adaptive final fusion.

    Args:
        c1: input channels
        c2: output channels
        n: number of DGF blocks
        e: expansion ratio
        k: directional kernel size
        ratio: partial channel ratio
        shortcut: residual connection
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        e=0.5,
        k=7,
        ratio=0.5,
        shortcut=True,
    ):
        super().__init__()

        self.c = int(c2 * e)

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            DGFBlock(
                self.c,
                k=k,
                ratio=ratio,
                shortcut=shortcut,
            )
            for _ in range(n)
        )

        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        self.af = AdaptiveFusion(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))

        for block in self.blocks:
            y.append(block(y[-1]))

        out = self.cv2(torch.cat(y, dim=1))
        out = self.af(out)

        return out

class DirectionalStructureBlock(nn.Module):
    """
    Directional Structure Block.

    Used inside DSC3k2 for elongated structured industrial targets.

    It enhances:
    1. local structural texture;
    2. vertical long-range continuity;
    3. horizontal structural response;
    4. large receptive-field context.
    """

    def __init__(self, c, k=15, shortcut=True):
        super().__init__()

        self.shortcut = shortcut

        # 1. Local texture branch
        self.local_branch = nn.Sequential(
            ColumnConv(c, c, k=3, s=1, g=c),
            ColumnConv(c, c, k=1, s=1),
        )

        # 2. Vertical long-range structure branch
        self.vertical_branch = nn.Sequential(
            ColumnConv(c, c, k=(k, 1), s=1, g=c),
            ColumnConv(c, c, k=1, s=1),
        )

        # 3. Horizontal structure response branch
        self.horizontal_branch = nn.Sequential(
            ColumnConv(c, c, k=(1, 7), s=1, g=c),
            ColumnConv(c, c, k=1, s=1),
        )

        # 4. Large receptive-field context branch
        self.context_branch = nn.Sequential(
            ColumnConv(c, c, k=3, s=1, g=c, d=2),
            ColumnConv(c, c, k=3, s=1, g=c, d=3),
            ColumnConv(c, c, k=1, s=1),
        )

        # Adaptive branch weighting
        hidden = max(8, c // 16)

        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=1, bias=True),
        )

        # Multi-branch fusion
        self.fuse = ColumnConv(c * 4, c, k=1, s=1)

        # Spatial recall gate
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=(7, 1),
                padding=(3, 0),
                bias=False,
            ),
            nn.BatchNorm2d(1),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                1,
                1,
                kernel_size=(1, 7),
                padding=(0, 3),
                bias=True,
            ),
            nn.Sigmoid(),
        )

        # Stable residual scaling
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        y1 = self.local_branch(x)
        y2 = self.vertical_branch(x)
        y3 = self.horizontal_branch(x)
        y4 = self.context_branch(x)

        # [B, 4, 1, 1]
        w = self.branch_gate(x)
        w = torch.softmax(w, dim=1)

        y1 = y1 * w[:, 0:1]
        y2 = y2 * w[:, 1:2]
        y3 = y3 * w[:, 2:3]
        y4 = y4 * w[:, 3:4]

        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.fuse(y)

        # Spatial recall enhancement
        avg_out = torch.mean(y, dim=1, keepdim=True)
        max_out, _ = torch.max(y, dim=1, keepdim=True)

        s = self.spatial_gate(torch.cat([avg_out, max_out], dim=1))

        # Recall-friendly enhancement:
        # use 1 + gate instead of pure suppression.
        y = y * (1.0 + s)

        if self.shortcut:
            return x + self.gamma * y
        else:
            return y


class DSC3k2(nn.Module):
    """
    Directional Structure Context C3k2.

    This module improves the original C3k2 by introducing:
    1. local texture modeling;
    2. vertical long-range structure modeling;
    3. horizontal structural response;
    4. large receptive-field context aggregation;
    5. adaptive branch weighting;
    6. spatial recall gating.

    Args:
        c1: input channels
        c2: output channels
        n: number of DirectionalStructureBlock
        shortcut: whether to use shortcut inside blocks
        e: expansion ratio
        k: vertical kernel size
    """

    def __init__(self, c1, c2, n=2, shortcut=True, e=0.5, k=15):
        super().__init__()

        self.c = int(c2 * e)

        # Split projection
        self.cv1 = ColumnConv(c1, 2 * self.c, k=1, s=1)

        # Directional structure blocks
        self.blocks = nn.ModuleList(
            DirectionalStructureBlock(
                self.c,
                k=k,
                shortcut=shortcut,
            )
            for _ in range(n)
        )

        # Final fusion
        self.cv2 = ColumnConv((2 + n) * self.c, c2, k=1, s=1)

        # Global structure gate
        hidden = max(8, c2 // 16)

        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Split features
        y = list(self.cv1(x).chunk(2, dim=1))

        # C3k2-like progressive feature aggregation
        for block in self.blocks:
            y.append(block(y[-1]))

        out = self.cv2(torch.cat(y, dim=1))

        # Global recall-friendly enhancement
        g = self.global_gate(out)

        # Use 1 + gate to enhance instead of suppressing weak responses
        out = out * (1.0 + g)

        return out


class GSCM(nn.Module):
    """
    Global Semantic Calibration Module.

    This module is designed as a lightweight residual plug-in.
    It does not use ColumnConv, directional convolution, or vertical/horizontal branches.

    It only performs global semantic calibration and lightweight spatial recalibration.

    Args:
        c1: input channels
        c2: output channels
        reduction: channel reduction ratio
    """

    def __init__(self, c1, c2, reduction=16):
        super().__init__()

        self.same_channels = c1 == c2

        if self.same_channels:
            self.short = nn.Identity()
        else:
            self.short = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )

        hidden = max(8, c2 // reduction)

        # Channel semantic calibration
        # Uses both average and max global descriptors.
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(c2 * 2, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Lightweight spatial calibration
        # No large kernel, no directional kernel.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # Stable residual scale.
        # Starts close to the original feature.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = self.short(x)

        # Global channel descriptors
        avg_pool = torch.mean(identity, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(identity, dim=(2, 3), keepdim=True)

        channel_gate = self.channel_mlp(torch.cat([avg_pool, max_pool], dim=1))

        # Channel recalibration
        y = identity * (1.0 + channel_gate)

        # Spatial descriptors
        avg_spatial = torch.mean(y, dim=1, keepdim=True)
        max_spatial, _ = torch.max(y, dim=1, keepdim=True)

        spatial_gate = self.spatial_gate(torch.cat([avg_spatial, max_spatial], dim=1))

        # Spatial recalibration
        y = y * (1.0 + spatial_gate)

        # Residual semantic calibration
        return identity + self.gamma * y

def autopad_sc(k, p=None, d=1):
    """Auto padding."""
    if d > 1:
        if isinstance(k, int):
            k = d * (k - 1) + 1
        else:
            k = [d * (x - 1) + 1 for x in k]
    if p is None:
        if isinstance(k, int):
            p = k // 2
        else:
            p = [x // 2 for x in k]
    return p


class ConvBNAct(nn.Module):
    """Standard Conv + BN + SiLU. No ColumnConv."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=autopad_sc(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SRU(nn.Module):
    """
    Spatial Reconstruction Unit.

    It separates informative and less-informative spatial responses,
    then reconstructs them to reduce spatial redundancy.
    """

    def __init__(self, channels, group_num=16, gate_threshold=0.5):
        super().__init__()

        group_num = min(group_num, channels)
        while channels % group_num != 0:
            group_num -= 1

        self.gn = nn.GroupNorm(num_groups=group_num, num_channels=channels)
        self.gate_threshold = gate_threshold

    def forward(self, x):
        gn_x = self.gn(x)

        gamma = self.gn.weight / torch.sum(self.gn.weight)
        gamma = gamma.view(1, -1, 1, 1)

        reweight = torch.sigmoid(gn_x * gamma)

        info_mask = reweight >= self.gate_threshold
        non_info_mask = reweight < self.gate_threshold

        x_info = info_mask * x
        x_non_info = non_info_mask * x

        x1, x2 = torch.chunk(x_info, 2, dim=1)
        x3, x4 = torch.chunk(x_non_info, 2, dim=1)

        return torch.cat([x1 + x4, x2 + x3], dim=1)


class CRU(nn.Module):
    """
    Channel Reconstruction Unit.

    It splits channels into two parts and reconstructs channel responses
    through group-wise convolution and point-wise convolution.
    """

    def __init__(
        self,
        channels,
        alpha=0.5,
        squeeze_ratio=2,
        group_size=2,
    ):
        super().__init__()

        self.up_channel = int(alpha * channels)
        self.low_channel = channels - self.up_channel

        up_squeezed = max(1, self.up_channel // squeeze_ratio)
        low_squeezed = max(1, self.low_channel // squeeze_ratio)

        self.squeeze_up = nn.Conv2d(
            self.up_channel,
            up_squeezed,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.squeeze_low = nn.Conv2d(
            self.low_channel,
            low_squeezed,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        group_size = math.gcd(group_size, math.gcd(up_squeezed, channels))
        group_size = max(1, group_size)

        self.group_conv = nn.Conv2d(
            up_squeezed,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=group_size,
            bias=False,
        )

        self.point_conv_up = nn.Conv2d(
            up_squeezed,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.point_conv_low = nn.Conv2d(
            low_squeezed,
            channels - low_squeezed,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_up, x_low = torch.split(
            x,
            [self.up_channel, self.low_channel],
            dim=1,
        )

        x_up = self.squeeze_up(x_up)
        x_low = self.squeeze_low(x_low)

        y_up = self.group_conv(x_up) + self.point_conv_up(x_up)
        y_low = torch.cat([self.point_conv_low(x_low), x_low], dim=1)

        y = torch.cat([y_up, y_low], dim=1)

        # channel-wise soft selection
        weight = torch.softmax(self.pool(y), dim=1)
        y = y * weight

        y1, y2 = torch.chunk(y, 2, dim=1)

        return y1 + y2


class SCConv(nn.Module):
    """
    Spatial-Channel Reconstruction Convolution.

    SRU: spatial redundancy reconstruction.
    CRU: channel redundancy reconstruction.
    """

    def __init__(
        self,
        channels,
        group_num=16,
        gate_threshold=0.5,
        alpha=0.5,
        squeeze_ratio=2,
        group_size=2,
    ):
        super().__init__()

        self.sru = SRU(
            channels=channels,
            group_num=group_num,
            gate_threshold=gate_threshold,
        )

        self.cru = CRU(
            channels=channels,
            alpha=alpha,
            squeeze_ratio=squeeze_ratio,
            group_size=group_size,
        )

    def forward(self, x):
        x = self.sru(x)
        x = self.cru(x)
        return x


class SCBottleneck(nn.Module):
    """
    Bottleneck with SCConv.

    It replaces the normal 3x3 convolution inside bottleneck with SCConv.
    """

    def __init__(self, c1, c2, shortcut=True, e=1.0):
        super().__init__()

        c_ = int(c2 * e)

        if c_ % 2 != 0:
            c_ += 1

        self.cv1 = ConvBNAct(c1, c_, k=1, s=1)
        self.scconv = SCConv(c_)
        self.cv2 = ConvBNAct(c_, c2, k=1, s=1)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.scconv(self.cv1(x)))
        return x + y if self.add else y


class SC_C3k2(nn.Module):
    """
    SCConv-enhanced C3k2.

    It keeps the split-concat-fusion structure of C3k2,
    but replaces the internal normal convolutional bottleneck
    with SCConv-based feature reconstruction.

    Args:
        c1: input channels
        c2: output channels
        n: number of internal SCBottleneck blocks
        shortcut: whether to use shortcut
        e: expansion ratio
    """

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()

        self.c = int(c2 * e)

        if self.c % 2 != 0:
            self.c += 1

        self.cv1 = ConvBNAct(c1, 2 * self.c, k=1, s=1)

        self.m = nn.ModuleList(
            SCBottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                e=1.0,
            )
            for _ in range(n)
        )

        self.cv2 = ConvBNAct((2 + n) * self.c, c2, k=1, s=1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))

        for block in self.m:
            y.append(block(y[-1]))

        return self.cv2(torch.cat(y, dim=1))

class APBottleneck(nn.Module):
    """
    AP Bottleneck: Asymmetric Padding Bottleneck.
    复现论文中的 AP 结构：
    Input -> 4 asymmetric padding branches -> Conv -> Concat -> 3x3 Conv -> Add
    """

    def __init__(self, c1, c2, shortcut=True, g=1):
        super().__init__()
        self.add = shortcut and c1 == c2

        # 如果输入输出通道不同，先调整通道；C3k2内部通常 c1 == c2
        self.cv0 = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # 将通道分成4份
        c = c2
        c_ = c // 4
        self.ch = [c_, c_, c_, c - 3 * c_]

        # ZeroPad2d顺序: left, right, top, bottom
        # 对应4个方向的非对称填充
        self.pad = nn.ModuleList([
            nn.ZeroPad2d((2, 0, 2, 0)),  # left-top
            nn.ZeroPad2d((0, 2, 2, 0)),  # right-top
            nn.ZeroPad2d((2, 0, 0, 2)),  # left-bottom
            nn.ZeroPad2d((0, 2, 0, 2)),  # right-bottom
        ])

        # 每个分支单独卷积
        self.cv1 = nn.ModuleList([
            Conv(self.ch[0], self.ch[0], 1, 1),
            Conv(self.ch[1], self.ch[1], 1, 1),
            Conv(self.ch[2], self.ch[2], 1, 1),
            Conv(self.ch[3], self.ch[3], 1, 1),
        ])

        # 4个分支 concat 后用 3x3 Conv 融合
        # 因为前面 ZeroPad 多了2个像素，所以这里 p=0 后尺寸会恢复
        self.cv2 = Conv(c2, c2, 3, 1, p=0, g=g)

    def forward(self, x):
        identity = x
        x = self.cv0(x)

        x1, x2, x3, x4 = torch.split(x, self.ch, dim=1)

        y1 = self.cv1[0](self.pad[0](x1))
        y2 = self.cv1[1](self.pad[1](x2))
        y3 = self.cv1[2](self.pad[2](x3))
        y4 = self.cv1[3](self.pad[3](x4))

        y = torch.cat((y1, y2, y3, y4), dim=1)
        y = self.cv2(y)

        return identity + y if self.add else y

class C3k2_AP(C2f):
    """
    AP-C3k2:
    保留 YOLO11 C3k2 / C2f 的外部结构，
    将内部 Bottleneck 替换成 APBottleneck。
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)

        # c3k 参数保留，用来兼容原 YOLO11 YAML 写法
        # 这里统一使用 APBottleneck
        self.m = nn.ModuleList(
            APBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class IRContrastGuide(nn.Module):
    """
    Infrared contrast-guided descriptor.
    用局部平均与全局响应差异增强红外弱目标。
    """

    def __init__(self, c):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c * 2, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        contrast = torch.cat([avg, maxv], dim=1)
        contrast = contrast.repeat(1, x.shape[1] // 2, 1, 1)
        return self.proj(torch.cat([x, contrast], dim=1))

class MSLKAConv(nn.Module):
    """
    Multi-Scale Large Kernel Attention Convolution

    非竖向卷积模块：
    1. 不使用 k×1 / 1×k
    2. 使用普通 k×k depthwise convolution
    3. 用多尺度感受野增强上下文
    4. 适合中等 / 偏大红外目标召回率提升
    """

    def __init__(self, c1, c2, reduction=4, shortcut=True):
        super().__init__()

        self.shortcut = shortcut and c1 == c2

        hidden = max(c2 // 2, 16)

        self.pre = ConvBNAct(c1, hidden, k=1)

        # local branch: 局部细节
        self.branch3 = nn.Sequential(
            ConvBNAct(hidden, hidden, k=3, g=hidden),
            ConvBNAct(hidden, hidden, k=1)
        )

        # medium branch: 中尺度上下文
        self.branch5 = nn.Sequential(
            ConvBNAct(hidden, hidden, k=5, g=hidden),
            ConvBNAct(hidden, hidden, k=1)
        )

        # large branch: 大感受野，上下文增强
        # 7×7 dilation=2，等效感受野约 13×13
        self.branch7_d2 = nn.Sequential(
            ConvBNAct(hidden, hidden, k=7, d=2, g=hidden),
            ConvBNAct(hidden, hidden, k=1)
        )

        self.num_branches = 3
        self.hidden = hidden

        # selective kernel attention
        attn_hidden = max(hidden // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.branch_attn = nn.Sequential(
            nn.Conv2d(hidden, attn_hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(attn_hidden, hidden * self.num_branches, kernel_size=1, bias=True)
        )

        # LKA-style attention，非方向性大核注意力
        self.lka = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=7, padding=9, dilation=3, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.post = ConvBNAct(hidden, c2, k=1)

    def forward(self, x):
        identity = x

        x = self.pre(x)

        f3 = self.branch3(x)
        f5 = self.branch5(x)
        f7 = self.branch7_d2(x)

        feats = [f3, f5, f7]

        # 用多分支特征之和生成选择权重
        u = f3 + f5 + f7

        attn = self.avg_pool(u)
        attn = self.branch_attn(attn)

        b, _, _, _ = attn.shape
        attn = attn.view(b, self.num_branches, self.hidden, 1, 1)
        attn = torch.softmax(attn, dim=1)

        out = 0
        for i in range(self.num_branches):
            out = out + feats[i] * attn[:, i]

        # 大核空间注意力
        spatial_weight = self.lka(out)
        out = out * spatial_weight + out

        out = self.post(out)

        if self.shortcut:
            out = out + identity

        return out
class IRContrastGuide(nn.Module):
    """
    Infrared Contrast-Guided Descriptor.

    作用：
    1. 计算局部对比度，突出红外弱目标与背景之间的热响应差异；
    2. 生成对比度描述符，用于多尺度分支权重分配；
    3. 生成轻量空间引导图，增强局部显著区域。
    """

    def __init__(self, c, k=5):
        super().__init__()

        self.k = k
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 使用 contrast 的 avg/max 生成空间引导图
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # local background response
        local_mean = F.avg_pool2d(
            x,
            kernel_size=self.k,
            stride=1,
            padding=self.k // 2
        )

        # infrared local contrast
        contrast = torch.abs(x - local_mean)

        # global response descriptor
        avg_desc = self.pool(x)

        # contrast descriptor
        contrast_desc = self.pool(contrast)

        # spatial contrast map
        contrast_avg = torch.mean(contrast, dim=1, keepdim=True)
        contrast_max, _ = torch.max(contrast, dim=1, keepdim=True)

        spatial_weight = self.spatial_gate(
            torch.cat([contrast_avg, contrast_max], dim=1)
        )

        # residual enhancement, avoid suppressing weak targets
        guided_x = x * (1.0 + 0.5 * spatial_weight)

        # [B, 2C, 1, 1]
        descriptor = torch.cat([avg_desc, contrast_desc], dim=1)

        return guided_x, descriptor

class IRMSLKAConv(nn.Module):
    """
    Infrared Contrast-Guided Multi-Scale Large Kernel Attention Convolution.

    Compared with original MSLKAConv:
    1. Uses 3x3, 5x5 and dilated 7x7 depthwise branches.
    2. Uses infrared local contrast descriptor instead of plain global average pooling.
    3. Generates adaptive multi-scale branch weights.
    4. Uses residual LKA-style spatial enhancement to avoid suppressing weak targets.

    Args:
        c1: input channels
        c2: output channels
        reduction: channel reduction ratio for branch attention
        shortcut: whether to use residual connection
    """

    def __init__(self, c1, c2, reduction=4, shortcut=True):
        super().__init__()

        self.shortcut = shortcut and c1 == c2

        hidden = max(c2 // 2, 16)

        self.hidden = hidden
        self.num_branches = 3

        self.pre = ConvBNAct(c1, hidden, k=1)

        # Branch 1: local detail
        self.branch3 = nn.Sequential(
            ConvBNAct(hidden, hidden, k=3, g=hidden),
            ConvBNAct(hidden, hidden, k=1)
        )

        # Branch 2: medium-scale context
        self.branch5 = nn.Sequential(
            ConvBNAct(hidden, hidden, k=5, g=hidden),
            ConvBNAct(hidden, hidden, k=1)
        )

        # Branch 3: large receptive field
        # 7x7 dilation=2, effective receptive field about 13x13
        self.branch7_d2 = nn.Sequential(
            ConvBNAct(hidden, hidden, k=7, d=2, g=hidden),
            ConvBNAct(hidden, hidden, k=1)
        )

        # Infrared contrast-guided descriptor
        self.ir_guide = IRContrastGuide(hidden, k=5)

        attn_hidden = max(hidden // reduction, 8)

        # Note:
        # input channel is 2 * hidden because descriptor = [avg_desc, contrast_desc]
        self.branch_attn = nn.Sequential(
            nn.Conv2d(hidden * 2, attn_hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                attn_hidden,
                hidden * self.num_branches,
                kernel_size=1,
                bias=True
            )
        )

        # LKA-style spatial attention
        self.lka = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=5,
                padding=2,
                groups=hidden,
                bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=7,
                padding=9,
                dilation=3,
                groups=hidden,
                bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.post = ConvBNAct(hidden, c2, k=1)

    def forward(self, x):
        identity = x

        x = self.pre(x)

        f3 = self.branch3(x)
        f5 = self.branch5(x)
        f7 = self.branch7_d2(x)

        # multi-scale feature aggregation
        u = f3 + f5 + f7

        # infrared contrast-guided descriptor
        guided_u, descriptor = self.ir_guide(u)

        # branch attention
        attn = self.branch_attn(descriptor)

        b, _, _, _ = attn.shape
        attn = attn.view(b, self.num_branches, self.hidden, 1, 1)
        attn = torch.softmax(attn, dim=1)

        feats = torch.stack([f3, f5, f7], dim=1)

        out = (feats * attn).sum(dim=1)

        # add guided response
        out = out + 0.5 * guided_u

        # residual LKA spatial enhancement
        spatial_weight = self.lka(out)
        out = out * (1.0 + 0.5 * spatial_weight)

        out = self.post(out)

        if self.shortcut:
            out = out + identity

        return out

__all__ = (
    "RectangularAttention",
    "VAM",
    "DSC3k2",
    "GSCM",
    "SC_C3k2",
    "C3k2_AP",
    "MSLKAConv",

)