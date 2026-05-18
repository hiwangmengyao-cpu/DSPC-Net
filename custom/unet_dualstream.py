import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Basic Blocks
# -------------------------
def _act():
    return nn.SiLU(inplace=True)

class ConvBNAct(nn.Module):
    """Conv2d + BN + SiLU"""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1):
        super().__init__()
        if p is None:
            p = (k // 2) * d
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = _act()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class GateFuse(nn.Module):
    """
    Fuse two feature maps using a learnable attention gate.
    """
    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        inter = max(channels // r, 8)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1, 1, 0, bias=True),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        att = self.local_att(x1 + x2) + self.global_att(x1 + x2)
        w = self.sigmoid(att)
        out = w * x1 + (1.0 - w) * x2
        return out

class DetailBranch(nn.Module):
    """Detail stream for edge and fine-grained structures."""
    def __init__(self, c: int):
        super().__init__()
        self.b1 = ConvBNAct(c, c, k=3, s=1)
        self.b2 = ConvBNAct(c, c, k=3, s=1)

    def forward(self, x):
        return self.b2(self.b1(x))

class StructureBranch(nn.Module):
    """
    Structure stream:
    Integrates Hybrid Dilated Convolution (HDC) and Depthwise Large Kernel Convolution
    to maximize the receptive field while minimizing parameters.
    """
    def __init__(self, c: int, k_struct: int = 7, dil: int = 2):
        super().__init__()
        self.proj = ConvBNAct(c, c, k=1, s=1, p=0)

        self.hdc1 = ConvBNAct(c, c, k=3, s=1, p=1, g=c, d=1)
        self.hdc2 = ConvBNAct(c, c, k=3, s=1, p=2, g=c, d=2)
        self.hdc3 = ConvBNAct(c, c, k=3, s=1, p=5, g=c, d=5)

        self.large = ConvBNAct(c, c, k=k_struct, s=1, p=k_struct // 2, g=c, d=1)

        self.fusion = ConvBNAct(c * 2, c, k=1, s=1, p=0)

    def forward(self, x):
        x_proj = self.proj(x)

        # HDC cascade to expand the receptive field
        h = self.hdc1(x_proj)
        h = self.hdc2(h)
        h = self.hdc3(h)

        # Large kernel branch
        l = self.large(x_proj)

        out = torch.cat([h, l], dim=1)
        return self.fusion(out) + x_proj

class DualStreamEncBlock(nn.Module):
    """
    Encoder block with parallel structure and detail streams.
    """
    def __init__(self, in_ch: int, out_ch: int, k_struct: int = 7, dil: int = 2, r_gate: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, out_ch, k=3, s=1),
            ConvBNAct(out_ch, out_ch, k=3, s=1),
        )
        self.struct = StructureBranch(out_ch, k_struct=k_struct, dil=dil)
        self.detail = DetailBranch(out_ch)
        self.fuse = GateFuse(out_ch, r=r_gate)
        self.out_proj = ConvBNAct(out_ch, out_ch, k=1, s=1, p=0)

    def forward(self, x):
        x = self.stem(x)
        xs = self.struct(x)
        xd = self.detail(x)

        xf = self.fuse(xs, xd)
        y = self.out_proj(xf) + x
        return y

class ConvBlock(nn.Module):
    """Standard convolution block for the decoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# -------------------------
# LCSC Module
# -------------------------
class LCSCSkipFuse3(nn.Module):
    """
    Local Context Skip Connection (LCSC) Module.
    Fuses features from three scales. skip_same provides the highest resolution base size.
    """
    def __init__(self, c_same, c_deep, c_deeper, c_out=None, r=4):
        super().__init__()
        if c_out is None:
            c_out = c_same
        self.c_out = c_out
        inter = max(c_out // r, 8)

        self.align_same = nn.Sequential(
            nn.Conv2d(c_same, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            _act(),
        )
        self.align_deep = nn.Sequential(
            nn.Conv2d(c_deep, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            _act(),
        )
        self.align_deeper = nn.Sequential(
            nn.Conv2d(c_deeper, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            _act(),
        )

        def _mlp():
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c_out, inter, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter, 1, 1, bias=True),
            )

        self.w_same = _mlp()
        self.w_deep = _mlp()
        self.w_deeper = _mlp()

        self.out_proj = ConvBNAct(c_out, c_out, k=1, s=1, p=0)

    def forward(self, skip_same, skip_deep, skip_deeper):
        s = self.align_same(skip_same)
        d = self.align_deep(skip_deep)
        dd = self.align_deeper(skip_deeper)

        # skip_same provides the base size. Deep and deeper features are upsampled to match it.
        d = F.interpolate(d, size=s.shape[-2:], mode="nearest")
        dd = F.interpolate(dd, size=s.shape[-2:], mode="nearest")

        l1 = self.w_same(s)
        l2 = self.w_deep(d)
        l3 = self.w_deeper(dd)

        logits = torch.cat([l1, l2, l3], dim=1)
        w = torch.softmax(logits, dim=1)

        fused = w[:, 0:1] * s + w[:, 1:2] * d + w[:, 2:3] * dd
        out = self.out_proj(fused)
        return out