import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3      # 导入官方的 C3 模块

__all__ = (
    "C3SE",  # 自定义 C3SE 模块
    "SEBlock"     # 自定义 SE 模块
)


# 定义 SE (Squeeze-and-Excitation) 模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        s = self.squeeze(x).view(b, c)
        # Excitation
        e = self.excitation(s).view(b, c, 1, 1)
        return x * e

# 定义使用 SE 模块的自定义 C3 模块 (C3SE)
class C3SE(C3):  # 继承自官方的 C3 模块
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 在官方 C3 的基础上添加 SE 模块
        self.se = SEBlock(c2)

    def forward(self, x):
        # 先通过官方 C3 的前向传播
        out = super().forward(x)
        # 然后应用 SE 注意力
        return self.se(out)
