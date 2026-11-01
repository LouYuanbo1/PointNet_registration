"""配准相关几何变换：SE(3) 随机真值生成、刚体施加。

原先位于 ``train_pointnetlk.py``，现抽为独立工具模块供各入口脚本复用。
"""

from __future__ import annotations

import torch

from lie.se3 import se3_exponential_map
from utils.builders import get_default_device


@torch.no_grad()
def random_igt_twist(batch_size, mag=1.0, mag_randomly=True, device=None):
    """官方式 SE(3) 随机变换（对齐原论文 RandomTransformSE3）。

    从 6 维 twist 一体生成旋转+平移：
      x = randn(B,6)，单位化后乘长度 mag（可随机放大），再 se3 指数映射得 4x4。
    旋转方向覆盖任意轴、无固定角度上限，幅度由 mag 控制；
    平移与旋转联合缩放。模型因此在训练时见过更大、更广的扰动分布，泛化更强。

    Parameters
    ----------
    mag : float
        twist 向量的目标模长（幅度）。默认 1.0；
        实际角度范围约为 [0, mag] 弧度的任意方向随机旋转 + 相应平移。
    mag_randomly : bool
        是否对每个样本随机取幅度（amp = rand * mag）。默认 True，
        使训练批次内覆盖多种扰动强度，进一步提升泛化。
    """
    if device is None:
        device = get_default_device()
    B = batch_size
    amp = torch.rand(B, 1, device=device) * mag if mag_randomly else mag
    x = torch.randn(B, 6, device=device)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8) * amp       # [B, 6]
    g = se3_exponential_map(x)                    # [B, 4, 4]
    return g


def apply_rt(points, igt):
    """points [B, N, 3], igt [B, 4, 4] -> [B, N, 3]，即 p1 = igt * p0。"""
    R = igt[:, 0:3, 0:3]   # [B, 3, 3]
    t = igt[:, 0:3, 3]     # [B, 3]
    return points @ R.transpose(1, 2) + t.unsqueeze(1)