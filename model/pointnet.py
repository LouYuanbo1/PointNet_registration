""" PointNet 兼容层

本模块为 PointNetLK (:mod:`pointnetlk`) 提供统一的 PointNet 编码器接口，
它内部接收两种现有风格的 PointNet：

1. ``pointnet_lk_style.PointNet_features`` —— 输入 [B, N, 3]，输出 [B, K]，
   已与 PointNetLK 约定一致。
2. ``pointnet_yanx27_style.PointNetEncoder`` —— 输入 [B, D, N]（通道在前），
   返回 ``(feat, trans)`` 的元组。

``PointNetLK`` 通过 ``ptnet`` 依赖如下统一接口：
- ``ptnet(points)``：``points`` 形状 [B, N, 3] -> 返回特征 [B, K]；
- ``ptnet.training`` / ``ptnet.eval()`` / ``ptnet.train(mode)``：
  BatchNorm 等模块在推理/训练之间切换。

本模块提供 :class:`PointNet` 包装器，自动识别底层风格并做输入/输出的
规范化，使得任意一种风格的网络都能被 PointNetLK 直接使用。
"""

from __future__ import annotations

import torch

from . import pointnet_lk_style, pointnet_yanx27_style


# 统一风格常量，供工厂函数使用
STYLE_LK = "lk"            # 对应 pointnet_lk_style.PointNet_features
STYLE_YANX27 = "yanx27"    # 对应 pointnet_yanx27_style.PointNetEncoder


class PointNet(torch.nn.Module):
    """PointNet 兼容包装器。

    将 ``pointnet_lk_style.PointNet_features`` 或
    ``pointnet_yanx27_style.PointNetEncoder`` 包装成 PointNetLK 所需接口。
    底层网络作为子模块注册，因此 ``.train()`` / ``.eval()`` / ``.training``
    会自动作用到内部网络（从而正确控制 BatchNorm）。

    Attributes:
        ptnet: 被包装的底层 PointNet 网络。
        style: 已检测出的风格，取值 :data:`STYLE_LK` 或 :data:`STYLE_YANX27`。
    """

    def __init__(self, ptnet: torch.nn.Module) -> None:
        """包装一个现成的 PointNet 网络实例。

        Args:
            ptnet: ``PointNet_features`` 或 ``PointNetEncoder`` 实例。

        Raises:
            TypeError: 传入的网络不属于任何受支持的风格。
        """
        super().__init__()
        self.ptnet = ptnet
        self.style = self._detect_style(ptnet)

    @staticmethod
    def _detect_style(ptnet: torch.nn.Module) -> str:
        """根据实例类型自动识别 PointNet 风格。"""
        if isinstance(ptnet, pointnet_lk_style.PointNet_features):
            return STYLE_LK
        if isinstance(ptnet, pointnet_yanx27_style.PointNetEncoder):
            return STYLE_YANX27
        raise TypeError(
            "Unsupported PointNet: {}. Use either "
            "pointnet_lk_style.PointNet_features or "
            "pointnet_yanx27_style.PointNetEncoder.".format(type(ptnet).__name__)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """从点云 [B, N, 3] 提取全局特征 [B, K]。

        Args:
            points: 输入点云，形状 (batch_size, num_points, 3)。

        Returns:
            feat: 全局特征，形状 (batch_size, K)。

        Note:
            - lk 风格已直接接受 [B, N, 3]，直接透传；
            - yanx27 风格需要将输入转置为 [B, 3, N]，且其前向返回
              ``(feat, trans)`` 元组，这里取第一个元素作为特征。
        """
        if self.style == STYLE_LK:
            return self.ptnet(points)

        # yanx27: [B, N, 3] -> [B, 3, N]，返回 (feat, trans)
        feat, _ = self.ptnet(points.transpose(1, 2).contiguous())
        return feat


def create_pointnet(
    style: str = STYLE_LK,
    **kwargs: object,
) -> PointNet:
    """按风格名创建并包装一个 PointNet 编码器。

    Args:
        style: 风格名，取值 :data:`STYLE_LK` 或 :data:`STYLE_YANX27`。
        **kwargs: 传递给底层网络构造器的其余参数。

    Returns:
        包装好的 :class:`PointNet` 实例，可直接交给 PointNetLK。

    Example:
        >>> ptnet = create_pointnet(STYLE_LK)
        >>> lk = PointNetLK(ptnet)
    """
    if style == STYLE_LK:
        net = pointnet_lk_style.PointNet_features(**kwargs)      # type: ignore[arg-type]
    elif style == STYLE_YANX27:
        net = pointnet_yanx27_style.PointNetEncoder(**kwargs)    # type: ignore[arg-type]
    else:
        raise ValueError("Unknown style: {!r}".format(style))
    return PointNet(net)