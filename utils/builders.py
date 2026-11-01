"""配准网络构造工厂。

将 PointNetLK / PointNetLKV2 的装配逻辑（含可选预训练特征权重加载）
从 ``train_pointnetlk.py`` 抽出，供训练 / 评测 / 测试脚本复用。

本模块只负责「装配」，与 :mod:`model` 中「网络定义」解耦：
若机器支持 GPU，默认在 GPU 上构建，否则回退 CPU。
"""

from __future__ import annotations

import os

import torch

from model.pointnet import STYLE_LK, create_pointnet
from model.pointnetlk import PointNetLK
from model.pointnetlk_v2 import PointNetLKV2


def get_default_device() -> str:
    """返回默认设备：CUDA 可用则用 GPU，否则回退 CPU。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_lk(version, pre_weights=None, device=None, **kw):
    """构造 lk 或 v2 配准网络，可选加载预训练特征权重。

    device 缺省时自动取 :func:`get_default_device`（GPU 优先，无则 CPU）。
    """
    if device is None:
        device = get_default_device()
    net = create_pointnet(style=STYLE_LK, dim_k=1024).to(device)
    if pre_weights and os.path.exists(pre_weights):
        state = torch.load(pre_weights, map_location=device)
        if isinstance(state, dict):
            state = state.get("net_features", state)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        net.ptnet.load_state_dict(state, strict=False)
        print(f"已加载初始特征权重: {pre_weights}")
    cls = PointNetLK if version == "lk" else PointNetLKV2
    return cls(net, **kw).to(device)