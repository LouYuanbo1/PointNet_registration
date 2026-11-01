"""ModelNet40 点云数据加载与数据集封装。

将原先散落在 ``train_pointnetlk.py`` 中的数据加载逻辑集中于此，
供训练/评测脚本复用。包含：

- :func:`locate_modelnet40` / :func:`modelnet_classes`：定位数据集目录、枚举类别；
- :class:`ModelNet40PointCloud`：批量读取本地点云，做去中心/尺度归一/降采样。
"""

from __future__ import annotations

import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset


def locate_modelnet40(root):
    data_dir = os.path.join(root, "modelnet40_normal_resampled")
    marker = os.path.join(data_dir, "modelnet40_train.txt")
    if not (os.path.isdir(data_dir) and os.path.isfile(marker)):
        raise SystemExit(
            "\n未找到本地 ModelNet40 数据。"
            f"\n请将已解压的 modelnet40_normal_resampled 数据放到:"
            f"\n    {os.path.abspath(data_dir)}"
        )
    return data_dir


def modelnet_classes(data_dir):
    cats = set()
    for split in ("train", "test"):
        with open(os.path.join(data_dir, f"modelnet40_{split}.txt")) as f:
            for line in f:
                line = line.strip()
                if line:
                    cats.add(re.sub(r"_\d+$", "", line))
    return sorted(cats)


class ModelNet40PointCloud(Dataset):
    """读取本地 ModelNet40 点云作为配准模板，归一化到零均值单位尺度。"""

    def __init__(self, data_dir, split="train", num_points=1024,
                 aug=True, seed=42):
        with open(os.path.join(data_dir, f"modelnet40_{split}.txt")) as f:
            names = [line.strip() for line in f if line.strip()]
        rng = random.Random(seed)
        rng.shuffle(names)
        self.data_dir = data_dir
        self.names = names
        self.num_points = num_points
        self.aug = aug

    def read_pts(self, name):
        cat = re.sub(r"_\d+$", "", name)
        path = os.path.join(self.data_dir, cat, name + ".txt")
        pts = torch.from_numpy(np.loadtxt(path, dtype=np.float32, delimiter=","))[:, :3]
        pts = pts - pts.mean(0)
        pts = pts / (pts.abs().max() + 1e-6)
        n = pts.shape[0]
        if n >= self.num_points:
            idx = torch.randperm(n)[: self.num_points]
        else:
            idx = torch.arange(n).repeat((self.num_points + n - 1) // n)[: self.num_points]
        pts = pts[idx]
        if self.aug:
            pts = self._rotate_aug(pts)
        return pts

    def _rotate_aug(self, pts):
        ax = torch.randn(3)
        ax = ax / (ax.norm() + 1e-8)
        a = torch.rand(1).item() * 2 * torch.pi
        K = torch.tensor(
            [[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]],
            dtype=pts.dtype,
        )
        R = torch.eye(3) + a * K + (1 - torch.cos(torch.tensor(a))) * (K @ K)
        return pts @ R.T

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return self.read_pts(self.names[i])