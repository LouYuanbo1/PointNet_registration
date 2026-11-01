"""配准网络模型包：PointNet 特征网络、PointNetLK 配准网络及其变体。

- :mod:`model.pointnet`          统一 PointNet 兼容层，自动适配 lk / yanx27 两种风格
- :mod:`model.pointnet_lk_style` PointNet_features（论文 PointNetLK 原实现风格）
- :mod:`model.pointnet_yanx27_style` PointNetEncoder（第三方 PointNet++ 精简版风格）
- :mod:`model.pointnetlk`        PointNetLK 配准网络（忠实复刻论文实现）
- :mod:`model.pointnetlk_v2`     PointNetLK V2 优化版（向量化雅可比 + 逐样本收敛）

本包只负责「网络定义」，网络的装配由 :mod:`utils.builders` 完成。
"""