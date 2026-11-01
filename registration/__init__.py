"""点云配准包：PointNetLK 粗配准 + PyTorch3D ICP 精细配准。

所有算法/计算已下沉到 :mod:`model`；本包只负责编排，并复用
:mod:`model.pointnet` 兼容层统一 lk/yanx27 两种 PointNet 风格。

建议直接使用子模块导入：
    from registration.pipeline import (
        PointCloudRegistrationPipeline,   # 粗配准 + 精配准 一体化流程
        iterative_fine,                   # 独立 PyTorch3D ICP 精配准
    )
"""