"""基于 :mod:`model` 的点云配准流水线：PointNetLK 粗配准 + PyTorch3D ICP 精配准。

设计说明（相对旧版的重构）：
- 所有「算法/计算」部分已全部下沉到 :mod:`model`：
    - PointNet 全局特征网络  -> :func:`model.pointnet.create_pointnet`（内置
      ``lk`` / ``yanx27`` 两种风格的自动适配，含输入转置、元组取值）；
    - 粗配准（SE(3) 李代数）-> :class:`model.pointnetlk.PointNetLK`；
    - SE(3)/SO(3) 算子       -> :mod:`lie.se3` / :mod:`lie.so3`。
  registration 不再复刻任何旋转/平移公式，也不维护 torch 版 ICP。
- 本模块只做「编排」：
    1. 用 ``PointNetLK`` 对 ``src`` 与 ``tgt`` 做粗配准，得到粗位姿；
    2. 把粗位姿作为初值，交给 ``pytorch3d.ops.iterative_closest_point`` 精化；
    3. 组合两次变换，返回最终刚体位姿。
- 因此旧的 ``registration.adapters`` 不再需要 —— 两种 PointNet 风格的
  输入规范化已由 :class:`model.pointnet.PointNet` 兼容层统一处理。

用法：
    >>> from registration.pipeline import PointCloudRegistrationPipeline
    >>> pipe = PointCloudRegistrationPipeline()            # 默认 lk 风格编码器
    >>> out = pipe.register(src, tgt)                      # src/tgt 为 (N,3) 点云
    >>> R, t = out["R_fine"], out["T_fine"]
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from model.pointnet import PointNet, create_pointnet, STYLE_LK
from model.pointnetlk import PointNetLK


def _run_pytorch3d_icp(
    src_pre: Tensor,
    tgt: Tensor,
    *,
    max_iterations: int,
    relative_rmse_thr: float,
    estimate_scale: bool,
    verbose: bool,
) -> object:
    """调用 pytorch3d 的 ICP，惰性导入以降低模块首载开销。

    src_pre / tgt: (B, N, 3)；初值已提前显式施加到 src_pre。
    返回 pytorch3d 的 ICPSolution（其字段 RTs.R / RTs.T / RTs.s / rmse /
    converged 被上层读取）。
    """
    from pytorch3d.ops import iterative_closest_point

    return iterative_closest_point(
        src_pre,
        tgt,
        init_transform=None,      # 初值已在外部显式施加，这里从单位阵起做 ICP
        max_iterations=max_iterations,
        relative_rmse_thr=relative_rmse_thr,
        estimate_scale=estimate_scale,
        verbose=verbose,
    )


def iterative_fine(
    src: Tensor,
    tgt: Tensor,
    R: Optional[Tensor] = None,
    T: Optional[Tensor] = None,
    *,
    max_iterations: int = 50,
    relative_rmse_thr: float = 5.0e-3,
    estimate_scale: bool = False,
    verbose: bool = False,
) -> dict:
    """独立的 PyTorch3D ICP 精配准（可先用粗位姿初始化）。

    Args:
        src: 待配准（移动）点云，形状 (N, 3) 或 (B, N, 3)。
        tgt: 模板（固定）点云，形状 (N, 3) 或 (B, N, 3)。
        R:   初值旋转，形状 (3, 3) 或 (B, 3, 3)；None 表示单位阵。
        T:   初值平移，形状 (3,) 或 (B, 3)；None 表示零平移。
        max_iterations:   ICP 最大迭代次数。
        relative_rmse_thr: 相对 RMSE 收敛阈值。
        estimate_scale:     是否估计相似变换缩放（默认否，保持刚体）。
        verbose:            是否打印每轮迭代信息。

    Returns:
        dict，键为：
            ``R`` / ``T``：  组合后的最终旋转/平移（维度与输入一致，单样本已压平）；
            ``s``：           缩放（estimate_scale=False 时为 1）；
            ``rmse``：         最终最近邻 RMSE；
            ``converged``：    是否收敛；
            ``Xt``：           施加最终变换后的源点云。
    """
    single = src.dim() == 2
    S = src.unsqueeze(0) if single else src                  # (B, N, 3)
    Tg = tgt.unsqueeze(0) if single else tgt
    batch = S.shape[0]
    device, dtype = S.device, S.dtype

    # 初值：(B, 3, 3) 旋转、(B, 3) 平移
    if R is None:
        R0 = torch.eye(3, device=device, dtype=dtype).expand(batch, 3, 3).clone()
    else:
        R0 = R.unsqueeze(0) if single else R
    if T is None:
        T0 = torch.zeros(batch, 3, device=device, dtype=dtype)
    else:
        T0 = T.unsqueeze(0) if T.dim() == 1 else T
    R0 = R0.to(device=device, dtype=dtype)
    T0 = T0.to(device=device, dtype=dtype)

    # 显式初值：先把粗位姿施加到源点云，再从单位阵初值做 ICP，
    # 避免依赖不同 pytorch3d 版本对 init_transform 的语义差异，保证初值一定生效。
    src_pre = torch.matmul(S, R0.transpose(1, 2)) + T0.unsqueeze(1)

    try:
        sol = _run_pytorch3d_icp(
            src_pre, Tg,
            max_iterations=max_iterations,
            relative_rmse_thr=relative_rmse_thr,
            estimate_scale=estimate_scale,
            verbose=verbose,
        )
    except Exception:
        # 常见情形：输入在 CUDA 但没有编译 knn 的 CUDA 内核 -> 回退 CPU。
        if verbose:
            print("PyTorch3D 无法在输入设备执行 ICP，自动回退到 CPU。")
        sol = _run_pytorch3d_icp(
            src_pre.detach().cpu(), Tg.detach().cpu(),
            max_iterations=max_iterations,
            relative_rmse_thr=relative_rmse_thr,
            estimate_scale=estimate_scale,
            verbose=verbose,
        )

    # 组合两次变换：y = R_icp (R0 x + T0) + T_icp = (R_icp R0) x + (R_icp T0 + T_icp)
    R_icp = sol.RTs.R.to(device=device, dtype=dtype)         # (B, 3, 3)
    T_icp = sol.RTs.T.to(device=device, dtype=dtype)         # (B, 3)
    R_fin = torch.matmul(R_icp, R0)
    T_fin = torch.matmul(R_icp, T0.unsqueeze(-1)).squeeze(-1) + T_icp
    Xt = torch.matmul(S, R_fin.transpose(1, 2)) + T_fin.unsqueeze(1)

    s = sol.RTs.s.to(device=device, dtype=dtype)
    rmse = sol.rmse.to(device=device, dtype=dtype)
    conv = sol.converged

    def _sq(x: Tensor) -> Tensor:
        """单样本输入时把批维去掉，保持与输入维度一致。"""
        return x[0] if single else x

    return {
        "R": _sq(R_fin),
        "T": _sq(T_fin),
        "s": _sq(s),
        "rmse": rmse if isinstance(rmse, list) else _sq(rmse),
        "converged": conv if isinstance(conv, list) else _sq(conv),
        "Xt": _sq(Xt),
    }


class PointCloudRegistrationPipeline:
    """粗配准(PointNetLK) + 精配准(PyTorch3D ICP) 一体化流程。

    通过 :class:`model.pointnet.PointNet` 兼容层统一管理 lk/yanx27 两种
    PointNet；粗配准用 :class:`model.pointnetlk.PointNetLK`，精细配准用
    :func:`iterative_fine`。

    Attributes:
        pointnet: 已包装的特征网络（:class:`model.pointnet.PointNet`）。
        lk:       粗配准用的 :class:`model.pointnetlk.PointNetLK` 实例。
    """

    def __init__(
        self,
        pointnet: Optional[PointNet] = None,
        *,
        style: str = STYLE_LK,
        dim_k: int = 1024,
        max_coarse_iter: int = 10,
        coarse_xtol: float = 1.0e-7,
        **pointnet_kwargs: object,
    ) -> None:
        """构造流水线。

        Args:
            pointnet: 现成的 :class:`model.pointnet.PointNet` 实例；若为
                None，则按 style/dim_k 自动构造。
            style: 点 net 为 None 时使用的编码器风格（lk 或 yanx27）。
            dim_k: 全局特征维度（默认 1024）。
            max_coarse_iter: PointNetLK 粗配准最大迭代次数。
            coarse_xtol: 粗配准收敛阈值（twist 增量范数）。
            **pointnet_kwargs: 传给 :func:`model.pointnet.create_pointnet` 的参数。
        """
        if pointnet is None:
            pointnet = create_pointnet(style=style, dim_k=dim_k, **pointnet_kwargs)
        self.pointnet = pointnet
        self.lk = PointNetLK(pointnet)
        self.max_coarse_iter = max_coarse_iter
        self.coarse_xtol = coarse_xtol

    def _coarse(self, src: Tensor, tgt: Tensor):
        """用 PointNetLK 求映射 src→tgt 的粗位姿。

        ``PointNetLK.forward(p0, p1)`` 估计使 ``g(p1) ≈ p0`` 的位姿 g，
        因此把 ``p0=tgt``、``p1=src`` 传入，得到把 src 对齐到 tgt 的变换。

        Returns:
            (R, t, history)：R (B,3,3)、t (B,3)、history (M,B,4,4)。
        """
        self.lk(tgt, src, maxiter=self.max_coarse_iter, xtol=self.coarse_xtol)
        g = self.lk.g                                  # (B, 4, 4)
        R = g[:, :3, :3].contiguous()                  # (B, 3, 3)
        t = g[:, :3, 3].contiguous()                   # (B, 3)
        return R, t, self.lk.g_series

    def register(
        self,
        src: Tensor,
        tgt: Tensor,
        *,
        max_icp_iter: int = 50,
        relative_rmse_thr: float = 5.0e-3,
        estimate_scale: bool = False,
        verbose: bool = True,
    ) -> dict:
        """对 src 到 tgt 执行完整配准：先粗后精。

        Args:
            src: 待配准（移动）点云，形状 (N, 3) 或 (B, N, 3)。
            tgt: 模板（固定）点云，形状 (N, 3) 或 (B, N, 3)。
            max_icp_iter:     ICP 最大迭代次数。
            relative_rmse_thr: ICP 相对 RMSE 收敛阈值。
            estimate_scale:    是否估计缩放。
            verbose:           是否打印 ICP 细节。

        Returns:
            dict：
                ``R_coarse`` / ``T_coarse``：粗配准旋转/平移；
                ``R_fine`` / ``T_fine``：    精配准（组合后）旋转/平移；
                ``s_fine``：                 精配准缩放；
                ``rmse``：                   精配准最终 RMSE；
                ``converged``：              精配准是否收敛；
                ``Xt``：                     施加最终变换后的源点云；
                ``history``：                PointNetLK 的位姿序列 (M, B, 4, 4)。
            单样本输入时各张量压平（不带批维）。
        """
        src = src.detach().float()
        tgt = tgt.detach().float()
        single = src.dim() == 2
        S = src.unsqueeze(0) if single else src
        Tg = tgt.unsqueeze(0) if single else tgt

        # 1. 粗配准（model.pointnetlk.PointNetLK）
        R_c, T_c, history = self._coarse(S, Tg)

        # 2. 精配准（pytorch3d ICP，以粗位姿为初值）
        fine = iterative_fine(
            S, Tg, R_c, T_c,
            max_iterations=max_icp_iter,
            relative_rmse_thr=relative_rmse_thr,
            estimate_scale=estimate_scale,
            verbose=verbose,
        )

        return {
            "R_coarse": R_c[0] if single else R_c,
            "T_coarse": T_c[0] if single else T_c,
            "R_fine": fine["R"],
            "T_fine": fine["T"],
            "s_fine": fine["s"],
            "rmse": fine["rmse"],
            "converged": fine["converged"],
            "Xt": fine["Xt"],
            "history": history,
        }