"""端到端配准演示示例：PointNet 特征粗配准(PointNetLK) + PyTorch3D ICP 精细配准。

展示多组不同场景下三种配准策略的对比：
    A) 仅粗配准（PointNetLK 特征）
    B) 完整流水线：粗配准 + PyTorch3D ICP 精配准
    C) 仅 PyTorch3D ICP（单位阵初值，不做粗配准）

基于 :mod:`model` 重新设计：
- 特征网络用 :func:`model.pointnet.create_pointnet` 构造（兼容层自动处理
  lk/yanx27 两种风格，不再需要 registration.adapters）；
- 真值刚体变换用 :func:`utils.transforms.random_igt_twist`（SE(3) twist 一体采样）
  生成，与 `train_pointnetlk.py` / `benchmark_registration.py` 保持一致：
  标准正态采样 6 维 twist -> 归一化后按 U(0, mag) 随机缩放 -> SE(3) 指数映射。

运行方式（在项目根目录 PointNet_registration/ 下）：
    python scripts/example_pipeline.py [--model lk|yanx27] [--weights path.pt]

设备：默认优先使用 GPU，机器人无 GPU 时自动回退 CPU。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.pointnet import STYLE_LK, PointNet, create_pointnet
from utils.builders import get_default_device
from utils.transforms import random_igt_twist
from registration.pipeline import PointCloudRegistrationPipeline, iterative_fine


def make_source_pointcloud(n: int = 1024, device: str = "cpu") -> torch.Tensor:
    """构造一个刚性点云（实心立方体表面采样）。"""
    m = int(round(n ** (1.0 / 3.0)))
    assert m >= 3, "n 太小，无法构建立方体网格"
    axis = torch.linspace(-1.0, 1.0, steps=m, device=device)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    if pts.shape[0] > n:
        idx = torch.randperm(pts.shape[0], device=device)[:n]
        pts = pts[idx]
    return pts  # (n, 3)


def apply_rt(points: torch.Tensor, R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """用旋转 R(3,3) 和平移 T(3,) 变换点云 (N,3)。"""
    return points @ R.T + T


def rotation_error(R_gt: torch.Tensor, R_est: torch.Tensor) -> torch.Tensor:
    """返回旋转角度误差（度）。"""
    M = R_est @ R_gt.T
    cos = torch.clamp((torch.trace(M) - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def axis_angle_from_rot(
    R: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """从旋转矩阵 R 精确提取 旋转轴(单位向量) + 旋转角(度)。

    返回 (axis, angle_deg)；angle_deg ∈ [0, 180]°。对 0° 与 180° 也能稳定给出轴。
    """
    cos = torch.clamp((torch.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos)
    angle_deg = torch.rad2deg(theta)

    if angle_deg.item() < 1e-3:  # 0°：轴无定义，给单位 z
        return torch.zeros(3, dtype=R.dtype, device=R.device), 0.0

    sin_theta = torch.sin(theta)
    if torch.abs(sin_theta) > 1e-4:  # 常规情形：用反对称部分
        axis = torch.stack(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        ) / (2.0 * sin_theta)
    else:  # 接近 180°：用 R + I 恢复轴（vv^T 的比例）
        A = R + torch.eye(3, dtype=R.dtype, device=R.device)
        axis = A[A.norm(dim=1).argmax()]
        axis = axis / (axis.norm() + 1e-12)
    return axis, angle_deg


def run_case(
    pipeline: PointCloudRegistrationPipeline,
    src: torch.Tensor,
    tgt: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    max_icp_iter: int = 80,
) -> Dict[str, object]:
    """对同一组 src/tgt 跑三种策略，返回各自误差指标。"""
    out = {}

    # B) 完整流水线（粗配准 + ICP），返回里包含粗配准与精配准结果
    full = pipeline.register(
        src, tgt, max_icp_iter=max_icp_iter, estimate_scale=False, verbose=False
    )
    # A) 仅粗配准：直接取完整流程里保存的粗配准结果
    out["A_coarse_err"] = (
        rotation_error(R_gt, full["R_coarse"].to(torch.float32)).item(),
        torch.norm(full["T_coarse"].to(torch.float32) - t_gt).item(),
    )

    # C) 仅 PyTorch3D ICP：单位阵初值，不做粗配准
    sol = iterative_fine(
        src, tgt, max_iterations=max_icp_iter, estimate_scale=False, verbose=False
    )
    out["C_icp_err"] = (
        rotation_error(R_gt, sol["R"].to(torch.float32)).item(),
        torch.norm(sol["T"].to(torch.float32) - t_gt).item(),
        sol["rmse"].item(),
        bool(sol["converged"][0]) if isinstance(sol["converged"], torch.Tensor) else bool(sol["converged"]),
    )

    # B) 完整流程的精配准误差
    out["B_full_err"] = (
        rotation_error(R_gt, full["R_fine"].to(torch.float32)).item(),
        torch.norm(full["T_fine"].to(torch.float32) - t_gt).item(),
        full["rmse"].item(),
        bool(full["converged"][0]) if isinstance(full["converged"], torch.Tensor) else bool(full["converged"]),
    )
    return out


def build_pointnet(model: str = "lk", weights: Optional[str] = None) -> PointNet:
    """按模型名构造统一 PointNet 兼容网络，可选加载训练好的特征权重。

    model: “lk” 或 “yanx27”（model.pointnet 自动适配）。
    weights: 训练好的特征权重 .pt 文件路径。
    网络默认放在 GPU（可用时），否则回退 CPU。
    """
    device = get_default_device()
    if model == "yanx27":
        net = create_pointnet(style="yanx27", channel=3)
    else:
        net = create_pointnet(style=STYLE_LK, dim_k=1024)
    net = net.to(device)
    if weights:
        state = torch.load(weights, map_location=device)
        if isinstance(state, dict) and "net_features" in state:  # checkpoint 容器
            state = state["net_features"]
        # checkpoint 中的 feat 权重保存自 model.ptnet.state_dict()，键带
        # PointNet 包装器的 "ptnet." 前缀（如 ptnet.h1.*）；加载到内层
        # PointNet_features 时移除该前缀。
        state = {
            (k[len("ptnet."):] if k.startswith("ptnet.") else k): v
            for k, v in state.items()
        }
        net.ptnet.load_state_dict(state)
        print(f"已加载特征权重: {weights}")
    return net


def run_suite(pointnet: PointNet, title: str = "") -> None:
    """用给定 PointNet 兼容网络跑完整的配准场景对比，并打印结果。"""
    device = torch.device(get_default_device())
    print("=" * 100)
    print(f"特征模型: {type(pointnet.ptnet).__name__}   {title}")
    print(f"torch 设备: {device}")
    print(
        "PyTorch3D: ICP 将在输入设备执行；若 CUDA 内核缺失则自动回退 CPU。"
    )
    print("=" * 100)

    pipeline = PointCloudRegistrationPipeline(
        pointnet,
        max_coarse_iter=10,
        coarse_xtol=1.0e-7,
    )

    # 测试场景：真值变换统一用 random_igt_twist 采样（任意轴旋转 + 平移、按样本随机缩放）；
    # mag 控制 SE(3) twist 幅度（弧度），越大配准越难；n 为源点数、noise 为施加的高斯噪声。
    cases = [
        dict(name="小幅度SE3", n=800, mag=0.3, noise=0.0),
        dict(name="中幅度SE3", n=800, mag=1.0, noise=0.0),
        dict(name="大幅度SE3", n=800, mag=1.5, noise=0.0),
        dict(name="更大角度SE3", n=800, mag=2.0, noise=0.0),
        dict(name="稀疏点云 n=300", n=300, mag=1.0, noise=0.0),
        dict(name="带高斯噪声", n=1000, mag=1.0, noise=0.02),
    ]

    print("=" * 100)
    print(
        f"{'场景':<14}{'真值旋转':<12}{'A.仅粗配准':<16}"
        f"{'B.粗+精':<16}{'C.仅ICP':<16}"
    )
    print(
        f"{'':<14}{'(度)':<12}{'(旋°|平)':<16}{'(旋°|平|rmse)':<24}{'(旋°|平|rmse|conv)':<28}"
    )

    for c in cases:
        print("-" * 100)
        print(f"场景: {c['name']}   源点数 n={c['n']}   噪声={c['noise']}   mag={c['mag']}")
        igt = random_igt_twist(1, mag=c["mag"], mag_randomly=True,
                               device=device)[0]  # (4,4)
        R_gt = igt[:3, :3]   # (3,3)，SE(3) 指数映射得到的旋转矩阵
        t_gt = igt[:3, 3]    # (3,)，SE(3) 指数映射得到的平移

        src = make_source_pointcloud(n=c["n"], device=device).to(torch.float32)
        tgt = apply_rt(src, R_gt, t_gt)
        if c["noise"] > 0:
            tgt = tgt + torch.randn_like(tgt) * c["noise"]

        try:
            r = run_case(pipeline, src, tgt, R_gt, t_gt)
            a_rot, a_t = r["A_coarse_err"]
            b_rot, b_t, b_rmse, b_conv = r["B_full_err"]
            c_rot, c_t, c_rmse, c_conv = r["C_icp_err"]

            ax, ang_deg = axis_angle_from_rot(R_gt)
            tx, ty, tz = t_gt[0].item(), t_gt[1].item(), t_gt[2].item()
            print(
                f"  真值变换(轴角) : 轴=({ax[0]:+.3f},{ax[1]:+.3f},{ax[2]:+.3f})"
                f"  角={ang_deg:6.2f}°  平移=({tx:+.3f},{ty:+.3f},{tz:+.3f})"
            )
            print(f"  A.仅粗配准      : 旋误差 {a_rot:8.3f}°  平移误差 {a_t:.5f}")
            print(
                f"  B.粗配准+ICP    : 旋误差 {b_rot:8.3f}°  平移误差 {b_t:.5f}  "
                f"rmse {b_rmse:.2e}  conv={b_conv}"
            )
            print(
                f"  C.仅PyTorch3DICP: 旋误差 {c_rot:8.3f}°  平移误差 {c_t:.5f}  "
                f"rmse {c_rmse:.2e}  conv={c_conv}"
            )
        except Exception as e:
            print(f"  !! 该场景运行失败: {type(e).__name__}: {e}")
        sys.stdout.flush()

    print("=" * 100)
    print(
        "说明: 对同一点云, A=只用PointNet粗配准, B=粗配准+ICP精修, "
        "C=完全跳过粗配准、直接对单位阵初值做PyTorch3D ICP。"
    )
    print(
        "B 相比 C 的优势: 在旋转接近/超过 90°、或离初值很远时, "
        "ICP(局部方法)常陷入错误局部最优, 而粗配准先把它拉进收敛盆。"
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="端到端配准演示示例")
    p.add_argument("--model", choices=["default", "lk", "yanx27"], default="lk")
    p.add_argument("--weights", default="checkpoints/pointnetlk_v2.pt",
                   help="加载训练好的特征权重 .pt（默认 checkpoints/pointnetlk_v2.pt）")
    args = p.parse_args()

    model = "lk" if args.model == "default" else args.model
    net = build_pointnet(model, args.weights)
    run_suite(net, title="配准演示示例")


if __name__ == "__main__":
    main()