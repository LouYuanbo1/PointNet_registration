"""批量评测：PointNetLK 粗配准 + PyTorch3D ICP  vs  仅 PyTorch3D ICP。

对同一批点云 (B, N, 3) 做配准，比较三种策略：
    A) 仅 PointNetLK 粗配准
    B) 仅 PyTorch3D ICP（单位阵初值，跳过 PointNetLK）
    C) 完整流水线：PointNetLK 粗配准 -> PyTorch3D ICP 精配准

每个策略都测量并在打印：
    - 耗时（秒）：整批 wall 时间 + 每个样本平均时间
    - 旋转误差（度）、平移误差、最终最近邻 rmse、是否收敛
    - 旋转/平移对应的 Hausdorff 距离 与 Chamfer 距离（在配准后的点云 vs 目标上）

数据与真值生成完全复用 train_pointnetlk.py：
    - ModelNet40PointCloud 批量加载真实点云（已做去中心/尺度归一/降采样）
    - random_igt_twist / apply_rt 生成官方 twist 式随机刚体真值并构造配对点云

用法（在项目根目录 PointNet_registration/ 下运行）:
    # 未训练：未提供权重时仍可运行（PointNetLK 用随机初始化，仅体现流程图）
    python benchmark_registration.py --batch 8 --num-samples 64

    # 加载训练权重（训练后推荐）
    python benchmark_registration.py --weights checkpoints/pointnetlk_v2.pt --batch 8 --num-samples 64

参数说明（均可选，均有默认值）:
    --weights      训练好的 PointNetLK 特征权重 .pt 路径（默认 None=随机初始化）
    --version      模型版本，lk 或 v2（默认 v2）
    --root         数据根目录，其下需含 modelnet40_normal_resampled（默认 ./data）
    --split        评测数据集划分，train / test（默认 test）
    --batch        批大小，控制显存占用，不影响总量（默认 8）
    --num-samples  参与评测的点云样本总数（默认 64）
    --num-points   每个点云采样的点数 N（默认 1024）
    --mag          官方式 SE(3) 扰动幅度(弧度)，越大配准越难（默认 1.0）
    --max-coarse-iter  PointNetLK 粗配准迭代上限（默认 10）
    --max-icp-iter     ICP 精配准迭代上限（默认 50）
    --relative-rmse-thr ICP 相对 RMSE 收敛阈值（默认 5e-3，越大越容易收敛）

常用示例:
    # 快速验证流程（少量样本+小批）
    python benchmark_registration.py --num-samples 16 --batch 4

    # 仅用更多点云样本测试
    python benchmark_registration.py --num-samples 128

    # 调整每点云点数
    python benchmark_registration.py --num-points 2048 --num-samples 32

    # 更难的配准任务（扰动更大）
    python benchmark_registration.py --mag 1.5 --num-samples 64

    # 在 Colab 中可通过
    #   %cd /content/PointNet_registration
    #   %run benchmark_registration.py --weights checkpoints/pointnetlk_v2.pt --batch 8 --num-samples 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data import ModelNet40PointCloud  # noqa: E402  # 复用数据与变换工具
from utils.builders import build_lk, get_default_device  # noqa: E402
from utils.transforms import apply_rt, random_igt_twist  # noqa: E402
from registration.pipeline import iterative_fine  # noqa: E402


def chamfer_distance(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """对称 Chamfer 距离（平均最近邻平方距离），支持批量。

    src/tgt: (B, N, 3)。逐批次计算每点最近邻，度量点云之间接近程度。
    返回: (B,) 的对称 Chamfer 距离（欧氏距离均值，已开方）。
    """
    B = src.shape[0]
    res = []
    for b in range(B):
        s, t = src[b], tgt[b]                     # (N, 3)
        d_ss = (s.unsqueeze(1) - t.unsqueeze(0)).pow(2).sum(-1)  # (N, N)
        min_st = d_ss.min(dim=1).values           # 每点在 t 最近邻平方距离
        min_ts = d_ss.min(dim=0).values           # 每点在 s 最近邻平方距离
        cham = (min_st.mean() + min_ts.mean()) / 2.0
        res.append(cham.clamp(min=0.0).sqrt())
    return torch.stack(res)


def hausdorff_distance(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """单向 Hausdorff 距离的最大值（max over 双向），支持批量。

    src/tgt: (B, N, 3)。hausdorff = max( max_a min_b d, max_b min_a d )。
    返回: (B,) 的 Hausdorff 距离（欧氏距离，已开方）。
    """
    B = src.shape[0]
    res = []
    for b in range(B):
        s, t = src[b], tgt[b]
        d = (s.unsqueeze(1) - t.unsqueeze(0)).pow(2).sum(-1)
        h1 = d.min(dim=1).values.max()            # 从 s 到 t
        h2 = d.min(dim=0).values.max()            # 从 t 到 s
        res.append(torch.maximum(h1, h2).clamp(min=0.0).sqrt())
    return torch.stack(res)


def rotation_error_batch(R_gt: torch.Tensor, R_est: torch.Tensor) -> torch.Tensor:
    """批量旋转角度误差（度）。R_gt/R_est: (B, 3, 3)。"""
    M = torch.bmm(R_est, R_gt.transpose(1, 2))
    cos = torch.clamp((M.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


@torch.no_grad()
def evaluate_batch(
    pipe,
    data_dir: str,
    split: str,
    num_points: int,
    batch: int,
    num_samples: int,
    mag: float,
    max_icp_iter: int,
    relative_rmse_thr: float,
    device: str,
) -> Tuple[dict, dict, dict]:
    """对一批随机生成的配准任务，同时跑 仅PointNetLK / 仅ICP / PointNetLK+ICP，汇总指标。"""
    ds = ModelNet40PointCloud(data_dir, split=split, num_points=num_points,
                              aug=False, seed=42)
    n = min(num_samples, len(ds))
    idx = torch.randperm(len(ds))[:n]
    samples = torch.stack([ds[int(i)] for i in idx]).to(device)  # (n,N,3)

    # 统一探针：测量三种策略的时间、位姿误差、Hausdorff、Chamfer
    keys = ["time", "rot_err", "trans_err", "rmse", "haus", "cham"]
    agg_a = {k: [] for k in keys}     # PointNetLK + ICP
    agg_b = {k: [] for k in keys}     # 仅 ICP
    agg_lk = {k: [] for k in keys}    # 仅 PointNetLK
    icp_b_conv = []
    icp_a_conv = []
    lk_conv = []

    # 一次性预生成全部 n 个随机扰动，再在循环内按 batch 切片，
    # 保证 samples[i] 始终配同一个 igt[i]，与 batch 大小无关（可复现）。
    igt_all = random_igt_twist(n, mag=mag, mag_randomly=True, device=device)  # (n,4,4)
    for st in range(0, n, batch):
        p0 = samples[st:st + batch]                       # (B,N,3)
        B = p0.shape[0]
        # 原方法：每批重新随机生成扰动，结果依赖 batch（batch 越大越偏移随机流），已弃用以保可复现。
        # igt = random_igt_twist(B, mag=mag, mag_randomly=True, device=device)
        igt = igt_all[st:st + batch]                      # (B,4,4) = 对应当前 batch 的扰动真值
        p1 = apply_rt(p0, igt)                            # 目标点云

        R_gt = igt[:, :3, :3]
        t_gt = igt[:, :3, 3]

        # ---- A) PointNetLK 粗配准 + ICP 精配准 ----
        torch.cuda.synchronize(device) if device == "cuda" else None
        t0 = time.perf_counter()
        out = pipe.register(p0, p1, max_icp_iter=max_icp_iter,
                            relative_rmse_thr=relative_rmse_thr,
                            estimate_scale=False, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        tA = time.perf_counter() - t0
        XtA = out["Xt"].to(p0.dtype)
        agg_a["time"].append(np.full(B, tA / B))
        agg_a["rot_err"].append(rotation_error_batch(R_gt, out["R_fine"].float()).cpu().numpy())
        agg_a["trans_err"].append(
            torch.norm(out["T_fine"].float() - t_gt, dim=1).cpu().numpy())
        agg_a["rmse"].append(np.asarray(out["rmse"].cpu().numpy()).reshape(-1))
        agg_a["haus"].append(hausdorff_distance(XtA, p1).cpu().numpy())
        agg_a["cham"].append(chamfer_distance(XtA, p1).cpu().numpy())
        icp_a_conv.append(out["converged"])

        # ---- B) 仅 ICP（单位阵初值）----
        torch.cuda.synchronize(device) if device == "cuda" else None
        t0 = time.perf_counter()
        sol = iterative_fine(p0, p1, max_iterations=max_icp_iter,
                             relative_rmse_thr=relative_rmse_thr,
                             estimate_scale=False, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        tB = time.perf_counter() - t0
        XtB = sol["Xt"].to(p0.dtype)
        agg_b["time"].append(np.full(B, tB / B))
        agg_b["rot_err"].append(rotation_error_batch(R_gt, sol["R"].float()).cpu().numpy())
        agg_b["trans_err"].append(torch.norm(sol["T"].float() - t_gt, dim=1).cpu().numpy())
        agg_b["rmse"].append(np.asarray(sol["rmse"].cpu().numpy()).reshape(-1))
        agg_b["haus"].append(hausdorff_distance(XtB, p1).cpu().numpy())
        agg_b["cham"].append(chamfer_distance(XtB, p1).cpu().numpy())
        icp_b_conv.append(sol["converged"])

        # ---- C) 仅 PointNetLK 粗配准 ----
        torch.cuda.synchronize(device) if device == "cuda" else None
        t0 = time.perf_counter()
        R_lk, T_lk, _ = pipe._coarse(p0, p1)
        if device == "cuda":
            torch.cuda.synchronize()
        tL = time.perf_counter() - t0
        XtL = torch.matmul(p0, R_lk.transpose(1, 2)) + T_lk.unsqueeze(1)
        agg_lk["time"].append(np.full(B, tL / B))
        agg_lk["rot_err"].append(rotation_error_batch(R_gt, R_lk.float()).cpu().numpy())
        agg_lk["trans_err"].append(torch.norm(T_lk.float() - t_gt, dim=1).cpu().numpy())
        agg_lk["rmse"].append(np.zeros(B))
        agg_lk["haus"].append(hausdorff_distance(XtL, p1).cpu().numpy())
        agg_lk["cham"].append(chamfer_distance(XtL, p1).cpu().numpy())
        # PointNetLK 收敛判据：实际迭代轮数 < 上限（提前退出即视为收敛）
        lk_conv.append(bool(pipe.lk.itr < pipe.max_coarse_iter))

    def _finalize(agg):
        return {k: np.concatenate(v) for k, v in agg.items()}

    out_a = _finalize(agg_a)
    out_b = _finalize(agg_b)
    out_lk = _finalize(agg_lk)
    out_a["conv"] = icp_a_conv
    out_b["conv"] = icp_b_conv
    out_lk["conv"] = lk_conv
    return out_a, out_b, out_lk


def _match_conv(conv) -> List[bool]:
    """把 ICP converged 结果展开为逐样本 bool 列表。"""
    out = []
    for c in conv:
        if isinstance(c, torch.Tensor):
            out += c.flatten().tolist()
        elif isinstance(c, list):
            for x in c:
                out.append(bool(x))
    return [bool(x) for x in out]


def _mean_std(a: np.ndarray) -> str:
    return f"{a.mean():.4f}±{a.std():.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description="批量评测 PointNetLK+ICP vs 仅ICP")
    p.add_argument("--weights", default="checkpoints/pointnetlk_v2.pt",
                   help="训练好的特征权重 .pt（默认 checkpoints/pointnetlk_v2.pt）")
    p.add_argument("--version", choices=["lk", "v2"], default="v2")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--mag", type=float, default=1.0, help="官方式 SE(3) 扰动幅度(弧度)")
    p.add_argument("--max-coarse-iter", type=int, default=10)
    p.add_argument("--max-icp-iter", type=int, default=50)
    p.add_argument("--relative-rmse-thr", type=float, default=5.0e-3)
    args = p.parse_args()

    torch.manual_seed(42)
    device = get_default_device()
    print(f"设备: {device}   配准模型: {args.version}")
    print(f"权重: {args.weights or '（未加载，PointNetLK 用随机初始化）'}")

    from registration.pipeline import PointCloudRegistrationPipeline

    # 构造 PointNetLK 流水线（复用 train_pointnetlk 的 build_lk 加载权重）
    lk = build_lk(args.version, pre_weights=args.weights, device=device)
    lk.eval()
    pipe = PointCloudRegistrationPipeline(
        lk.ptnet, max_coarse_iter=args.max_coarse_iter, coarse_xtol=1.0e-7)
    pipe.pointnet.eval()          # pipeline 非 nn.Module，需显式关掉编码器训练态(BatchNorm)

    data_dir = os.path.join(args.root, "modelnet40_normal_resampled")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"未找到数据目录: {data_dir}")

    out_a, out_b, out_lk = evaluate_batch(
        pipe, data_dir, args.split, args.num_points, args.batch,
        args.num_samples, args.mag, args.max_icp_iter, args.relative_rmse_thr, device,
    )

    conv_a = _match_conv(out_a["conv"])     # PointNetLK + ICP 的 ICP 收敛
    conv_b = _match_conv(out_b["conv"])     # 仅 ICP 的收敛
    conv_lk = _match_conv(out_lk["conv"])   # 仅 PointNetLK 的收敛
    na, nb, nlk = len(out_a["rot_err"]), len(out_b["rot_err"]), len(out_lk["rot_err"])

    print("=" * 96)
    print("配准评测结果（数值格式: 均值 ± 标准差）")
    print("=" * 96)
    _keys = ["time", "rot_err", "trans_err", "haus", "cham"]
    _colnames = ["耗时(s/样本)", "旋转误差(°)", "平移误差", "Hausdorff", "Chamfer"]
    print(f"{'策略':<20}" + "\t".join(_colnames))
    for _label, _agg in (("A) 仅PointNetLK", out_lk),
                         ("B) 仅ICP", out_b),
                         ("C) PointNetLK+ICP", out_a)):
        print(f"{_label:<20}" + "\t".join(_mean_std(_agg[k]) for k in _keys))
    print("总耗时(全部样本): " + "  |  ".join(
        f"{_label} = {_agg['time'].sum():.2f}s" for _label, _agg in
        (("A", out_lk), ("B", out_b), ("C", out_a))))
    print("=" * 96)
    print(f"收敛率:  A = {sum(conv_lk)}/{nlk}    B = {sum(conv_b)}/{nb}    C = {sum(conv_a)}/{na}")
    print(f"样本数: {na}（split={args.split}, batch={args.batch}）")
    print("=" * 96)
    print("说明:")
    print("  - A: 仅 PointNetLK 粗配准（不接 ICP）")
    print("  - B: 单位阵初值直接跑 PyTorch3D ICP（跳过粗配准）")
    print("  - C: PointNetLK 粗配准给出初值 → PyTorch3D ICP 精配准")
    print("  - Hausdorff / Chamfer 越小说明配准后点云与目标贴合越好")
    print("  - 点云已去中心：ICP 的质心对齐是解析精确解，平移误差通常很小且")
    print("    不含判别信息，请以旋转误差 / Hausdorff / Chamfer 判断配准好坏")


if __name__ == "__main__":
    main()