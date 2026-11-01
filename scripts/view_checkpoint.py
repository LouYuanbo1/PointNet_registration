"""查看/对比已保存的 PointNetLK 模型 checkpoint 参数信息。

读取 <save> 目录下的 checkpoint 文件并打印其元信息：
    - 图档的类型（特征权重 .pt / 完整快照 _opt.pt）、键名
    - Epoch（若存档包含）、模型版本、最佳验证旋转误差（若存档包含）
    - 网络参数量、可训练参数量、参数字典各张量形状
    - （可选）--detail 打印每个张量的 shape

文件命名规则与 train_pointnetlk.py 保持一致：
    <save>/pointnetlk_<version>.pt        特征权重（仅 net_features 单字段）
    <save>/pointnetlk_<version>_opt.pt    完整快照（含 optimizer/scheduler/epoch）

用法（在项目根目录 PointNet_registration/ 下运行）:
    python view_checkpoint.py                          # 查看 checkpoints/ 下全部已存在文件
    python view_checkpoint.py --version v2             # 只看 v2 的两个文件
    python view_checkpoint.py --detail                 # 并打印每个权重张量的 shape
    python view_checkpoint.py --checkpoint checkpoints/pointnetlk_v2.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

VERSIONS = ("lk", "v2")


def _count_params(state_dict: dict) -> tuple[int, int]:
    """统计总参数量与可训练参数量。"""
    total = int(sum(v.numel() for v in state_dict.values()))
    trainable = int(
        sum(v.numel() for k, v in state_dict.items() if not (
            k.endswith("num_batches_tracked") or k.endswith(".running_mean")
            or k.endswith(".running_var")))
    )
    return total, trainable


def _print_checkpoint(path: str, detail: bool) -> None:
    if not os.path.isfile(path):
        print(f"[跳过] 不存在: {path}")
        return

    ck = torch.load(path, map_location="cpu", weights_only=False)
    print(f"{'=' * 72}")
    print(f"文件: {path}")
    if not isinstance(ck, dict):
        print(f"  - 类型: {type(ck).__name__}（非 dict，无法解析字段）")
        return

    keys = list(ck.keys())
    print(f"  - 键名: {keys}")

    # 元信息字段（完整快照才有 epoch / version / best_err）
    epoch = ck.get("epoch")
    version = ck.get("version")
    best_err = ck.get("best_err")
    print(f"  - epoch    : {epoch if epoch is not None else '（未记录）'}")
    print(f"  - version  : {version if version is not None else '（未记录）'}")
    print(f"  - best_err : {best_err:.4f}°" if best_err is not None
          else "  - best_err : （未记录）")

    # 定位网络参数字典：完整快照存于 "ptnet"，特征权重存于 "net_features"
    state_key = None
    for cand in ("net_features", "ptnet", "model", "state_dict"):
        if isinstance(ck.get(cand), dict) and any(
                isinstance(v, torch.Tensor) for v in ck[cand].values()):
            state_key = cand
            break

    if state_key is None:
        print("  - 未找到网络参数字典（net_features / ptnet）。")
        return

    sd = ck[state_key]
    total, trainable = _count_params(sd)
    print(f"  - 参数来源 : {state_key}")
    print(f"  - 参数量   : {total:,}（可训练约 {trainable:,}）")

    if detail:
        for k, v in sd.items():
            print(f"      {k:<40} shape={tuple(v.shape)}  numel={v.numel()}")


def _collect_targets(args) -> list[str]:
    """按参数组装要查看的文件列表。

    --checkpoint 指定时只看该文件；否则扫描 <save> 下 pointnetlk_<version>[,_opt].pt。
    """
    if args.checkpoint:
        return [args.checkpoint]

    targets = []
    versions = (args.version,) if args.version else VERSIONS
    for v in versions:
        targets.append(os.path.join(args.save, f"pointnetlk_{v}.pt"))
        targets.append(os.path.join(args.save, f"pointnetlk_{v}_opt.pt"))
    return targets


def main() -> None:
    p = argparse.ArgumentParser(description="查看 / 对比 PointNetLK checkpoint 的模型参数信息")
    p.add_argument("--checkpoint", default=None,
                   help="指定单个 .pt 文件；不传则扫描 <save> 目录")
    p.add_argument("--save", type=str, default="./checkpoints",
                   help="checkpoint 所在目录（默认 ./checkpoints）")
    p.add_argument("--version", choices=VERSIONS, default=None,
                   help="只看指定版本的两个文件（默认两个版本都看）")
    p.add_argument("--detail", action="store_true",
                   help="额外打印每个权重张量的 shape")
    args = p.parse_args()

    if args.checkpoint:
        _print_checkpoint(os.path.abspath(args.checkpoint), args.detail)
        return

    if not os.path.isdir(args.save):
        raise SystemExit(f"未找到 checkpoint 目录: {args.save}")

    targets = _collect_targets(args)
    for t in targets:
        _print_checkpoint(os.path.abspath(t), args.detail)

    if not any(os.path.isfile(t) for t in targets):
        print(f"提示: 在目录 {args.save} 下未发现任何 pointnetlk_*.pt 文件，"
              f"请先运行 train_pointnetlk.py 训练。")


if __name__ == "__main__":
    main()