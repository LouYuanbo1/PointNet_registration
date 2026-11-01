"""端到端点云配准训练：直接基于 PointNetLK / PointNetLKV2 训练特征网络。

这里的优化目标就是配准本身：

- 数据：读取本地 ModelNet40 点云作为模板 p0；对每个样本随机生成一个真值
  刚体变换 igt（随机旋转 + 平移），得到配对点云 p1 = igt * p0。
- 模型：PointNetLK(class) 或其变体 PointNetLKV2，内部含 PointNet 特征网络。
- 损失：配准误差 comp(g, igt) = |g * igt - I|，其中 g 为网络估计位姿。
  梯度经 g 回传到特征网络，端到端学习「利于 LK 迭代配准」的特征。

用法:
    python train_pointnetlk.py --version lk    --epochs 30
    python train_pointnetlk.py --version v2    --epochs 30 --resume
    python train_pointnetlk.py --version lk    --weights ./checkpoints/checkpoint_lk.pt  # 用预训练特征初始化

可调参数（均可通过命令行 --args 覆盖）:
    --version        模型版本，lk 或 v2（默认 lk）       --epochs   训练轮数（默认 30）
    --batch          批大小（默认 8）                    --num_points 每点云采样点数（默认 1024）
    --num_workers    数据加载线程数（默认取 CPU 核数）    --lr       学习率（默认 1e-3；若给 --weights 会自动降为 1e-4 微调）
    --maxiter        LK 迭代上限（默认 10）               --learn_delta 是否将差分步长作为可学习参数（默认关）
    --twist_mag      官方式 SE(3) 随机扰动幅度(弧度)（默认 1.0，对齐原论文 RandomTransformSE3）
    --weights        预训练特征权重路径（可选，默认无）    --root   ModelNet40 数据目录（默认 ./data）
    --save           权重保存目录（默认 ./checkpoints）
    --resume         从 <save>/pointnetlk_<version>_opt.pt 断点续训（默认关）
    --seed           随机种子（默认 42）

    说明:
    - 真值扰动采用官方式 twist 一体采样（任意方向旋转 + 平移，幅度按样本随机放大），
      训练与验证共用同一幅度，twist_mag 越大配准越难。
    - --weights 只用于预训练初始化；续训请用 --resume（两者可叠加）。

产物:
    <save>/pointnetlk_<version>.pt    特征权重（仅 net_features 单字段，供 example_pipeline.run_suite 评测）
    <save>/pointnetlk_<version>_opt.pt 完整快照（含优化器/调度器/轮次，仅用于 --resume 续训）
"""

import argparse
import os
import random
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data import ModelNet40PointCloud, locate_modelnet40
from utils.builders import build_lk, get_default_device
from utils.transforms import apply_rt, random_igt_twist


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def rotation_error(R_gt, R_est):
    M = R_est @ R_gt.transpose(-1, -2)
    cos = torch.clamp((torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2, -1, 1)
    return torch.rad2deg(torch.acos(cos))


@torch.no_grad()
def evaluate(model, loader, device, maxiter, mag=1.0):
    model.eval()
    rot_errs = []
    trans_errs = []
    for p0 in loader:
        p0 = p0.to(device)
        B = p0.size(0)
        igt = random_igt_twist(B, mag=mag, mag_randomly=True, device=device)
        p1 = apply_rt(p0, igt)
        model(p0, p1, maxiter=maxiter, xtol=1.0e-7)   # 返回残差
        g = model.g
        # 方向约定:
        #   数据侧  igt 是「p0 -> p1」的正变换（p1 = igt * p0）;
        #   网络侧  g  在 iclk 中把 p1 变换回 p0，因此 g ≈ igt⁻¹。
        # 所以评测真值必须取 igt 的逆，否则即便 g 完美拟合，旋转误差也会被
        # 内部一个「正变换与逆变换之差」的恒定偏置抬高，永远无法降到接近 0。
        igt_inv = igt.inverse()
        R_err = rotation_error(igt_inv[:, 0:3, 0:3], g[:, 0:3, 0:3])
        T_err = (igt_inv[:, 0:3, 3] - g[:, 0:3, 3]).norm(dim=1)
        rot_errs.append(R_err)
        trans_errs.append(T_err)
    model.train()
    rot = torch.cat(rot_errs).mean().item()
    trans = torch.cat(trans_errs).mean().item()
    return rot, trans


def main():
    p = argparse.ArgumentParser(description="PointNetLK / PointNetLKV2 端到端配准训练")
    p.add_argument("--version", choices=["lk", "v2"], default="lk")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=os.cpu_count())
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--maxiter", type=int, default=10, help="LK 迭代上限")
    p.add_argument("--twist_mag", type=float, default=1.0,
                   help="官方式 SE(3) twist 幅度(弧度，默认1.0，对齐原论文)")
    p.add_argument("--learn_delta", action="store_true", help="将差分步长作为可学习参数")
    p.add_argument("--weights", default=None, help="预训练特征权重(可选)")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--save", type=str, default="./checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_default_device()
    print(f"设备: {device}    配准模型: {args.version}")

    data_dir = locate_modelnet40(args.root)
    train_ds = ModelNet40PointCloud(data_dir, split="train",
                                    num_points=args.num_points, aug=True, seed=args.seed)
    val_ds = ModelNet40PointCloud(data_dir, split="test",
                                  num_points=args.num_points, aug=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers)
    print(f"训练样本: {len(train_ds)}   测试样本: {len(val_ds)}")

    model = build_lk(args.version, args.weights, device, learn_delta=args.learn_delta)
    # 若已给出预训练权重，则用较小的学习率在配准目标上微调
    lr = 1e-4 if args.weights else args.lr
    optimizer = torch.optim.Adam(model.ptnet.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs(args.save, exist_ok=True)
    ckpt_path = os.path.join(args.save, f"pointnetlk_{args.version}_opt.pt")
    feat_path = os.path.join(args.save, f"pointnetlk_{args.version}.pt")

    start_epoch = 1
    best_err = float("inf")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.ptnet.load_state_dict(ckpt["ptnet"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_err = ckpt["best_err"]
        print(f"续训: 从 epoch {start_epoch} 继续, best_rot_err={best_err:.3f}°")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for p0 in train_loader:
            p0 = p0.to(device)
            B = p0.size(0)
            igt = random_igt_twist(B, mag=args.twist_mag, mag_randomly=True, device=device)
            p1 = apply_rt(p0, igt)

            # 前向配准：返回残差 r，估计位姿存入 model.g
            model(p0, p1, maxiter=args.maxiter, xtol=1.0e-7)
            g = model.g
            loss = model.comp(g, igt)   # |g * igt - I| -> 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * B
            count += B
        scheduler.step()

        rot, trans = evaluate(model, val_loader, device, args.maxiter, mag=args.twist_mag)
        print(f"[{args.version}] epoch {epoch:03d}/{args.epochs} "
              f"loss={running / max(count, 1):.4f} "
              f"val_rot_err={rot:.3f}°  val_trans_err={trans:.4f}")

        if rot < best_err:
            best_err = rot
            torch.save({"net_features": model.ptnet.state_dict()}, feat_path)

        torch.save({
            "ptnet": model.ptnet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "version": args.version,
            "best_err": best_err,
        }, ckpt_path)

    print(f"[{args.version}] 完成. 最优验证旋转误差 = {best_err:.3f}°")
    print(f"特征权重已保存: {feat_path}")


if __name__ == "__main__":
    main()