# PointNet_registration

基于 [PointNet](https://arxiv.org/abs/1612.00593) 的点云配准（Point Cloud Registration）研究项目，
实现了 **PointNetLK** 这一基于李群/李代数、通过特征空间最小化实现配准的深度网络，
并在此基础上进行优化与拓展，同时提供了与 **ICP 迭代最近点算法** 的完整对比。

本项目以**点云配准本身**为训练目标（而非分类/分割），通过端到端训练学习「利于 LK 迭代配准」的全局特征，
支持：
- **PointNetLK（忠实复刻论文）** 与 **PointNetLKV2（向量化优化版）** 两种配准网络；
- **PointNetLK 粗配准 + PyTorch3D ICP 精配准** 的完整流水线；
- 多种配准策略的批量评测与可视化对比（旋转/平移误差、RMSE、Hausdorff / Chamfer 距离等）。

---

## 目录结构

```
PointNet_registration/
├── model/                    # 网络定义（仅负责网络，装配由 utils.builders 完成）
│   ├── pointnet.py           #   统一 PointNet 兼容层，自动适配 lk / yanx27 两种风格
│   ├── pointnet_lk_style.py  #   PointNet_features（PointNetLK 论文原实现风格）
│   ├── pointnet_yanx27_style.py  # PointNetEncoder（第三方 PointNet++ 精简版风格）
│   ├── pointnetlk.py         #   PointNetLK（忠实复刻论文实现）
│   ├── pointnetlk_v2.py      #   PointNetLK V2（向量化雅可比 + 逐样本收敛）
│   └── __init__.py
├── lie/                      # 李群 / 李代数算子（SE(3) / SO(3)）
│   ├── se3.py                #   SE(3) 指数映射 / 李代数（twist）
│   ├── so3.py                #   SO(3) 罗德里格斯旋转等
│   ├── invmat.py
│   └── sinc.py
├── registration/             # 配准编排（编排，不含算法复刻）
│   ├── pipeline.py           #   PointNetLK 粗配准 + PyTorch3D ICP 精配准流水线
│   └── __init__.py
├── utils/                    # 工具（数据 / 构建 / 变换）
│   ├── data.py               #   ModelNet40 点云数据加载与数据集封装
│   ├── builders.py           #   配准网络构建工厂（加载预训练权重、选择设备）
│   ├── transforms.py         #   随机刚体真值生成、点云变换等
│   └── __init__.py
├── scripts/                  # 可执行脚本
│   ├── check_env.py          #   环境检测（Python / CUDA / PyTorch / PyTorch3D）
│   ├── train_pointnetlk.py   #   端到端配准训练
│   ├── example_pipeline.py   #   端到端配准演示（PointNetLK + ICP 对比）
│   ├── benchmark_registration.py  # 批量评测（多种策略对比）
│   ├── view_checkpoint.py    #   查看 / 对比 checkpoint 信息
│   └── ...
├── data/                     # 数据集（仓库外维护，见 .gitignore）
│   └── modelnet40_normal_resampled/
├── checkpoints/              # 训练产物 .pt 权重（仓库外维护）
│   ├── pointnetlk_v2.pt
│   └── pointnetlk_v2_opt.pt
├── requirements.txt          # 跨平台运行时依赖
├── requirements-cu118-linux.txt  # 本机环境冻结备份（Linux + CUDA 11.8）
├── LICENSE
└── README.md
```

---

## 核心概念

### 配准网络：PointNetLK

[PointNetLK](https://arxiv.org/abs/1903.00311) 将点云配准建模为**点云全局特征空间的配准问题**：

1. 待求解的刚体变换被参数化为 SE(3) 的李代数 twist `(w₁,w₂,w₃,v₁,v₂,v₃)`（6 维）；
2. 用 PointNet 提取源点云与目标点云的全局特征 `f(p0)`、`f(p1)`；
3. 用**后向差分**近似特征关于该 6 维参数的一阶雅可比矩阵 `J`；
4. 通过伪逆求解线性化方程 `J·dx = -r`（`r` 为特征残差），迭代更新位姿直至收敛。

在本项目中实现了两个版本：

| 版本 | 文件 | 说明 |
| ---- | ---- | ---- |
| `PointNetLK` | `model/pointnetlk.py` | **论文原实现的忠实复刻**，保留 batch 级收敛门限、逐样本循环构造雅可比扰动等原始语义，便于复现论文实验。 |
| `PointNetLKV2` | `model/pointnetlk_v2.py` | **优化实现**：把雅可比扰动矩阵构造成负对角张量一次调用 `se3.Exp` 去掉逐样本循环（数学等价、批量更快）；引入逐样本收敛掩码 `active`，难样本不再拖慢全局，已收敛样本提前终止。 |

两种网络共用同一套李群/李代数算子（`lie/` 模块）。

### PointNet 特征网络兼容层

`model/pointnet.py` 提供统一的 `PointNet` 包装器，使任意一种 PointNet 风格都能被 PointNetLK 直接使用（自动统一输入转置、元组取值与训练/推理切换）：

- `pointnet_lk_style.PointNet_features` —— 输入 `[B, N, 3]`，输出 `[B, K]`（PointNetLK 论文约定）；
- `pointnet_yanx27_style.PointNetEncoder` —— 输入 `[B, D, N]`（通道在前），返回 `(feat, trans)`。

### 配准流水线

`registration/pipeline.py` 将算法全部下沉到 `model` 与 `lie`，仅做「编排」：

1. 用 `PointNetLK` 对 `src` 与 `tgt` 做粗配准，得到粗位姿；
2. 把粗位姿作为初值，交给 **PyTorch3D 的 ICP**（`pytorch3d.ops.iterative_closest_point`）精化；
3. 组合两次变换，返回最终刚体位姿。

与「仅 ICP（单位阵初值）」相比，粗配准能把接近/超过 90° 或远离初值的目标先拉进收敛盆，从而显著提升局部方法 ICP 的鲁棒性。

---

## 安装与环境

### 依赖

| 包 | 说明 |
| ---- | ---- |
| `numpy` | 基础数值计算 |
| `torch` | 深度学习框架 |
| `pytorch3d` | ICP 精配准（若仅做训练/评测，不含 ICP，可去掉） |

```bash
# 跨平台运行时依赖
pip install -r requirements.txt

# 或在本机（Linux + CUDA 11.8）还原完整环境
pip install -r requirements-cu118-linux.txt
```

> **注意**：`pytorch3d` 与 `torch` 版本强绑定，安装前请参考官方兼容矩阵，建议先运行 `python scripts/check_env.py` 检测本机 torch / CUDA 版本，再选择匹配的 `pytorch3d` 版本。

### 检查环境

```bash
python scripts/check_env.py
```

该脚本会打印 Python、CUDA、PyTorch、PyTorch3D 是否可用及设备信息；支持 GPU 时自动用 GPU，无 GPU 自动回退 CPU。

---

## 数据准备

本项目使用 **ModelNet40 点云数据集**，约定数据目录结构如下：

```
data/
└── modelnet40_normal_resampled/
    ├── modelnet40_train.txt
    ├── modelnet40_test.txt
    └── <类别>/
        └── <类别>_打开编号.txt   # 每行一个点：x y z
```

可通过以下方式获取：

```bash
# 下载并解压后放到项目 data/ 目录下
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
mkdir -p data
unzip modelnet40_normal_resampled.zip -d data/
```

数据集加载逻辑封装在 `utils/data.py`（`locate_modelnet40` / `ModelNet40PointCloud`），会自动做去中心、尺度归一与降采样。

---

## 快速开始

### 1. 训练 PointNetLK 特征网络

训练直接以**配准本身**为优化目标：读取 ModelNet40 点云作为模板 `p0`，随机生成真值刚体变换 `igt` 得到配对点云 `p1 = igt * p0`，损失为使网络估计位姿 `g` 满足 `comp(g, igt) = |g·igt - I| → 0`。

```bash
cd PointNet_registration

# 训练 lk 版本（30 轮）
python scripts/train_pointnetlk.py --version lk    --epochs 30

# 训练 v2 优化版，并断点续训
python scripts/train_pointnetlk.py --version v2    --epochs 30 --resume

# 用预训练特征权重初始化（微调时学习率会自动降为 1e-4）
python scripts/train_pointnetlk.py --version lk --weights ./checkpoints/checkpoint_lk.pt
```

常用参数（均可用 `--args` 覆盖）：

| 参数 | 默认值 | 说明 |
| ---- | ---- | ---- |
| `--version` | `lk` | 模型版本：`lk` 或 `v2` |
| `--epochs` | `30` | 训练轮数 |
| `--batch` | `8` | 批大小 |
| `--num_points` | `1024` | 每个点云采样点数 |
| `--lr` | `1e-3` | 学习率（指定 `--weights` 时自动降为 `1e-4`） |
| `--maxiter` | `10` | LK 迭代上限 |
| `--learn_delta` | 关 | 是否将差分步长作为可学习参数 |
| `--twist_mag` | `1.0` | 官方式 SE(3) 随机扰动幅度（弧度，对齐原论文 `RandomTransformSE3`），越大配准越难 |
| `--weights` | 无 | 预训练特征权重路径（可选） |
| `--root` | `./data` | ModelNet40 数据目录 |
| `--save` | `./checkpoints` | 权重保存目录 |
| `--resume` | 关 | 从 `--save` 下的 `_opt.pt` 断点续训 |
| `--seed` | `42` | 随机种子 |

**产物**：

- `<save>/pointnetlk_<version>.pt`：特征权重（仅 `net_features` 单字段，用于评测）；
- `<save>/pointnetlk_<version>_opt.pt`：完整快照（含优化器/调度器/轮次，仅用于 `--resume` 续训）。

### 2. 端到端配准演示

```bash
python scripts/example_pipeline.py [--model lk|yanx27] [--weights path.pt]
```

对配准场景，对比三种策略：

- **A. 仅粗配准（PointNetLK 特征）**
- **B. 完整流水线（PointNetLK 粗配准 + PyTorch3D ICP 精配准）**
- **C. 仅 PyTorch3D ICP（单位阵初值，跳过粗配准）**

### 3. 批量评测

```bash
# 未训练：未提供权重时仍可运行（PointNetLK 用随机初始化，仅体现流程图）
python scripts/benchmark_registration.py --batch 8 --num-samples 64

# 加载训练权重（训练后推荐）
python scripts/benchmark_registration.py --weights checkpoints/pointnetlk_v2.pt --batch 8 --num-samples 64
```

对同一批点云 `(B, N, 3)`，逐策略打印：耗时、旋转误差（度）、平移误差、最终最近邻 RMSE、是否收敛，以及配准后点云 vs 目标的 **Hausdorff 距离**与 **Chamfer 距离**。

常用参数：

| 参数 | 默认值 | 说明 |
| ---- | ---- | ---- |
| `--weights` | 无 | 训练好的 PointNetLK 特征权重 `.pt`（默认随机初始化） |
| `--version` | `v2` | 模型版本：`lk` 或 `v2` |
| `--root` | `./data` | 数据根目录 |
| `--split` | `test` | 评测数据划分：`train` / `test` |
| `--batch` | `8` | 批大小（控制显存占用） |
| `--num-samples` | `64` | 参与评测的点云样本总数 |
| `--num-points` | `1024` | 每个点云采样点数 N |
| `--mag` | `1.0` | 官方式 SE(3) 扰动幅度（弧度），越大配准越难 |
| `--max-coarse-iter` | `10` | PointNetLK 粗配准迭代上限 |
| `--max-icp-iter` | `50` | ICP 精配准迭代上限 |
| `--relative-rmse-thr` | `5e-3` | ICP 相对 RMSE 收敛阈值 |

### 4. 查看 checkpoint

```bash
python scripts/view_checkpoint.py                           # 查看 checkpoints/ 下全部文件
python scripts/view_checkpoint.py --version v2              # 只看 v2 的两个文件
python scripts/view_checkpoint.py --detail                  # 并打印每个权重张量 shape
python scripts/view_checkpoint.py --checkpoint checkpoints/pointnetlk_v2.pt
```

---


## 参考文献

- Charles R. Qi, Hao Su, Kaichun Mo and Leonidas J. Guibas. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*. **CVPR 2017**.
- Yuki Aoki†, Hunter Goforth†, Rangaprasad Arun Srivatsan and Simon Lucey. *PointNetLK: Robust & Efficient Point Cloud Registration using PointNet*. **CVPR 2019**.
- Yue Wang, Justin Solomon. *PRNet: Self-Supervised Learning for Partial-to-Partial Registration*.
- PyTorch3D `iterative_closest_point` 文档：https://pytorch3d.org/

---

## 致谢与引用

本项目在实现过程中参考并借鉴了以下开源项目（均为 **MIT 协议**）：

- **PointNetLK**（[github.com/hmgoforth/PointNetLK](https://github.com/hmgoforth/PointNetLK)）—— PointNetLK 点云配准网络的论文官方实现，本项目的 `PointNetLK` / `PointNetLKV2` 及李群李代数算子均以其为参考基准。
- **Pointnet_Pointnet2_pytorch**（[github.com/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)）—— 第三方 PointNet / PointNet++ 的 PyTorch 实现，本项目的 `pointnet_yanx27_style.PointNetEncoder` 以其 `pointnet_utils.py` 为蓝本做了配准场景精简。

（两者均以 MIT 协议发布，本项目的参考与改写遵循其许可条款。）

---

## 许可

本项目基于 **MIT License** 发布，详见 [LICENSE](LICENSE)。