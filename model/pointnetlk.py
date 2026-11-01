"""
点云配准网络：PointNetLK（本文实现，与论文原实现一致）。

对应论文：
    Aoki, Yuki, Hunter Goforth, Rangaprasad Arun Srivatsan, and Simon Lucey.
    "PointNetLK: Robust & Efficient Point Cloud Registration using PointNet."
    CVPR 2019.

本文件是**论文原实现的忠实复刻**：保留原始算法的行为语义（batch 级收敛门限、
逐样本循环构造雅可比扰动等），便于复现论文实验或与优化版 ——
见 :mod:`model.pointnetlk_v2` —— 做对比。

使用「后向差分」近似雅可比矩阵的迭代最近点（ICP）式点云配准网络。

算法要点：
- 将待求解的刚体变换参数化为 SE(3) 的李代数 twist (w1,w2,w3,v1,v2,v3)；
- 用后向差分近似特征关于该 6 维参数的一阶雅可比 J；
- 通过伪逆求解线性化方程 J * dx = -r（r 为残差）；迭代更新位姿直至收敛。

与优化版（PointNetLKV2）的主要差异见 :mod:`model.pointnetlk_v2` 顶部说明。
"""

import torch
from lie import invmat, se3


class PointNetLK(torch.nn.Module):
    """基于李群李代数的点云配准网络（近似雅可比版 PointNetLK）。

    输入两个任意姿态的点云 p0 / p1，通过迭代估计相对刚体变换 g，
    使特征空间残差 r = f(g * p1) - f(p0) 尽可能小。
    """

    def __init__(
        self,
        ptnet: torch.nn.Module,
        delta: float = 1.0e-2,
        learn_delta: bool = False,
    ) -> None:
        """构造 PointNetLK。

        Args:
            ptnet: PointNet 特征提取网络（见 pointnet 模块），接受
                [B, N, 3] 点云并返回 [B, K] 全局特征。
            delta: 数值差分步长，用于近似雅可比。
            learn_delta: 是否将差分步长作为可学习参数（默认不学习）。
        """
        super().__init__()
        self.ptnet = ptnet  # [B, N, 3] -> [B, K]

        # SE(3) 所需的算子
        self.inverse = invmat.InvMatrix.apply                       # H -> H^-1
        self.exp = se3.Exp                                          # [B, 6] -> [B, 4, 4]
        self.transform = se3.apply_rigid_transform_to_points                  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

        # 差分步长：6 维 twist 向量 (w1,w2,w3,v1,v2,v3)，可视 `learn_delta` 选择是否可学
        w1 = delta
        w2 = delta
        w3 = delta
        v1 = delta
        v2 = delta
        v3 = delta
        twist = torch.Tensor([w1, w2, w3, v1, v2, v3])
        self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=learn_delta)

        # 运行时状态 / 调试信息
        self.last_err: object | None = None   # 最近一次异常（如矩阵奇异）
        self.g_series: torch.Tensor | None = None  # 每次迭代的位姿序列（调试用）
        self.prev_r: torch.Tensor | None = None    # 上一次残差
        self.g: torch.Tensor | None = None         # 最终估计的位姿 [B, 4, 4]
        self.itr: int = 0                          # 实际迭代轮数

    @staticmethod
    def comp(g: torch.Tensor, igt: torch.Tensor) -> torch.Tensor:
        """配准误差度量：|g * igt - I|（应趋于 0）。

        Args:
            g:   估计的变换 [B, 4, 4]。
            igt: 真值变换 [B, 4, 4]。

        Returns:
            归一化均方误差 * 16（标量）。
        """
        """ |g*igt - I| (should be 0) """
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)                       # [B, 4, 4]
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        # 弃用写法（size_average 参数已废弃）：return torch.nn.functional.mse_loss(A, I, size_average=True) * 16
        return torch.nn.functional.mse_loss(A, I, reduction="mean") * 16

    def forward(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor,
        maxiter: int = 10,
        xtol: float = 1.0e-7,
    ) -> torch.Tensor:
        """执行配准，返回最终残差，并把结果存入 self.g / self.itr。

        Args:
            p0: 模板点云 [B, N, 3]。
            p1: 待配准点云 [B, N, 3]。
            maxiter: 最大迭代次数。
            xtol:   收敛阈值。

        Returns:
            r: 最终残差 [B, K]。
        """
        g0 = torch.eye(4).to(p0).view(1, 4, 4).expand(p0.size(0), 4, 4).contiguous()  # 初始为单位变换
        r, g, itr = self.iclk(g0, p0, p1, maxiter, xtol)

        self.g = g
        self.itr = itr
        return r

    def update(self, g: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """用李代数增量 dx 左乘更新位姿。

        等价于在 SE(3) 上做一次李群左扰动：g <- Exp(dx) * g。

        Args:
            g:  当前位姿 [B, 4, 4]。
            dx: 李代数增量 [B, 6]。

        Returns:
            更新后的位姿 [B, 4, 4]。
        """
        dg = self.exp(dx)        # [B, 6] -> [B, 4, 4]
        return dg.matmul(g)      # Exp(dx) * g

    def approx_Jic(self, p0: torch.Tensor, f0: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """用后向差分近似雅可比矩阵 J。

        思路：对每个 6 维 twist 分量 k，给 p0 施加一个 -dt[k] 的小扰动并重新提取特征，
        通过 Jk = (f0 - f(p(-dt[k], p0))) / dt[k] 近似特征对参数的导数。

        Args:
            p0: 模板点云 [B, N, 3]。
            f0: p0 的特征 [B, K]。
            dt: 差分步长 [B, 6]。

        Returns:
            J: 雅可比 [B, K, 6]，J[b, k, m] = d f_k / d twist_m。
        """
        batch_size = p0.size(0)
        num_points = p0.size(1)

        # 构造 6 个微小扰动对应的 SE(3) 变换：[B, 6, 4, 4]
        transf = torch.zeros(batch_size, 6, 4, 4).to(p0)
        for b in range(p0.size(0)):
            d = torch.diag(dt[b, :])          # [6, 6] 对角步长矩阵
            D = self.exp(-d)                  # [6, 4, 4] 每个扰动的变换
            transf[b, :, :, :] = D[:, :, :]
        transf = transf.unsqueeze(2).contiguous()          # [B, 6, 1, 4, 4]
        p = self.transform(transf, p0.unsqueeze(1))        # [B, 6, N, 3]：6 个扰动后的点云

        # f0 = self.ptnet(p0).unsqueeze(-1)  # [B, K, 1]（可改用这段）
        f0 = f0.unsqueeze(-1)                                # [B, K, 1]
        f = self.ptnet(p.view(-1, num_points, 3)).view(batch_size, 6, -1).transpose(1, 2)  # [B, K, 6]

        df = f0 - f                 # [B, K, 6] 后向差分的分子
        J = df / dt.unsqueeze(1)    # 除以步长得到近似雅可比

        return J

    def iclk(
        self,
        g0: torch.Tensor,
        p0: torch.Tensor,
        p1: torch.Tensor,
        maxiter: int,
        xtol: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """点云迭代最近点（ICLK）配准核心流程。

        步骤：
        1. 训练态先跑一次前向，让 BatchNorm 统计量更新；
        2. 之后固定网络（eval），重算 f0；
        3. 用后向差分求雅可比 J，求其伪逆 pinv；
        4. 迭代：用位姿 g 变换 p1 -> 提取特征 -> 残差 r -> 求增量 dx = -pinv * r
           -> 若 |dx| < xtol 收敛 -> 否则 update(g, dx) 继续。

        Args:
            g0:     初始位姿 [B, 4, 4]（通常为单位阵）。
            p0:     模板点云 [B, N, 3]。
            p1:     待配准点云 [B, N, 3]。
            maxiter: 最大迭代次数。
            xtol:   收敛阈值。

        Returns:
            (r, g, itr)：
                r:   最终残差 [B, K]；
                g:   估计位姿 [B, 4, 4]；
                itr: 实际迭代次数（矩阵奇异时返回 -1）。
        """
        training = self.ptnet.training
        batch_size = p0.size(0)

        g = g0
        self.g_series = torch.zeros(maxiter + 1, *g0.size(), dtype=g0.dtype)  # 记录每步位姿
        self.g_series[0] = g0.clone()

        if training:
            # 训练阶段先跑前向以更新 BatchNorm 的统计量，随后再置为 eval 固定之
            f0 = self.ptnet(p0)
            f1 = self.ptnet(p1)
        self.ptnet.eval()  # 固定网络（含 BatchNorm）

        # 用当前网络（已固定）重算模板特征
        f0 = self.ptnet(p0)   # [B, N, 3] -> [B, K]

        # 用后向差分近似雅可比 J
        dt = self.dt.to(p0).expand(batch_size, 6)
        J = self.approx_Jic(p0, f0, dt)

        self.last_err = None
        itr = -1
        # 计算 J 的伪逆：pinv = (J'J)^-1 J'，用于解 J*x = -r
        try:
            Jt = J.transpose(1, 2)        # [B, K, 6] -> [B, 6, K]
            H = Jt.bmm(J)                 # [B, 6, 6]
            B = self.inverse(H)           # (J'J)^-1
            pinv = B.bmm(Jt)              # [B, 6, K]
        except RuntimeError as err:
            # 出现奇异（如步长过小、H 不可逆）时的兜底：直接用特征差作为输出、返回 itr=-1
            self.last_err = err
            # print(err)
            # 或许可使用 Moore-Penrose 伪逆，但这里先采用简单处理；
            # 极可能是 self.dt 取值过小所致。
            f1 = self.ptnet(p1)           # [B, N, 3] -> [B, K]
            r = f1 - f0
            self.ptnet.train(training)
            return r, g, itr

        itr = 0
        r = None
        for itr in range(maxiter):
            self.prev_r = r
            p = self.transform(g.unsqueeze(1), p1)   # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            f = self.ptnet(p)                        # [B, N, 3] -> [B, K]
            r = f - f0                               # 残差

            dx = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)  # 求解增量 J*x = -r

            # DEBUG. 输出每次迭代残差范数
            # norm_r = r.norm(p=2, dim=1)
            # print('itr,{},|r|,{}'.format(itr+1, ','.join(map(str, norm_r.data.tolist()))))

            check = dx.norm(p=2, dim=1, keepdim=True).max()   # 本次增量范数
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0   # 一次迭代就收敛，说明无更新
                break

            g = self.update(g, dx)
            self.g_series[itr + 1] = g.clone()

        # 将未继续迭代的剩余步骤位姿补平为最终值，保证 g_series 长度固定
        rep = len(range(itr, maxiter))
        self.g_series[(itr + 1):] = g.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

        self.ptnet.train(training)   # 恢复原训练状态
        return r, g, (itr + 1)