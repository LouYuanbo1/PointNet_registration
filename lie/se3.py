""" 3-d rigid body transfomation group and corresponding Lie algebra. """
import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx
from .sinc import sinc1, sinc2, sinc3
from . import so3


def se3_hat_map(x: Tensor) -> Tensor:
    """6 维 twist -> se(3) 的李代数矩阵（hat 算子）。

    输入 shape [*, 6]，输出 shape [*, 4, 4]：
    X = [[ 0, -w3,  w2, v1],
         [ w3,  0, -w1, v2],
         [-w2,  w1,  0, v3],
         [ 0,   0,   0,  0]]
    """
    # 展平为 [N, 6]，拆出三个角速度与三个线速度分量。
    x_ = x.view(-1, 6)
    w1, w2, w3 = x_[:, 0], x_[:, 1], x_[:, 2]
    v1, v2, v3 = x_[:, 3], x_[:, 4], x_[:, 5]
    O = torch.zeros_like(w1)

    # 按行堆叠成 [N, 4, 4] 的 se(3) 矩阵。
    X = torch.stack((
        torch.stack((  O, -w3,  w2, v1), dim=1),
        torch.stack(( w3,   O, -w1, v2), dim=1),
        torch.stack((-w2,  w1,   O, v3), dim=1),
        torch.stack((  O,   O,   O,  O), dim=1)), dim=1)
    # 恢复原始批维度形状。
    return X.view(*(x.size()[0:-1]), 4, 4)

def se3_basis_vectors() -> Tensor:
    """se(3) 的一组基向量（6 维单位矩阵的行）。"""
    return torch.eye(6)

def se3_generator_matrices() -> Tensor:
    """se(3) 的一组李代数生成元（矩阵形式）：对基向量取 hat 算子。"""
    return se3_hat_map(se3_basis_vectors())

def se3_exponential_map(x: Tensor) -> Tensor:
    """se(3) -> SE(3) 的指数映射。

    输入 twist x（shape [*, 6]，含角速度 w 与线速度 v），
    输出 4x4 齐次变换矩阵 g = [[R, p], [0, 1]]（shape [*, 4, 4]）。
    """
    # 展平为 [N, 6]，拆出角速度 w 与线速度 v。
    x_ = x.view(-1, 6)
    w, v = x_[:, 0:3], x_[:, 3:6]
    # 转角 t = ||w||，并构造 X = so3.so3_hat_map(w)、S = X²。
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    X = so3.so3_hat_map(w)
    S = X.bmm(X)
    I = torch.eye(3).to(w)

    # 旋转部分：R = I + sinc1(t)·X + sinc2(t)·X²（Rodrigues 公式）。
    R = I + sinc1(t)*X + sinc2(t)*S

    # 平动部分：V = I + sinc2(t)·X + sinc3(t)·X²，p = V·v。
    V = I + sinc2(t)*X + sinc3(t)*S

    p = V.bmm(v.contiguous().view(-1, 3, 1))

    # 最底行 [0,0,0,1]，用于构造齐次矩阵。
    z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(x_.size(0), 1, 1).to(x)
    # 拼接 [R | p]，再加最底行。
    Rp = torch.cat((R, p), dim=2)
    g = torch.cat((Rp, z), dim=1)

    # 恢复原始批维度形状。
    return g.view(*(x.size()[0:-1]), 4, 4)

def apply_rigid_transform_to_points(g: Tensor, a: Tensor) -> Tensor:
    """SE(3) 变换 g 作用到点 a（先旋转后平移）。

    g 为 [*, 4, 4]，a 为 [*, 3] 或 [*, 3, N]。
    结果为 R·a + p。
    """
    # 展平并拆出旋转 R 与平移 p，恢复原始批维度。
    g_ = g.view(-1, 4, 4)
    R = g_[:, 0:3, 0:3].contiguous().view(*(g.size()[0:-2]), 3, 3)
    p = g_[:, 0:3, 3].contiguous().view(*(g.size()[0:-2]), 3)
    # 维度一致则直接旋转加平移；否则对 a 补一维再压缩。
    if len(g.size()) == len(a.size()):
        b = R.matmul(a) + p.unsqueeze(-1)
    else:
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p
    return b


class ExpMap(torch.autograd.Function):
    """Exp: se(3) -> SE(3) 的自定义 autograd 函数。"""
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        """前向：R⁶ -> M(4)。

        size: [B, 6] -> [B, 4, 4]，或 [B, 1, 6] -> [B, 1, 4, 4]。
        """
        # 保存输入供反向使用。
        ctx.save_for_backward(x)
        g = se3_exponential_map(x)
        return g

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        # 取回前向保存的 x，并重算 g = exp(se3)(x)。
        x, = ctx.saved_tensors
        g = se3_exponential_map(x)
        # 六个生成元的矩阵形式，用于构造雅可比。
        gen_k = se3_generator_matrices().to(x)

        # 令 z = f(g) = f(exp(x))，则
        # dz = Σ df/dg_ij · dg_ij/dx_k · dx_k
        #    = Σ df/dg_ij · [gen_k · g]_ij · dx_k
        # 即 dg/dx_k = gen_k · g。
        dg = gen_k.matmul(g.view(-1, 1, 4, 4))
        # 形状为 (k, i, j)。
        dg = dg.to(grad_output)

        # 上游梯度与雅可比逐元素相乘，再对 (i,j) 求和，得到对 x 的梯度。
        go = grad_output.contiguous().view(-1, 1, 4, 4)
        dd = go * dg
        grad_input = dd.sum(-1).sum(-1)

        return grad_input

Exp = ExpMap.apply