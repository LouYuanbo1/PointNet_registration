""" 3-d rotation group and corresponding Lie algebra """
import torch
from torch import Tensor
from . import sinc


def so3_hat_map(x: Tensor) -> Tensor:
    """轴向量 x -> so(3) 反对称矩阵（hat 算子，记作 x^ 或 [x]×）。

    输入 shape [*, 3]（* 为任意批维度），
    输出 shape [*, 3, 3]。
    X = [x]× = [[ 0, -x3,  x2],
                [ x3,  0, -x1],
                [-x2,  x1,  0]]
    """
    # 展平为 [N, 3]，取出三个分量。
    x_ = x.view(-1, 3)
    x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
    # 全 0 张量，用于填充对角。
    O = torch.zeros_like(x1)

    # 按行序堆叠成 [N, 3, 3] 的反对称矩阵。
    X = torch.stack((
        torch.stack((O, -x3, x2), dim=1),
        torch.stack((x3, O, -x1), dim=1),
        torch.stack((-x2, x1, O), dim=1)), dim=1)
    # 恢复原始批维度形状 [*, 3, 3]。
    return X.view(*(x.size()[0:-1]), 3, 3)

def so3_rodrigues_rotation(x: Tensor) -> Tensor:
    """Rodrigues 旋转（指数映射的带自定义 autograd 梯度版本）。

    输入轴角向量 w，输出旋转矩阵 R，shape [*, 3] -> [*, 3, 3]。
    R = exp([w]×) = I + sinc1(t)·[w]× + sinc2(t)·[w]×²
    其中 t = ||w||。
    """
    # 展平为 [N, 3]，求每个向量的 2-范数作为转角 t，形状 [N,1,1]。
    w = x.view(-1, 3)
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    # X = [w]×，S = X²。
    X = so3_hat_map(w)
    S = X.bmm(X)
    # 单位阵，广播到与 w 相同设备/类型。
    I = torch.eye(3).to(w)

    # 用 sinc 的 autograd 版本构成旋转。
    R = I + sinc.Sinc1(t)*X + sinc.Sinc2(t)*S

    # 恢复原始批维度形状 [*, 3, 3]。
    return R.view(*(x.size()[0:-1]), 3, 3)