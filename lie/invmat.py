""" inverse matrix

本模块基于 torch 实现了「批量矩阵求逆」及其梯度计算，并封装成可被
PyTorch 自动求导框架识别的自定义反向传播函数 :class:`InvMatrix`。

主要包含：
- :func:`batch_inverse` ：批量矩阵求逆 x -> x^-1
- :func:`batch_inverse_dx`：批量矩阵求逆的导数 dy(j,k)/dx(m,n)
- :class:`InvMatrix`     ：PyTorch autograd.Function，供用户在模型中使用
"""

from __future__ import annotations

import torch
from torch import Tensor


def batch_inverse(x: Tensor) -> Tensor:
    """批量矩阵求逆 M(n) -> M(n); x -> x^-1

    Args:
        x: 输入张量，形状为 (batch_size, h, w)，其中 h == w（方阵）。
           实际调用 torch 内置的 ``.inverse()`` 逐样本求逆。

    Returns:
        y: 形状与 x 相同的张量 (batch_size, h, w)，即每个样本的逆矩阵。

    Note:
        对每个 batch 样本调用 ``x[i].inverse()``。若矩阵不可逆，
        ``torch.inverse`` 会抛出 ``linalg.LinAlgError``。
    """
    batch_size, h, w = x.size()
    assert h == w, "matrix must be square (h == w)"
    y = torch.zeros_like(x)
    for i in range(batch_size):
        y[i, :, :] = x[i, :, :].inverse()
    return y


def batch_inverse_dx(y: Tensor) -> Tensor:
    """计算批量矩阵求逆的导数（用于反向传播）。

    设 y(x) = x^-1，则导数计算如下：

        dy(j,k)/dx(m,n) = - y(j,m) * y(n,k)

    来源推导：
        d(y * x) = 0  =>  dy * x + y * dx = 0
        => dy = - y * dx * y
        => dy(j,k) = - sum_mn y(j,m) * dx(m,n) * y(n,k)
        => dy(j,k)/dx(m,n) = - y(j,m) * y(n,k)

    Args:
        y: 逆矩阵，形状 (batch_size, h, h)（方阵）。

    Returns:
        dy: 形状 (batch_size, h, h, h, h) 的雅可比张量，满足
            dy[b, j, k, m, n] = dy(j,k)/dx(m,n)。
    """
    batch_size, h, w = y.size()
    assert h == w, "matrix must be square (h == w)"

    # 计算 dy(j,k,m,n) = dy(j,k)/dx(m,n) = - y(j,m) * y(n,k)
    #   = - (y(j,:))' * y'(k,:)
    #
    # yl  行索引为 (j)，形状扩展为 [batch_size*h*h, h, 1]，取值 y(j,:)
    # yr  列索引为 (k)，形状扩展为 [batch_size*h*h, 1, h]，取值 y(k,:)
    # 二者做批量矩阵乘法得到 dy(j,k,m,n)，最后还原为 5 维张量。
    yl = y.repeat(1, 1, h).view(batch_size * h * h, h, 1)
    yr = y.transpose(1, 2).repeat(1, h, 1).view(batch_size * h * h, 1, h)
    dy = -yl.bmm(yr).view(batch_size, h, h, h, h)

    # 另一种等价的索引组织方式（dy(m,n,j,k) 的布局），已被上面的实现取代。
    # 二者数学等价，只是 5 维张量的索引顺序不同，保留供参考：
    # yl = y.transpose(1, 2).repeat(1, 1, h).view(batch_size*h*h, h, 1)
    # yr = y.repeat(1, h, 1).view(batch_size*h*h, 1, h)
    # dy = - yl.bmm(yr).view(batch_size, h, h, h, h)

    return dy


class InvMatrix(torch.autograd.Function):
    """自定义反向传播的矩阵求逆算子 M(n) -> M(n); x -> x^-1。

    在 pytorch 中通过 ``InvMatrix.apply(x)`` 调用，等价于 ``x.inverse()``，
    但梯度通过下方 :meth:`backward` 显式定义，可与自定义算子组合使用。
    """

    @staticmethod
    def forward(ctx: object, x: Tensor) -> Tensor:
        """前向传播：计算 x 的逆矩阵。

        Args:
            ctx: autograd 上下文，用于在 forward/backward 之间保存数据。
            x:  输入方阵，形状 (batch_size, h, h)。

        Returns:
            y: 逆矩阵，(batch_size, h, h)。
        """
        y = batch_inverse(x)
        ctx.save_for_backward(y)  # 保存 y 供 backward 使用
        return y

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> Tensor:
        """反向传播：计算输入 x 的梯度。

        设 y(x) = x^-1，对任意函数 f(y(x))，有
            df/dx(m,n) = df/dy(j,k) * dy(j,k)/dx(m,n)
        其中 df/dy = grad_output 由上层传入，dy(j,k)/dx(m,n) 来自
        :func:`batch_inverse_dx`，二者做批量矩阵乘后还原为 [m, n] 布局。

        Args:
            ctx: 前向保存的上下文。
            grad_output: 上游梯度，形状 (batch_size, h, h)。

        Returns:
            grad_input: 关于输入 x 的梯度，形状 (batch_size, h, h)。
                注意：由于 forward 只有 x 一个非 ctx 输入，因此返回单个张量。
        """
        y, = ctx.saved_tensors  # v0.4 之后使用 saved_tensors
        # y, = ctx.saved_variables  # v0.3.1 时代的旧写法
        batch_size, h, w = y.size()
        assert h == w, "matrix must be square (h == w)"

        # Let y(x) = x^-1 and assume any function f(y(x)).
        # compute df/dx(m,n)...
        #   df/dx(m,n) = df/dy(j,k) * dy(j,k)/dx(m,n)
        # well, df/dy is 'grad_output'
        # and so we will return 'grad_input = df/dy(j,k) * dy(j,k)/dx(m,n)'

        dy = batch_inverse_dx(y)                  # dy(j,k,m,n) = dy(j,k)/dx(m,n)
        go = grad_output.contiguous().view(batch_size, 1, h * h)  # [1, (j*k)]
        ym = dy.view(batch_size, h * h, h * h)    # [(j*k), (m*n)]
        r = go.bmm(ym)                            # [1, (m*n)]
        grad_input = r.view(batch_size, h, h)     # [m, n]

        return grad_input