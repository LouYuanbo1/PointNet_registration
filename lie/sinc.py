""" 
"sinc" 是 sine cardinal 的缩写，其中：
sine 来自它由正弦函数构成；
cardinal 来自拉丁语 cardo(铰链/枢轴），
在数学中 "cardinal function" 指一类在整数点上取值为 0 或 1 的插值函数。
sinc 函数正是信号处理中理想的插值/采样"枢轴"函数。
这个名称最早由 Philip Woodward 在 1952 年的信息论工作中提出。
sinc(t) := sin(t) / t 
"""
import torch
from torch import sin, cos
from torch import Tensor
from torch.autograd.function import FunctionCtx


"""
关于 Tensor 的数据结构（torch.Tensor）：

1. shape 与 ndim
   - ndim / dim(): 维度数。shape 是描述每一维尺寸的元组。
   - 标量(0维)          shape=()                          -> Tensor 元素个数
   - 向量(1维)          shape=(N,)         e.g. (3,)
   - 矩阵(2维)          shape=(N, M)       e.g. (3, 5)
   - 高维张量           shape=(B, C, N)    e.g. (32, 3, 1024)

2. dtype（数据类型）
   - 浮点: float16 / float32 / float64(bfloat16)
   - 其它: int8/16/32/64, bool, complex64/128 等
   - 运算会尽量保持 dtype，必要时自动提升。

3. device（设备）
   - 数据可位于 CPU 或 GPU(cuda)，跨设备运算会报错，需先 .to(device)。

4. 支持的索引（本文件中大量使用）
   - 布尔掩码: t[a < e] 选出满足条件的元素，得到一维子张量。
   - 就地写入: r[s] = ... 仅写入被掩码选中的位置。

5. 关键性质（与 sinc1 相关的运算）
   - 逐元素运算: sin, cos, abs, ** 均按元素操作，shape 不变。
   - torch.zeros_like(t): 生成与 t 相同 shape / dtype / device 的全 0 张量。

本文件的 sinc1(t) 要求 t 为 dtype 为浮点的 Tensor（任意 shape），
计算 sin(t)/t，在 t≈0 处用泰勒级数近似以避免除零。
"""
def sinc1(t: Tensor) -> Tensor:
    """ sinc1: t -> sin(t)/t """
    # 阈值 e：|t| 小于该值的点视为“接近 0”，改用泰勒展开以避免除零。
    e = 0.01
    # 申请与 t 形状/类型/设备完全相同的全 0 张量，作为输出容器。
    r = torch.zeros_like(t)
    # 对 t 逐元素取绝对值，用于判断哪些位置接近 0。
    a = torch.abs(t)

    # s（small）为布尔掩码：a < e 的位置为 True，代表接近 0 的点。
    # s bool tensor
    s = a < e
    # c（common）为另一布尔掩码：s 为 False 的位置（即普通点，s != 0 时取补集）。
    c = ~s
    # 仅对接近 0 的子集取平方，参与泰勒展开计算。
    t2 = t[s] ** 2
    # 用泰勒级数（O(t^8)）近似 sin(t)/t 在小 t 时的值，只写入 s 标记的位置。
    r[s] = 1 - t2/6*(1 - t2/20*(1 - t2/42))  # Taylor series O(t^8)
    # 对普通点直接用原始定义 sin(t)/t，只写入 c 标记的位置。
    r[c] = sin(t[c]) / t[c]

    return r

def sinc1_dt(t: Tensor) -> Tensor:
    """ d/dt(sinc1) """
    # 阈值 e：|t| 小于该值视为接近 0，用泰勒展开近似导数。
    e = 0.01
    # 申请全 0 输出容器，shape/dtype/device 与 t 一致。
    r = torch.zeros_like(t)
    # 逐元素取绝对值，用于划分“接近 0”和“普通点”。
    a = torch.abs(t)

    # s：接近 0 的点（布尔掩码）。
    s = a < e
    # c：普通点（s 的补集）。
    c = ~s
    # 对全部 t 统一求平方，供后续两种分支使用。
    t2 = t ** 2
    # d/dt(sin(t)/t) 在 t→0 的泰勒展开（O(t^8)），只写入 s 位置。
    r[s] = -t[s]/3*(1 - t2[s]/10*(1 - t2[s]/28*(1 - t2[s]/54)))  # Taylor series O(t^8)
    # 常规求导公式 (cos t)/t - (sin t)/(t^2)，只写入 c 位置。
    r[c] = cos(t[c])/t[c] - sin(t[c])/t2[c]

    return r

def sinc2(t: Tensor) -> Tensor:
    """ sinc2: t -> (1 - cos(t)) / (t**2) """
    # 阈值 e：|t| 小于该值视为接近 0，用泰勒展开近似。
    e = 0.01
    # 申请全 0 输出容器。
    r = torch.zeros_like(t)
    # 逐元素取绝对值，划分接近 0 的点与普通点。
    a = torch.abs(t)

    # s：接近 0 的点。
    s = a < e
    # c：普通点。
    c = ~s
    # 对全部 t 统一求平方，供分母与泰勒展开使用。
    t2 = t ** 2
    # (1-cos t)/t^2 在 t→0 的泰勒展开（O(t^8)），只写入 s 位置。
    r[s] = 1/2*(1-t2[s]/12*(1-t2[s]/30*(1-t2[s]/56)))  # Taylor series O(t^8)
    # 普通点：直接按定义 (1-cos t)/t^2 计算。
    r[c] = (1-cos(t[c]))/t2[c]

    return r

def sinc2_dt(t: Tensor) -> Tensor:
    """ d/dt(sinc2) """
    # 阈值 e：|t| 小于该值视为接近 0，用泰勒展开近似导数。
    e = 0.01
    # 申请全 0 输出容器。
    r = torch.zeros_like(t)
    # 逐元素取绝对值，划分接近 0 的点与普通点。
    a = torch.abs(t)

    # s：接近 0 的点。
    s = a < e
    # c：普通点。
    c = ~s
    # 对全部 t 统一求平方，供分母与泰勒展开使用。
    t2 = t ** 2
    # d/dt[(1-cos t)/t^2] 在 t→0 的泰勒展开（O(t^8)），只写入 s 位置。
    r[s] = -t[s]/12*(1 - t2[s]/5*(1.0/3 - t2[s]/56*(1.0/2 - t2[s]/135)))  # Taylor series O(t^8)
    # 普通点：对 (1-cos t)/t^2 求导，即 sin t/t^2 - 2(1-cos t)/(t^3)。
    r[c] = sin(t[c])/t2[c] - 2*(1-cos(t[c]))/(t2[c]*t[c])

    return r


def sinc3(t: Tensor) -> Tensor:
    """ sinc3: t -> (t - sin(t)) / (t**3) """
    # 阈值 e：|t| 小于该值视为接近 0，用泰勒展开近似。
    e = 0.01
    # 申请全 0 输出容器。
    r = torch.zeros_like(t)
    # 逐元素取绝对值，划分接近 0 的点与普通点。
    a = torch.abs(t)

    # s：接近 0 的点。
    s = a < e
    # c：普通点。
    c = ~s
    # 仅对接近 0 的子集求平方，用于泰勒展开。
    t2 = t[s] ** 2
    # (t - sin t)/t^3 在 t→0 的泰勒展开（O(t^8)），只写入 s 位置。
    r[s] = 1/6*(1-t2/20*(1-t2/42*(1-t2/72)))  # Taylor series O(t^8)
    # 普通点：直接按定义 (t - sin t)/t^3 计算。
    r[c] = (t[c]-sin(t[c]))/(t[c]**3)

    return r


class Sinc1_autograd(torch.autograd.Function):
    """ sinc1 的自定义 autograd 函数，实现前向计算与反向传播。"""
    @staticmethod
    def forward(ctx: FunctionCtx, theta: Tensor) -> Tensor:
        # 保存 theta 供 backward 阶段使用（无需梯度的中间量）。
        ctx.save_for_backward(theta)
        # 前向直接复用普通函数 sinc1。
        return sinc1(theta)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        # 取出 forward 保存的 theta。
        theta, = ctx.saved_tensors
        # 默认梯度为 None；若确实需要输入的梯度才计算。
        grad_theta = None
        # 仅当该输入需要梯度时计算，乘以链式法则中的上游梯度 grad_output。
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * sinc1_dt(theta).to(grad_output)
        return grad_theta

Sinc1 = Sinc1_autograd.apply

class Sinc2_autograd(torch.autograd.Function):
    """ sinc2 的自定义 autograd 函数，实现前向计算与反向传播。"""
    @staticmethod
    def forward(ctx: FunctionCtx, theta: Tensor) -> Tensor:
        # 保存 theta 供 backward 阶段使用。
        ctx.save_for_backward(theta)
        # 前向直接复用普通函数 sinc2。
        return sinc2(theta)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        # 取出 forward 保存的 theta。
        theta, = ctx.saved_tensors
        # 默认梯度为 None。
        grad_theta = None
        # 仅当该输入需要梯度时计算，乘以上游梯度 grad_output。
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * sinc2_dt(theta).to(grad_output)
        return grad_theta

Sinc2 = Sinc2_autograd.apply