"""环境检测脚本：测试 CUDA、PyTorch、PyTorch3D是否可用。

用法:
    python check_env.py
"""

import platform
import sys


def print_header(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_python() -> None:
    print_header("Python")
    print(f"Python 版本     : {sys.version}")
    print(f"系统平台       : {platform.platform()}")
    print()


def check_cuda() -> None:
    print_header("CUDA")
    try:
        import torch

        print(f"CUDA 可用      : {torch.cuda.is_available()}")
        print(f"CUDA 版本      : {torch.version.cuda}")
        print(f"GPU 数量       : {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  - 设备 {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                total_mem_gb = props.total_memory / (1024 ** 3)
                print(f"      显存(约)  : {total_mem_gb:.2f} GB")
                print(f"      算力      : {props.major}.{props.minor}")
        else:
            print("当前没有可用的 CUDA 设备（可能为 CPU-only 安装或因驱动问题不可用）。")
    except ImportError:
        print("未导入 torch，无法检测 CUDA。")
    print()


def check_pytorch() -> None:
    print_header("PyTorch")
    try:
        import torch

        print(f"PyTorch 版本   : {torch.__version__}")
        print(f"安装路径       : {torch.__file__}")
        cuda_built = torch.backends.cuda.is_built()
        print(f"CUDA 构建     : {'是' if cuda_built else '否'}")
        cudnn_ver = torch.backends.cudnn.version()
        print(f"cuDNN 版本    : {cudnn_ver if cudnn_ver else '未检测到'}")

        # 用 GPU 做一次简单张量运算验证（若可用）
        if torch.cuda.is_available():
            device = torch.device("cuda")
            a = torch.randn(1024, 1024, device=device)
            b = torch.randn(1024, 1024, device=device)
            c = a @ b
            torch.cuda.synchronize()
            print(f"GPU 矩阵乘法验证: OK, 结果 shape = {tuple(c.shape)}, "
                  f"sum = {c.sum().item():.4f}")
        else:
            a = torch.randn(1024, 1024)
            c = a @ a
            print(f"CPU 矩阵乘法验证: OK, 结果 shape = {tuple(c.shape)}")

        # 验证自动求导是否正常
        x = torch.randn(3, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        print(f"自动求导验证   : OK, grad = {x.grad.tolist()}")
    except ImportError:
        print("PyTorch 未安装！请先安装：pip install torch")
    print()


def check_pytorch3d() -> None:
    print_header("PyTorch3D")
    try:
        import pytorch3d

        print(f"PyTorch3D 版本 : {pytorch3d.__version__}")
        print(f"安装路径       : {pytorch3d.__file__}")

        from pytorch3d.transforms import rotation_6d_to_matrix
        import torch

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"PyTorch3D 测试设备: {device}")

        # 简单验证：6D 旋转 -> 旋转矩阵
        rot6d = torch.randn(2, 6, device=device)
        R = rotation_6d_to_matrix(rot6d)
        # 检查正交性：R^T @ R 应接近单位阵
        I = torch.eye(3, device=device)
        err = (R.transpose(1, 2) @ R - I).abs().max().item()
        print(f"rotation_6d_to_matrix 验证: OK, 正交误差 = {err:.2e}")
    except ImportError:
        print("PyTorch3D 未安装或导入失败。")
    print()


def main() -> None:
    print_header("环境检测开始")
    print()
    check_python()
    check_cuda()
    check_pytorch()
    check_pytorch3d()
    print_header("检测结束")


if __name__ == "__main__":
    main()