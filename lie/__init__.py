"""李群/李代数数学工具包。

SO(3) / SE(3) 及其李代数（so(3) / se(3)）的纯数学算子：
hat 映射、指数映射（Rodrigues）、sinc 及批量矩阵求逆等。
本包不依赖任何神经网络，供 :mod:`model` 与 :mod:`utils` 共用。

- :mod:`lie.so3`    三维特殊正交群与其李代数
- :mod:`lie.se3`    三维刚体变换群与其李代数
- :mod:`lie.sinc`   sinc 族函数
- :mod:`lie.invmat` 批量矩阵求逆
"""