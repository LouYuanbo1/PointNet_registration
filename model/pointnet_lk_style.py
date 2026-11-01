""" PointNet
    References.
        .. [1] Charles R. Qi, Hao Su, Kaichun Mo and Leonidas J. Guibas,
        "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation",
        CVPR (2017)
"""

import torch
from typing import List, Sequence

def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def mlp_layers(
    nch_input: int,
    nch_layers: Sequence[int],
    b_shared: bool = True,
    bn_momentum: float = 0.1,
    dropout: float = 0.0,
) -> List[torch.nn.Module]:
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers

class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(
        self,
        nch_input: int,
        nch_layers: Sequence[int],
        b_shared: bool = True,
        bn_momentum: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = self.layers(inp)
        return out


def symfn_max(x: torch.Tensor) -> torch.Tensor:
    # [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    #a, _ = torch.max(x, dim=-1, keepdim=True)
    return a

class TNet(torch.nn.Module):
    """ [B, K, N] -> [B, K, K]
    """
    def __init__(self, K: int) -> None:
        super().__init__()
        # [B, K, N] -> [B, K*K]
        self.mlp1 = torch.nn.Sequential(*mlp_layers(K, [64, 128, 1024], b_shared=True))
        self.mlp2 = torch.nn.Sequential(*mlp_layers(1024, [512, 256], b_shared=False))
        self.lin = torch.nn.Linear(256, K*K)

        for param in self.mlp1.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.mlp2.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.lin.parameters():
            torch.nn.init.constant_(param, 0.0)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        K = inp.size(1)
        N = inp.size(2)
        eye = torch.eye(K).unsqueeze(0).to(inp) # [1, K, K]

        x = self.mlp1(inp)
        x = flatten(torch.nn.functional.max_pool1d(x, N))
        x = self.mlp2(x)
        x = self.lin(x)

        x = x.view(-1, K, K)
        x = x + eye
        return x


class PointNet_features(torch.nn.Module):
    def __init__(
        self, dim_k: int = 1024, use_tnet: bool = False, sym_fn: callable = symfn_max, scale: int = 1
    ) -> None:
        super().__init__()
        mlp_h1 = [int(64/scale), int(64/scale)]
        mlp_h2 = [int(64/scale), int(128/scale), int(dim_k/scale)]

        self.h1 = MLPNet(3, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers
        self.sy = sym_fn

        self.tnet1 = TNet(3) if use_tnet else None
        self.tnet2 = TNet(mlp_h1[-1]) if use_tnet else None

        self.t_out_t2 = None
        self.t_out_h1 = None

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        x = points.transpose(1, 2) # [B, 3, N]
        if self.tnet1:
            t1 = self.tnet1(x)
            x = t1.bmm(x)

        x = self.h1(x)
        if self.tnet2:
            t2 = self.tnet2(x)
            self.t_out_t2 = t2
            x = t2.bmm(x)
        self.t_out_h1 = x # local features

        x = self.h2(x)
        #x = flatten(torch.nn.functional.max_pool1d(x, x.size(-1)))
        x = flatten(self.sy(x))

        #if self.ret_global:
        #    pass
        #else:
        #    # local + global
        #    l0 = self.t_out_h1 # [B, 64, N]
        #    g0 = x # [B, K]
        #    x = torch.cat((l0, g0.unsqueeze(2).repeat(1, 1, num_points)), dim=1)

        return x