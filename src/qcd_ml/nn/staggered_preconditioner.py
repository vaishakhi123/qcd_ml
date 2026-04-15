"""
Staggered Preconditioner Network
================================

Neural network preconditioner for staggered fermions.
Mirrors the Wilson clover Preconditioner_network architecture.
"""

import torch
from .dense_staggered import s_Dense
from .s_pt_buffer import s_PT_buffer


class StaggeredPreconditionerNetwork(torch.nn.Module):
    """
    Preconditioner network for staggered fermion Dirac operator.

    Uses staggered-aware parallel transport layers (s_PT_buffer) and
    staggered dense layers (s_Dense) since staggered fermions have
    no spin index, only color.

    Field shape: (Lx, Ly, Lz, Lt, 3) - no spin index
    """

    def __init__(self, U, nr_layers, long_range_paths=False, smaller=False, eta=None):
        super().__init__()
        self.nr_layers = nr_layers

        paths = [[]]
        for mu in range(4):
            pathlength = 1
            max_pathlength = U.shape[mu + 1] // 2
            if smaller:
                max_pathlength //= 2
            while pathlength <= max_pathlength:
                paths.extend([[(mu, pathlength)], [(mu, -pathlength)]])
                pathlength *= 2
                if long_range_paths == False:
                    break
        nr_paths = len(paths)

        self.pt = s_PT_buffer(paths, U, eta=eta)
        self.dense_layers = torch.nn.ModuleList(
            [
                s_Dense(1, nr_paths),
                *[
                    s_Dense(nr_paths, nr_paths)
                    for _ in range(self.nr_layers - 1)
                ],
                s_Dense(nr_paths, 1),
            ]
        )

    def forward(self, v):
        """
        Input:  (Lx, Ly, Lz, Lt, 3) - staggered field
        Output: (Lx, Ly, Lz, Lt, 3) - preconditioned field
        """
        vprev = torch.stack([v])

        for i in range(self.nr_layers + 1):
            v = self.dense_layers[i](vprev)
            v[0] += vprev[0]
            if i == self.nr_layers:
                break
            v = self.pt(v)
            vprev = v

        return v[0]
