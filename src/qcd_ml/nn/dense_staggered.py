"""
qcd_ml.nn.staggered_dense
==========================
Dense layer for staggered fermion fields.
Staggered fields have no spin index - just color (G=3).
Field shape: (n_features, Lx, Ly, Lz, Lt, 3)
"""

import torch

class s_Dense(torch.nn.Module):
    r"""
    Dense Layer for staggered fermion fields.

    ``s_Dense.forward(features_in)`` computes

    .. math::
        \phi_o(x) = \sum\limits_i W_{io} \phi_i(x)

    where :math:`W_{io}` are complex scalars (not spin matrices,
    since staggered fields have no spin index).

    ``s_Dense.reverse(features_in)`` computes the hermitian adjoint:

    .. math::
        \phi_i(x) = \sum\limits_o W_{io}^* \phi_o(x)
    """

    def __init__(self, n_feature_in, n_feature_out):
        super().__init__()
        # scalar weights — no 4x4 spin structure
        self.weights = torch.nn.Parameter(
            torch.randn(n_feature_in, n_feature_out, dtype=torch.cdouble)
        )
        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out

    def forward(self, features_in):
        r"""
        features_in shape: (n_feature_in, Lx, Ly, Lz, Lt, 3)
        output shape:      (n_feature_out, Lx, Ly, Lz, Lt, 3)

        phi_o(x) = sum_i W_io * phi_i(x)
        """
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]}"
                f" but expected {self.n_feature_in}"
            )
        # W: (i, o)
        # features_in: (i, Lx, Ly, Lz, Lt, G)
        # output: (o, Lx, Ly, Lz, Lt, G)
        return torch.einsum("io,iabcdG->oabcdG", self.weights, features_in)

    def reverse(self, features_in):
        r"""
        Hermitian adjoint: phi_i = sum_o W_io^* phi_o
        features_in shape: (n_feature_out, Lx, Ly, Lz, Lt, 3)
        output shape:      (n_feature_in,  Lx, Ly, Lz, Lt, 3)
        """
        if features_in.shape[0] != self.n_feature_out:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]}"
                f" but expected {self.n_feature_out}"
            )
        return torch.einsum("io,oabcdG->iabcdG", self.weights.conj(), features_in)