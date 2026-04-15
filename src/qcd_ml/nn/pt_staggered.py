"""
qcd_ml.nn.staggered_pt
=======================
Parallel transport layer for staggered fermion fields.
Includes staggered phase factors eta_mu(x).
Field shape: (n_features, Lx, Ly, Lz, Lt, 3)
"""
import torch

from ..base.operations import make_eta

class s_PT(torch.nn.Module):
    """
    Parallel transport layer for staggered fermion fields.
    No learnable weights.

    Each channel i is transported along paths[i] with
    staggered phase factors included at each hop.

    paths: list of paths, each path = [(mu, direction), ...]
           empty list = identity
    U:     gauge field, shape (4, Lx, Ly, Lz, Lt, 3, 3)
    """
    def __init__(self, paths, U):
        super().__init__()
        self.paths = paths
        self.n_feature_in  = len(paths)
        self.n_feature_out = len(paths)
        self.lattice_sizes = U.shape[1:5]

        # register U and eta as buffers (move with .to(device))
        self.register_buffer("U", U)
        self.register_buffer(
            "eta",
            make_eta(self.lattice_sizes, U.device, U.real.dtype)
        )

    # ── single hop ───────────────────────────────────────────────────────
    def _hop_forward(self, field, mu):
        """eta_mu(x) * U_mu(x) * field(x + mu_hat)"""
        shifted   = torch.roll(field, shifts=-1, dims=mu)
        transport = torch.einsum("...ab,...b->...a", self.U[mu], shifted)
        eta_mu    = self.eta[mu].unsqueeze(-1).to(field.dtype)
        return eta_mu * transport

    def _hop_backward(self, field, mu):
        """eta_mu(x-mu_hat) * U_mu†(x-mu_hat) * field(x - mu_hat)"""
        U_dag     = self.U[mu].conj().transpose(-2, -1)
        U_shifted = torch.roll(U_dag,  shifts=+1, dims=mu)
        f_shifted = torch.roll(field,  shifts=+1, dims=mu)
        transport = torch.einsum("...ab,...b->...a", U_shifted, f_shifted)
        eta_mu    = torch.roll(self.eta[mu], shifts=+1, dims=mu).unsqueeze(-1).to(field.dtype)
        return eta_mu * transport

    def _transport(self, field, path):
        """Transport field along path = [(mu0,d0),(mu1,d1),...]"""
        result = field
        for (mu, direction) in path:
            if direction == 1:
                result = self._hop_forward(result, mu)
            else:
                result = self._hop_backward(result, mu)
        return result

    # ── forward / reverse ────────────────────────────────────────────────
    def forward(self, features_in):
        """
        features_in: (n_features, Lx, Ly, Lz, Lt, 3)
        returns:     (n_features, Lx, Ly, Lz, Lt, 3)
        Each channel transported along its assigned path.
        """
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]}"
                f" but expected {self.n_feature_in}"
            )
        return torch.stack([
            self._transport(features_in[i], self.paths[i])
            for i in range(self.n_feature_in)
        ])

    def reverse(self, features_in):
        """Hermitian adjoint — reverse each path."""
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]}"
                f" but expected {self.n_feature_in}"
            )
        return torch.stack([
            self._transport(
                features_in[i],
                [(mu, -d) for (mu, d) in reversed(self.paths[i])]
            )
            for i in range(self.n_feature_in)
        ])

    def gauge_transform_using_transformed(self, U_transformed):
        """Update gauge links. eta unchanged (site-dependent only)."""
        self.U = U_transformed