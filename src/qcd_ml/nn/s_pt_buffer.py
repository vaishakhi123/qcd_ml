"""
s_PT_buffer
============

Parallel Transport Layer for staggered fermion fields.
Uses StaggeredPathBuffer for consistent interface.
"""

import torch

from ..base.paths import StaggeredPathBuffer
from ..base.operations import make_eta


class s_PT_buffer(torch.nn.Module):
    """
    Parallel Transport Layer for staggered fermion fields.

    Field shape: (n_features, Lx, Ly, Lz, Lt, 3)
    U shape:     (4, Lx, Ly, Lz, Lt, 3, 3)

    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    """

    def __init__(self, paths, U, eta=None, lattice_sizes=None):
        super().__init__()
        self.n_feature_in = len(paths)
        self.n_feature_out = len(paths)
        self.lattice_sizes = lattice_sizes or tuple(U.shape[1:5])
        self.paths = paths
        
        self.register_buffer("_U", U)
        _eta = eta if eta is not None else make_eta(self.lattice_sizes, U.device, U.real.dtype)
        self.register_buffer("_eta", _eta)

        self.path_buffers = [
            StaggeredPathBuffer(U, pi, eta=_eta, lattice_sizes=self.lattice_sizes)
            for pi in paths
        ]

    @property
    def U(self):
        return self._U
    
    @U.setter
    def U(self, value):
        self._U = value

    @property 
    def eta(self):
        return self._eta
    
    @eta.setter
    def eta(self, value):
        self._eta = value

    def forward(self, features_in):
        """
        Transport each feature channel along its corresponding path.
        features_in: (n_features, Lx, Ly, Lz, Lt, 3)
        returns:     (n_features, Lx, Ly, Lz, Lt, 3)
        """
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}"
            )

        features_out = [None] * self.n_feature_out
        for i, p in enumerate(self.path_buffers):
            result = p.v_transport(features_in[i])
            features_out[i] = result

        return torch.stack(features_out)

    def reverse(self, features_in):
        """
        Hermitian adjoint — reverse each path.
        """
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(
                f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}"
            )

        features_out = [None] * self.n_feature_out
        for i, p in enumerate(self.path_buffers):
            result = p.v_reverse_transport(features_in[i])
            features_out[i] = result

        return torch.stack(features_out)

    def gauge_transform_using_transformed(self, U_transformed):
        """Update gauge links."""
        self.U = U_transformed
        self.path_buffers = [
            StaggeredPathBuffer(U_transformed, pi, eta=self.eta, lattice_sizes=self.lattice_sizes)
            for pi in self.paths
        ]
