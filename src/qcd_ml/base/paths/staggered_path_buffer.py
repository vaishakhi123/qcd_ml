import torch

from ..operations import make_eta
from .compile import compile_path


class StaggeredPathBuffer:
    """
    Pre-computes data for staggered fermion path transport.
    
    For staggered fermions, eta factors are site-dependent and applied at each hop.
    This class stores U and eta, providing transport methods.
    
    Field shapes:
        psi: (Lx, Ly, Lz, Lt, 3)  # single feature
        U:   (4, Lx, Ly, Lz, Lt, 3, 3)
    """
    def __init__(self, U, path, eta=None, lattice_sizes=None):
        if isinstance(U, list):
            U = torch.stack(U)

        self.U = U
        self.path = path
        self.lattice_sizes = lattice_sizes if lattice_sizes is not None else tuple(U.shape[1:5])
        self.eta = eta if eta is not None else make_eta(self.lattice_sizes, U.device, U.real.dtype)

        if len(self.path) > 0:
            self.path = compile_path(self.path)

    def _hop_forward(self, field, mu):
        """eta_mu(x) * U_mu(x) * field(x + mu_hat)"""
        shifted = torch.roll(field, shifts=-1, dims=mu)
        transport = torch.einsum("...ab,...b->...a", self.U[mu], shifted)
        eta_mu = self.eta[mu].unsqueeze(-1).to(field.dtype)
        return eta_mu * transport

    def _hop_backward(self, field, mu):
        """eta_mu(x-mu_hat) * U_mu†(x-mu_hat) * field(x - mu_hat)"""
        U_dag = self.U[mu].conj().transpose(-2, -1)
        U_shifted = torch.roll(U_dag, shifts=+1, dims=mu)
        f_shifted = torch.roll(field, shifts=+1, dims=mu)
        transport = torch.einsum("...ab,...b->...a", U_shifted, f_shifted)
        eta_mu = torch.roll(self.eta[mu], shifts=+1, dims=mu).unsqueeze(-1).to(field.dtype)
        return eta_mu * transport

    def _transport(self, field, path, forward=True):
        result = field
        if forward:
            for mu, nhops in path:
                direction = -1 if nhops < 0 else 1
                nhops = abs(nhops)
                for _ in range(nhops):
                    result = self._hop_forward(result, mu) if direction == 1 else self._hop_backward(result, mu)
        else:
            for mu, nhops in reversed(path):
                nhops *= -1
                direction = -1 if nhops < 0 else 1
                nhops = abs(nhops)
                for _ in range(nhops):
                    result = self._hop_forward(result, mu) if direction == 1 else self._hop_backward(result, mu)
        return result

    @property
    def gauge_transport_matrix(self):
        return self.U

    def v_transport(self, psi):
        """Transport staggered field along path."""
        return self._transport(psi, self.path, forward=True)

    def v_reverse_transport(self, psi):
        """Inverse of v_transport."""
        return self._transport(psi, self.path, forward=False)

    def m_transport(self, m):
        """Transport matrix-like field."""
        return self._transport(m, self.path, forward=True)

    def m_reverse_transport(self, m):
        """Inverse of m_transport."""
        return self._transport(m, self.path, forward=False)
