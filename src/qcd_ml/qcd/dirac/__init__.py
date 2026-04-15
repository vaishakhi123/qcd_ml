import torch

from ..static import gamma
from ...base.operations import v_spin_const_transform, mspin_const_group_compose, make_eta
from ...base.hop import v_hop
from ...base.hop import stag_hop
from ...base.hop import naik
from ...base.paths import PathBuffer

from ...util.comptime import comptime
from ...util import get_device_by_reference

"""
qcd_ml.qcd.dirac
================

Dirac operators.
"""


@comptime([(mu, nu) for mu in range(4) for nu in range(4)])
def sigmamunu(mu, nu):
    return (mspin_const_group_compose(gamma[mu], gamma[nu]) 
            - mspin_const_group_compose(gamma[nu], gamma[mu])) / 2


class dirac_wilson:
    """
    Dirac Wilson operator. See arXiv:2302.05419.
    """
    def __init__(self, U, mass_parameter):
        self.U = U
        self.mass_parameter = mass_parameter

        # copy gamma to local device.
        self.gamma = torch.stack(gamma).to(get_device_by_reference(U[0]))


    def __call__(self, v):
        result = (4 + self.mass_parameter) * v 
        for mu in range(4):
            result -= v_hop(self.U, mu, 1, v) / 2
            result -= v_hop(self.U, mu, -1, v) / 2

            result += v_spin_const_transform(gamma[mu], v_hop(self.U, mu, -1, v)) / 2
            result -= v_spin_const_transform(gamma[mu], v_hop(self.U, mu, 1, v)) / 2

        return result


class dirac_wilson_clover:
    """
    Dirac Wilson operator with clover term improvement.

    See arXiv:2302.05419.
    """
    def __init__(self, U, mass_parameter, csw):
        self.U = U
        self.mass_parameter = mass_parameter
        self.csw = csw

        # copy both gamma and sigma to local device.
        self.gamma = torch.stack(gamma).to(get_device_by_reference(U[0]))

        self.sigmamunu = torch.stack([
                                torch.stack([sigmamunu(mu, nu) for nu in range(4)])
                            for mu in range(4)]).to(get_device_by_reference(U[0]))

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        self.plaquette_path_buffers = [[[PathBuffer(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

    def Qmunu(self, mu, nu, v):
        paths = self.plaquette_path_buffers[mu][nu]
        return (
                paths[0].v_transport(v)
                + paths[1].v_transport(v)
                + paths[2].v_transport(v)
                + paths[3].v_transport(v)
                )

    def field_strength(self, mu, nu, v):
        return (self.Qmunu(mu, nu, v) - self.Qmunu(nu, mu, v)) / 8

    def __call__(self, v):
        result = (4 + self.mass_parameter) * v
        for mu in range(4):
            result -= v_hop(self.U, mu, 1, v) / 2
            result -= v_hop(self.U, mu, -1, v) / 2

            result += v_spin_const_transform(self.gamma[mu], v_hop(self.U, mu, -1, v)) / 2
            result -= v_spin_const_transform(self.gamma[mu], v_hop(self.U, mu, 1, v)) / 2

        improvement = 0
        for mu in range(4):
            for nu in range(mu):
                # sigma and field_strength are both anti symmetric.
                improvement = (improvement
                               + 2*v_spin_const_transform(self.sigmamunu[mu, nu], self.field_strength(mu, nu, v))
                               )

        return result - self.csw / 4 * improvement



class dirac_staggered:
    """
    Kogut-Susskind staggered Dirac operator with separate thin and fat links arXiv:hep-lat/9712010v2.

    Implements:
        D psi(x) = m * psi(x)
                 + sum_mu eta_mu(x) * c1/2 * [U_fat_mu(x) psi(x+mu) 
                                               - U_fat_mu†(x-mu) psi(x-mu)]
                 + sum_mu eta_mu(x) * c2/2 * [U_thin(x)U_thin(x+mu)U_thin(x+2mu) psi(x+3mu)
                                               - U_thin†(x-mu)U_thin†(x-2mu)U_thin†(x-3mu) psi(x-3mu)]

    Mirrors GPT convention:
        g.qcd.fermion.staggered(U_thin + U_fat, mass, c1, c2, u0)
        first 4 links  → U_thin  (Naik 3-link term)
        next  4 links  → U_fat   (1-link term)

    Usage:
        # naive staggered (no smearing, no Naik)
        D = dirac_staggered(U, U, mass=0.1, c1=1.0, c2=0.0)

        # improved staggered (fat 1-link + Naik)
        D = dirac_staggered(U_thin, U_fat, mass=0.1, c1=9/8, c2=-1/24)

        # naive shorthand (thin=fat=U)
        D = dirac_staggered.naive(U, mass=0.1)

        # from GPT-style combined list (8 links)
        D = dirac_staggered.from_combined(U_combined, mass=0.1, c1=9/8, c2=-1/24)
    """

    def __init__(self, U_thin, U_fat, mass, u0, c1=1.0, c2=0.0):
        """
        U_thin: gauge links for Naik 3-link term, shape (4, Lx, Ly, Lz, Lt, 3, 3)
        U_fat:  gauge links for 1-link term,      shape (4, Lx, Ly, Lz, Lt, 3, 3)
        mass:   fermion mass
        c1:     1-link coefficient  (1.0 for naive, 9/8 for Naik-improved)
        c2:     3-link coefficient  (0.0 for naive, -1/24 for Naik-improved)
        """
        self.U_thin = U_thin
        self.U_fat  = U_fat
        self.mass   = mass
        self.c1     = c1
        self.c2     = c2
        self.u0     = u0

        self.lattice_sizes = U_thin.shape[1:5]  # (Lx, Ly, Lz, Lt)
        self.eta = make_eta(self.lattice_sizes, U_thin.device, U_thin.real.dtype)

    # ── Constructors ────────────────────────────────────────────────────────

    @classmethod
    def naive(cls, U, mass, u0):
        """
        Naive staggered — thin=fat=U, no Naik term.
        Matches: g.qcd.fermion.staggered(U + U, mass, c1=1.0, c2=0.0, u0=1.0)
        """
        return cls(U, U, mass, u0, c1=1.0, c2=0.0)

    @classmethod
    def from_combined(cls, U_combined, mass, u0, c1=9/8, c2=-1/24):
        """
        Build from GPT-style combined 8-link list/tensor.
        U_combined: shape (8, Lx, Ly, Lz, Lt, 3, 3)
                    first 4  → thin (Naik)
                    next  4  → fat  (1-link)

        Matches: g.qcd.fermion.staggered(U_thin + U_fat, mass, c1, c2, u0=1.0)
        """
        if isinstance(U_combined, (list, tuple)):
            U_combined = torch.stack(U_combined)
        U_thin = U_combined[:4]
        U_fat  = U_combined[4:]
        return cls(U_thin, U_fat, mass, u0, c1=c1, c2=c2)


    def __call__(self, psi, adjoint=False):
        """
        psi shape: (Lx, Ly, Lz, Lt, 3)
        returns:   (Lx, Ly, Lz, Lt, 3)
        
        adjoint: if True, applies D^dagger instead of D
        """
        result = self.mass * psi
        u0 = self.u0
        
        for mu in range(4):
            eta_mu = self.eta[mu].unsqueeze(-1).to(psi.dtype)
            if adjoint:
                eta_mu = torch.roll(eta_mu, shifts=+1, dims=mu)

            # 1-link term — uses fat links
            if self.c1 != 0.0:
                if adjoint:
                    fwd = stag_hop(self.U_fat, mu, +1, psi) 
                    bwd = stag_hop(self.U_fat, mu, -1, psi)
                else:
                    fwd = stag_hop(self.U_fat, mu, -1, psi) 
                    bwd = stag_hop(self.U_fat, mu, +1, psi)
                result = result + (self.c1 / 2/u0) * eta_mu * (fwd - bwd)

            # Naik 3-link term — uses thin links
            if self.c2 != 0.0:
                if adjoint:
                    fwd3 = naik(self.U_thin, mu, +1, psi)
                    bwd3 = naik(self.U_thin, mu, -1, psi)
                else:
                    fwd3 = naik(self.U_thin, mu, -1, psi)
                    bwd3 = naik(self.U_thin, mu, +1, psi)
                result = result + (self.c2 / (2*u0**3)) * eta_mu * (fwd3 - bwd3)

        return result
