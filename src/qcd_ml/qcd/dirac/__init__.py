import torch

from ..static import gamma
from ...base.operations import v_spin_const_transform, mspin_const_group_compose
from ...base.hop import v_hop
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
    Staggered Dirac operator with optional fat links.
    """

    def __init__(self, U, mass_parameter):
        """
        U: gauge field (4, Nx, Ny, Nz, Nt, 3, 3)
        mass_parameter: fermion mass
        """
        self.U = U
        self.mass_parameter = mass_parameter

        # device
        self.device = get_device_by_reference(U[0])

        # lattice sizes
        self.Lx, self.Ly, self.Lz, self.Lt = U.shape[1:5]

        # precompute staggered phases
        self.eta = self._compute_eta().to(self.device)

    def _compute_eta(self):
        x = torch.arange(self.Lx).view(-1,1,1,1)
        y = torch.arange(self.Ly).view(1,-1,1,1)
        z = torch.arange(self.Lz).view(1,1,-1,1)
        t = torch.arange(self.Lt).view(1,1,1,-1)
    
        eta = torch.ones((4, self.Lx, self.Ly, self.Lz, self.Lt), dtype=torch.cdouble)
        # eta_0 = 1 (no preceding coordinates)
        eta[1] = (-1) ** (x)              # sum of coords[:1] = x
        eta[2] = (-1) ** (x + y)          # sum of coords[:2]
        eta[3] = (-1) ** (x + y + z)      # sum of coords[:3]
        return eta
    # def _compute_eta(self):
    #     """
    #     Compute staggered phase factors eta_mu(x)
    #     shape: (4, Nx, Ny, Nz, Nt)
    #     """
    #     eta = torch.ones((4, self.Lx, self.Ly, self.Lz, self.Lt), dtype=torch.cdouble)

    #     for x in range(self.Lx):
    #         for y in range(self.Ly):
    #             for z in range(self.Lz):
    #                 for t in range(self.Lt):
    #                     coords = [x, y, z, t]
    #                     for mu in range(4):
    #                         phase = sum(coords[:mu]) % 2
    #                         if phase == 1:
    #                             eta[mu, x, y, z, t] = -1

    #     return eta


    def __call__(self, v):
        """
        Apply staggered Dirac operator.
        v: fermion field (Nx, Ny, Nz, Nt, Nc)
        """
        result = self.mass_parameter * v

        for mu in range(4):
            # forward hop
            forward = v_hop(self.U, mu, 1, v)

            # backward hop
            backward = v_hop(self.U, mu, -1, v)

            # apply staggered phase
            eta_mu = self.eta[mu].unsqueeze(-1)  # match color dim

            result += eta_mu * (forward - backward) / 2
            # eta_fwd = self . eta [ mu ]. unsqueeze ( -1)
            # #shift eta back by 1 along dimension mu to get eta ( x -
            # mu_hat )
            # eta_bwd = torch . roll ( self . eta [ mu ] , shifts =1 , dims = mu ) .
            # unsqueeze ( -1)
            # result += ( eta_fwd * forward - eta_bwd * backward ) / 2

        return result
