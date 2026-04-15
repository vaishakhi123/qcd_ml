#!/usr/bin/env python3
"""
qcd_ml.base.operations
======================

Provides

- matrix-matrix multiplication for
  - SU3 fields
  - spin matrices
- gauge transformation of
  - vector-like fields
  - link-like fields
- group action of
  - spin matrices on vector-like fields
  - spin fields on vector-like fields

See also: :ref:`doc-datatypes:qcd_ml Datatypes`.
"""


import torch


def make_eta(lattice_sizes, device, dtype):
    """
    Staggered phase factors eta_mu(x):
        eta_0(x) = 1
        eta_1(x) = (-1)^x0
        eta_2(x) = (-1)^(x0+x1)
        eta_3(x) = (-1)^(x0+x1+x2)
    Returns tensor shape (4, Lx, Ly, Lz, Lt).
    """
    Lx, Ly, Lz, Lt = lattice_sizes
    x0 = torch.arange(Lx, device=device).view(Lx,  1,  1,  1)
    x1 = torch.arange(Ly, device=device).view( 1, Ly,  1,  1)
    x2 = torch.arange(Lz, device=device).view( 1,  1, Lz,  1)

    eta = torch.ones(4, Lx, Ly, Lz, Lt, dtype=dtype, device=device)
    eta[1] = (-1.0) ** x0
    eta[2] = (-1.0) ** (x0 + x1)
    eta[3] = (-1.0) ** (x0 + x1 + x2)
    return eta

def _mul(iterable):
    res = 1
    for i in iterable:
        res *= i
    return res


def _es_SU3_group_compose(A, B):
    return torch.einsum("abcdij,abcdjk->abcdik", A, B)


def SU3_group_compose(A, B):
    """
    :math:`SU(3)` group composition of two :math:`SU(3)` fields.
    """
    vol = _mul(A.shape[:4])
    old_shape = A.shape
    return torch.bmm(A.reshape((vol, *(A.shape[4:])))
                     , B.reshape((vol, *(A.shape[4:])))).reshape(old_shape)


def _es_v_gauge_transform(Umu, v):
    return torch.einsum("abcdij,abcdSj->abcdSi", Umu, v)


def v_gauge_transform(Umu, v):
    """
    Gauge transformation of vector-like fields.
    """
    vol = _mul(v.shape[:4])
    old_shape = v.shape
    return torch.bmm(Umu.reshape((vol, *(Umu.shape[4:])))
                     , v.reshape((vol, *(v.shape[4:]))).transpose(-1, -2)
                     ).transpose(-1, -2).reshape(old_shape)


def _es_v_spin_transform(M, v):
    return torch.einsum("abcdij,abcdjG->abcdiG", M, v)


def v_spin_transform(M, v):
    """
    Applies a spin matrix field to a vector field.
    """
    vol = _mul(v.shape[:4])
    old_shape = v.shape
    return torch.bmm(M.reshape((vol, *(M.shape[4:])))
                     , v.reshape((vol, *(v.shape[4:])))
                     ).reshape(old_shape)


def v_spin_const_transform(M, v):
    """
    Applies a spin matrix to a vector field.
    """
    return torch.einsum("ij,abcdjG->abcdiG", M, v)


def v_ng_spin_transform(M, v):
    """
    Applies a spin matrix field to a vector field without gauge freedom.
    """
    return torch.einsum("abcdij,abcdj->abcdi", M, v)


def v_ng_spin_const_transform(M, v):
    """
    Applies a spin matrix to a vector field without gauge freedom.
    """
    return torch.einsum("ij,abcdj->abcdi", M, v)


def link_gauge_transform(U, V):
    """
    Gauge-transforms a link-like field.
    A link-like is typically a gauge configuration.
    """
    Vdg = V.adjoint()
    U_trans = [SU3_group_compose(V, Umu) for Umu in U]
    for mu, U_transmu in enumerate(U_trans):
        U_trans[mu] = SU3_group_compose(U_transmu, torch.roll(Vdg, -1, mu))
    return U_trans


def mspin_const_group_compose(A, B):
    """
    Matrix-matrix multiplication for spin matrices.
    """
    return torch.einsum("ij,jk->ik", A, B)


def _es_m_gauge_transform(Umu, m):
    return torch.einsum("abcdij,abcdjk,abcdkl->abcdil", Umu, m, Umu.adjoint())


def m_gauge_transform(Umu, m):
    vol = _mul(m.shape[:4])
    old_shape = m.shape
    Umu_reshaped = Umu.reshape((vol, *(Umu.shape[4:])))
    return torch.bmm(torch.bmm(Umu_reshaped
                     , m.reshape((vol, *(m.shape[4:])))), Umu_reshaped.adjoint()).reshape(old_shape)


def stag_v_gauge_transform(Umu, v):
    """
    Gauge transformation for staggered vector fields.
    Field shape: (n_features, Lx, Ly, Lz, Lt, 3)
    U shape:     (Lx, Ly, Lz, Lt, 3, 3)
    """
    vol = 1
    for s in v.shape[1:5]:
        vol *= s
    n_feat = v.shape[0]
    Umu_3x3 = Umu.reshape((vol, 3, 3))
    v_perm = v.permute(1, 2, 3, 4, 5, 0).reshape((vol, 3, n_feat))
    result = torch.bmm(Umu_3x3, v_perm)
    return result.reshape((v.shape[1], v.shape[2], v.shape[3], v.shape[4], 3, n_feat)).permute(5, 0, 1, 2, 3, 4)


def stag_m_gauge_transform(Umu, m):
    """
    Gauge transformation for staggered matrix fields.
    Field shape: (n_features, Lx, Ly, Lz, Lt, 3, 3)
    U shape:     (Lx, Ly, Lz, Lt, 3, 3)
    """
    vol = 1
    for s in m.shape[1:5]:
        vol *= s
    n_feat = m.shape[0]
    Umu_3x3 = Umu.reshape((vol, 3, 3))
    m_perm = m.permute(1, 2, 3, 4, 5, 6, 0).reshape((vol, 3, 3 * n_feat))
    temp = torch.bmm(Umu_3x3, m_perm.reshape((vol, 3, 3)))
    result = torch.bmm(temp, Umu_3x3.adjoint())
    return result.reshape((m.shape[1], m.shape[2], m.shape[3], m.shape[4], 3, 3, n_feat)).permute(6, 0, 1, 2, 3, 4, 5)
