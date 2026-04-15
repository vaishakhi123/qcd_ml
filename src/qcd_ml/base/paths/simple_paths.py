import torch

from ..hop import v_hop, v_ng_hop, m_hop, stag_hop
from ..operations import SU3_group_compose, make_eta
from ..operations import v_gauge_transform, stag_v_gauge_transform

def v_evaluate_path(U, path, v):
    """
    Gauge-equivariantly evaluate a path on a vector-like field.

    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    for mu, nhops in path:
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            v = v_hop(U, mu, direction, v)
    return v

def v_ng_evaluate_path(path, v):
    """
    Evaluate a path on a vector-like field without gauge degrees of freedom.

    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    if len(path) > 0:
        mus = [mu for mu,_ in path]
        hops = [nhops for _,nhops in path]
        return torch.roll(v, shifts=hops, dims=mus)
    return v


def slow_v_ng_evaluate_path(path, v):
    """
    XXX: deprecated; only used for testing purposes.

    Evaluate a path on a vector-like field without gauge degrees of freedom.

    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    for mu, nhops in path:
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            v = v_ng_hop(mu, direction, v)
    return v


def v_reverse_evaluate_path(U, path, v):
    """
    Gauge-equivariantly evaluate a path on a vector-like field.
    This is the inverse of ``v_evaluate_path``.

    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    for mu, nhops in reversed(path):
        nhops *= -1
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            v = v_hop(U, mu, direction, v)
    return v


def v_ng_reverse_evaluate_path(path, v):
    """
    Inverse of ``v_ng_evaluate_path``.
    """
    if len(path) > 0:
        mus = [mu for mu,_ in path]
        hops = [-nhops for _,nhops in path]
        return torch.roll(v, shifts=hops, dims=mus)
    return v


def slow_v_ng_reverse_evaluate_path(path, v):
    """
    XXX: Deprecated; used for testing.
    """
    for mu, nhops in reversed(path):
        nhops *= -1
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            v = v_ng_hop(mu, direction, v)
    return v


def m_evaluate_path(U, path, m):
    """
    Gauge-equivariantly evaluate a path on a matrix-like field.

    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    for mu, nhops in path:
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            m = m_hop(U, mu, direction, m)
    return m


def m_reverse_evaluate_path(U, path, m):
    """
    Gauge-equivariantly evaluate a path on a matrix-like field.
    This is the inverse of ``m_evaluate_path``.

    paths is a list of paths. Every path is a list [(mu, nhops)].
    An empty list is the path that does not perform any hops.
    
    If nhops is negative, the hop is made in negative mu direction.
    """
    for mu, nhops in reversed(path):
        nhops *= -1
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1

        for _ in range(nhops):
            m = m_hop(U, mu, direction, m)
    return m


def stag_v_evaluate_path(U, path, psi, eta):
    """
    Gauge-equivariantly evaluate a path on a staggered fermion field.
    
    psi shape: (n_features, Lx, Ly, Lz, Lt, 3)
    U shape:   (4, Lx, Ly, Lz, Lt, 3, 3)
    eta shape: (4, Lx, Ly, Lz, Lt)
    
    Applies: eta_mu(x) * U_mu(x) * psi(x + mu_hat)
    """
    for mu, nhops in path:
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1
        for _ in range(nhops):
            psi = stag_hop(U, mu, direction, psi)
            eta_mu = eta[mu].unsqueeze(-1).to(psi.dtype)
            psi = eta_mu * psi
    return psi


def stag_v_reverse_evaluate_path(U, path, psi, eta):
    """
    Inverse of stag_v_evaluate_path.
    """
    for mu, nhops in reversed(path):
        nhops *= -1
        if nhops < 0:
            direction = -1
            nhops *= -1
        else:
            direction = 1
        for _ in range(nhops):
            eta_mu = eta[mu].unsqueeze(-1).to(psi.dtype)
            psi = eta_mu * psi
            psi = stag_hop(U, mu, direction, psi)
    return psi
