#!/usr/bin/env python3
"""
qcd_ml.base.hop
===============

Gauge-equivariant hops.

"""

import torch

from .operations import v_gauge_transform, m_gauge_transform

def v_hop(U, mu, direction, v):
    """
    Gauge-equivariant hop for a vector-like field.
    """
    if direction == -1:
        result = torch.roll(v, -1, mu)
        return v_gauge_transform(U[mu], result)
    else:
        Umudg = U[mu].adjoint()
        result = v_gauge_transform(Umudg, v)
        return torch.roll(result, 1, mu)


def v_ng_hop(mu, direction, v):
    """
    Hop for a vector-like field without gauge degrees of freedom.
    """
    return torch.roll(v, direction,  mu)


def m_hop(U, mu, direction, m):
    """
    Gauge-equivariant hop for a matrix-like field.
    """
    if direction == -1:
        result = torch.roll(m, -1, mu)
        return m_gauge_transform(U[mu], result)
    else:
        Umudg = U[mu].adjoint()
        result = m_gauge_transform(Umudg, m)
        return torch.roll(result, 1, mu)

def stag_hop(U, mu, direction, psi):
    
    """forward 1-hop: U_mu(x) * psi(x + mu_hat)
       backward 1-hop: U_mu†(x - mu_hat) * psi(x - mu_hat)"""
    
    if direction == -1:
        psi_shifted = torch.roll(psi, shifts=direction, dims=mu)
        return torch.einsum("...ab,...b->...a", U[mu], psi_shifted)
        
    else:
        U_dag     = U[mu].conj().transpose(-2, -1)
        U_shifted = torch.roll(U_dag,  shifts=direction, dims=mu)
        p_shifted = torch.roll(psi,    shifts=direction, dims=mu)
        return torch.einsum("...ab,...b->...a", U_shifted, p_shifted)
        

def naik(U_thin, mu, direction, psi):
    
    """forward 3-links: U_thin(x) U_thin(x+mu) U_thin(x+2mu) psi(x+3mu)
       backward 3-links: U_thin†(x-mu) U_thin†(x-2mu) U_thin†(x-3mu) psi(x-3mu)"""
    
    tmp = stag_hop(U_thin, mu, direction, psi)
    tmp = stag_hop(U_thin, mu, direction, tmp)
    tmp = stag_hop(U_thin, mu, direction, tmp)
    return tmp

    
