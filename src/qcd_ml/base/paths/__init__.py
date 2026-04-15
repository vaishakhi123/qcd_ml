"""
qcd_ml.base.paths
=================

Gauge-equivariant parallel transport paths.

The function ``v_evaluate_path`` is memory effective but slow.

The class ``PathBuffer`` can be used to speed up path evaluation
but may be more memory intensve.

For staggered fermions, use ``StaggeredPathBuffer`` which includes
staggered phase factors eta_mu(x) at each hop.
"""

from .simple_paths import (
        v_evaluate_path, v_ng_evaluate_path, v_reverse_evaluate_path, v_ng_reverse_evaluate_path, m_evaluate_path
        )

from .path_buffer import PathBuffer
from .staggered_path_buffer import StaggeredPathBuffer


def path_get_orig_point(path):
    """
    This funciton returns the point that will be transported to the point [0,0,0,0]
    by the path.
    """
    point = [0] * 4
    for mu, nhops in path:
        point[mu] -= nhops
    return point
