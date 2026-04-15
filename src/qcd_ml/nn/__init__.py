"""
This module provides neural networks for lattice QCD. 
The modules ``ptc`` and ``lptc`` provide parallel transport pooling and local parallel transport pooling, respectively.
The module ``pt_pool`` provides the ``v_ProjectLayer`` class for parallel transport pooling and some utility functions for 
paralell transport pooling.
The modules ``dense`` and ``pt`` provide dense layers and parallel transport layers, which can be used to build more general gauge-equivariant neural networks.

Staggered fermion modules:
- ``dense_staggered`` provides dense layers for staggered fields
- ``pt_staggered`` provides parallel transport layers with staggered phase factors
- ``s_pt_buffer`` provides pre-computed parallel transport using StaggeredPathBuffer
- ``staggered_preconditioner`` provides preconditioner network architecture for staggered fermions
"""

import qcd_ml.nn.dense
import qcd_ml.nn.ptc
import qcd_ml.nn.lptc
import qcd_ml.nn.pt
import qcd_ml.nn.matrix_layers
import qcd_ml.nn.non_gauge
import qcd_ml.nn.dense_staggered
import qcd_ml.nn.pt_staggered
import qcd_ml.nn.s_pt_buffer
import qcd_ml.nn.staggered_preconditioner
