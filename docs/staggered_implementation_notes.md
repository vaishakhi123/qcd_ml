# Staggered Fermion Neural Network Implementation Notes

## Overview

This document explains the implementation of gauge-equivariant neural networks for staggered fermions. Staggered fermions are a formulation of lattice QCD where the 4-component Dirac spinor is decomposed into 4 single-component fields, with the spin degrees of freedom encoded in the lattice sites via phase factors.

## Wilson vs Staggered Fermions

### Wilson Fermions (Existing Implementation)

In the Wilson formulation, the fermion field `ψ(x)` has:
- **Spin index**: 4 components
- **Color index**: 3 components (SU(3))
- **Field shape**: `(Lx, Ly, Lz, Lt, 4, 3)`

The Dirac operator acts on the full spinor. Gauge transport can be pre-computed because the gauge links `U_μ(x)` are site-independent - once you compute the path-ordered product of links, you can apply it directly to any field at any site.

### Staggered Fermions (New Implementation)

In the staggered formulation:
- **No spin index**: 1 component per site
- **Color index**: 3 components (SU(3))
- **Field shape**: `(Lx, Ly, Lz, Lt, 3)` or with features: `(n_features, Lx, Ly, Lz, Lt, 3)`

The spin degrees of freedom are encoded in **staggered phase factors** `η_μ(x)`.

## The Staggered Phase Factors η_μ(x)

The staggered phase factors are defined as:

```
η_0(x) = 1
η_1(x) = (-1)^x₀
η_2(x) = (-1)^(x₀ + x₁)
η_3(x) = (-1)^(x₀ + x₁ + x₂)
```

These factors implement the "spin diagonalization" on the lattice. They appear in the staggered Dirac operator:

```
D_μ(x) ψ(x) → η_μ(x) U_μ(x) ψ(x + μ̂) - η_μ(x - μ̂) U_μ†(x - μ̂) ψ(x - μ̂)
```

### Why η Factors Must Be Applied at Each Hop

**Critical difference from Wilson fermions**: The eta factors `η_μ(x)` depend on the lattice site `x`. This means:

1. **Wilson**: `result(x) = U_path(x) * ψ(source_point)`
   - The gauge transport `U_path` is pre-computable
   - Applied uniformly at all sites

2. **Staggered**: `result(x) = η_μ(x) * U_μ(x) * ψ(x + μ̂)`
   - The eta factor `η_μ(x)` varies with position
   - Must be applied together with the gauge link at each hop
   - Cannot pre-compute just the gauge transport

## Field Shapes

### Wilson (existing)
```python
psi:  (Lx, Ly, Lz, Lt, 4, 3)  # spin × color
U:    (4, Lx, Ly, Lz, Lt, 3, 3)  # gauge links
v:    (n_features, Lx, Ly, Lz, Lt, 4, 3)
```

### Staggered (new)
```python
psi:  (Lx, Ly, Lz, Lt, 3)  # color only, no spin
U:    (4, Lx, Ly, Lz, Lt, 3, 3)  # gauge links
v:    (n_features, Lx, Ly, Lz, Lt, 3)
```

## Implementation Details

### 1. make_eta (base/operations.py)

```python
def make_eta(lattice_sizes, device, dtype):
    """
    Creates staggered phase factors η_μ(x).

    Args:
        lattice_sizes: (Lx, Ly, Lz, Lt)
        device: torch device
        dtype: tensor dtype

    Returns:
        Tensor of shape (4, Lx, Ly, Lz, Lt)
    """
```

**Usage:**
```python
from qcd_ml.base.operations import make_eta

eta = make_eta((8, 8, 8, 8), device='cpu', dtype=torch.float64)
# eta[0] = 1 everywhere
# eta[1] = (-1)^x
# eta[2] = (-1)^(x+y)
# eta[3] = (-1)^(x+y+z)
```

### 2. stag_v_gauge_transform and stag_m_gauge_transform (base/operations.py)

```python
def stag_v_gauge_transform(Umu, v):
    """
    Gauge transformation for staggered vector fields.

    Args:
        Umu: (Lx, Ly, Lz, Lt, 3, 3) - single gauge link
        v:   (n_features, Lx, Ly, Lz, Lt, 3) - staggered field

    Returns:
        Transformed field with same shape as v
    """

def stag_m_gauge_transform(Umu, m):
    """
    Gauge transformation for staggered matrix fields.

    Args:
        Umu: (Lx, Ly, Lz, Lt, 3, 3) - single gauge link
        m:   (n_features, Lx, Ly, Lz, Lt, 3, 3) - staggered matrix field

    Returns:
        Transformed field with same shape as m
    """
```

**Why needed**: These handle the extra feature dimension `(n_features, ...)` that Wilson functions don't support.

### 3. stag_v_evaluate_path and stag_v_reverse_evaluate_path (base/paths/simple_paths.py)

```python
def stag_v_evaluate_path(U, path, psi, eta):
    """
    Gauge-equivariantly evaluate a path on a staggered fermion field.

    Args:
        U:   (4, Lx, Ly, Lz, Lt, 3, 3) - gauge links
        path: [(mu, nhops), ...] - e.g., [(1, 1)] for one hop in mu=1 direction
        psi: (n_features, Lx, Ly, Lz, Lt, 3) - staggered field
        eta: (4, Lx, Ly, Lz, Lt) - staggered phase factors

    Returns:
        Transported field

    Operation at each hop:
        forward:  η_μ(x) * U_μ(x) * ψ(x + μ̂)
        backward: η_μ(x - μ̂) * U_μ†(x - μ̂) * ψ(x - μ̂)
    """
```

### 4. StaggeredPathBuffer (base/paths/staggered_path_buffer.py)

```python
class StaggeredPathBuffer:
    """
    Pre-computes data for staggered fermion path transport.

    Unlike Wilson PathBuffer, we cannot pre-compute the full gauge
    transport matrix because eta factors depend on position.

    Instead, we store both U and eta, and apply them together
    at each hop.
    """

    def __init__(self, U, path, eta=None, lattice_sizes=None):
        """
        Args:
            U:  (4, Lx, Ly, Lz, Lt, 3, 3) - gauge field
            path: [(mu, nhops), ...] - path to transport along
            eta: (4, Lx, Ly, Lz, Lt) - staggered phase factors
            lattice_sizes: (Lx, Ly, Lz, Lt)
        """

    def v_transport(self, psi):
        """Transport staggered field along path."""
        # Applies: hop + eta at each step

    def v_reverse_transport(self, psi):
        """Inverse transport (reverse path with adjoint links)."""
```

**Key methods:**

```python
# Single hop forward (eta_mu(x) * U_mu(x) * psi(x + mu_hat))
def _hop_forward(self, field, mu):
    shifted = torch.roll(field, shifts=-1, dims=mu)  # psi(x + mu_hat)
    transport = torch.einsum("...ab,...b->...a", self.U[mu], shifted)  # U_mu(x) * psi(x + mu_hat)
    eta_mu = self.eta[mu].unsqueeze(-1)  # η_μ(x)
    return eta_mu * transport  # η_μ(x) * U_μ(x) * ψ(x + μ̂)

# Single hop backward (eta_mu(x-mu_hat) * U_mu†(x-mu_hat) * psi(x - mu_hat))
def _hop_backward(self, field, mu):
    U_dag = self.U[mu].conj().transpose(-2, -1)
    U_shifted = torch.roll(U_dag, shifts=+1, dims=mu)  # U_μ†(x - μ̂)
    f_shifted = torch.roll(field, shifts=+1, dims=mu)  # ψ(x - μ̂)
    transport = torch.einsum("...ab,...b->...a", U_shifted, f_shifted)
    eta_mu = torch.roll(self.eta[mu], shifts=+1, dims=mu)  # η_μ(x - μ̂)
    return eta_mu * transport
```

### 5. s_PT_buffer (nn/s_pt_buffer.py)

```python
class s_PT_buffer(torch.nn.Module):
    """
    Parallel Transport Layer for staggered fermion fields.

    Similar to v_PT for Wilson, but includes staggered eta factors.

    Field shape: (n_features, Lx, Ly, Lz, Lt, 3)
    U shape:    (4, Lx, Ly, Lz, Lt, 3, 3)
    """

    def __init__(self, paths, U, eta=None, lattice_sizes=None):
        """
        Args:
            paths: list of paths, one per feature channel
                   e.g., [[], [(0, 1)], [(1, 1)]] means:
                   - feature 0: identity (no transport)
                   - feature 1: forward hop in mu=0
                   - feature 2: forward hop in mu=1
            U: gauge field
            eta: optional phase factors (computed if None)
        """
```

## Usage Example

```python
import torch
from qcd_ml.nn import s_pt_buffer, dense_staggered

# Gauge field
Lx, Ly, Lz, Lt = 8, 8, 8, 8
U = torch.rand(4, Lx, Ly, Lz, Lt, 3, 3, dtype=torch.cdouble)

# Define paths - one per input feature
# Feature 0: identity
# Features 1-4: single hops in each direction
paths = [
    [],           # feature 0: no transport
    [(0, 1)],    # feature 1: forward hop in mu=0 (x-direction)
    [(1, 1)],    # feature 2: forward hop in mu=1 (y-direction)
    [(2, 1)],    # feature 3: forward hop in mu=2 (z-direction)
    [(3, 1)],    # feature 4: forward hop in mu=3 (t-direction)
]

# Create layers
pt_layer = s_pt_buffer.s_PT_buffer(paths, U)
dense1 = dense_staggered.s_Dense(5, 16)
dense2 = dense_staggered.s_Dense(16, 16)

# Input: 5 features, 8^4 lattice, 3 colors
x = torch.rand(5, Lx, Ly, Lz, Lt, 3, dtype=torch.cdouble)

# Forward pass
x = pt_layer(x)    # Transport each feature along its path
x = dense1(x)      # Linear combination: (n_features_in, 3) → (n_features_out, 3)
x = torch.relu(x.real) + 1j * torch.relu(x.imag)  # Non-linear activation
x = dense2(x)

print(f"Output shape: {x.shape}")  # torch.Size([16, 8, 8, 8, 8, 3])
```

## Comparison with s_PT (nn/pt_staggered.py)

There are two implementations for staggered parallel transport:

| Feature | s_PT | s_PT_buffer |
|---------|------|-------------|
| Pre-computation | No | Partially (U and eta stored) |
| Speed | Slower | Faster |
| Memory | Lower | Higher |
| Interface | Simple | Same |

Both produce identical results. `s_PT_buffer` is useful when the same paths are used repeatedly.

## The Dimension Convention

**Critical**: The lattice dimensions in the field tensor are ordered as:

```python
psi.shape = (n_features, Lx, Ly, Lz, Lt, 3)
# dims:                    0       1   2   3   4   5
```

When specifying paths, `mu` refers to lattice direction:
- `mu=0` → roll in dim 1 (Lx)
- `mu=1` → roll in dim 2 (Ly)
- `mu=2` → roll in dim 3 (Lz)
- `mu=3` → roll in dim 4 (Lt)

The `_hop_forward` function uses `dims=mu` directly because the field tensor lacks the mu dimension that `U[mu]` has.

## Reverse Transport

The reverse operation implements the hermitian adjoint of the forward transport:

```python
forward:  ψ_out(x) = η_μ(x) U_μ(x) ψ_in(x + μ̂)

reverse:  ψ_in(x) = η_μ(x - μ̂) U_μ†(x - μ̂) ψ_out(x - μ̂)
```

This is implemented by:
1. Rolling the field backward
2. Applying U† shifted backward
3. Applying η shifted backward

## Gauge Transformation

To update gauge links during training:

```python
# Create new gauge configuration
U_new = ...  # transformed gauge field

# Update the layer
layer.gauge_transform_using_transformed(U_new)
```

The eta factors are site-dependent only and don't change under gauge transformations.

## Testing

Run the test suite:

```bash
cd /home/vaishakhi/GENN/qcd_ml
python -c "
import sys
sys.path.insert(0, 'src')

import torch
from qcd_ml.nn import pt_staggered, s_pt_buffer

# Create test gauge field with unitary links
Lx = Ly = Lz = Lt = 4
U = torch.rand(4, Lx, Ly, Lz, Lt, 3, 3, dtype=torch.cdouble)

# Normalize to be approximately unitary
for mu in range(4):
    Q, _ = torch.linalg.qr(U[mu].reshape(-1, 3))
    U[mu] = Q.reshape(Lx, Ly, Lz, Lt, 3, 3)

# Test paths
paths = [[], [(0, 1)], [(1, 1)], [(2, 1)], [(3, 1)]]

# Compare implementations
s_pt = pt_staggered.s_PT(paths, U)
s_pt_buf = s_pt_buffer.s_PT_buffer(paths, U)

v = torch.rand(5, Lx, Ly, Lz, Lt, 3, dtype=torch.cdouble)

# Check roundtrip
forward = s_pt_buf(v)
reverse = s_pt_buf.reverse(forward)
diff = torch.max(torch.abs(v - reverse))
print(f'Roundtrip error: {diff.item():.2e}')

# Compare with s_PT
diff_forward = torch.max(torch.abs(s_pt(v) - s_pt_buf(v)))
diff_reverse = torch.max(torch.abs(s_pt.reverse(v) - s_pt_buf.reverse(v)))
print(f's_PT vs s_PT_buffer forward diff: {diff_forward.item():.2e}')
print(f's_PT vs s_PT_buffer reverse diff: {diff_reverse.item():.2e}')
"
```

## Path Lengths vs Naik Term

**Important clarification**:

### Current Implementation: Arbitrary Path Lengths
The current implementation **already supports** arbitrary path lengths:
```python
path = [(0, 1)]      # 1-hop in mu=0
path = [(0, 2)]      # 2-hop in mu=0  
path = [(0, 4)]      # 4-hop in mu=0
path = [(0, 1), (1, 1)]  # L-shaped path
```

These implement **sequential 1-link operations**: applying the 1-link operator multiple times.

### Naik Term: Special 3-Link Product
The Naik term is a **specific physical operator** for improved staggered fermions:
```python
# Naik 3-link: U_μ(x) U_μ(x+μ̂) U_μ(x+2μ̂) ψ(x+3μ̂)
# This is a SINGLE operator, not 3 sequential 1-link hops!

def naik_hop(U, mu, psi):
    """Single 3-link product (not 3 sequential hops)"""
    # Product of 3 links: U(x) @ U(x+μ) @ U(x+2μ)
    U0 = U[mu]
    U1 = torch.roll(U[mu], shifts=-1, dims=mu+1)
    U2 = torch.roll(U[mu], shifts=-2, dims=mu+1)
    U_product = torch.einsum("...ab,...bc,...cd->...ad", U0, U1, U2)
    psi_shifted = torch.roll(psi, shifts=-3, dims=mu+1)
    return torch.einsum("...ab,...b->...a", U_product, psi_shifted)
```

### Why the Difference Matters
```
Sequential 3× 1-link:  U(x)ψ(x+μ) → U(x+μ)ψ(x+2μ) → U(x+2μ)ψ(x+3μ)
                        = U(x+2μ)U(x+μ)U(x) ψ(x+3μ)  ← ORDER MATTERS!

Naik 3-link:          U(x)U(x+μ)U(x+2μ) ψ(x+3μ)
                        = U(x)U(x+μ)U(x+2μ) ψ(x+3μ)  ← Different order!
```

The current implementation does **sequential** hops. For Naik, you'd need a special operator.

### Would Naik Extension Help?
For neural network preconditioners: **Probably not much**.
- The NN learns the effective action anyway
- Sequential hops can approximate Naik behavior
- Naik is mainly important for exact fermion actions

For exact staggered action: **Yes**, if you're solving the Dirac equation exactly.

## Future Extensions

1. **Naik operator**: For improved staggered fermions (if exact action needed)

2. **Fat links**: Separate thin and fat gauge links as in `dirac_staggered`

3. **Multiple eta factors**: For rooted staggered fermions, multiple square-root factors

4. **Optimized batched operations**: Process multiple paths simultaneously
