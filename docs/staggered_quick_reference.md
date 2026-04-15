# Staggered Fermion Quick Reference

## Field Shapes

| Type | Wilson | Staggered |
|------|--------|-----------|
| Fermion field | `(Lx, Ly, Lz, Lt, 4, 3)` | `(Lx, Ly, Lz, Lt, 3)` |
| With features | `(n_feat, Lx, Ly, Lz, Lt, 4, 3)` | `(n_feat, Lx, Ly, Lz, Lt, 3)` |
| Gauge links | `(4, Lx, Ly, Lz, Lt, 3, 3)` | Same |

## Key Difference: η Factors

```
Wilson:   result(x) = U_path(x) * ψ(source)
Staggered: result(x) = η_μ(x) * U_μ(x) * ψ(x + μ̂)
```

**η factors are site-dependent, so they must be applied at every hop.**

## Quick Import

```python
from qcd_ml.nn import s_pt_buffer, dense_staggered
from qcd_ml.nn import pt_staggered  # alternative implementation
from qcd_ml.base.paths import StaggeredPathBuffer
from qcd_ml.base.operations import make_eta
```

## Creating a Staggered Network

```python
import torch
from qcd_ml.nn import s_pt_buffer, dense_staggered

# Gauge field
U = torch.rand(4, 8, 8, 8, 8, 3, 3, dtype=torch.cdouble)

# Paths for parallel transport
paths = [
    [],           # feature 0: identity
    [(0, 1)],    # feature 1: hop +x
    [(1, 1)],    # feature 2: hop +y
    [(2, 1)],    # feature 3: hop +z
    [(3, 1)],    # feature 4: hop +t
]

# Layers
pt = s_pt_buffer.s_PT_buffer(paths, U)
dense1 = dense_staggered.s_Dense(5, 16)
dense2 = dense_staggered.s_Dense(16, 5)

# Input
x = torch.rand(5, 8, 8, 8, 8, 3, dtype=torch.cdouble)

# Forward
x = pt(x)       # shape: (5, 8, 8, 8, 8, 3)
x = dense1(x)   # shape: (16, 8, 8, 8, 8, 3)
```

## Path Format

```python
path = [(mu, nhops), ...]

# Examples:
[]           # identity (no transport)
[(1, 1)]    # forward hop in mu=1 direction
[(1, -1)]   # backward hop in mu=1 direction
[(0, 2)]    # two hops forward in mu=0 direction
[(0, 1), (1, 1)]  # two sequential hops
[(0, 1), (1, -1)] # hop +x then -y
```

## Dimension Mapping

```
Field tensor: (n_feat, Lx, Ly, Lz, Lt, 3)
              dim:      0    1   2   3   4   5

mu=0 → dim 1 (rolls in Lx direction)
mu=1 → dim 2 (rolls in Ly direction)
mu=2 → dim 3 (rolls in Lz direction)
mu=3 → dim 4 (rolls in Lt direction)
```

## Available Classes

### Neural Network Layers

| Class | File | Description |
|-------|------|-------------|
| `s_PT` | `nn/pt_staggered.py` | Basic staggered PT layer |
| `s_PT_buffer` | `nn/s_pt_buffer.py` | Buffered staggered PT layer |
| `s_Dense` | `nn/dense_staggered.py` | Dense layer for staggered fields |

### Low-Level Utilities

| Function | File | Description |
|----------|------|-------------|
| `make_eta` | `base/operations.py` | Create staggered phase factors |
| `stag_v_gauge_transform` | `base/operations.py` | Gauge transform vector field |
| `stag_m_gauge_transform` | `base/operations.py` | Gauge transform matrix field |
| `StaggeredPathBuffer` | `base/paths/staggered_path_buffer.py` | Path transport buffer |

## Testing Roundtrip

```python
import torch
from qcd_ml.nn import s_pt_buffer

U = torch.rand(4, 4, 4, 4, 4, 3, 3, dtype=torch.cdouble)
paths = [[], [(1, 1)], [(2, 1)]]
layer = s_pt_buffer.s_PT_buffer(paths, U)

v = torch.rand(3, 4, 4, 4, 4, 3, dtype=torch.cdouble)

# Forward and reverse
v_out = layer(v)
v_back = layer.reverse(v_out)

# Check roundtrip
diff = torch.max(torch.abs(v - v_back))
print(f"Roundtrip error: {diff.item():.2e}")  # Should be ~0
```

## Gauge Transformation

```python
# Create new gauge field
U_new = ...  # transformed U

# Update layer (eta unchanged)
layer.gauge_transform_using_transformed(U_new)
```

## Comparison: s_PT vs s_PT_buffer

| Aspect | s_PT | s_PT_buffer |
|--------|------|--------------|
| Implementation | Direct | Buffer-based |
| Speed | Slower | Faster for reuse |
| Memory | Lower | Higher |
| Results | Identical | Identical |

Both implementations are mathematically equivalent.

## Path Lengths

Current implementation supports **arbitrary path lengths**:

```python
# Examples:
[(0, 1)]      # 1-hop forward in x
[(0, 2)]      # 2-hop forward in x
[(0, -1)]     # 1-hop backward in x
[(0, 1), (1, 1)]  # L-shaped: +x then +y
```

**Note**: These are sequential 1-link hops. The Naik 3-link term (improved staggered) is a special single operator, not implemented yet.

## Common Patterns

### Single Hop Parallel Transport
```python
paths = [[], [(0, 1)], [(1, 1)], [(2, 1)], [(3, 1)]]
# Standard 1-hop stencil including identity
```

### Multi-Hop Stencil
```python
paths = [
    [],
    [(0, 1)], [(0, -1)],
    [(1, 1)], [(1, -1)],
    [(2, 1)], [(2, -1)],
    [(3, 1)], [(3, -1)],
]
# 1-hop in all 8 directions
```

### Complex Path
```python
paths = [
    [],                            # identity
    [(0, 1), (1, 1)],            # L-shaped path
    [(0, 2), (1, -1), (2, 1)],   # 3-step path
]
```

## Error Checking

```python
# Shape validation
assert x.shape[0] == layer.n_feature_in, "Feature count mismatch"
assert x.shape[1:] == (Lx, Ly, Lz, Lt, 3), "Lattice shape mismatch"

# Check for NaN/Inf
assert torch.isfinite(x).all(), "NaN or Inf detected"

# Check gauge invariance (advanced)
# Transport field, gauge transform, transport back
# Should give same result
```
