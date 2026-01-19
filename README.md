# Smatrix_2D: Operator-Factorized 2D Proton Transport System

**Version**: 7.2
**Status**: Implementation Complete (Alpha)
**Last Updated**: 2026-01-19

## Overview

A deterministic transport engine for proton beam simulation using operator factorization. The system simulates proton beam transport through water using continuous slowing-down approximation (CSDA) and multiple Coulomb scattering (MCS) with strict probability conservation.

**Key Features**:
- **Operator factorization**: `psi_next = A_s(A_E(A_theta(psi)))`
- **No global S-matrix construction**: Memory-efficient, GPU-friendly
- **GPU-optimized memory layout**: `psi[E, theta, z, x]` with canonical ordering
- **Physics-validated**: Uses NIST PSTAR stopping power data and Highland scattering formula
- **Exact mass conservation**: With escape tracking across 4 channels

### Physics Implementation

The simulation models proton transport through a 4D phase space:
- **x**: Lateral position [mm]
- **z**: Depth position [mm] (beam direction)
- **theta**: Scattering angle [degrees] (0° to 180°)
- **E**: Proton energy [MeV]

**Transport operators applied sequentially:**
1. **Angular Scattering (A_θ)**: Multiple Coulomb scattering via Highland formula
2. **Energy Loss (A_E)**: CSDA with NIST PSTAR stopping power data
3. **Spatial Streaming (A_s)**: Bilinear interpolation advection

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Smatrix_2D

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

**Requirements**:
- Python >= 3.8
- NumPy
- CuPy (optional, for GPU acceleration)
- PyYAML
- h5py

---

## Quick Start

### Running a Simulation

The main simulation entry point is `run_simulation.py`:

```bash
python run_simulation.py initial_info.yaml
```

### Configuration File

The simulation is configured via `initial_info.yaml`:

```yaml
# Grid parameters
grid:
  Nx: 12           # Lateral grid points
  Nz: 60           # Depth grid points
  Ntheta: 41       # Angular grid points
  Ne: 35           # Energy grid points
  x_min: 0.0       # [mm]
  x_max: 12.0      # [mm]
  z_min: 0.0       # [mm]
  z_max: 60.0      # [mm]
  E_min: 0.1       # [MeV]
  E_max: 70.0      # [MeV]
  E_cutoff: 0.1    # [MeV]

# Beam parameters
beam:
  E_init: 70.0     # Initial energy [MeV]
  theta_init: 0.0 # Initial angle [degrees] (forward along +z)
  beam_width_sigma: 1.0  # Gaussian width [mm]

# Transport parameters
transport:
  delta_s: auto     # Step length [mm] (auto or manual)
  n_buckets: 32    # Sigma bucket count for scattering
```

### Python API Usage

```python
from smatrix_2d.transport import create_simulation
from smatrix_2d.config import load_config

# Load configuration
config = load_config("initial_info.yaml")

# Create simulation
sim = create_simulation(config)

# Run simulation
result = sim.run()

# Access results
dose_profile = result.dose
bragg_peak_z = result.bragg_peak_position
```

---

## Physics Documentation

For detailed physics implementation, see **[PHYSICS_INTERACTIONS.md](PHYSICS_INTERACTIONS.md)**.

### Physics Summary

| Physics | Implementation | Data Source |
|---------|----------------|-------------|
| **Stopping Power** | CSDA with NIST PSTAR LUT | NIST PSTAR database |
| **Multiple Coulomb Scattering** | Highland formula | Molière theory |
| **Scattering Kernel** | Gaussian convolution | High-energy approximation |
| **Energy Grid** | Range-based logarithmic | CSDA range calculation |
| **Radiation Length** | X₀ = 360.8 mm (water) | ICRU Report 49 |

### Key Equations

**Angular Scattering (Highland formula)**:
```
sigma_theta = (13.6 MeV / beta*cp) * sqrt(L/X0) * [1 + 0.038*ln(L/X0)]
```

**Energy Loss (CSDA)**:
```
dE/ds = -S(E)
E_new = E_old - S(E) * delta_s
```

**Spatial Advection**:
```
r_new = r_old + v * delta_s
v = (sin(theta), cos(theta))  # vx=sin for lateral (x), vz=cos for forward (z)
```

---

## Implementation Structure

```
Smatrix_2D/
├── smatrix_2d/
│   ├── __init__.py
│   ├── core/
│   │   ├── constants.py          # PhysicsConstants2D
│   │   ├── grid.py              # GridSpecs2D, PhaseSpaceGrid2D
│   │   ├── state.py             # TransportState
│   │   └── config.py            # Configuration loading
│   ├── operators/
│   │   ├── angular_scattering.py # A_theta operator
│   │   ├── spatial_streaming.py  # A_s operator
│   │   ├── energy_loss.py       # A_E operator
│   │   └── sigma_buckets.py     # Scattering kernel LUT
│   ├── transport/
│   │   ├── simulation.py        # Main simulation loop
│   │   └── runners/             # Workflow orchestration
│   ├── gpu/
│   │   ├── kernels.py           # GPU transport step
│   │   ├── accumulators.py      # Zero-sync accumulators
│   │   ├── cuda_kernels/        # CUDA kernel sources
│   │   │   ├── angular_scattering.cu
│   │   │   ├── energy_loss.cu
│   │   │   └── spatial_streaming.cu
│   │   └── kernel_loader.py     # Dynamic kernel loading
│   ├── physics_data/
│   │   ├── fetchers/            # NIST data fetching
│   │   └── processors/          # LUT generation
│   └── data/
│       ├── raw/                 # Original NIST data
│       └── processed/           # Generated LUT files
├── run_simulation.py            # Main entry point
├── initial_info.yaml            # Configuration file
├── PHYSICS_INTERACTIONS.md      # Detailed physics documentation
└── README.md                    # This file
```

---

## Output Files

The simulation generates several output files in the `output/` directory:

| File | Description |
|------|-------------|
| `dose_profile.h5` | 3D dose distribution [MeV] |
| `lateral_profile_per_step.csv` | Lateral beam profile evolution |
| `transport_statistics.json` | Mass conservation, escape channels |
| `checkpoints/*.npz` | Simulation state checkpoints |

### Escape Channels

Mass conservation is tracked through 4 escape channels:

| Index | Channel | Description |
|-------|---------|-------------|
| 0 | `theta_cutoff` | Scattering kernel truncation |
| 1 | `theta_boundary` | Angular domain boundary loss |
| 2 | `energy_stopped` | Particles reaching E_cutoff |
| 3 | `spatial_leak` | Particles leaving spatial domain |

**Conservation equation**: `mass_in = mass_out + sum(escapes)`

---

## GPU Acceleration

### GPU Implementation

The codebase includes GPU-accelerated kernels using CuPy:

```python
from smatrix_2d.gpu import create_gpu_transport_step

if GPU_AVAILABLE:
    gpu_step = create_gpu_transport_step(
        Ne=100, Ntheta=72, Nz=100, Nx=50,
    )

    # Run on GPU
    psi_out, escapes, dose = gpu_step.apply_step(
        psi_gpu, E_grid, sigma_theta, theta_beam, delta_s,
        stopping_power, E_cutoff,
    )
```

### GPU Kernels

| Kernel | File | Features |
|--------|------|----------|
| **A_θ** | `angular_scattering.cu` | FFT-based convolution, sparse gather |
| **A_E** | `energy_loss.cu` | Binary search LUT, bin splitting, dose deposition |
| **A_s** | `spatial_streaming.cu` | Bilinear scatter, atomic accumulation |

### Memory Layout

Canonical GPU layout: `psi[E, theta, z, x]`

**Performance characteristics**:
- Expected speedup: 10-30× vs NumPy
- Memory bandwidth: ~400 GB/s on RTX 3090
- Atomic overhead: <15% with deterministic mode

---

## Validation

### Physics Verification

Compare against established benchmarks:

| Metric | Expected (70 MeV) | Verification Source |
|--------|------------------|---------------------|
| **Bragg Peak (R90)** | ~40 mm | NIST PSTAR CSDA range |
| **Stopping Power** | 13.7 MeV²cm²/g | NIST PSTAR at 70 MeV |
| **Lateral Spread** | x_rms ~1-3 mm | MC simulations |

**For detailed validation procedures**, see PHYSICS_INTERACTIONS.md.

### Conservation Tests

The system includes automatic validation:
- **Column sum verification**: `sum(psi_out) = sum(psi_in) - escapes`
- **Positivity check**: `psi >= 0` everywhere
- **Mass balance**: All 4 escape channels tracked

---

## Configuration Parameters

### Grid Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `Nx` | 12 | 6-24 | Lateral grid points |
| `Nz` | 60 | 30-120 | Depth grid points |
| `Ntheta` | 41 | 21-181 | Angular grid points |
| `Ne` | 35 | 20-100 | Energy grid points |
| `x_max` | 12 mm | 6-24 mm | Lateral domain extent |
| `z_max` | 60 mm | 30-120 mm | Depth domain extent |
| `E_max` | 70 MeV | 50-250 MeV | Maximum energy |
| `E_cutoff` | 0.1 MeV | 0.01-1 MeV | Energy cutoff |

### Beam Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `E_init` | 70.0 MeV | 50-250 MeV | Initial beam energy |
| `theta_init` | 0.0° | -180 to 180° | Initial beam angle (0° = forward along +z) |
| `beam_width_sigma` | 1.0 mm | 0.5-5 mm | Gaussian beam width |

### Transport Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `delta_s` | auto | 0.1-2 mm | Step length |
| `n_buckets` | 32 | 16-64 | Sigma bucket count |
| `k_cutoff` | 5.0 | 3-7 | Kernel cutoff [sigma] |

---

## Current Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Numerical diffusion** | Grid-based characteristic | Use finer grids |
| **Ray effect** | Coarse angular resolution | Use Ntheta >= 72 |
| **No energy straggling** | Missing stochastic energy loss | Planned for v8.0 |
| **No nuclear interactions** | Missing fragmentation channels | Planned for future |
| **2D geometry** | Azimuthal symmetry only | Use for symmetric problems |

---

## Future Work

- [ ] Energy straggling models (Vavilov distribution)
- [ ] Nuclear interaction operators
- [ ] 3D generalization
- [ ] Adaptive angular quadrature
- [ ] Multi-material support
- [ ] Multi-GPU support

---

## Physics Data Sources

### NIST PSTAR
**Stopping Power Data**: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
- Material: Liquid water (H2O)
- Energy range: 0.01 - 200 MeV
- Processed to: `data/processed/stopping_power_water.npy`

### ICRU Report 49
**Radiation Length**: X₀ = 360.8 mm for water

### PDG
**Fundamental Constants**: https://pdg.lbl.gov/
- Proton mass: 938.27 MeV/c²
- Highland constant: 13.6 MeV

---

## References

### Documentation
- **PHYSICS_INTERACTIONS.md**: Detailed physics implementation guide
- `spec.md`: Implementation specification v7.2

### Physics Theory
1. Bethe-Bloch equation for stopping power
2. Molière theory for multiple Coulomb scattering
3. Highland formula for practical MCS approximation
4. CSDA (ICRU Report 37)

### Code Implementation
| File | Purpose |
|------|---------|
| `smatrix_2d/operators/angular_scattering.py` | CPU reference for scattering |
| `smatrix_2d/operators/energy_loss.py` | CPU reference for energy loss |
| `smatrix_2d/operators/spatial_streaming.py` | CPU reference for streaming |
| `smatrix_2d/gpu/cuda_kernels/*.cu` | GPU kernel implementations |
| `smatrix_2d/transport/simulation.py` | Main simulation loop |

---

## License

Same as parent Smatrix project.

---

*For detailed physics implementation, operator-by-operator explanations, and verification procedures, see [PHYSICS_INTERACTIONS.md](PHYSICS_INTERACTIONS.md).*
