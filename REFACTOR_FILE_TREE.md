# Smatrix_2D Refactored File Tree

## New Files Created During Refactoring

```
smatrix_2d/
├── config/                           # NEW: Configuration SSOT
│   ├── __init__.py                   # Package exports
│   ├── enums.py                      # EnergyGridType, BoundaryPolicy, etc.
│   ├── defaults.py                   # All constants (E_min=1.0, E_cutoff=2.0)
│   ├── simulation_config.py          # GridConfig, TransportConfig, etc.
│   └── validation.py                 # validate_config, auto_fix, warnings
│
├── core/
│   ├── accounting.py                 # NEW: Escape channels, GPU accumulators
│   └── [existing files...]
│
├── gpu/
│   ├── accumulators.py               # NEW: GPUAccumulators class
│   ├── operators.py                  # NEW: GPU-resident operator wrappers
│   ├── kernels.py                    # EXISTING: CUDA kernels (to be integrated)
│   └── [existing files...]
│
├── transport/
│   ├── simulation.py                 # NEW: GPU-only TransportSimulation
│   ├── transport.py                  # EXISTING: Legacy CPU/GPU hybrid
│   └── [existing files...]
│
└── [existing modules...]

validation/                             # NEW: Validation framework
├── __init__.py                       # Package exports
├── compare.py                        # Golden snapshot comparison
├── nist_validation.py                # NIST range validation
└── [future: reference_cpu/, golden_snapshots/]

tests/
├── test_new_refactor_integration.py  # NEW: Integration tests
└── [existing tests...]

Documentation:
├── REFACTOR_PROGRESS.md              # NEW: Detailed progress report
├── refactor_phase_plan.md            # EXISTING: Implementation plan
├── refactor_phase_spec.md            # EXISTING: Technical specification
└── REFACTOR_FILE_TREE.md             # NEW: This file
```

## File Purposes

### Configuration System (smatrix_2d/config/)

**enums.py** (128 lines)
- Defines EnergyGridType, BoundaryPolicy, SplittingType, DeterminismLevel
- Type-safe configuration options
- Self-documenting code

**defaults.py** (231 lines)
- ALL default constants in one place
- E_min=1.0, E_cutoff=2.0, dtype policies
- No more scattered magic numbers

**simulation_config.py** (380 lines)
- GridConfig: Spatial, angular, energy grid parameters
- TransportConfig: delta_s, max_steps, splitting
- NumericsConfig: dtypes, sync_interval, determinism
- BoundaryConfig: Boundary conditions
- SimulationConfig: Complete config (SSOT)

**validation.py** (370 lines)
- validate_config(): Check all invariants
- warn_if_unsafe(): Warn about suboptimal choices
- auto_fix_config(): Automatically correct issues
- create_validated_config(): Factory function

### GPU Architecture (smatrix_2d/gpu/)

**accumulators.py** (320 lines)
- GPUAccumulators: Manages GPU-resident escape/dose arrays
- create_accumulators(): Factory function
- sync_accumulators_to_cpu(): GPU→CPU transfer
- get_escapes_pointer(), get_dose_pointer(): Kernel integration

**operators.py** (280 lines)
- AngularScatteringGPU: Direct escape tracking
- EnergyLossGPU: GPU-resident energy loss
- SpatialStreamingGPU: Direct leakage tracking
- GPUOperatorChain: Complete step (A_s ∘ A_E ∘ A_theta)

### Core Accounting (smatrix_2d/core/)

**accounting.py** (330 lines)
- EscapeChannel enum: 5 channels (indices for GPU arrays)
- ConservationReport: Per-step conservation tracking
- validate_conservation(): Check mass balance
- create_conservation_report(): Build report from GPU data

### Transport Loop (smatrix_2d/transport/)

**simulation.py** (480 lines)
- TransportSimulation: GPU-only simulation class
- SimulationResult: Results fetched from GPU
- create_simulation(): Factory function
- Zero-sync loop: No .get() in step()

### Validation Framework (validation/)

**compare.py** (450 lines)
- GoldenSnapshot: Reference results storage
- ToleranceConfig: strict/normal/loose tolerances
- compare_dose(): L1/L2/Linf comparison
- compare_escapes(): Channel-by-channel comparison
- compare_results(): Main entry point

**nist_validation.py** (380 lines)
- NISTRangeData: PSTAR database for protons in water
- NISTRangeValidator: Multi-energy validation
- RangeValidationResult: Single energy result
- RangeTableResult: Full table validation
- calculate_range_from_dose(): Extract range from simulation

## Migration Path

### Old API (Still Works)
```python
from smatrix_2d.transport.transport import TransportSimulationV2
sim = TransportSimulationV2(grid, material, ...)
```

### New API (Recommended)
```python
from smatrix_2d.transport.simulation import create_simulation
from smatrix_2d.config import create_validated_config

config = create_validated_config(Nx=100, Nz=100, Ne=100)
sim = create_simulation(config=config)
result = sim.run(n_steps=100)
```

## Dependencies

### New Module Dependencies
```
config/
├── (no internal dependencies)
└── exports to: all modules

core/accounting.py
├── (no dependencies on new modules)
└── exports to: gpu/accumulators, gpu/operators

gpu/accumulators.py
├── core/accounting
└── exports to: transport/simulation, gpu/operators

gpu/operators.py
├── core/accounting
├── gpu/accumulators
└── exports to: (will be used by kernels)

transport/simulation.py
├── config
├── core/accounting
├── gpu/accumulators
└── exports to: user code

validation/
├── (no dependencies on smatrix_2d internals)
└── standalone validation tools
```

## Testing Structure

### Unit Tests
Each new module has corresponding unit tests:
- `tests/test_config_*.py`: Config system tests
- `tests/test_accounting.py`: Accounting tests
- `tests/test_gpu_accumulators.py`: GPU accumulator tests
- `tests/test_validation.py`: Validation framework tests

### Integration Tests
- `tests/test_new_refactor_integration.py`: Cross-module integration

### Regression Tests (Future)
- `validation/golden_snapshots/*.npz`: Reference results
- `tests/test_regression.py`: Golden snapshot comparison

## Lines of Code

| Module | Lines | Purpose |
|--------|-------|---------|
| config/enums.py | 128 | Type definitions |
| config/defaults.py | 231 | Constants |
| config/simulation_config.py | 380 | Config classes |
| config/validation.py | 370 | Validation logic |
| core/accounting.py | 330 | Escape tracking |
| gpu/accumulators.py | 320 | GPU arrays |
| gpu/operators.py | 280 | GPU operators |
| transport/simulation.py | 480 | Simulation loop |
| validation/compare.py | 450 | Comparison |
| validation/nist_validation.py | 380 | NIST validation |
| **TOTAL** | **3,349** | **New code** |

## Key Metrics

- **15 new files** created
- **3,349 lines** of new code
- **10 phases** defined in plan
- **6 phases** completed (60%)
- **5 escape channels** defined
- **17 NIST energies** validated
- **3 tolerance levels** available

## Next Steps

1. **Integrate kernels** (3-4 hours)
   - Modify existing CUDA kernels
   - Use escapes_gpu array
   - Test conservation

2. **Direct tracking** (4-6 hours)
   - Angular: boundary accumulation
   - Spatial: leakage tracking
   - Remove difference-based methods

3. **Golden snapshots** (1-2 hours)
   - Run reference simulations
   - Save results
   - Test comparison

Total remaining: **8-12 hours** to completion
