# Golden Snapshots for Regression Testing

This directory contains golden snapshots for regression testing per DOC-1 Phase A specification (R-PROF-002).

## Snapshot Format

Each snapshot follows the v2 format specification:

```
{snapshot_name}/
├── config.yaml      # Full simulation configuration
├── dose.npy         # Final dose distribution [Nz, Nx]
├── escapes.npy      # Final escape values [5 channels]
├── psi_final.npy    # Optional: Final phase space [Ne, Ntheta, Nz, Nx]
└── metadata.json    # CUDA version, CuPy version, GPU name, timestamp
```

## Predefined Snapshots

### Config-S (config_s/)
- **Purpose**: Quick regression tests
- **Grid**: 32×32 spatial, 32 energy bins, 45 angles
- **Steps**: 20
- **Runtime**: Fast (~1-2 seconds)

### Config-M (config_m/)
- **Purpose**: Standard validation
- **Grid**: 64×64 spatial, 64 energy bins, 90 angles
- **Steps**: 20
- **Runtime**: Medium (~10-20 seconds)

### Config-L (config_l/)
- **Purpose**: Comprehensive validation
- **Grid**: 128×128 spatial, 100 energy bins, 180 angles
- **Steps**: 20
- **Runtime**: Long (~1-2 minutes)

## Generating New Snapshots

### Generate All Predefined Snapshots

```bash
python validation/generate_snapshots.py
```

### Generate a Specific Snapshot

```python
from validation.generate_snapshots import generate_snapshot, CONFIG_S

snapshot_path = generate_snapshot(
    config_name='my_snapshot',
    grid_specs=CONFIG_S,
    simulation_config={},
    n_steps=20,
)
```

### Generate Custom Snapshot

```python
from validation.generate_snapshots import generate_custom_snapshot

config = {
    "Nx": 48, "Nz": 48, "Ne": 100, "Ntheta": 72,
    "E_min": 1.0, "E_cutoff": 2.0, "E_max": 70.0,
    "delta_s": 0.5, "sync_interval": 0,
}

path = generate_custom_snapshot(
    name="proton_70MeV",
    config=config,
    n_steps=20,
    description="70 MeV proton therapy beam",
)
```

## Using Snapshots for Regression Testing

```python
from validation.compare import GoldenSnapshot, compare_results, ToleranceConfig

# Load snapshot
snapshot = GoldenSnapshot.load(
    Path("validation/golden_snapshots"),
    "config_s"
)

# Run your simulation
sim = create_simulation(**CONFIG_S)
result = sim.run(n_steps=20)
dose_test = result.dose_final
escapes_test = sim.accumulators.get_escapes_cpu()

# Compare against golden snapshot
comparison = compare_results(
    dose_test=dose_test,
    escapes_test=escapes_test,
    snapshot=snapshot,
    tolerance=ToleranceConfig.normal(),
)

# Check results
print(comparison)
assert comparison.passed, "Regression detected!"
```

## Metadata Format

The `metadata.json` file contains:

```json
{
  "description": "Config-S: Small grid (32x32) for quick regression tests",
  "n_steps": 20,
  "runtime_seconds": 1.234,
  "conservation_valid": true,
  "timestamp": "2025-01-15T10:30:00.123456",
  "gpu_name": "NVIDIA GeForce RTX 3080",
  "cuda_version": "12.2",
  "cupy_version": "12.0.0",
  "version": "2.0"
}
```

## Pass/Fail Criteria

Per R-PROF-003 specification:

- **Dose L2 error**: `< 1e-4`
- **Dose max error**: `< 1e-4`
- **Escape relative error**: `< 1e-5`
- **Weight relative error**: `< 1e-6`

### Tolerance Adjustments

- **Different GPU**: Multiply tolerances by 10
- **Debug mode**: Divide tolerances by 10

```python
# Loose tolerance (different GPU)
tolerance = ToleranceConfig.loose()

# Strict tolerance (debug mode)
tolerance = ToleranceConfig.strict()

# Normal tolerance (same GPU, production)
tolerance = ToleranceConfig.normal()
```

## File Format Versions

### V2 Format (Current Standard)
- Separate files for dose, escapes, psi_final
- JSON metadata
- Per R-PROF-002 specification

### V1 Format (Legacy)
- Combined `results.npz` file
- YAML metadata
- Still supported for backward compatibility

## Requirements

### Generation Conditions (R-PROF-002)

- **GPU**: NVIDIA RTX 3080 or equivalent
- **Driver**: CUDA 12.x
- **CuPy**: Version specified in metadata
- **Determinism**: No random seeds (deterministic algorithm)

### Reproducibility

For bitwise reproducibility (V-PROF-001):
- Same GPU model
- Same CUDA version
- Same CuPy version
- Same configuration

## Troubleshooting

### Snapshots Not Reproducing

If you get different results on different hardware:
1. Check GPU model in metadata
2. Use loose tolerance: `ToleranceConfig.loose()`
3. Verify CUDA/CuPy versions match

### Conservation Check Failing

If conservation check fails:
1. Check E_cutoff > E_min
2. Verify delta_s <= min(delta_x, delta_z)
3. Review escape channel values

## Reference

- **Specification**: DOC-1 Phase A, Section A-3 (R-PROF-002, R-PROF-003)
- **Validation**: Section A-4 (V-PROF-001)
- **Implementation**: `/workspaces/Smatrix_2D/validation/generate_snapshots.py`
