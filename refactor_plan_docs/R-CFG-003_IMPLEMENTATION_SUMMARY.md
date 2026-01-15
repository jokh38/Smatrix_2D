# R-CFG-003: SimulationConfig → GridSpecsV2 Factory Functions

## Implementation Status: ✅ COMPLETE

**Date:** 2026-01-15
**Location:** `/workspaces/Smatrix_2D/smatrix_2d/core/grid.py`
**Requirement:** R-CFG-003 from refactoring plan

---

## Overview

Implemented bidirectional factory functions to convert between `SimulationConfig` (SSOT configuration) and `GridSpecsV2` (grid specification). This enables seamless integration between the configuration system and the core grid infrastructure.

---

## Factory Functions

### 1. `GridSpecsV2.from_simulation_config()`

**Type:** Classmethod
**Location:** `smatrix_2d/core/grid.py:142-201`
**Purpose:** Convert SimulationConfig to GridSpecsV2

**Signature:**
```python
@classmethod
def from_simulation_config(cls, config: SimulationConfig) -> GridSpecsV2
```

**Features:**
- Extracts all 13 grid parameters from `SimulationConfig.grid`
- Calculates `delta_x` and `delta_z` from spatial domain bounds
- Handles enum conversion between `config.enums.EnergyGridType` and `core.grid.EnergyGridType`
- Type-safe with `TypeError` for invalid inputs
- Sets `use_texture_memory=False` by default

**Parameters Extracted:**
| Parameter | Source | Description |
|-----------|--------|-------------|
| Nx | grid.Nx | Number of x spatial bins |
| Nz | grid.Nz | Number of z spatial bins |
| Ntheta | grid.Ntheta | Number of angular bins |
| Ne | grid.Ne | Number of energy bins |
| x_min | grid.x_min | Minimum x coordinate [mm] |
| x_max | grid.x_max | Maximum x coordinate [mm] |
| z_min | grid.z_min | Minimum z coordinate [mm] |
| z_max | grid.z_max | Maximum z coordinate [mm] |
| theta_min | grid.theta_min | Minimum angle [degrees] |
| theta_max | grid.theta_max | Maximum angle [degrees] |
| E_min | grid.E_min | Minimum energy [MeV] |
| E_max | grid.E_max | Maximum energy [MeV] |
| E_cutoff | grid.E_cutoff | Energy cutoff [MeV] |
| delta_x | **CALCULATED** | (x_max - x_min) / Nx |
| delta_z | **CALCULATED** | (z_max - z_min) / Nz |

**Example Usage:**
```python
from smatrix_2d.config.simulation_config import SimulationConfig
from smatrix_2d.core.grid import GridSpecsV2

# Create configuration
config = SimulationConfig()

# Convert to GridSpecsV2
grid_specs = GridSpecsV2.from_simulation_config(config)

# Use grid specs
print(f"Grid: {grid_specs.Nx}x{grid_specs.Nz} bins")
print(f"Resolution: delta_x={grid_specs.delta_x} mm")
```

---

### 2. `GridSpecsV2.to_simulation_config()`

**Type:** Instance method
**Location:** `smatrix_2d/core/grid.py:203-245`
**Purpose:** Convert GridSpecsV2 to SimulationConfig

**Signature:**
```python
def to_simulation_config(self) -> SimulationConfig
```

**Features:**
- Creates `GridConfig` from `GridSpecsV2` instance
- Converts `core.grid.EnergyGridType` to `config.enums.EnergyGridType`
- Other config sections (transport, numerics, boundary) use defaults
- Returns fully valid `SimulationConfig` instance

**Example Usage:**
```python
from smatrix_2d.core.grid import GridSpecsV2

# Create GridSpecsV2 directly
grid_specs = GridSpecsV2(
    Nx=128, Nz=128,
    Ntheta=180, Ne=100,
    delta_x=0.5, delta_z=0.5,
    x_min=-32.0, x_max=32.0,
    z_min=-32.0, z_max=32.0,
)

# Convert to SimulationConfig
config = grid_specs.to_simulation_config()

# Validate configuration
errors = config.validate()
if errors:
    for err in errors:
        print(f"Error: {err}")
```

---

## Technical Details

### Enum Conversion

The implementation handles the fact that `EnergyGridType` exists in two modules:
- `smatrix_2d.config.enums.EnergyGridType` (configuration layer)
- `smatrix_2d.core.grid.EnergyGridType` (core layer)

**Solution:** Convert via string values:
```python
# Config → Grid
grid_type = GridEnergyGridType(config.energy_grid_type.value)

# Grid → Config
config_type = ConfigEnergyGridType(grid.energy_grid_type.value)
```

### Delta Calculation

Spatial resolution is calculated from domain bounds:
```python
delta_x = (x_max - x_min) / Nx
delta_z = (z_max - z_min) / Nz
```

This ensures consistency with the actual grid spacing.

### Type Safety

Both factory functions include type checking:
```python
if not isinstance(config, SimulationConfig):
    raise TypeError(f"Expected SimulationConfig, got {type(config).__name__}")
```

---

## Testing Results

All tests passed successfully:

✅ **Test 1:** Basic conversion (SimulationConfig → GridSpecsV2)
✅ **Test 2:** Reverse conversion (GridSpecsV2 → SimulationConfig)
✅ **Test 3:** Round-trip preservation (all 13 parameters)
✅ **Test 4:** Type safety (TypeError for invalid inputs)
✅ **Test 5:** Delta calculation correctness
✅ **Test 6:** Custom configuration support
✅ **Test 7:** Integration with PhaseSpaceGrid
✅ **Test 8:** EnergyGridType enum conversion

**Round-trip Verification:**
```
Original:  Nx=100, Nz=100, Ntheta=180, Ne=100
           x=[-50, 50], z=[-50, 50]
           theta=[0, 180], E=[1, 100], E_cutoff=2.0

           ↓ (from_simulation_config)

GridSpecsV2: Nx=100, Nz=100, Ntheta=180, Ne=100
             x=[-50, 50], z=[-50, 50]
             theta=[0, 180], E=[1, 100], E_cutoff=2.0

             ↓ (to_simulation_config)

Reconstructed: Nx=100, Nz=100, Ntheta=180, Ne=100
               x=[-50, 50], z=[-50, 50]
               theta=[0, 180], E=[1, 100], E_cutoff=2.0

Result: ✓ All parameters preserved
```

---

## Usage Examples

### Example 1: Load Configuration from File

```python
import yaml
from smatrix_2d.config.simulation_config import SimulationConfig
from smatrix_2d.core.grid import GridSpecsV2, create_phase_space_grid

# Load configuration from YAML
with open('simulation_config.yaml') as f:
    config_dict = yaml.safe_load(f)

# Create SimulationConfig
config = SimulationConfig.from_dict(config_dict)

# Convert to GridSpecsV2
grid_specs = GridSpecsV2.from_simulation_config(config)

# Create phase space grid
phase_space = create_phase_space_grid(grid_specs)
print(f"Phase space shape: {phase_space.shape}")
```

### Example 2: Direct GridSpecsV2 Creation

```python
from smatrix_2d.core.grid import GridSpecsV2
from smatrix_2d.config.simulation_config import SimulationConfig

# Create GridSpecsV2 directly
grid_specs = GridSpecsV2(
    Nx=200, Nz=150,
    Ntheta=90, Ne=50,
    delta_x=0.3, delta_z=0.53,
    x_min=-30.0, x_max=30.0,
    z_min=-40.0, z_max=40.0,
)

# Convert to SimulationConfig for validation
config = grid_specs.to_simulation_config()
errors = config.validate()

if not errors:
    print("Configuration is valid!")
```

### Example 3: Workflow Integration

```python
from smatrix_2d.config.simulation_config import SimulationConfig
from smatrix_2d.core.grid import GridSpecsV2, create_phase_space_grid

def create_simulation_from_config(config_dict):
    """Complete workflow from config dict to phase space grid."""

    # Step 1: Create SimulationConfig
    config = SimulationConfig.from_dict(config_dict)

    # Step 2: Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {errors}")

    # Step 3: Convert to GridSpecsV2
    grid_specs = GridSpecsV2.from_simulation_config(config)

    # Step 4: Create phase space grid
    phase_space = create_phase_space_grid(grid_specs)

    return config, grid_specs, phase_space

# Usage
config, grid_specs, phase_space = create_simulation_from_config({
    'grid': {
        'Nx': 150,
        'Nz': 150,
        'Ntheta': 180,
        'Ne': 100,
    }
})
```

---

## Files Modified

1. **`/workspaces/Smatrix_2D/smatrix_2d/core/grid.py`**
   - Added `GridSpecsV2.from_simulation_config()` classmethod (lines 142-201)
   - Added `GridSpecsV2.to_simulation_config()` method (lines 203-245)

2. **`/workspaces/Smatrix_2D/smatrix_2d/core/grid_factory_example.py`** (NEW)
   - Comprehensive usage examples
   - 5 complete examples demonstrating different use cases

---

## Integration Points

### Works With:
- ✅ `SimulationConfig` (SSOT configuration)
- ✅ `GridConfig` (grid configuration)
- ✅ `create_phase_space_grid()` (grid creation)
- ✅ `create_default_grid_specs()` (default specs)
- ✅ `SimulationConfig.from_dict()` (YAML/JSON loading)

### Enables:
- ✅ Configuration-driven grid creation
- ✅ Bidirectional conversion between config and grid specs
- ✅ Validation of grid parameters via SimulationConfig
- ✅ Seamless integration with existing codebase

---

## Compliance

✅ **R-CFG-001:** Uses SSOT defaults from `config/defaults.py`
✅ **R-CFG-002:** Follows SimulationConfig SSOT principle
✅ **R-CFG-003:** Implements factory functions as specified
✅ **SPEC v2.1:** Grid specs follow v2.1 conventions (absolute angles, centered coordinates)

---

## Future Enhancements

Potential improvements for future iterations:

1. **Partial Configuration:** Support converting only specific sections (e.g., just energy grid)
2. **Validation Warnings:** Add warnings when delta_x/delta_z seem inconsistent
3. **Custom Defaults:** Allow customizing `use_texture_memory` in `from_simulation_config()`
4. **Bulk Conversion:** Support converting multiple configs at once

---

## Conclusion

The R-CFG-003 implementation successfully provides bidirectional conversion between `SimulationConfig` and `GridSpecsV2`, enabling seamless integration between the configuration SSOT and the core grid infrastructure. The implementation is type-safe, handles enum conversions correctly, and preserves all grid parameters through round-trip conversions.

**Status:** Ready for production use ✅
