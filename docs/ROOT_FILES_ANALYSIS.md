# Root Directory Files Analysis & Recommendations

**Date:** 2026-01-10
**Action:** Evaluate test/analysis files in root directory

---

## Current Files in Root Directory

### 📁 Project Files (Keep in Root)

| File | Purpose | Action |
|------|---------|--------|
| `README.md` | Main project documentation | ✅ **KEEP** |
| `spec.md` | Specification v7.2 | ✅ **KEEP** |
| `setup.py` | Python package setup | ✅ **KEEP** |
| `__init__.py` | Package initialization | ✅ **KEEP** |

### 🧪 Test/Analysis Scripts (7 files)

| File | Purpose | Status | Recommendation |
|------|---------|--------|----------------|
| `run_proton_pdd.py` | **Main proton PDD simulation** | ✅ Primary | **KEEP in root** |
| `quick_pdd_test.py` | Quick PDD test | ❌ Duplicate | Remove |
| `test_proton_pdd.py` | Fast PDD test | ❌ Duplicate | Remove |
| `benchmark_grid.py` | Grid benchmarking | ⚠️ Useful | Move to `scripts/` |
| `analyze_energy_conservation.py` | Debug energy conservation | ❌ Debug | Remove |
| `analyze_interpolation.py` | Debug interpolation | ❌ Debug | Remove |
| `check_energy_grid.py` | Debug grid structure | ❌ Debug | Remove |

### 🗑️ Junk/Temporary Files (8 files)

| File | Type | Recommendation |
|------|------|----------------|
| `apt-get` | Empty file | **DELETE** |
| `sudo` | Empty file | **DELETE** |
| `update` | Empty file | **DELETE** |
| `cuda-keyring_1.1-1_all.deb` | Debian package | **DELETE** (install via apt) |
| `comprehensive_review.toon` | Analysis output | **DELETE** |
| `tags` | Ctags file | **DELETE** (add to .gitignore) |
| `proton_dose_map.png` | Output plot | Move to `output/` |
| `proton_pdd.png` | Output plot | Move to `output/` |
| `proton_pdd_quick.png` | Output plot | Move to `output/` |

---

## Detailed Analysis

### 1. Main Scripts (Essential)

#### `run_proton_pdd.py` ✅ KEEP
- **Purpose:** Primary proton PDD simulation script
- **Status:** Well-documented, production-ready
- **Usage:** Main entry point for proton simulations
- **Action:** **KEEP in root directory**

### 2. Duplicate Test Scripts (Remove)

#### `quick_pdd_test.py` ❌ REMOVE
- **Purpose:** Quick PDD test with minimal grid
- **Status:** Superseded by `run_proton_pdd.py`
- **Reason:** Functionality duplicated
- **Action:** **DELETE**

#### `test_proton_pdd.py` ❌ REMOVE
- **Purpose:** Fast proton PDD simulation
- **Status:** Superseded by `run_proton_pdd.py`
- **Reason:** Same functionality, less complete
- **Action:** **DELETE**

### 3. Debug Scripts (Remove)

#### `analyze_energy_conservation.py` ❌ REMOVE
- **Purpose:** Debug energy conservation in interpolation
- **Status:** Historical debug script
- **Reason:** Bug was fixed, script no longer needed
- **Action:** **DELETE**

#### `analyze_interpolation.py` ❌ REMOVE
- **Purpose:** Debug interpolation energy conservation
- **Status:** Historical debug script
- **Reason:** Bug was fixed, script no longer needed
- **Action:** **DELETE**

#### `check_energy_grid.py` ❌ REMOVE
- **Purpose:** Check energy grid structure
- **Status:** One-time debug script
- **Reason:** For development only, not production use
- **Action:** **DELETE**

### 4. Benchmark Script (Move)

#### `benchmark_grid.py` ⚠️ MOVE
- **Purpose:** Benchmark grid performance
- **Status:** Useful for performance testing
- **Reason:** Development/analysis tool, not main script
- **Action:** **Move to `scripts/` folder**

### 5. Output Files (Move)

#### PNG Files (3)
- `proton_dose_map.png`
- `proton_pdd.png`
- `proton_pdd_quick.png`

**Action:** **Move to `output/` folder**

### 6. Junk Files (Delete)

#### Empty Files (3)
- `apt-get` (empty)
- `sudo` (empty)
- `update` (empty)

**Action:** **DELETE**

#### Package Files
- `cuda-keyring_1.1-1_all.deb` (Debian package)

**Action:** **DELETE** (install via apt-get if needed)

#### Development Artifacts
- `comprehensive_review.toon` (analysis output)
- `tags` (ctags file)

**Action:** **DELETE** (add `tags` to .gitignore)

---

## Recommended Actions

### Step 1: Create `scripts/` Folder

```bash
mkdir -p scripts
```

### Step 2: Move Benchmark Script

```bash
mv benchmark_grid.py scripts/
```

### Step 3: Move Output Files

```bash
mv proton_dose_map.png output/
mv proton_pdd.png output/
mv proton_pdd_quick.png output/
```

### Step 4: Remove Unnecessary Files

```bash
# Duplicate test scripts
rm quick_pdd_test.py
rm test_proton_pdd.py

# Debug scripts
rm analyze_energy_conservation.py
rm analyze_interpolation.py
rm check_energy_grid.py

# Junk files
rm apt-get sudo update
rm cuda-keyring_1.1-1_all.deb
rm comprehensive_review.toon
rm tags
```

### Step 5: Update .gitignore

Add to `.gitignore`:
```
# Output files
*.png
output/

# Development artifacts
*.toon
tags
.DS_Store

# Package files
*.deb

# Empty files
apt-get
sudo
update
```

---

## Final Root Directory Structure

After cleanup, root directory should contain:

```
Smatrix_2D/
├── README.md                    # Main documentation
├── spec.md                      # Specification
├── setup.py                     # Package setup
├── __init__.py                  # Package init
├── run_proton_pdd.py            # Main simulation script
│
├── docs/                        # Documentation
├── examples/                    # Example scripts
├── scripts/                     # Utility scripts (NEW)
│   └── benchmark_grid.py       # Performance benchmarking
├── smatrix_2d/                  # Source code
├── tests/                       # Test suite
└── output/                      # Simulation outputs
```

---

## File Count Summary

| Category | Before | After | Change |
|----------|--------|-------|--------|
| **Main scripts** | 1 | 1 | 0 |
| **Test scripts** | 7 | 0 | -7 |
| **Debug scripts** | 3 | 0 | -3 |
| **Junk files** | 8 | 0 | -8 |
| **Output files** | 3 (root) | 3 (output/) | 0 |
| **Total extras** | 22 | 0 | -22 |

**Result:** Clean root directory with only essential files.

---

## Rationale

### Why Keep `run_proton_pdd.py` in Root?

1. **User-Facing**: Main entry point for simulations
2. **Well-Documented**: Comprehensive docstring
3. **Production-Ready**: Optimized and validated
4. **Common Pattern**: Users expect main script in root

### Why Remove Debug Scripts?

1. **Historical**: Bugs already fixed
2. **One-Time**: Used for specific debugging
3. **Dated**: Not maintained with code changes
4. **Confusing**: Too many similar scripts

### Why Move `benchmark_grid.py`?

1. **Development Tool**: Not for end users
2. **Utility Script**: Belongs in `scripts/`
3. **Still Useful**: Keep for performance testing

### Why Delete Junk?

1. **Empty Files**: No purpose
2. **Packages**: Install via package manager
3. **Output**: Move to `output/` folder
4. **Artifacts**: Not part of source code

---

## Benefits

### ✅ Clean Root Directory
- Only essential files visible
- Clearer project structure
- Easier for new users

### ✅ Better Organization
- Scripts in `scripts/`
- Outputs in `output/`
- Tests in `tests/`

### ✅ Less Confusion
- No duplicate scripts
- No debug artifacts
- Clear purpose for each file

### ✅ Better Version Control
- `.gitignore` for generated files
- Only track source code
- Cleaner git history

---

## Conclusion

**Recommended Actions:**
1. ✅ Keep `run_proton_pdd.py` in root (main script)
2. ❌ Remove 6 debug/duplicate scripts
3. 📁 Move 1 benchmark script to `scripts/`
4. 📁 Move 3 output plots to `output/`
5. ❌ Delete 8 junk files

**Result:** 22 fewer files in root, clean and organized structure.
