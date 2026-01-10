# Smatrix_2D Documentation

This directory contains comprehensive documentation for the Smatrix_2D operator-factorized transport system.

---

## Core Documentation

### [GPU.md](GPU.md)
Complete guide to GPU acceleration using CuPy:
- Installation and setup
- Performance benchmarks (30-120× speedup)
- Implementation details
- Troubleshooting
- Multi-GPU roadmap

### [PERFORMANCE.md](PERFORMANCE.md)
Comprehensive performance optimization guide:
- Quick reference table
- GPU acceleration (best)
- Grid reduction strategies
- Numba JIT limitations
- Why multiprocessing doesn't work
- Accuracy vs speed tradeoffs
- Implementation examples

### [VALIDATION.md](SPEC_EVALUATION_REPORT.md)
Specification v7.2 compliance evaluation:
- Executive summary (99% compliance)
- Critical area analysis
- All issues RESOLVED
- Test results
- Recommendations

### [FIXES_SUMMARY.md](FIXES_SUMMARY.md)
Detailed summary of all fixes applied:
- Variable naming clarification
- Angular accuracy cap implementation
- Non-uniform grid support
- CPU-GPU validation test
- Usage examples

### [PROTON_PDD_BUG_REPORT.md](PROTON_PDD_BUG_REPORT.md)
Historical bug report:
- Energy gain bug (FIXED)
- Bragg peak validation
- Root cause analysis
- Resolution

---

## Main Project Documentation

- **[README.md](../README.md)** - Project overview, installation, quick start
- **[spec.md](../spec.md)** - Complete specification v7.2 (authoritative reference)

---

## Documentation Structure

```
Smatrix_2D/
├── README.md              # Main project readme
├── spec.md                # Specification v7.2
└── docs/
    ├── README.md          # This file
    ├── GPU.md             # GPU acceleration guide
    ├── PERFORMANCE.md     # Performance optimization
    ├── SPEC_EVALUATION_REPORT.md  # Compliance evaluation
    ├── FIXES_SUMMARY.md   # Fixes applied
    └── PROTON_PDD_BUG_REPORT.md  # Historical bugs
```

---

## Quick Links

### For Users
- [Installation](../README.md#installation)
- [Quick Start](../README.md#quick-start)
- [GPU Setup](GPU.md#requirements)
- [Performance Tuning](PERFORMANCE.md#quick-reference)

### For Developers
- [Specification](../spec.md)
- [GPU Implementation](GPU.md#implementation-details)
- [Performance Analysis](PERFORMANCE.md#bottleneck-analysis)
- [Validation Tests](VALIDATION.md#validation-requirements-status)

### For Contributors
- [Code Quality](VALIDATION.md#strengths)
- [Known Issues](FIXES_SUMMARY.md#next-steps)
- [Performance Goals](PERFORMANCE.md#recommendations-summary)

---

## Maintenance

### Adding Documentation

1. Choose appropriate location:
   - User-facing → Main README.md
   - Technical/specification → spec.md
   - Implementation guides → docs/

2. Follow existing structure:
   - Use Markdown formatting
   - Include code examples
   - Add cross-references

3. Update this README if adding new docs.

### Removing Documentation

Before removing:
1. Check if content is referenced elsewhere
2. Consolidate useful information into other docs
3. Update cross-references
4. Update this README

---

## Version History

- **2026-01-10**: Documentation reorganization
  - Consolidated GPU docs (9 files → 1)
  - Consolidated performance docs (7 files → 1)
  - Moved evaluation reports to docs/
  - Removed outdated install guides
