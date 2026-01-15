# Benchmarking Suite - Examples and Usage Guide

This document provides practical examples for using the Smatrix_2D benchmarking suite.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [CI/CD Integration](#cicd-integration)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Quick Test (5 seconds)

Run the quick test to verify everything works:

```bash
cd benchmarks
python quick_test.py
```

Expected output:
```
Smatrix_2D Quick Benchmark Test (Config-S)
======================================================================
Grid: 32×32×45×35
DOF: 1,584,000
Running 50 transport steps...
  Step   1/50: 48.23 ms
  Step  10/50: 49.12 ms
  ...
  Step  50/50: 48.87 ms

RESULTS
======================================================================
Total time:         2.456 s
Avg step time:      49.12 ms
Steps/sec:          20.4
Conservation:       PASS
======================================================================
```

### 2. Full Benchmark (2-5 minutes)

Run all configurations and compare against baseline:

```bash
python run_benchmark.py
```

This will:
- Run Config-S, Config-M, and Config-L
- Compare each against baseline
- Generate JSON results and regression reports
- Exit with error code 1 if regression detected

### 3. Generate HTML Report

Create an interactive visualization:

```bash
python generate_html_report.py
```

Open `benchmark_report.html` in a browser to see:
- Performance overview
- Detailed metrics per configuration
- Regression highlights
- Performance comparison charts

## Basic Usage

### Run Single Configuration

```bash
# Small only (fastest)
python run_benchmark.py --config S

# Medium only (standard)
python run_benchmark.py --config M

# Large only (slowest)
python run_benchmark.py --config L
```

### Update Baseline

After performance improvements, update the baseline:

```bash
python run_benchmark.py --update-baseline
```

This saves current results as the new reference point.

### Custom Output Location

```bash
python run_benchmark.py --output-dir /tmp/my_results
python generate_html_report.py --result-dir /tmp/my_results --output /tmp/report.html
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -e .
        pip install cupy-cuda11x psutil

    - name: Run quick benchmark
      run: |
        cd benchmarks
        python quick_test.py

    - name: Run full benchmarks
      run: |
        cd benchmarks
        python run_benchmark.py --config S --config M

    - name: Upload results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: benchmark-results
        path: benchmarks/results/*.json

    - name: Generate report
      if: always()
      run: |
        cd benchmarks
        python generate_html_report.py

    - name: Upload report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: benchmark-report
        path: benchmarks/benchmark_report.html
```

### GitLab CI Example

```yaml
stages:
  - test

benchmark:
  stage: test
  image: nvidia/cuda:11.8.0-devel-ubuntu22.04
  script:
    - apt-get update && apt-get install -y python3 python3-pip
    - pip3 install -e .
    - pip3 install cupy-cuda11x psutil
    - cd benchmarks
    - python quick_test.py
    - python run_benchmark.py --config S
  artifacts:
    paths:
      - benchmarks/results/*.json
      - benchmarks/benchmark_report.html
    expire_in: 1 week
```

### Pre-commit Hook

Add `.git/hooks/pre-commit`:

```bash
#!/bin/bash
cd benchmarks
python quick_test.py
if [ $? -ne 0 ]; then
    echo "Benchmark test failed"
    exit 1
fi
```

## Advanced Usage

### Custom Configuration

Create `benchmarks/configs/config_custom.json`:

```json
{
  "config_name": "Config-Custom",
  "description": "Custom test configuration",
  "grid": {
    "Nx": 64,
    "Nz": 64,
    "Ntheta": 90,
    "Ne": 50,
    "delta_x": 1.0,
    "delta_z": 1.0,
    "x_min": -32.0,
    "x_max": 32.0,
    "z_min": -32.0,
    "z_max": 32.0,
    "theta_min": 0.0,
    "theta_max": 180.0,
    "E_min": 1.0,
    "E_max": 100.0,
    "E_cutoff": 2.0
  },
  "beam": {
    "E0": 100.0,
    "x0": 0.0,
    "z0": -20.0,
    "theta0": 0.0,
    "w0": 1.0
  },
  "simulation": {
    "delta_s": 1.0,
    "n_steps": 75,
    "n_buckets": 32,
    "k_cutoff": 5.0
  },
  "performance_targets": {
    "max_step_time_ms": 150,
    "max_total_time_sec": 15.0,
    "max_memory_mb": 1000
  },
  "regression_thresholds": {
    "step_time_pct": 5.0,
    "memory_pct": 10.0,
    "kernel_time_pct": 5.0
  }
}
```

Run with:
```bash
python run_benchmark.py --config Custom
```

### Analyzing Results with Python

```python
import json
from pathlib import Path

# Load results
result_file = Path("benchmarks/results/S_20260115_120000.json")
with open(result_file) as f:
    data = json.load(f)

# Extract metrics
steps_per_sec = data['steps_per_second']
conservation_error = data['conservation_error']

print(f"Performance: {steps_per_sec:.1f} steps/sec")
print(f"Conservation: {conservation_error:.2e}")

# Load baseline
baseline_file = Path("benchmarks/results/baseline/config_S_baseline.json")
with open(baseline_file) as f:
    baseline = json.load(f)

# Compare
change = (data['avg_step_time_ms'] - baseline['avg_step_time_ms']) / baseline['avg_step_time_ms'] * 100
print(f"Step time change: {change:+.1f}%")
```

### Batch Comparison

Compare multiple runs:

```python
import json
from pathlib import Path

results_dir = Path("benchmarks/results")
configs = ['S', 'M', 'L']

for config in configs:
    # Get all runs for this config
    runs = list(results_dir.glob(f"{config}_*.json"))
    runs = [r for r in runs if 'baseline' not in str(r) and 'report' not in str(r)]

    print(f"\n{config}: {len(runs)} runs")

    # Show trend
    for run in sorted(runs)[-5:]:  # Last 5 runs
        with open(run) as f:
            data = json.load(f)
        print(f"  {run.name}: {data['avg_step_time_ms']:.2f} ms")
```

## Troubleshooting

### Issue: "No module named 'psutil'"

**Solution:**
```bash
pip install psutil
```

### Issue: "CUDA out of memory"

**Solution:** Run smaller configs only:
```bash
python run_benchmark.py --config S  # Skip M and L
```

### Issue: "CuPy not available"

**Solution:** Install CuPy for your CUDA version:
```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x
```

Or run with CPU fallback (slower):
```bash
# Benchmark will automatically fall back to CPU
```

### Issue: "No baseline found"

**Solution:** Create initial baseline:
```bash
python run_benchmark.py --update-baseline
```

### Issue: Regression detected but it's expected

**Solution:** Update baseline after intentional changes:
```bash
# Make performance improvements
# Run tests
python run_benchmark.py
# If satisfied, update baseline
python run_benchmark.py --update-baseline
```

### Issue: Conservation validation fails

**Possible causes:**
1. Numerical instability in current code
2. Grid configuration too coarse
3. Energy cutoff too high

**Debug:**
```python
# Check conservation error details
print(f"Mass balance: {final_mass} + {escapes} = {final_mass + escapes}")
print(f"Expected: {initial_mass}")
print(f"Error: {abs(final_mass + escapes - initial_mass)}")
```

## Performance Tips

### 1. Warm-up Runs

GPU performance can vary on first runs. Add warm-up:

```python
# In benchmark runner, add warm-up steps
for _ in range(5):  # Warm-up
    psi_gpu = transport_step.apply(psi_gpu, accumulators)
    cp.cuda.Stream.null.synchronize()

# Then start timing
```

### 2. Reduce I/O Overhead

Minimize GPU-CPU transfers during benchmarking:

```python
# Keep everything on GPU
# Only sync for timing
# Only transfer to CPU at end
```

### 3. Use Fixed Random Seed

For reproducible benchmarks:

```python
np.random.seed(42)
if GPU_AVAILABLE:
    cp.random.seed(42)
```

## FAQ

**Q: How often should I run benchmarks?**

A: At minimum:
- Before every commit to main branch
- After any performance-related changes
- Weekly in CI/CD pipeline

**Q: What if I don't have a GPU?**

A: The suite will fall back to CPU automatically. Results will be slower but still valid for regression testing.

**Q: Can I add custom metrics?**

A: Yes! Extend the `BenchmarkMetrics` dataclass in `run_benchmark.py` with your custom fields.

**Q: How do I share results?**

A: Share the JSON files from `results/` or the generated HTML report. All results are self-contained.

**Q: Can I run benchmarks in parallel?**

A: Yes, but be careful with GPU memory:
```bash
# Run different configs in parallel
python run_benchmark.py --config S &
python run_benchmark.py --config M &
python run_benchmark.py --config L &
wait
```

## Getting Help

For issues or questions:
1. Check this document
2. Check main README.md
3. Check master specification: `refactor_plan_docs/DOC-0_MASTER_SPEC_v2.1_REVISED.md`
4. Open an issue on GitHub

## See Also

- [Main README](README.md) - Overview and configuration reference
- [Master Spec](../refactor_plan_docs/DOC-0_MASTER_SPEC_v2.1_REVISED.md) - Standard configurations
- [Source Code](run_benchmark.py) - Implementation details
