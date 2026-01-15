# Smatrix_2D Automated Benchmarking Suite

Performance regression testing system for GPU transport simulations.

## Overview

This benchmarking suite provides automated performance regression testing for Smatrix_2D, supporting three standard configurations (Config-S, Config-M, Config-L) as defined in the master specification (DOC-0).

## Features

- **Multi-configuration testing**: Config-S (fast), Config-M (standard), Config-L (high-res)
- **Performance metrics**: Step time, total runtime, memory usage, kernel timings
- **Regression detection**: Automatic comparison against baseline with configurable thresholds
- **Conservation validation**: Mass and energy conservation checking
- **JSON output**: Machine-readable results for CI/CD integration
- **HTML reports**: Interactive visualization with comparison views
- **GPU/CPU support**: Automatic fallback to CPU if GPU unavailable

## Directory Structure

```
benchmarks/
├── configs/
│   ├── config_S.json    # Small configuration (32×32×45×35)
│   ├── config_M.json    # Medium configuration (100×100×180×100)
│   └── config_L.json    # Large configuration (200×200×360×200)
├── results/
│   ├── baseline/        # Baseline (golden) results
│   └── *.json           # Benchmark results and reports
├── run_benchmark.py     # Main benchmark runner
├── generate_html_report.py  # HTML report generator
└── README.md            # This file
```

## Quick Start

### Run All Benchmarks

```bash
cd benchmarks
python run_benchmark.py
```

### Run Specific Configuration

```bash
python run_benchmark.py --config S    # Small only
python run_benchmark.py --config M    # Medium only
python run_benchmark.py --config L    # Large only
```

### Update Baseline

```bash
python run_benchmark.py --update-baseline
```

This saves current results as the new baseline for future comparisons.

### Generate HTML Report

```bash
python generate_html_report.py
```

Generates `benchmark_report.html` with interactive visualization.

## Configuration

Each benchmark configuration is defined in JSON format under `configs/`:

### Config-S (Small)
- **Purpose**: Fast validation during development
- **Grid**: 32×32×45×35 (~3mm resolution)
- **Expected runtime**: < 5 seconds
- **Memory**: ~500 MB

### Config-M (Medium)
- **Purpose**: Standard validation for production
- **Grid**: 100×100×180×100 (1mm resolution)
- **Expected runtime**: < 30 seconds
- **Memory**: ~2 GB

### Config-L (Large)
- **Purpose**: High-resolution validation
- **Grid**: 200×200×360×200 (0.5mm resolution)
- **Expected runtime**: < 5 minutes
- **Memory**: ~8 GB

## Regression Thresholds

Default thresholds (configurable in each config file):

- **Step time**: > 5% slowdown triggers regression
- **Memory**: > 10% increase triggers regression
- **Conservation**: Failure to validate triggers regression

## Output Format

### JSON Results

Each benchmark run generates a JSON file with:

```json
{
  "config_name": "Config-S",
  "timestamp": "2026-01-15T12:00:00",
  "git_commit": "a1b2c3d4",
  "gpu_info": {
    "device": "NVIDIA RTX 3080",
    "compute_capability": "8.6"
  },
  "avg_step_time_ms": 45.2,
  "total_time_sec": 2.26,
  "steps_per_second": 22.1,
  "memory_usage": {
    "total_mb": 512.0
  },
  "conservation_valid": true,
  "conservation_error": 1.2e-7
}
```

### Regression Report

If baseline exists, a regression report is generated:

```json
{
  "config_name": "Config-S",
  "has_regression": false,
  "regressions": [],
  "warnings": [],
  "improvements": [
    {
      "metric": "avg_step_time_ms",
      "change_pct": -8.5
    }
  ]
}
```

## CI/CD Integration

### Example GitHub Actions Workflow

```yaml
name: Benchmark

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install cupy-cuda11x

      - name: Run benchmarks
        run: |
          cd benchmarks
          python run_benchmark.py --config S

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/results/*.json
```

## Performance Targets

Per DOC-0 Master Specification:

| Config | Grid Size | Resolution | Target Runtime | Target Memory |
|--------|-----------|------------|----------------|---------------|
| S      | 32×32×45×35 | 3mm | < 5 s | < 500 MB |
| M      | 100×100×180×100 | 1mm | < 30 s | < 2 GB |
| L      | 200×200×360×200 | 0.5mm | < 5 min | < 8 GB |

*Targets based on RTX 3080 baseline*

## Interpreting Results

### Exit Codes

- `0`: All benchmarks passed, no regressions
- `1`: Regression detected or benchmark failed

### Conservation Validation

- **PASS**: Mass balance error < 1e-5
- **FAIL**: Mass balance error >= 1e-5

### Regression Categories

1. **Regressions**: Performance degradation exceeding thresholds
2. **Warnings**: Concerning trends but below threshold
3. **Improvements**: Performance gains

## Troubleshooting

### "No baseline found" Warning

Run with `--update-baseline` to create initial baseline:

```bash
python run_benchmark.py --update-baseline
```

### GPU Out of Memory

Skip Config-L or reduce batch size:

```bash
python run_benchmark.py --config S  # Skip L
```

### CuPy Not Available

Suite automatically falls back to CPU (slower but functional).

## Advanced Usage

### Custom Output Directory

```bash
python run_benchmark.py --output-dir /path/to/results
```

### Custom Baseline Directory

```bash
python run_benchmark.py --baseline-dir /path/to/baseline
```

### Generate Report from Custom Results

```bash
python generate_html_report.py --result-dir /path/to/results --output report.html
```

## Reference Implementation

- **Runner**: `run_benchmark.py` - Main benchmark execution
- **Generator**: `generate_html_report.py` - HTML report generation
- **Configs**: `configs/*.json` - Standard configurations
- **Spec**: `refactor_plan_docs/DOC-0_MASTER_SPEC_v2.1_REVISED.md`

## Contributing

When adding new benchmark configurations:

1. Create JSON config in `configs/`
2. Follow naming convention: `config_*.json`
3. Include all required fields (see existing configs)
4. Set appropriate performance targets
5. Update this README

## License

Same as parent Smatrix_2D project.
