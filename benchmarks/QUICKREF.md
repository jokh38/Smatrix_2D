# Benchmarking Suite - Quick Reference

## Commands

```bash
# Quick test (5 seconds)
python quick_test.py

# Run all benchmarks
python run_benchmark.py

# Run specific configuration
python run_benchmark.py --config S  # or M, L

# Update baseline
python run_benchmark.py --update-baseline

# Generate HTML report
python generate_html_report.py
```

## Exit Codes

- `0` - Success, no regressions
- `1` - Regression detected or benchmark failed

## Configurations

| Config | Grid Size | Resolution | Runtime | Memory |
|--------|-----------|------------|---------|--------|
| S      | 32×32×45×35 | 3mm | < 5 s | < 500 MB |
| M      | 100×100×180×100 | 1mm | < 30 s | < 2 GB |
| L      | 200×200×360×200 | 0.5mm | < 5 min | < 8 GB |

## Regression Thresholds

- Step time: > 5% slowdown
- Memory: > 10% increase
- Conservation: Validation failure

## Output Files

```
results/
├── baseline/
│   ├── config_S_baseline.json
│   ├── config_M_baseline.json
│   └── config_L_baseline.json
├── S_20260115_120000.json           # Benchmark results
├── S_20260115_120000_report.json    # Regression report
└── benchmark_report.html            # HTML visualization
```

## Quick CI/CD Integration

```yaml
- name: Run benchmarks
  run: |
    cd benchmarks
    python run_benchmark.py --config S

- name: Upload results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: benchmarks/results/*.json
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Module 'psutil' not found | `pip install psutil` |
| CUDA out of memory | Run `--config S` only |
| No baseline found | Run `--update-baseline` |
| Conservation failed | Check grid configuration |

## Key Metrics

- **avg_step_time_ms**: Average step execution time
- **steps_per_second**: Throughput metric
- **conservation_error**: Mass balance error (target: < 1e-5)
- **memory_usage.total_mb**: Total memory consumption

## Documentation

- `README.md` - Complete user guide
- `EXAMPLES.md` - Detailed examples and CI/CD
- `IMPLEMENTATION_SUMMARY.md` - Implementation details

## Support

For issues, check:
1. Quick reference (this file)
2. README.md
3. EXAMPLES.md
4. Master spec: `refactor_plan_docs/DOC-0_MASTER_SPEC_v2.1_REVISED.md`
