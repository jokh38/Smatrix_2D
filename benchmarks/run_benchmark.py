#!/usr/bin/env python3
"""
Smatrix_2D Automated Benchmarking Suite

Performance regression testing system for GPU transport simulations.
Runs Config-S, Config-M, and Config-L simulations, collects metrics,
and compares against baseline results.

Usage:
    python run_benchmark.py [--config S|M|L|all] [--update-baseline] [--output-dir DIR]
"""

import json
import time
import gc
import psutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: CuPy not available, falling back to CPU")

# Import Smatrix_2D components
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from smatrix_2d import (
    GridSpecsV2,
    create_phase_space_grid,
    create_water_material,
    PhysicsConstants2D,
    create_water_stopping_power_lut,
    SigmaBuckets,
)
from smatrix_2d.gpu.kernels import create_gpu_transport_step_v3
from smatrix_2d.gpu.accumulators import GPUAccumulators


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class KernelTimings:
    """GPU kernel execution timings."""
    angular_scattering_ms: float
    energy_loss_ms: float
    spatial_streaming_ms: float
    total_step_ms: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class MemoryUsage:
    """Memory usage statistics."""
    gpu_memory_mb: float
    cpu_memory_mb: float
    phase_space_mb: float
    luts_mb: float
    total_mb: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BenchmarkMetrics:
    """Complete benchmark metrics for a single run."""
    config_name: str
    timestamp: str
    git_commit: str
    gpu_info: Dict[str, str]

    # Performance metrics
    setup_time_sec: float
    total_time_sec: float
    avg_step_time_ms: float
    min_step_time_ms: float
    max_step_time_ms: float
    steps_per_second: float

    # Kernel timings
    kernel_timings: KernelTimings

    # Memory usage
    memory_usage: MemoryUsage

    # Conservation metrics
    final_mass: float
    total_deposited_energy: float
    conservation_error: float
    conservation_valid: bool

    # Grid info
    grid_size_mb: float
    total_dof: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['kernel_timings'] = self.kernel_timings.to_dict()
        data['memory_usage'] = self.memory_usage.to_dict()
        return data


@dataclass
class RegressionReport:
    """Regression detection report."""
    config_name: str
    has_regression: bool
    regressions: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    improvements: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """Main benchmark execution engine."""

    def __init__(self, config_path: str, output_dir: Optional[Path] = None):
        """Initialize benchmark runner.

        Args:
            config_path: Path to benchmark configuration JSON
            output_dir: Directory for results (default: benchmarks/results/)
        """
        self.config_path = Path(config_path)
        self.output_dir = output_dir or Path(__file__).parent / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        with open(self.config_path) as f:
            self.config = json.load(f)

        self.config_name = self.config['config_name']

        # Get git commit
        self.git_commit = self._get_git_commit()

        # Get GPU info
        self.gpu_info = self._get_gpu_info()

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            return result.stdout.strip()[:8]
        except:
            return "unknown"

    def _get_gpu_info(self) -> Dict[str, str]:
        """Get GPU device information."""
        if not GPU_AVAILABLE:
            return {"device": "CPU fallback", "compute_capability": "N/A"}

        try:
            device = cp.cuda.Device()
            mem_info = device.mem_info
            return {
                "device": device.name,
                "compute_capability": f"{device.compute_capability[0]}.{device.compute_capability[1]}",
                "total_memory_mb": f"{mem_info[0] / 1024**2:.0f}",
                "free_memory_mb": f"{mem_info[1] / 1024**2:.0f}",
            }
        except:
            return {"device": "Unknown GPU", "compute_capability": "N/A"}

    def _get_memory_usage(self) -> Tuple[float, float]:
        """Get current GPU and CPU memory usage in MB."""
        cpu_mem = psutil.Process().memory_info().rss / 1024**2

        if GPU_AVAILABLE:
            gpu_mem = cp.cuda.Device().mem_info[0] - cp.cuda.Device().mem_info[1]
            gpu_mem = gpu_mem / 1024**2
        else:
            gpu_mem = 0.0

        return gpu_mem, cpu_mem

    def run(self) -> BenchmarkMetrics:
        """Run benchmark and collect metrics.

        Returns:
            BenchmarkMetrics with complete results
        """
        print(f"\n{'='*70}")
        print(f"Running Benchmark: {self.config_name}")
        print(f"{'='*70}")

        # Track setup time
        setup_start = time.time()

        # Create grid
        grid_specs = GridSpecsV2(**self.config['grid'])
        grid = create_phase_space_grid(grid_specs)

        # Create material and LUTs
        material = create_water_material()
        constants = PhysicsConstants2D()
        stopping_power_lut = create_water_stopping_power_lut()

        # Create sigma buckets
        sigma_buckets = SigmaBuckets(
            grid=grid,
            material=material,
            constants=constants,
            n_buckets=self.config['simulation']['n_buckets'],
            k_cutoff=self.config['simulation']['k_cutoff'],
            delta_s=self.config['simulation']['delta_s'],
        )

        # Create transport step
        transport_step = create_gpu_transport_step_v3(
            grid=grid,
            sigma_buckets=sigma_buckets,
            stopping_power_lut=stopping_power_lut,
            delta_s=self.config['simulation']['delta_s'],
        )

        # Create accumulators
        accumulators = GPUAccumulators.create(
            spatial_shape=(grid.Nz, grid.Nx),
            enable_history=False
        )

        # Initialize beam
        beam = self.config['beam']
        psi = np.zeros(grid.shape, dtype=np.float32)

        # Find nearest indices
        ix0 = np.argmin(np.abs(grid.x_centers - beam['x0']))
        iz0 = np.argmin(np.abs(grid.z_centers - beam['z0']))
        ith0 = np.argmin(np.abs(grid.th_centers_rad - np.deg2rad(beam['theta0'])))
        iE0 = np.argmin(np.abs(grid.E_centers - beam['E0']))

        psi[iE0, ith0, iz0, ix0] = beam['w0']

        # Move to GPU
        if GPU_AVAILABLE:
            psi_gpu = cp.asarray(psi)
        else:
            psi_gpu = psi

        setup_time = time.time() - setup_start

        # Calculate memory usage
        gpu_mem_after_setup, cpu_mem_after_setup = self._get_memory_usage()

        phase_space_mb = psi.nbytes / 1024**2
        luts_mb = (
            sigma_buckets.kernel_lut.nbytes / 1024**2 +
            stopping_power_lut.stopping_power.nbytes / 1024**2
        )
        grid_size_mb = phase_space_mb + luts_mb

        print(f"Grid: {grid.Nx}×{grid.Nz}×{grid.Ntheta}×{grid.Ne}")
        print(f"DOF: {grid.total_dof:,}")
        print(f"Phase space: {phase_space_mb:.1f} MB")
        print(f"LUTs: {luts_mb:.1f} MB")
        print(f"Setup time: {setup_time:.3f} s")

        # Run simulation and collect timings
        n_steps = self.config['simulation']['n_steps']

        step_times = []
        kernel_times = []

        print(f"\nRunning {n_steps} steps...")

        total_start = time.time()

        for step in range(n_steps):
            step_start = time.time()

            # Apply transport step
            psi_gpu = transport_step.apply(psi_gpu, accumulators)

            # Sync to get accurate timing
            if GPU_AVAILABLE:
                cp.cuda.Stream.null.synchronize()

            step_time = (time.time() - step_start) * 1000  # Convert to ms
            step_times.append(step_time)

            if (step + 1) % 10 == 0 or step == 0:
                print(f"  Step {step+1:3d}/{n_steps}: {step_time:.2f} ms")

        total_time = time.time() - total_start

        # Get final state
        if GPU_AVAILABLE:
            psi_final = cp.asnumpy(psi_gpu)
            dose_final = accumulators.get_dose_cpu()
            escapes_final = accumulators.get_escapes_cpu()
        else:
            psi_final = psi_gpu
            dose_final = accumulators.get_dose_cpu() if hasattr(accumulators, 'get_dose_cpu') else np.zeros((grid.Nz, grid.Nx))
            escapes_final = accumulators.get_escapes_cpu() if hasattr(accumulators, 'get_escapes_cpu') else np.zeros(5)

        # Get final memory usage
        gpu_mem_final, cpu_mem_final = self._get_memory_usage()

        # Calculate statistics
        avg_step_time = np.mean(step_times)
        min_step_time = np.min(step_times)
        max_step_time = np.max(step_times)
        steps_per_sec = n_steps / total_time

        # Conservation metrics
        final_mass = np.sum(psi_final)
        total_deposited = np.sum(dose_final)
        total_escapes = np.sum(escapes_final)
        initial_mass = beam['w0']

        conservation_error = abs(final_mass + total_escapes - initial_mass) / initial_mass
        conservation_valid = conservation_error < 1e-5

        print(f"\n{'='*70}")
        print(f"Results:")
        print(f"  Total time: {total_time:.3f} s")
        print(f"  Avg step time: {avg_step_time:.2f} ms")
        print(f"  Steps/sec: {steps_per_sec:.1f}")
        print(f"  Final mass: {final_mass:.6e}")
        print(f"  Deposited energy: {total_deposited:.6e}")
        print(f"  Conservation error: {conservation_error:.6e} [{'PASS' if conservation_valid else 'FAIL'}]")
        print(f"{'='*70}\n")

        # Create metrics object
        metrics = BenchmarkMetrics(
            config_name=self.config_name,
            timestamp=datetime.now().isoformat(),
            git_commit=self.git_commit,
            gpu_info=self.gpu_info,

            setup_time_sec=setup_time,
            total_time_sec=total_time,
            avg_step_time_ms=avg_step_time,
            min_step_time_ms=min_step_time,
            max_step_time_ms=max_step_time,
            steps_per_second=steps_per_sec,

            kernel_timings=KernelTimings(
                angular_scattering_ms=0.0,  # Would need instrumented kernels
                energy_loss_ms=0.0,
                spatial_streaming_ms=0.0,
                total_step_ms=avg_step_time,
            ),

            memory_usage=MemoryUsage(
                gpu_memory_mb=gpu_mem_final,
                cpu_memory_mb=cpu_mem_final,
                phase_space_mb=phase_space_mb,
                luts_mb=luts_mb,
                total_mb=gpu_mem_final + cpu_mem_final,
            ),

            final_mass=final_mass,
            total_deposited_energy=total_deposited,
            conservation_error=conservation_error,
            conservation_valid=conservation_valid,

            grid_size_mb=grid_size_mb,
            total_dof=grid.total_dof,
        )

        return metrics


# ============================================================================
# Regression Detection
# ============================================================================

def compare_with_baseline(
    current: BenchmarkMetrics,
    baseline: BenchmarkMetrics,
    thresholds: Dict[str, float]
) -> RegressionReport:
    """Compare current benchmark results with baseline.

    Args:
        current: Current benchmark metrics
        baseline: Baseline (golden) metrics
        thresholds: Regression thresholds

    Returns:
        RegressionReport with detected regressions
    """
    regressions = []
    warnings = []
    improvements = []

    # Check step time regression
    if baseline.avg_step_time_ms > 0:
        step_time_change = (
            (current.avg_step_time_ms - baseline.avg_step_time_ms) /
            baseline.avg_step_time_ms * 100
        )

        if step_time_change > thresholds['step_time_pct']:
            regressions.append({
                'metric': 'avg_step_time_ms',
                'baseline': baseline.avg_step_time_ms,
                'current': current.avg_step_time_ms,
                'change_pct': step_time_change,
                'threshold': thresholds['step_time_pct'],
            })
        elif step_time_change < -thresholds['step_time_pct']:
            improvements.append({
                'metric': 'avg_step_time_ms',
                'baseline': baseline.avg_step_time_ms,
                'current': current.avg_step_time_ms,
                'change_pct': step_time_change,
            })

    # Check memory regression
    if baseline.memory_usage.total_mb > 0:
        memory_change = (
            (current.memory_usage.total_mb - baseline.memory_usage.total_mb) /
            baseline.memory_usage.total_mb * 100
        )

        if memory_change > thresholds['memory_pct']:
            regressions.append({
                'metric': 'total_memory_mb',
                'baseline': baseline.memory_usage.total_mb,
                'current': current.memory_usage.total_mb,
                'change_pct': memory_change,
                'threshold': thresholds['memory_pct'],
            })

    # Check conservation validity
    if current.conservation_valid != baseline.conservation_valid:
        if not current.conservation_valid:
            regressions.append({
                'metric': 'conservation_valid',
                'baseline': baseline.conservation_valid,
                'current': current.conservation_valid,
                'message': 'Conservation validation failed',
            })

    # Check conservation error increase
    if current.conservation_error > baseline.conservation_error * 10:
        warnings.append({
            'metric': 'conservation_error',
            'baseline': baseline.conservation_error,
            'current': current.conservation_error,
            'message': 'Conservation error increased significantly',
        })

    return RegressionReport(
        config_name=current.config_name,
        has_regression=len(regressions) > 0,
        regressions=regressions,
        warnings=warnings,
        improvements=improvements,
    )


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description='Smatrix_2D Automated Benchmarking Suite'
    )
    parser.add_argument(
        '--config',
        choices=['S', 'M', 'L', 'all'],
        default='all',
        help='Configuration to run (default: all)'
    )
    parser.add_argument(
        '--update-baseline',
        action='store_true',
        help='Update baseline results with current run'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for results (default: benchmarks/results/)'
    )
    parser.add_argument(
        '--baseline-dir',
        type=Path,
        default=None,
        help='Baseline directory (default: benchmarks/results/baseline/)'
    )

    args = parser.parse_args()

    # Setup directories
    output_dir = args.output_dir or Path(__file__).parent / "results"
    baseline_dir = args.baseline_dir or output_dir / "baseline"

    # Determine which configs to run
    configs_to_run = []
    if args.config == 'all':
        configs_to_run = ['S', 'M', 'L']
    else:
        configs_to_run = [args.config]

    # Run benchmarks
    all_results = []
    all_reports = []

    for config_id in configs_to_run:
        config_path = Path(__file__).parent / "configs" / f"config_{config_id}.json"

        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            continue

        # Run benchmark
        runner = BenchmarkRunner(config_path, output_dir)
        metrics = runner.run()
        all_results.append(metrics)

        # Save current results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"{args.config}_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        print(f"Results saved to: {result_file}")

        # Load baseline and compare
        baseline_file = baseline_dir / f"config_{config_id}_baseline.json"

        if baseline_file.exists():
            with open(baseline_file) as f:
                baseline_data = json.load(f)

            # Reconstruct baseline metrics
            baseline = BenchmarkMetrics(
                config_name=baseline_data['config_name'],
                timestamp=baseline_data['timestamp'],
                git_commit=baseline_data['git_commit'],
                gpu_info=baseline_data['gpu_info'],
                setup_time_sec=baseline_data['setup_time_sec'],
                total_time_sec=baseline_data['total_time_sec'],
                avg_step_time_ms=baseline_data['avg_step_time_ms'],
                min_step_time_ms=baseline_data['min_step_time_ms'],
                max_step_time_ms=baseline_data['max_step_time_ms'],
                steps_per_second=baseline_data['steps_per_second'],
                kernel_timings=KernelTimings(**baseline_data['kernel_timings']),
                memory_usage=MemoryUsage(**baseline_data['memory_usage']),
                final_mass=baseline_data['final_mass'],
                total_deposited_energy=baseline_data['total_deposited_energy'],
                conservation_error=baseline_data['conservation_error'],
                conservation_valid=baseline_data['conservation_valid'],
                grid_size_mb=baseline_data['grid_size_mb'],
                total_dof=baseline_data['total_dof'],
            )

            # Get thresholds from config
            thresholds = runner.config['regression_thresholds']

            # Compare
            report = compare_with_baseline(metrics, baseline, thresholds)
            all_reports.append(report)

            # Print regression report
            print(f"\n{'='*70}")
            print(f"Regression Analysis: {config_id}")
            print(f"{'='*70}")

            if report.has_regression:
                print("REGRESSIONS DETECTED:")
                for reg in report.regressions:
                    print(f"  {reg['metric']}: {reg['change_pct']:+.1f}% "
                          f"(baseline: {reg['baseline']:.2f}, "
                          f"current: {reg['current']:.2f})")
            else:
                print("No regressions detected")

            if report.improvements:
                print("\nIMPROVEMENTS:")
                for imp in report.improvements:
                    print(f"  {imp['metric']}: {imp['change_pct']:+.1f}%")

            if report.warnings:
                print("\nWARNINGS:")
                for warn in report.warnings:
                    print(f"  {warn['metric']}: {warn['message']}")

            print(f"{'='*70}\n")

            # Save regression report
            report_file = output_dir / f"{args.config}_{timestamp}_report.json"
            with open(report_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)

            # Exit with error if regression detected
            if report.has_regression:
                print("ERROR: Performance regression detected!")
                sys.exit(1)
        elif args.update_baseline:
            print(f"No baseline found, creating new baseline")
        else:
            print(f"Warning: No baseline found at {baseline_file}")

        # Update baseline if requested
        if args.update_baseline:
            baseline_dir.mkdir(parents=True, exist_ok=True)
            baseline_file = baseline_dir / f"config_{config_id}_baseline.json"
            with open(baseline_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            print(f"Baseline updated: {baseline_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    for metrics in all_results:
        print(f"\n{metrics.config_name}:")
        print(f"  Steps/sec: {metrics.steps_per_second:.1f}")
        print(f"  Avg step: {metrics.avg_step_time_ms:.2f} ms")
        print(f"  Memory: {metrics.memory_usage.total_mb:.0f} MB")
        print(f"  Conservation: {'PASS' if metrics.conservation_valid else 'FAIL'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
