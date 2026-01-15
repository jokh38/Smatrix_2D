# DOC-4: Phase D SPEC — GPU Optimization & Hardening

**Document ID**: SMP-PH-D-2.1
**Status**: Final / Binding
**Depends on**: SMP-PH-A-2.1 (Phase A), SMP-PH-B1-2.1 (Phase B-1)
**Revision**: v2.1 (Initial release)

---

## D-0 Phase Context Summary

### 목적

Phase D의 목적은 "실제 GPU 하드웨어에서의 최적 성능을 끌어내는 것"이다.

Phase A/B에서 기능적 정확성과 LUT 기반 구현을 완료했으므로, Phase D에서는:
1. Memory hierarchy 최적화 (shared/constant memory 활용)
2. Thread block/warp 수준 최적화
3. GPU 자원 활용률 개선 (occupancy, SM utilization)
4. Production-ready hardening (견고함, 안정성)

### 완료 조건

Phase D는 다음이 모두 충족될 때 완료된다:

- [ ] R-GPU-001 ~ R-GPU-007 구현
- [ ] R-OPT-001 ~ R-OPT-004 구현
- [ ] V-GPU-001, V-OPT-001 통과
- [ ] P-GPU-001, P-GPU-002 성능 목표 달성

---

## D-1 Memory Hierarchy Optimization

### R-GPU-001 — Shared Memory Tiling for Spatial Streaming

Spatial streaming kernel에서 shared memory를 사용하여 global memory access를 줄인다.

**현재 문제점:**
- 모든 데이터를 global memory에서 직접 읽음
- 각 스레드가 4개의 인접 셀을 읽을 때 중복된 읽기 발생
- Poor cache utilization

**최적화 전략:**
```
Shared memory tile: [TILE_Z + 1, TILE_X + 1]
1. 각 block이 해당 영역의 데이터를 shared memory로 로드
2. Halo region 포함 (boundary interpolation용)
3. Shared memory 내에서 interpolation 수행
4. 결과만 global memory에 write-back
```

**기대 효과:**
- Global memory bandwidth 감소: 50-70%
- Spatial kernel speedup: 2-3×

**구현 영역:**
- `smatrix_2d/gpu/kernels.py`: `spatial_streaming_kernel_v3`
- Tile size: 16×16 또는 32×32 (shared memory 48KB 제한 고려)

### R-GPU-002 — Constant Memory for LUTs

자주 참조되는 LUT 데이터를 constant memory로 이전한다.

**대상 LUTs:**
- Stopping power LUT: ~200 energies × 4 bytes = 800 bytes
- Scattering sigma LUT: ~200 energies × 4 materials × 4 bytes = 3.2 KB
- Velocity/Energy relationship: ~200 entries × 4 bytes = 800 bytes

**Constant memory 특성:**
- 크기: 64KB (전체 warp가 broadcast로 동시 읽기)
- Latency: 최소 (GPU on-chip)
- Bandwidth: 모든 스레드가 동일 값을 읽을 때 최적

**구현 방법:**
```cuda
__constant__ float stopping_power_lut[MAX_LUT_SIZE];
__constant__ float sigma_lut[MAX_MATERIALS][MAX_LUT_SIZE];
```

**기대 효과:**
- LUT lookup latency: 5-10× 감소
- Global memory bandwidth 감소: ~10%

### R-GPU-003 — Memory Coalescing 검증

모든 global memory access가 fully coalesced되도록 보장한다.

**검증 항목:**
- [ ] x-direction (fastest index): 32 consecutive threads → 1 transaction
- [ ] z-direction: 32 consecutive threads → 1-2 transactions
- [ ] theta/E direction: strided access 최소화

**Coalescing 규칙:**
- Thread ID가 contiguous memory에 매핑되도록 설계
- Stride가 128 bytes 이상인 access restructure
- Vectorized load/store 사용 (float4)

---

## D-2 Thread-Level Optimization

### R-GPU-004 — Warp-Level Primitives for Scatter

Atomic operation contention을 줄이기 위해 warp-level primitives를 활용한다.

**현재 문제점:**
- `atomicAdd(&escapes_gpu[channel], weight)`에서 contention
- Multiple warp이 동일 escape channel에 기록
- Serialization으로 성능 저하

**최적화 전략:**
```cuda
// Warp-level reduction
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp별로 하나의 atomic만 수행
float warp_sum = warp_reduce_sum(local_escape);
if (lane_id == 0) {
    atomicAdd(&escapes_gpu[channel], warp_sum);
}
```

**기대 효과:**
- Atomic operation 횟수: 32× 감소 (warp size = 32)
- Escape tracking speedup: 5-10×

### R-GPU-005 — Dynamic Block Sizing

GPU architecture에 따라 최적 block size를 자동 선택한다.

**현재 문제점:**
- Fixed block size (256 threads)가 모든 GPU에 최적이 아님
- High-end GPU (A100): 큰 block 유리
- Low-end GPU (GTX 1650): 작은 block 필요

**Auto-tuning 전략:**
```python
def get_optimal_block_size(gpu_properties, kernel_type):
    # Compute capability 기반 heuristics
    sm_count = gpu_properties.multi_processor_count
    max_threads_per_sm = gpu_properties.max_threads_per_multiprocessor

    # Kernel type별 register 사용량 추정
    register_per_thread = {
        'angular': 32,
        'energy': 40,
        'spatial': 48
    }[kernel_type]

    # Occupancy 계산
    for block_size in [128, 192, 256, 384, 512, 1024]:
        occupancy = calculate_occupancy(block_size, register_per_thread)
        if occupancy > 0.8:
            return block_size
    return 256  # Fallback
```

**기대 효과:**
- Cross-GPU 성능 편차 감소
- Low-end GPU에서 10-20% speedup

---

## D-3 Occupancy and Resource Optimization

### R-GPU-006 — Occupancy Profiling and Targeting

각 kernel의 theoretical occupancy를 측정하고 목표치를 달성한다.

**Occupancy 목표:**
| Kernel | Target Occupancy | Minimum Acceptable |
|--------|------------------|--------------------|
| Angular | ≥ 75% | 50% |
| Energy | ≥ 75% | 50% |
| Spatial | ≥ 50% | 33% |

**측정 방법:**
```python
import cupy as cp

def measure_occupancy(kernel, grid_size, block_size):
    # CuPy/CUDA profiler 사용
    with cp.cuda.Profile():
        kernel(...)
    # Nsight Compute 또는 cupy.cuda.profiler로 분석
```

**최적화 레버:**
- Block size 조절
- Register 사용량 감소 (kernel code refactoring)
- Shared memory 사용량 감소

### R-GPU-007 — Register Pressure Reduction

Register 사용량을 줄여 occupancy를 개선한다.

**분석 방법:**
```bash
nvcc --ptxas-options=-v kernel.cu
# Register usage: 64 registers per thread
```

**최적화 기법:**
1. 불필요한 변수 제거
2. Loop invariant code motion
3. Array indexing 최적화 (연산 reduction)
4. `#pragma unroll` 제어

---

## D-4 Profiling Enhancement

### R-OPT-001 — GPU-Specific Profiling Metrics

GPU architecture 관련 메트릭을 추가로 측정한다.

**추가 메트릭:**
- `sm_efficiency`: Streaming multiprocessor 활용률 (%)
- `warp_efficiency`: Warp가 active인 시간 비율 (%)
- `memory_bandwidth`: 실제 사용된 대역폭 (GB/s)
- `dram_read_transactions`, `dram_write_transactions`: DRAM 트랜잭션 수
- `l2_cache_hit_rate`: L2 cache hit rate (%)

**구현:**
```python
@dataclass
class GPUMetrics:
    sm_efficiency: float
    warp_efficiency: float
    memory_bandwidth: float
    l2_cache_hit_rate: float
    occupancy: float
```

### R-OPT-002 — Automated Benchmarking Suite

성능 regression을 자동으로 검출하는 벤치마크 시스템을 구축한다.

**벤치마크 구조:**
```
benchmarks/
├── run_benchmark.py           # Main runner
├── results/
│   ├── baseline.json          # Reference performance
│   └── current.json           # Current run results
└── configs/
    ├── benchmark_small.yaml   # Config-S based
    ├── benchmark_medium.yaml  # Config-M based
    └── benchmark_large.yaml   # Config-L based
```

**Pass/Fail 기준:**
- Step time regression < 5%
- Memory usage regression < 10%
- Conservation error 변화 없음

---

## D-5 Hardening and Production Readiness

### R-OPT-003 — GPU Architecture Detection

런타임에 GPU 아키텍처를 감지하고 최적 설정을 적용한다.

**감지 항목:**
- Compute capability (major.minor)
- SM count
- Max threads per SM
- Shared memory per SM
- L2 cache size

**설정 프로파일:**
```python
GPU_PROFILES = {
    'A100': {'block_size': 512, 'tile_size': 32},
    'RTX_3080': {'block_size': 256, 'tile_size': 16},
    'GTX_1650': {'block_size': 128, 'tile_size': 16},
}
```

### R-OPT-004 — Error Handling and Recovery

GPU 오류를 graceful하게 처리한다.

**오류 시나리오:**
- CUDA out of memory
- Kernel launch failure
- Invalid memory access
- Timeout

**처리 전략:**
```python
try:
    run_simulation(config)
except cp.cuda.memory.OutOfMemoryError:
    # Fallback to reduced resolution
    config = fallback_to_lower_resolution(config)
    run_simulation(config)
except cp.cuda.driver.CUDAError as e:
    # Log error and provide diagnostics
    log_gpu_diagnostics()
    raise
```

---

## D-6 Validation Criteria

### V-GPU-001 — Bitwise Equivalence with Optimizations

모든 최적화가 bitwise equivalence를 유지함을 검증한다.

**테스트 방법:**
1. Optimized kernel 결과를 baseline kernel 결과와 비교
2. Config-S, Config-M 모두에서 검증
3. All channels: dose, escapes, weight remaining

**Pass 기준:**
```
max(|optimized - baseline|) < 1e-12 (float32 epsilon)
```

### V-OPT-001 — Performance Regression Test

벤치마크 suite가 regression을 검출함을 검증한다.

**테스트 방법:**
1. Intentional slowdown injection (10% sleep)
2. 벤치마크가 이를 검출하는지 확인
3. False positive rate 확인

---

## D-7 Performance Targets

### P-GPU-001 — Overall Speedup

Phase D 최적화로 전체 속도 개선을 달성한다.

**목표 (Config-M 기준):**
| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Total step time | 100% | ≤ 60% | RTX 3080 |
| Angular kernel | 100% | ≤ 70% | RTX 3080 |
| Energy kernel | 100% | ≤ 80% | RTX 3080 |
| Spatial kernel | 100% | ≤ 50% | RTX 3080 |

**기대 총 speedup: 1.5-2.0×**

### P-GPU-002 — Memory Bandwidth Efficiency

Memory bandwidth를 효율적으로 사용한다.

**목표:**
- Achieved bandwidth / Peak bandwidth ≥ 40%
- L2 cache hit rate ≥ 60%

**측정 조건:**
- Config-M
- Nsight Compute 또는 nvprof로 측정

---

## D-8 Implementation Checklist

### Memory Hierarchy
- [ ] Shared memory tiling for spatial kernel
- [ ] Constant memory for LUTs
- [ ] Memory coalescing verification

### Thread-Level
- [ ] Warp-level primitives for scatter
- [ ] Dynamic block sizing
- [ ] Register pressure reduction

### Profiling
- [ ] GPU-specific metrics collection
- [ ] Automated benchmarking suite
- [ ] Performance regression tests

### Hardening
- [ ] GPU architecture detection
- [ ] Error handling and recovery
- [ ] Documentation and user guide

---

## D-9 Risk Assessment

### Risk 1: Optimization Breaks Physics

**위험:** Aggressive optimization으로 인해 수치적 정확도 손상
**완화:**
- Bitwise equivalence test 필수 통과
- Conservation test 각 step에서 실행

### Risk 2: GPU-Specific Optimization

**위험:** 특정 GPU에서만 작동하는 코드
**완화:**
- Fallback to unoptimized version
- Multiple GPU에서 테스트

### Risk 3: Occupancy Trade-off

**위험:** Occupancy 개선을 위해 register를 줄이면 instruction 증가
**완화:**
- Profiling으로 trade-off 분석
- 실제 runtime으로 판단

---

## D-10 Timeline Estimate

| Task | 예상 소요 시간 |
|------|----------------|
| Shared memory tiling | 2-3일 |
| Constant memory LUTs | 1일 |
| Warp-level primitives | 1-2일 |
| Dynamic block sizing | 1일 |
| Profiling enhancement | 1일 |
| Benchmarking suite | 1-2일 |
| Hardening and error handling | 1일 |
| Validation and testing | 1-2일 |
| 문서화 | 0.5일 |
| **Total** | **10-14일** |

---

*End of DOC-4*
