
## Phase 2 분석: Energy Monotonicity 위반 문제

### 문제 현상

```
Warning: Monotonicity violated in energy mapping, using scatter fallback
```

### 코드 분석: `_build_energy_gather_lut()`

```python
def _build_energy_gather_lut(self, E_grid, stopping_power, delta_s, E_cutoff, E_edges):
    # E_new 계산
    deltaE_raw = stopping_power_np * delta_s
    max_deltaE = np.maximum(E_grid_np - E_cutoff, 0.0)
    deltaE = np.minimum(deltaE_raw, max_deltaE)
    E_new = E_grid_np - deltaE
    
    # Monotonicity check
    if not np.all(np.diff(E_new) < 0):  # ← 문제 발생 지점
        print("Warning: Monotonicity violated...")
        return None, None, None
```

### 물리적 분석

**에너지 그리드:** `E_grid[i]`는 높은 에너지 → 낮은 에너지 순서 (i가 증가하면 E 감소)

**Stopping power 특성:** Bethe-Bloch 공식에 따르면:
- 낮은 에너지에서 S(E) 증가 (Bragg peak 근처)
- `deltaE = S(E) × delta_s`
- 낮은 E에서 더 큰 에너지 손실

**기대하는 E_new:**
```
E_new[i] = E_grid[i] - deltaE[i]
```

단조 감소 조건 `np.diff(E_new) < 0`의 의미:
```
E_new[i+1] - E_new[i] < 0
→ (E_grid[i+1] - deltaE[i+1]) - (E_grid[i] - deltaE[i]) < 0
→ (E_grid[i+1] - E_grid[i]) < (deltaE[i+1] - deltaE[i])
```

**위반 조건:** 인접 bin 간 에너지 차이보다 stopping power 차이가 더 클 때 발생

### 근본 원인

1. **에너지 그리드 해상도 부족:** `delta_E`가 stopping power 변화율보다 클 때
2. **Bragg peak 근처:** S(E)가 급격히 변하는 영역에서 발생 가능
3. **Step size 과대:** `delta_s`가 너무 커서 한 step에 여러 bin을 건너뜀

### 해결 방안

```python
# Option 1: Adaptive step size
delta_s_max = delta_E / max(stopping_power)  # CFL-like condition

# Option 2: Energy grid refinement near Bragg peak
# 낮은 에너지 영역에서 더 촘촘한 grid

# Option 3: Monotonicity check를 완화하고 multi-source gather 허용
```

---

## Phase 2 분석: Spatial Streaming Kernel

### 현재 CUDA Kernel 분석

```cuda
// kernels.py의 _cuda_streaming_gather_kernel
void spatial_streaming_gather(...) {
    // Thread indices: one thread per SOURCE (z, x) cell per angle
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int ith = blockIdx.z;
    
    // SOURCE cell center position
    float z_src = z_offset + iz * delta_z;
    float x_src = x_offset + ix * delta_x;
    
    // Compute TARGET position (forward displacement)
    float z_tgt = z_src + delta_s * sin_th;
    float x_tgt = x_src + delta_s * cos_th;
    
    // Scatter: read from source, atomically add to target
    atomicAdd(&psi_out[tgt_idx], psi_in[src_idx]);
}
```

### 문제점: 이름과 구현 불일치

**Spec의 Gather 정의 (Section 3.3):**
> "For each target location, find what contributes to it"
> - Coalesced memory writes (no atomics needed)

**실제 구현:** Source → Target 방향의 **Scatter** 패턴
- `atomicAdd` 사용 (atomic operation)
- Thread가 source cell 기준으로 할당

### 진정한 Gather 구현

```cuda
// 올바른 Gather pattern
void spatial_streaming_gather_correct(...) {
    // Thread per TARGET cell
    int ix_tgt = blockIdx.x * blockDim.x + threadIdx.x;
    int iz_tgt = blockIdx.y * blockDim.y + threadIdx.y;
    int ith = blockIdx.z;
    
    float z_tgt = z_offset + iz_tgt * delta_z;
    float x_tgt = x_offset + ix_tgt * delta_x;
    
    // Find SOURCE position (backward displacement)
    float z_src = z_tgt - delta_s * sin_th;  // 부호 반대
    float x_src = x_tgt - delta_s * cos_th;
    
    // Bounds check for source
    if (z_src < 0 || z_src >= Nz * delta_z || ...) {
        psi_out[tgt_idx] = 0.0f;
        return;
    }
    
    // Bilinear interpolation from source (no atomics!)
    int iz_src = (int)floorf(z_src / delta_z);
    int ix_src = (int)floorf(x_src / delta_x);
    
    // Direct write (coalesced, deterministic)
    psi_out[tgt_idx] = interpolate(psi_in, iz_src, ix_src, ...);
}
```

**핵심 차이:**
| 속성 | 현재 구현 (Scatter) | 올바른 Gather |
|------|---------------------|---------------|
| Thread 할당 | Source cell | Target cell |
| 변위 방향 | Forward (+delta_s) | Backward (-delta_s) |
| Memory write | Atomic (race condition) | Direct (deterministic) |
| Coalescing | Poor (random target) | Good (sequential target) |

---

## Phase 2 분석: Python Scatter Kernel

### 코드 분석: `_spatial_streaming_kernel()`

```python
def _spatial_streaming_kernel(self, psi_in, delta_s, sigma_theta, theta_beam):
    # ... coordinate computation ...
    
    # Only process non-zero weights
    valid_mask = psi_flat > 1e-12
    if cp.any(valid_mask):
        # ... index computation ...
        
        # Use scatter_add for atomic operations
        if self.accumulation_mode == AccumulationMode.FAST:
            indices = (final_iE, final_ith, final_iz, final_ix)
            cp.add.at(psi_out, indices, final_weights)
```

### Sparsity 활용 분석

현재 구현은 `valid_mask`로 non-zero 요소만 처리:
```python
valid_mask = psi_flat > 1e-12
```

**이것이 "sparse-aware"인가?**
- Yes: 0인 bin은 건너뜀
- But: 여전히 전체 배열 flatten 후 mask 적용 → O(N_total) 메모리 접근

**진정한 sparse 처리:**
```python
# COO format으로 non-zero만 저장
iE, ith, iz, ix = cp.nonzero(psi_in > threshold)
weights = psi_in[iE, ith, iz, ix]
# O(n_active) 연산
```

---

## Phase 3 분석: Angular Scattering 문제

### 코드 분석: `_angular_scattering_kernel()`

```python
def _angular_scattering_kernel(self, psi_in, sigma_theta):
    # Direct convolution along theta axis
    for ith_out in range(self.Ntheta):
        theta_out = theta_centers[ith_out]
        
        # Gaussian weights for all input bins
        theta_diff = theta_centers - theta_out
        weights = cp.exp(-0.5 * (theta_diff / sigma_theta) ** 2)
        weights = weights / weights.sum()
        
        # Weighted sum
        convolved = cp.sum(psi_in * weights, axis=1, keepdims=True)
        psi_out[:, ith_out:ith_out+1, :, :] = convolved
```

### 물리적 분석

이것은 **deterministic angular redistribution**입니다:
- 각 (E, z, x) 위치에서 θ 차원을 따라 convolution
- 전체 weight 보존 (normalization)
- θ bin 수 불변

### Sparse 표현에서의 문제

**Dense format:**
- `psi[E, θ, z, x]` 배열에서 θ 축 convolution
- Memory footprint 불변

**PHASE3_SPARSE_LIMITATION.md의 주장:**
> "Angular scattering spreads each particle to multiple theta bins"
> "Exponential particle growth: n × Ntheta^k"

**이 주장의 오류:**

Deterministic transport에서는:
1. "Particle"이 아닌 "weight distribution"
2. Convolution은 weight를 재분배, 총량 보존
3. Sparse COO에서도 동일 위치 (E, z, x)의 다른 θ bin들을 **merge** 가능

**실제 문제:**
```python
# Sparse에서 convolution 구현
for each (iE, iz, ix) with any nonzero weight:
    theta_weights = gather_all_theta_for_position(iE, iz, ix)  # O(Ntheta)
    convolved = convolve(theta_weights, kernel)
    scatter_to_theta_bins(convolved)  # 최대 Ntheta entries
```

문제는 "폭발"이 아니라:
- 같은 spatial position의 θ bins를 그룹화해야 함
- COO format에서 이 연산이 비효율적 (random access)

---

## Spec vs 구현 비교 분석

### Spec Section 3.3: Gather Kernel

| Spec 주장 | 구현 상태 | 평가 |
|-----------|-----------|------|
| "Coalesced memory writes" | atomicAdd 사용 | ❌ 미구현 |
| "No atomics needed" | atomicAdd 사용 | ❌ 미구현 |
| "Thread per target cell" | Thread per source cell | ❌ 미구현 |
| "4-5x speedup" | 1.11x | 당연한 결과 |

### Spec Section 3.4: Sparse Representation

| Spec 주장 | 구현 상태 | 평가 |
|-----------|-----------|------|
| "COO format" | 구현됨 | ✅ |
| "O(n_active) operations" | spatial/energy만 | ⚠️ 부분 |
| "Angular scattering in sparse" | 미구현 | ❌ |
| "10-50x speedup" | 288x (without angular) | ⚠️ 불완전 |

---

## 개선 권장사항

### 1. Gather Kernel 올바른 구현

```cuda
// Phase 2: Correct gather-based spatial streaming
__global__ void spatial_streaming_gather_v2(
    const float* psi_in, float* psi_out,
    const float* sin_theta, const float* cos_theta,
    int Ne, int Ntheta, int Nz, int Nx,
    float delta_x, float delta_z, float delta_s
) {
    // Thread per TARGET
    int ix_tgt = blockIdx.x * blockDim.x + threadIdx.x;
    int iz_tgt = blockIdx.y * blockDim.y + threadIdx.y;
    int ith = blockIdx.z;
    
    if (ix_tgt >= Nx || iz_tgt >= Nz || ith >= Ntheta) return;
    
    float z_tgt = (iz_tgt + 0.5f) * delta_z;
    float x_tgt = (ix_tgt + 0.5f) * delta_x;
    
    // BACKWARD displacement to find source
    float z_src = z_tgt - delta_s * sin_theta[ith];
    float x_src = x_tgt - delta_s * cos_theta[ith];
    
    // Bounds check
    if (z_src < 0 || z_src >= Nz * delta_z ||
        x_src < 0 || x_src >= Nx * delta_x) {
        // Source outside domain - this target gets 0
        for (int iE = 0; iE < Ne; iE++) {
            int tgt_idx = iE * Ntheta * Nz * Nx + ith * Nz * Nx + iz_tgt * Nx + ix_tgt;
            psi_out[tgt_idx] = 0.0f;
        }
        return;
    }
    
    // Bilinear interpolation indices
    float fz = z_src / delta_z - 0.5f;
    float fx = x_src / delta_x - 0.5f;
    int iz0 = max(0, min((int)floorf(fz), Nz-2));
    int ix0 = max(0, min((int)floorf(fx), Nx-2));
    float wz = fz - iz0;
    float wx = fx - ix0;
    
    // Gather with bilinear interpolation (NO ATOMICS)
    for (int iE = 0; iE < Ne; iE++) {
        float val = 0.0f;
        val += (1-wz)*(1-wx) * psi_in[IDX(iE,ith,iz0,ix0)];
        val += (1-wz)*wx     * psi_in[IDX(iE,ith,iz0,ix0+1)];
        val += wz*(1-wx)     * psi_in[IDX(iE,ith,iz0+1,ix0)];
        val += wz*wx         * psi_in[IDX(iE,ith,iz0+1,ix0+1)];
        
        psi_out[IDX(iE,ith,iz_tgt,ix_tgt)] = val;  // Direct write
    }
}
```

### 2. Energy Monotonicity 해결

```python
def _build_energy_gather_lut_robust(self, E_grid, stopping_power, delta_s, E_cutoff, E_edges):
    """Robust LUT that handles non-monotonic cases."""
    
    E_new = E_grid - stopping_power * delta_s
    E_new = np.maximum(E_new, E_cutoff)  # Clamp to cutoff
    
    # For each TARGET bin, find ALL sources that contribute
    gather_map = [[] for _ in range(self.Ne)]  # List of (source_idx, weight)
    
    for iE_src in range(self.Ne):
        E_after = E_new[iE_src]
        
        if E_after <= E_cutoff:
            # Absorbed - contributes to dose only
            continue
        
        # Find target bin(s) via interpolation
        iE_tgt = np.searchsorted(E_edges, E_after, side='right') - 1
        iE_tgt = np.clip(iE_tgt, 0, self.Ne - 1)
        
        # Interpolation weight
        if iE_tgt < self.Ne - 1:
            E_lo, E_hi = E_edges[iE_tgt], E_edges[iE_tgt + 1]
            w_lo = (E_hi - E_after) / (E_hi - E_lo)
            gather_map[iE_tgt].append((iE_src, w_lo))
            gather_map[iE_tgt + 1].append((iE_src, 1 - w_lo))
        else:
            gather_map[iE_tgt].append((iE_src, 1.0))
    
    return gather_map  # No monotonicity requirement
```

### 3. Sparse Angular Scattering 구현

```python
class SparseAngularScattering:
    """Efficient angular scattering for sparse representation."""
    
    def apply(self, sparse_state, sigma_theta):
        # Group by (E, z, x) position
        positions = {}  # (iE, iz, ix) -> list of (ith, weight)
        
        for i in range(sparse_state.n_active):
            key = (sparse_state.iE[i], sparse_state.iz[i], sparse_state.ix[i])
            if key not in positions:
                positions[key] = []
            positions[key].append((sparse_state.ith[i], sparse_state.weight[i]))
        
        # Apply convolution per position
        new_entries = []
        for (iE, iz, ix), theta_weights in positions.items():
            # Reconstruct theta distribution
            theta_dist = np.zeros(self.Ntheta)
            for ith, w in theta_weights:
                theta_dist[ith] = w
            
            # Convolve with Gaussian kernel
            convolved = convolve_periodic(theta_dist, self.kernel(sigma_theta))
            
            # Extract significant entries
            for ith in range(self.Ntheta):
                if convolved[ith] > self.threshold:
                    new_entries.append((iE, ith, iz, ix, convolved[ith]))
        
        # Update sparse state
        return SparsePhaseState.from_entries(new_entries)
```

### 4. Hybrid Dense-Sparse 전략

```python
class HybridTransport:
    """Use dense for angular, sparse for spatial/energy."""
    
    def apply_step(self, psi, ...):
        # Angular scattering: Dense (efficient convolution)
        psi_1 = self.angular_kernel_dense(psi, sigma_theta)
        
        # Convert to sparse for spatial/energy
        sparse = SparsePhaseState.from_dense(psi_1, threshold=1e-12)
        
        # Spatial streaming: Sparse (O(n_active))
        sparse = self.spatial_kernel_sparse(sparse, delta_s)
        
        # Energy loss: Sparse (O(n_active))
        sparse, dose = self.energy_kernel_sparse(sparse, stopping_power)
        
        # Convert back to dense for next iteration
        psi_out = sparse.to_dense(psi.shape)
        
        return psi_out, dose
```

---

## 요약

| 문제 | 원인 | 해결책 |
|------|------|--------|
| Phase 2 성능 미달 | Gather가 실제로는 Scatter 구현 | Backward displacement + direct write |
| Energy monotonicity 위반 | LUT 구축 알고리즘의 제약 | Multi-source gather 허용 |
| Phase 3 angular scattering | Dense convolution을 sparse로 잘못 변환 | Position 그룹화 + 1D convolution |
| 288x 성능 달성 불가 | Angular scattering 미구현 | Hybrid dense-sparse 전략 |


---
# 올바른 구현 시 예상 성능 분석

---

## 현재 상태 기준점

| 단계 | 현재 성능 | 구성 |
|------|-----------|------|
| Phase 1 (Baseline) | 803 ms/step | GPU-resident + scatter |
| Medium Grid | 98.3M elements | Ne=256, Nθ=80, Nz=200, Nx=24 |

---

## Phase 2: 올바른 Gather 구현 시

### Spatial Streaming (A_stream)

**현재 문제:**
- Scatter pattern with atomicAdd
- Random memory write (poor coalescing)

**올바른 Gather 구현 효과:**

| 요소 | 현재 | 개선 후 | 근거 |
|------|------|---------|------|
| Memory coalescing | ~25% | ~90% | Sequential target write |
| Atomic overhead | 있음 | 없음 | Direct assignment |
| Warp divergence | 높음 | 낮음 | Uniform control flow |

**예상 A_stream 성능:**
```
현재: ~780 ms (scatter)
개선: ~200-250 ms (gather with bilinear interpolation)
개선율: 3-4x
```

**근거:**
- Memory bandwidth limited 연산
- Coalesced write는 이론적 4x 효율 (128B transaction vs 32B)
- Bilinear interpolation 추가 연산 비용 ~20%

### Energy Loss (A_E)

**현재 문제:**
- Monotonicity 제약으로 scatter fallback
- Loop over source bins: O(Ne)

**Multi-source Gather 구현 효과:**

| 요소 | 현재 | 개선 후 | 근거 |
|------|------|---------|------|
| LUT 적용률 | 0% (fallback) | 100% | Monotonicity 제약 제거 |
| Memory pattern | Scatter | Gather | Coalesced write |
| Loop structure | O(Ne) source loop | O(2) per target | Sparse source mapping |

**예상 A_E 성능:**
```
현재: ~730 ms (scatter with loop)
개선: ~150-200 ms (gather with LUT)
개선율: 4-5x
```

### Phase 2 전체 예상

```
현재 Total:     803 ms
├── A_theta:     30 ms  →  30 ms (변경 없음, FFT 기반)
├── A_stream:   780 ms → 220 ms (gather)
└── A_E:        730 ms → 180 ms (gather with LUT)

예상 Total:    ~430 ms
개선율:        ~1.9x
```

---

## Phase 3: 올바른 Sparse 구현 시

### 활성 셀 분석

**Deterministic Transport 특성:**
- 초기: 1개 bin에서 시작
- Angular scattering: θ 방향 확산 (Gaussian, ~3σ 범위)
- Spatial streaming: (z, x) 이동
- Energy loss: E 감소

**활성 셀 수 추정:**

| Step | θ 확산 | (z,x) 확산 | E bins | 총 활성 셀 |
|------|--------|-----------|--------|-----------|
| 초기 | 1 | 1 | 1 | 1 |
| 10 | ~10 | ~100 | ~5 | ~5,000 |
| 50 | ~20 | ~2,000 | ~20 | ~800,000 |
| 100 | ~30 | ~5,000 | ~50 | ~7,500,000 |
| 200 (Bragg) | ~40 | ~8,000 | ~100 | ~32,000,000 |

**참고:** Full grid = 98.3M, 최대 점유율 ~33%

### Sparse 연산 복잡도

**Dense 연산:**
```
A_theta:  O(Ne × Nθ × Nz × Nx × Nθ) = O(98.3M × 80) convolution
A_stream: O(Ne × Nθ × Nz × Nx) = O(98.3M)
A_E:      O(Ne × Nθ × Nz × Nx) = O(98.3M)
```

**Sparse 연산 (올바른 구현):**
```
A_theta:  O(n_active × Nθ) - position grouping + 1D convolution
A_stream: O(n_active)
A_E:      O(n_active)
```

### Hybrid Dense-Sparse 전략 성능 예상

**Angular Scattering (Dense 유지):**
- FFT 기반 convolution이 이미 효율적
- Dense: ~30 ms (현재)

**Spatial Streaming (Sparse):**
```
Dense gather: ~220 ms (Phase 2)
Sparse:       O(n_active) / O(N_total) × 220 ms

Step 50 기준: 800K / 98.3M × 220 ms ≈ 1.8 ms
Step 200 기준: 32M / 98.3M × 220 ms ≈ 72 ms
평균 (weighted): ~30-50 ms
```

**Energy Loss (Sparse):**
```
Dense gather: ~180 ms (Phase 2)
Sparse:       O(n_active) / O(N_total) × 180 ms

Step 50 기준: 800K / 98.3M × 180 ms ≈ 1.5 ms
Step 200 기준: 32M / 98.3M × 180 ms ≈ 59 ms
평균 (weighted): ~25-40 ms
```

### Phase 3 전체 예상

```
Phase 2 Total:  430 ms
├── A_theta:     30 ms →  30 ms (dense 유지)
├── A_stream:   220 ms →  40 ms (sparse, 평균)
├── A_E:        180 ms →  35 ms (sparse, 평균)
└── Overhead:     0 ms →  15 ms (dense↔sparse 변환)

예상 Total:    ~120 ms
개선율 vs Phase 2: ~3.6x
개선율 vs Baseline: ~6.7x
```

---

## 추가 최적화: Kernel Fusion

### Fused A_stream + A_E

**현재 (분리):**
```
A_stream: psi → psi_1 (write 98.3M × 4B = 393 MB)
A_E:      psi_1 → psi_2 (read 393 MB, write 393 MB)
Total memory traffic: 1.2 GB
```

**Fused:**
```
Fused: psi → psi_2 (read once, write once)
Total memory traffic: 786 MB
절감: 35%
```

**Fused Sparse Kernel 예상:**
```
Phase 3 (분리): 40 + 35 = 75 ms
Fused:          75 × 0.65 ≈ 50 ms
절감: ~25 ms
```

---

## 최종 성능 예상 요약

| 최적화 단계 | Step Time | vs Baseline | 누적 개선 |
|-------------|-----------|-------------|-----------|
| **Baseline (Phase 1)** | 803 ms | 1.0x | - |
| **Phase 2: Correct Gather** | 430 ms | 1.9x | 1.9x |
| **Phase 3: Hybrid Sparse** | 120 ms | 6.7x | 6.7x |
| **+ Kernel Fusion** | 95 ms | 8.5x | 8.5x |
| **+ Mixed Precision (FP16)** | 60-70 ms | 11-13x | 11-13x |

---

## Spec 목표 대비 평가

| 목표 | Spec 예상 | 실제 예상 | 달성 가능성 |
|------|-----------|-----------|-------------|
| Phase 2 | <200 ms | ~430 ms | ❌ (목표 과대) |
| Phase 3 | <200 ms | ~120 ms | ✅ |
| 최종 목표 | <100 ms | ~60-95 ms | ✅ (추가 최적화 시) |

---

## 제약 사항 및 주의점

### 1. Step별 성능 변동

```
Early steps (sparse):  ~30 ms  (n_active << N_total)
Mid steps:             ~80 ms
Late steps (dense):    ~150 ms (n_active → N_total)
```

Bragg peak 근처에서 weight 분포가 넓어지면서 sparse 이점 감소

### 2. Memory Overhead

```
Dense:  393 MB (psi array)
Sparse: 5 arrays × n_active × 4B
        Step 200: 5 × 32M × 4B = 640 MB
```

활성 셀이 많아지면 sparse가 오히려 메모리 더 사용

### 3. 변환 비용

```
Dense → Sparse: O(N_total) scan for nonzero
Sparse → Dense: O(n_active) scatter

Step당 추가 비용: ~10-20 ms
```

### 4. Full Grid (190M) 예상

```
Medium (98.3M) 기준 스케일링:
Phase 3 예상: 120 ms × (190/98.3) ≈ 230 ms

Full grid에서 <200 ms 달성 어려움
→ Multi-GPU 또는 추가 최적화 필요
```

---

## 결론

| 구현 | Medium Grid (98.3M) | Full Grid (190M) |
|------|---------------------|------------------|
| 올바른 Phase 2 | ~430 ms | ~830 ms |
| 올바른 Phase 3 | ~120 ms | ~230 ms |
| + Fusion + FP16 | ~60-70 ms | ~120-140 ms |

**Spec의 <200 ms 목표:**
- Medium Grid: Phase 3로 달성 가능 ✅
- Full Grid: 추가 최적화 또는 Multi-GPU 필요 ⚠️