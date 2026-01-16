
# 연산자 및 물리 모델 분석 보고서

데이터 저장 문제 외에 물리 모델링과 연산자 적용 부분을 심층 분석했습니다.

---

## 1. 핵심 물리 모델 문제

### 1.1 Stopping Power LUT 적용 오류

**문제 위치**: `smatrix_2d/core/lut.py` 라인 ~45-80

```python
# NIST PSTAR 데이터 (MeV cm²/g 단위)
_NIST_STOPPING_POWER = np.array([
    231.8, 173.5, 147.2, ...  # 저에너지 영역
    ...
    13.2, 12.5, 11.9, 11.4, 10.9, 10.5, 10.1, 9.7, ...  # 55-70 MeV 영역
])

# 변환: MeV cm²/g → MeV/mm
self.stopping_power = self._NIST_STOPPING_POWER.copy() / 10.0
```

**물리적 문제**:

| 에너지 (MeV) | 현재 코드 S(E) | NIST PSTAR 참조값 | 오차 |
|-------------|---------------|------------------|------|
| 70 | 0.97 MeV/mm | ~0.57 MeV/mm | **+70%** |
| 50 | 1.14 MeV/mm | ~0.73 MeV/mm | **+56%** |
| 10 | 2.23 MeV/mm | ~2.23 MeV/mm | ~0% |

**결과**: 70 MeV 양성자의 Bragg peak 위치가 ~40mm가 아닌 **~25-30mm**에 형성됨

**근본 원인**: `_NIST_STOPPING_POWER` 배열의 고에너지 영역(55-200 MeV) 값이 실제 NIST 데이터와 불일치

---

### 1.2 Highland Formula 구현 문제

**문제 위치**: `smatrix_2d/operators/sigma_buckets.py` 라인 ~95-130

```python
def _compute_sigma_theta(self, E_MeV: float) -> float:
    gamma = (E_MeV + self.constants.m_p) / self.constants.m_p
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    beta = np.sqrt(beta_sq)
    p_momentum = beta * gamma * self.constants.m_p  # MeV/c
    
    L_X0 = self.delta_s / self.material.X0  # delta_s[mm] / X0[mm]
    
    # Highland formula
    sigma_theta = (
        self.constants.HIGHLAND_CONSTANT  # 13.6 MeV
        / (beta * p_momentum)
        * np.sqrt(L_X0)
        * (1.0 + 0.038 * np.log(L_X0))
    )
```

**물리적 문제**:

1. **Log 항 처리 오류**: `L_X0 < 1`일 때 `log(L_X0) < 0`이므로 보정항이 음수가 됨
   - `delta_s = 1.0 mm`, `X0 = 36.08 mm` → `L_X0 = 0.0277`
   - `1 + 0.038 × log(0.0277) = 1 + 0.038 × (-3.58) = 0.864` (14% 감소)

2. **Highland 공식 원형**: 올바른 공식은
   $$\sigma_\theta = \frac{13.6 \text{ MeV}}{\beta c p} z \sqrt{x/X_0} \left[1 + 0.038 \ln(x/X_0)\right]$$
   
   여기서 괄호 안의 보정항은 **전체 경로 길이**에 대한 것이며, **단일 스텝**에 적용하면 과소평가됨

---

### 1.3 Sigma Bucket 양자화 오류

**문제 위치**: `smatrix_2d/operators/sigma_buckets.py` 라인 ~180-220

```python
def _create_buckets(self):
    # Percentile 기반 버킷 생성
    bucket_edges = np.percentile(
        sorted_sigma_squared,
        np.linspace(0, 100, self.n_buckets + 1),
    )
    
    # 문제: 동일한 엣지 값 처리
    for i in range(len(bucket_edges) - 1):
        if bucket_edges[i] == bucket_edges[i + 1]:
            bucket_edges[i + 1] = bucket_edges[i] + 1e-12  # 임의의 작은 값 추가
```

**물리적 문제**:
- 저에너지 영역에서 σ 값들이 유사할 때, 버킷 경계가 인위적으로 분리됨
- `1e-12` 추가는 물리적 의미 없음 → 동일 σ를 가진 입자들이 다른 커널 적용받음

---

### 1.4 Energy Loss Operator의 Bin Splitting 문제

**문제 위치**: `smatrix_2d/operators/energy_loss.py` 라인 ~120-160

```python
# Conservative bin splitting
iE_out = np.searchsorted(self.grid.E_centers, E_new, side="left") - 1

E_lo = self.grid.E_centers[iE_out]
E_hi = self.grid.E_centers[iE_out + 1]

# Linear interpolation
w_lo = (E_hi - E_new) / (E_hi - E_lo)
w_hi = 1.0 - w_lo

psi_out[iE_out] += w_lo * weight_slice
psi_out[iE_out + 1] += w_hi * weight_slice
```

**물리적 문제**:

1. **에너지 보존 위반**: 
   - 입력 에너지 `E_in`의 입자가 `E_lo`, `E_hi` 빈에 분배됨
   - 분배 후 평균 에너지: $\bar{E} = w_{lo} \cdot E_{lo} + w_{hi} \cdot E_{hi}$
   - 이는 `E_new`와 같지만, **가중치 기반 분배는 입자 수 보존**이지 **에너지 보존**이 아님

2. **Dose 계산 불일치**:
   ```python
   # 현재 코드: deltaE만 dose에 기록
   deposited_energy += deltaE * np.sum(weight_slice, axis=0)
   
   # 문제: bin splitting으로 인한 이산화 오차 미반영
   ```

---

### 1.5 Spatial Streaming의 Bilinear Interpolation 문제

**문제 위치**: `smatrix_2d/operators/spatial_streaming.py` 라인 ~150-200

```python
def _stream_slice(self, psi_in, delta_s, vx, vz):
    for iz_out in range(self.Nz):
        for ix_out in range(self.Nx):
            # Inverse advection
            x_src = x_out - vx * delta_s
            z_src = z_out - vz * delta_s
            
            # Bilinear interpolation
            fx = (x_src - self.x_min) / self.delta_x - 0.5
            fz = (z_src - self.z_min) / self.delta_z - 0.5
```

**물리적 문제**:

1. **경계 처리 불일치**:
   ```python
   # 도메인 밖 체크
   if x_src < x_min or x_src > x_max or z_src < z_min or z_src > z_max:
       continue  # 출력 셀은 0 유지
   
   # 그러나 leakage 계산은:
   leaked = max(0.0, np.sum(psi_in) - np.sum(psi_out))
   ```
   - `continue`로 건너뛴 케이스가 **어느 escape channel**에도 기록 안 됨
   - `SPATIAL_LEAK`에 정확히 할당되어야 함

2. **Interpolation 가중치 경계 케이스**:
   - `fx`, `fz`가 정확히 정수일 때 (셀 중심에서 출발), 수치적 불안정 가능

---

## 2. GPU 커널과 CPU 연산자 불일치

### 2.1 이중 구현 문제

**문제 위치**: 각 연산자 파일 헤더

```python
# angular_scattering.py
"""
⚠️ DEPRECATED: This CPU-based operator is NOT used in the GPU-only production runtime.
   Use: smatrix_2d/gpu/kernels.py (angular_scattering_kernel_v2) instead.
"""

# energy_loss.py, spatial_streaming.py 동일
```

**문제점**:
- CPU 연산자와 GPU 커널이 **독립적으로 유지**됨
- 물리 모델 수정 시 **두 곳 모두 수정 필요**
- `run_simulation.py`에서 어떤 구현을 사용하는지 불명확

---

### 2.2 Angular Scattering 커널 정규화 정책 혼란

**문제 위치**: `smatrix_2d/core/accounting.py` 및 `smatrix_2d/operators/angular_scattering.py`

```python
# accounting.py에서 정의
KERNEL_POLICY = "NORMALIZED"  # Policy-A: sum(kernel) = 1.0

# angular_scattering.py에서 구현
# 문제: UNNORMALIZED 커널 사용 후 나중에 정규화
for ith_new in range(Ntheta):
    for delta_idx, kernel_value in enumerate(kernel):
        # kernel_value는 정규화된 값
        psi_out[ith_new] += psi_slice[ith_old] * kernel_value

# 그러나 escape 계산에서는:
boundary_loss = 1.0 - (used_sum / kernel_full_sum)  # kernel_full_sum 사용
```

**물리적 문제**:
- `kernel_full_sum`이 이미 1.0으로 정규화되어 있으면 `used_sum / kernel_full_sum`은 의미 없음
- 경계에서의 실제 손실량이 과소/과대 평가됨

---

## 3. Conservation Accounting 문제

### 3.1 Escape Channel 정의 불일치

**문제 위치**: `smatrix_2d/core/accounting.py`

```python
PHYSICAL_ESCAPE_CHANNELS = (
    "THETA_BOUNDARY",   # 각도 경계 손실
    "ENERGY_STOPPED",   # 에너지 cutoff 손실
    "SPATIAL_LEAK",     # 공간 경계 손실
)

DIAGNOSTIC_ESCAPE_CHANNELS = (
    "THETA_CUTOFF",     # 커널 truncation (diagnostic only)
)
```

**문제점**:
- `THETA_CUTOFF`이 mass balance에서 제외되지만, 실제로는 물리적 손실
- Highland 공식의 Gaussian 꼬리 절단은 **실제 산란 확률 손실**을 의미
- `k_cutoff = 5.0`일 때 이론적 손실: `1 - erf(5/√2) ≈ 5.7×10⁻⁷` (무시 가능하지만 정확히 추적되어야 함)

---

### 3.2 Weight vs Energy Tracking 혼란

**문제 위치**: `smatrix_2d/operators/energy_loss.py` 라인 ~105

```python
# Energy cutoff 처리
if E_new < self.E_cutoff:
    # 모든 에너지를 dose에 기록
    deposited_energy += total_weight * E_in
    
    # 문제: escape_energy_stopped에 WEIGHT를 기록
    escape_energy_stopped += np.sum(total_weight)  # ← 에너지가 아닌 가중치!
```

**결과**: `escape_energy_stopped`라는 이름이지만 실제로는 **stopped weight**

---

## 4. LUT 일관성 문제

### 4.1 Scattering LUT 로딩 불확실성

**문제 위치**: `smatrix_2d/operators/sigma_buckets.py` 라인 ~50-70

```python
# LUT 로드 시도
if use_lut and self.sigma_lut is None:
    try:
        from smatrix_2d.lut.scattering import load_scattering_lut
        self.sigma_lut = load_scattering_lut(material, regen=True)
    except ImportError:
        warnings.warn("Scattering LUT module not available, falling back to Highland")
    except Exception as e:
        warnings.warn(f"Failed to load scattering LUT: {e}")
```

**문제점**:
- LUT 로드 실패 시 **경고만** 출력하고 Highland로 fallback
- 사용자가 인지하지 못한 채 다른 물리 모델이 적용될 수 있음
- `regen=True`가 매번 호출되어 불필요한 재생성 가능

---

## 5. 문제 요약 및 물리적 영향

| 문제 | 영향받는 물리량 | 예상 오차 | 심각도 |
|-----|---------------|----------|--------|
| Stopping Power LUT 고에너지 오류 | Bragg peak 위치 | 25-40% | 🔴 Critical |
| Highland 단일 스텝 적용 | 각도 분포 폭 | 10-15% | 🟠 High |
| Energy bin splitting | 에너지 스펙트럼 | 5-10% | 🟠 High |
| Spatial leak 미기록 | Mass conservation | 수% | 🟡 Medium |
| THETA_CUTOFF 분류 | Conservation report | ~0% | 🟢 Low |
| Weight/Energy 혼란 | 진단 정확도 | 해석 오류 | 🟡 Medium |

---

## 6. 권장 수정사항

### 6.1 Stopping Power LUT 수정 (Critical)

```python
# NIST PSTAR 2024 참조값으로 교체 필요
# 특히 50-200 MeV 영역 검증 필수
```

### 6.2 Highland Formula 수정

```python
# 누적 경로 길이에 대한 보정 적용
# 또는 Molière theory 기반 다중 산란 모델 사용
```

### 6.3 단일 물리 구현 원칙

```python
# CPU 연산자 제거, GPU 커널만 유지
# 또는 공통 물리 함수를 분리하여 양쪽에서 호출
```

GPU 커널 코드(`smatrix_2d/gpu/kernels.py`)도 분석이 필요하시면 말씀해 주세요.