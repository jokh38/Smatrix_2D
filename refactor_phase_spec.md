아래는 “초기 개선안 + 평가 피드백 + 추가 제안”을 모두 반영해 **Smatrix_2D 리팩토링/구현을 실제로 진행할 수 있도록** 하나의 **통합 기술 문서(Spec)** 형태로 재작성한 것입니다. 마지막 섹션에, 제안안 자체를 더 안전/고성능으로 만들기 위한 **추가 수정 제안**도 포함했습니다.

---

# Smatrix_2D GPU-Only Runtime Refactor Specification (v1.0)

## 0. 목적과 범위

### 0.1 목적

* 런타임은 **GPU-only (CuPy/CUDA)**로 단일화하여 성능 및 유지보수성을 확보한다.
* 물리적/수치적 정합성을 보장하기 위해 **검증용 reference(비런타임)**을 유지한다.
* CPU/GPU 불일치(escape/leakage/energy 정책)를 제거하고, 보존량(conservation) 추적을 일관화한다.

### 0.2 범위 (In Scope)

* transport 루프(각 step) 및 operator 적용을 **GPU kernel 기반 단일 경로**로 통합
* 설정(그리드/물리/수치/경계/분할)을 SSOT(Single Source of Truth)로 중앙화
* escape/leakage를 **차이 기반**이 아닌 **직접 추적(direct tracking)**으로 변경
* step별 동기화를 제거하고, **GPU 누적 + 최소 동기화**로 변경
* 검증 프레임워크: reference CPU 또는 golden snapshot 기반 회귀 검증, NIST range 검증 포함

### 0.3 비범위 (Out of Scope)

* multi-GPU, MPI, distributed
* 3D 확장
* 정교한 재질/이종 매질(초기 버전에서는 물/균질을 기본으로 가정)
* 완전한 물리 모델 확장(energy straggling 등)은 “확장 포인트”로만 명시

---

## 1. 핵심 설계 원칙

### P1. GPU-only 런타임 단일 경로

* `transport/` 실행 경로에서 CPU operator/step 구현은 제거한다.
* CPU 코드는 런타임에 참여하지 않으며 `validation/`에서만 사용한다.

### P2. 검증 기준점은 유지(Regression 방지)

* “CPU reference 구현” 또는 “golden snapshot” 중 최소 1개는 반드시 유지한다.
* GPU 최적화(atomic/reduction/메모리 레이아웃 변경) 시 물리 퇴행을 자동 감지한다.

### P3. Host-device 동기화 최소화

* step 루프 내부에서 `.get()`, `float(cp.sum())` 등 동기화를 유발하는 호출을 금지한다.
* 보존/escape/leakage/dose 누적은 GPU에서 수행하고, CPU로의 전송은 종료 시 1회(또는 coarse interval)만 수행한다.

### P4. Escape/Leakage는 “direct tracking”

* `sum(in) - sum(out)` 같은 차이 기반 leakage는 금지한다.
* boundary-crossing으로 정의되는 물리 채널(공간/각/에너지)은 커널에서 직접 누적한다.
* 필요 시 “residual(수치 잔차)”을 별도 채널로 분리하여 보고한다.

### P5. Energy grid 정책: E_cutoff buffer 강제

* `E_cutoff > E_min`을 강제하고, **최소 buffer (예: 1 MeV)**를 권장/강제한다.
* cutoff 영역 처리는 “정책”으로 문서화하고, 코드에서 일관되게 구현한다.

---

## 2. 목표 디렉터리 구조

```
smatrix_2d/
├── config/
│   ├── defaults.py              # 상수만 (import * 금지)
│   ├── enums.py                 # EnergyGridType, BoundaryPolicy, SplittingType 등
│   ├── simulation_config.py     # SSOT (dataclass/pydantic)
│   └── validation.py            # config self-check & invariants
├── core/
│   ├── grid.py                  # grid construction, indexing helpers
│   ├── materials.py             # material props (rho 등)
│   ├── lut.py                   # stopping power LUT, range integration
│   ├── constants.py
│   └── accounting.py            # channel definitions, reporting schema (host-side)
├── gpu/
│   ├── kernels.py               # RawKernel/RawModule + launch wrappers
│   ├── memory_layout.py         # strides, packing, dtype policies
│   ├── accumulators.py          # GPU-side accum arrays + reset utilities
│   ├── tiling.py                # optional (block/grid tuning)
│   └── operators/
│       ├── angular_scatter.cu   # direct tracking 구현
│       ├── energy_loss.cu
│       └── spatial_stream.cu
├── transport/
│   ├── simulation.py            # GPU-only simulation loop (no CPU fallback)
│   └── api.py                   # convenience builders / CLI entry
└── validation/
    ├── reference_cpu/           # 테스트 전용 CPU 구현(선택)
    ├── golden_snapshots/        # 특정 버전 정답 데이터
    ├── compare.py               # tolerances & metrics
    └── nist_validation.py       # range 검증 (전 구간)
```

---

## 3. 설정(SSOT) 스펙

### 3.1 config/defaults.py

* 상수만 정의한다.
* `import *` 사용 금지(이름 충돌 및 추적 불가 방지).

권장 기본값(요지):

* `E_min = 1.0`
* `E_cutoff = 2.0` (buffer >= 1 MeV)
* dtype 정책: psi=float32, dose=float32(또는 float64 옵션), accumulator=float64

### 3.2 config/enums.py

* `EnergyGridType`: UNIFORM, RANGE_BASED(확장)
* `BoundaryPolicy`: ABSORB (기본), REFLECT(테스트용), PERIODIC(금지/실험용)
* `SplittingType`: FIRST_ORDER, STRANG
* `BackwardTransportPolicy`: HARD_REJECT 등(기존 설계에 맞춤)

### 3.3 config/simulation_config.py

#### 필수 필드

* Grid:

  * Nx, Nz, Ntheta, Ne
  * x/z/theta/E min/max
  * `E_min`, `E_cutoff`, `E_max`
  * energy_grid_type
  * boundary_policy (space/theta 각각 분리 가능)
* Transport:

  * delta_s, max_steps
  * splitting_type, sub_steps
  * sigma bucket: n_buckets, k_cutoff
* Numerics:

  * weight_threshold
  * beta_sq_min
  * accumulator_dtype
  * sync_interval (예: 0=종료 시 1회, N=매 N step마다만 sync)

#### 불변식(invariants) 검증

* `E_cutoff > E_min`
* `E_cutoff >= E_min + buffer_min` (기본 1.0 MeV)
* grid sizes > 0
* theta range와 Ntheta 일관성
* dtype 정책 유효성
* boundary policy 허용 여부(기본 ABSORB)

---

## 4. 데이터 레이아웃과 dtype 정책

### 4.1 psi 텐서 레이아웃

* 기본: `psi[Ne, Ntheta, Nz, Nx]` contiguous
* 이유:

  * 에너지/각/공간 연산을 순차적으로 적용하는 operator splitting과 맞춤
  * 커널에서 특정 축 반복 접근이 명확해짐

### 4.2 dtype

* `psi`: float32 권장 (성능/메모리)
* `dose`: float32 기본, 고정밀 검증 모드에서 float64 옵션
* `accumulators(escape/leakage/residual/mass)`: float64 권장

  * direct tracking을 float64로 누적해야 “물리 채널 vs 수치 잔차” 분리가 명확해짐

---

## 5. GPU Accumulator 설계 (동기화 최소화 핵심)

### 5.1 GPU-side accumulator arrays

* `escapes_gpu[ch]` : float64, ch = {THETA_BOUNDARY, THETA_CUTOFF, ENERGY_STOPPED, SPATIAL_LEAK, RESIDUAL, ...}
* `mass_in_gpu[step]`, `mass_out_gpu[step]` (옵션): float64 또는 float32
* `deposited_gpu[step]` 또는 누적 `deposited_total_gpu`: float64 권장

### 5.2 동기화 정책

* step 루프 내:

  * 금지: `cp.sum`, `.get()`, `float(...)` 등
  * 허용: kernel launch, GPU array in-place update
* sync는 아래 중 하나만:

  * run 종료 시 1회
  * 또는 `sync_interval` step마다 1회(디버그/모니터링 모드)

---

## 6. Operator 스펙 (GPU-only)

각 operator는 다음 인터페이스를 갖는다.

### 6.1 Angular Scattering (direct tracking)

#### 요구사항

* theta domain boundary를 넘어가는 분량은 **직접 escape로 누적**
* “used_sum/full_sum” 비율 기반 보정은 기본 버전에서 사용하지 않음(설정 변화에 취약)
* 커널 LUT는:

  * sigma bucket별 kernel weights를 global memory에 저장 (또는 shared로 block 캐싱)
  * constant memory 의존 금지(확장성 이슈 회피)

#### 직접 추적 방식(개념)

* 각 input bin (ith_src)에 대해 kernel support를 적용
* ith_target이 [0, Ntheta) 밖이면 `escape_theta_boundary += psi_in * kernel_w`
* 안이면 `psi_out += psi_in * kernel_w`

#### 추가 정책(선택)

* `theta_cutoff` 정책이 존재한다면(예: forward-only):

  * cutoff 밖은 별도 채널(THETA_CUTOFF)로 누적
  * boundary(격자 밖)와 cutoff(정책 제거)는 분리 보고

---

### 6.2 Energy Loss (dose in-place, cutoff 처리 일관화)

#### 요구사항

* `E < E_cutoff`로 내려가는 분량은 즉시 deposit 또는 energy-stopped escape로 누적 (정책 명시)
* `E_min`와 `E_cutoff`가 가까울 때 수치 민감도가 커지므로 buffer 강제 정책 적용

#### 권장 정책

* **정책 EL-1 (기본)**:

  * 한 step에서 energy advection 후, `E < E_cutoff`로 들어간 모든 weight는

    * `dose += deposited_energy(weight, dE)`로 누적하고
    * psi에서는 제거
    * `ENERGY_STOPPED` 채널로 weight(또는 에너지)를 누적(무엇을 누적할지 명시 필요)

> 중요한 결정: escape channel에 “weight”를 넣을지 “energy”를 넣을지 혼재하면 해석이 어려워집니다. 본 문서에서는 기본적으로 **weight(확률 질량)** 채널과 **energy(선량/에너지)** 채널을 분리할 것을 권장합니다.

---

### 6.3 Spatial Streaming (direct leakage tracking)

#### 요구사항

* 경계 밖으로 나가는 flux를 커널 내에서 **직접 누적**
* `sum(in)-sum(out)` 방식 금지
* 스트리밍 방식은 gather/scatter 선택 가능하나, leakage direct tracking이 자연스러운 방향을 우선

#### 권장 구현 방향

* **scatter형**이 leakage 직접 누적에 유리한 경우가 많음:

  * source가 domain 내부일 때만 target에 기여
  * domain 밖으로 나가는 경로는 boundary check로 direct leak 누적

#### residual 분리

* direct leak를 계산한 뒤에도, 수치적 이유로 보존이 어긋나면

  * `RESIDUAL = max(0, mass_in - mass_out - (sum_of_weight_escapes))`
  * residual은 경고/리포팅 대상으로만 사용(물리 채널과 분리)

---

## 7. Transport Loop 스펙 (GPU-only, sync 최소화)

### 7.1 simulation.py 책임

* config 로드 및 검증
* grid, LUT, sigma buckets 준비
* GPU state(psi, dose, accumulators) 생성 및 초기화
* run loop 수행(동기화 최소화)
* 종료 시 결과를 CPU로 가져와 리포트 생성

### 7.2 SplittingType

* FIRST_ORDER:

  * psi = A_stream(A_E(A_theta(psi))) 또는 repo 기존 순서 유지
* STRANG:

  * A_theta(Δs/2) → A_E(Δs) → A_stream(Δs) → A_theta(Δs/2)
* sub_steps:

  * 안정성을 위해 각 operator를 subcycling할 수 있도록 확장 포인트 제공

### 7.3 Conservation history 기록

* 기본 모드에서는 GPU에만 누적
* 종료 시 CPU로 가져와 step별 테이블 생성
* 디버그 모드(sync_interval>0)일 때만 중간 확인

---

## 8. Validation 스펙 (CPU reference / golden snapshot / NIST)

### 8.1 최소 요구사항

* 다음 중 최소 1개 이상:

  1. `validation/reference_cpu/` (테스트 전용)
  2. `validation/golden_snapshots/` (정답 데이터)

### 8.2 Golden snapshot 권장 포맷

* config YAML + 결과 npz:

  * `psi_final`, `dose_final`
  * `escapes_by_channel`
  * `mass_in/out per step`(선택)
* tolerance:

  * float32/atomic 비결정성을 감안한 상대 오차 기준(예: 1e-5~1e-3, 항목별로 다르게)

### 8.3 NIST range validation (전 구간)

* 단일 에너지 점 검증(예: 70 MeV)은 “스모크 테스트”
* 필수는:

  * `Range(E0) = ∫ dE / S(E)`를 전 구간에서 계산하고
  * NIST range table과 비교(오차 기준 설정)
* LUT 단위 혼동 방지:

  * mass stopping power vs linear stopping power 변환, 밀도 적용, cm↔mm 변환을 명확히 문서화

---

## 9. 구현 단계(Phase Plan)

### Phase 0: 구조 이동 및 SSOT 적용

* config 모듈 생성 + 기존 흩어진 defaults 제거
* `SimulationConfig`로 grid/LUT/operator 설정을 단일화
* 런타임 경로에서 CPU fallback 제거(하지만 validation에는 유지)

### Phase 1: GPU-only 루프 + accumulator 도입

* `TransportSimulation`에서 step loop 내 `.get()/sum()` 제거
* `escapes_gpu` 및 `deposited_total_gpu` 누적
* 종료 시 1회만 결과 fetch

### Phase 2: direct tracking 적용

* angular scattering direct escape
* spatial streaming direct leak
* residual 분리 보고

### Phase 3: 검증 체계 고도화

* golden snapshot 생성/비교 자동화
* NIST range 전 구간 검증 도입
* 성능 회귀(throughput)도 같이 측정(선택)

---

## 10. 위험요인 및 완화책

### R1. CPU 완전 삭제로 인한 regression 감지 불능

* 완화: validation/reference_cpu 또는 golden snapshot 필수 유지

### R2. step loop의 host sync로 인한 성능 붕괴

* 완화: accumulator GPU 누적 + 종료 시 1회 fetch, sync_interval 옵션

### R3. leakage/escape를 차이 기반으로 계산해 물리 채널 오염

* 완화: direct tracking + residual 분리

### R4. E_cutoff 경계 수치 민감도

* 완화: `E_cutoff >= E_min + buffer` 강제, 정책 문서화 및 단일 구현

### R5. atomicAdd 비결정성으로 인한 테스트 불안정

* 완화:

  * 비교 metric을 “절대 동일”이 아닌 tolerance 기반으로 설계
  * 주요 합산(accumulator)은 float64 사용
  * 필요 시 deterministic mode(느리지만 안정) 옵션을 별도 제공

---

# 추가 수정 제안(이번 통합안에 더해 권장)

아래는 사용자가 준 통합안(추가 제안 포함)을 “실제 구현 안정성/성능” 관점에서 한 단계 더 개선하는 제안입니다.

## A) EscapeAccounting의 “채널 정의(무엇을 누적하는가)”를 명확히 분리

현재 문서/코드 흐름에서는 escape 채널에 “weight”가 들어가기도 하고 “energy”가 들어가기도 쉬운데, 이는 분석/검증을 어렵게 합니다.

* **권장**: 채널을 2계열로 분리

  * `escape_weight[ch]` : 확률 질량(psi weight)
  * `escape_energy[ch]` : 에너지(선량/에너지 손실)
* 최소한 naming으로라도 강제:

  * `ENERGY_STOPPED_WEIGHT` vs `ENERGY_DEPOSITED_TOTAL`

## B) psi 업데이트 방식의 선택: scatter vs gather를 operator별로 분리

* angular scattering: scatter(atomic) vs gather(메모리 접근) 중 성능/정확성 트레이드오프가 큼
* spatial streaming: leakage direct tracking 관점에서 scatter가 유리한 경우가 많음
* 에너지 loss: 보통 gather/advection 형태가 직관적

**권장**: “operator별로 최적의 패턴”을 허용하고, 단 최종 보존 검증으로 안전장치 확보

## C) 커널 LUT 캐싱은 constant memory 고정 대신 “shared tile 캐싱” 우선

constant memory는 확장성(Ntheta 증가)에 취약합니다.

* shared 캐싱: block 단위로 kernel support만 로드(필요 크기만)
* 또는 sigma bucket kernel을 “압축 형태(half-width + weights)”로 저장하여 global traffic 감소

## D) “determinism level” 옵션을 명시

* Level 0 (fast): atomicAdd, float32 psi, float64 accum, tolerance 테스트
* Level 1 (more stable): block-level reduction 후 atomic, 더 좁은 tolerance
* Level 2 (debug): 일부 연산 float64, sync_interval 사용

이 옵션은 “성능 vs 재현성”을 팀/사용자가 통제할 수 있게 해줍니다.

---

## 최종 요약

이번 통합 스펙의 핵심 결론은 다음입니다.

1. 런타임은 GPU-only로 단일화하되, **검증 기준(reference CPU 또는 golden snapshot)은 반드시 유지**
2. GPU-resident를 진짜로 만들기 위해, **step loop 내부 동기화 금지 + GPU 누적 accumulator 채택**
3. escape/leakage는 **direct tracking**으로 구현하고, **수치 잔차(residual)는 별도 채널로 분리**
4. `E_cutoff`는 `E_min`보다 충분히 높게(버퍼 강제) 두어 **경계 수치 불안정**을 제거
5. 설정은 SSOT로 중앙화하되, repo의 옵션(energy grid type, splitting, boundary, backward 정책)을 **누락 없이 포함**
