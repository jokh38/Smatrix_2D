# Smatrix_2D GPU-Only Refactor Implementation Plan v1.0

## 0. 목표 산출물

### 0.1 최종 목표

* 런타임(`transport/`)은 **GPU-only 단일 경로**로 동작.
* step 루프에서 **host-device sync 제거**(기본: 종료 시 1회만 결과 fetch).
* escape/leakage는 **direct tracking**으로 구현(차이 기반 금지).
* `SimulationConfig`가 **SSOT**로 모든 설정을 관장.
* `validation/`에 **reference 기준점**(CPU reference 또는 golden snapshot) 유지.
* **NIST range 전 구간 검증**을 자동화.

### 0.2 Done 정의(완료 조건)

* `python -m smatrix_2d.transport.api run --config ...` 형태(또는 기존 진입점)로 GPU-only 실행 가능.
* `validation/compare.py`가 golden snapshot 대비 tolerance 내 통과.
* `validation/nist_validation.py`가 range 오차 기준 내 통과.
* 런타임 코드에서 CPU operator/step 클래스가 import 되지 않음(테스트 제외).
* step 루프에서 `.get()`, `cp.sum()` 등 동기화 호출이 기본 모드에서 0회.

---

## 1. 브랜치/작업 전략

### 1.1 브랜치

* `refactor/gpu-only-v1` (feature branch)
* merge 전 `main`과 rebase

### 1.2 단계별 머지(권장)

* Phase 0/1/2/3 별로 PR 분리

  * PR-0: config SSOT + 구조 이동(기능 변화 최소)
  * PR-1: GPU accumulator + sync 제거
  * PR-2: direct tracking(angular/streaming)
  * PR-3: validation(NIST/golden) + 문서화

---

## 2. 파일/모듈 변경 계획(파일 단위 TODO)

아래는 “새 구조” 기준입니다. 기존 파일은 단계적으로 이동/폐기합니다.

### 2.1 신규 생성

* `smatrix_2d/config/defaults.py`
* `smatrix_2d/config/enums.py`
* `smatrix_2d/config/simulation_config.py`
* `smatrix_2d/config/validation.py`
* `smatrix_2d/core/accounting.py`  (채널 정의/리포트 스키마)
* `smatrix_2d/gpu/accumulators.py`
* `smatrix_2d/transport/simulation.py` (GPU-only loop)
* `smatrix_2d/transport/api.py` (CLI/편의 생성)
* `validation/golden_snapshots/*` (샘플 1~3개)
* `validation/compare.py`
* `validation/nist_validation.py`

### 2.2 리팩토링/이동

* GPU 커널 코드(기존 `kernels.py`)를 다음으로 분해(가능하면):

  * `smatrix_2d/gpu/operators/angular_scatter.cu`
  * `smatrix_2d/gpu/operators/energy_loss.cu`
  * `smatrix_2d/gpu/operators/spatial_stream.cu`
  * `smatrix_2d/gpu/kernels.py`는 RawModule 로더/런처 역할만

### 2.3 런타임에서 제거(또는 validation으로 격리)

* CPU용 `TransportStepV2` 및 CPU operator(angular_scattering.py, energy_loss.py, spatial_streaming.py 등)

  * 런타임에서 import 금지
  * 필요 시 `validation/reference_cpu/`로 이동(Phase 3에서 결정)

---

## 3. Phase Plan

## Phase 0 — SSOT 설정 중앙화 + 구조 정리(기능 변화 최소)

**목표:** 실행 결과는 거의 유지하면서 “설정/구조 기반”을 먼저 고정.

### 작업 항목

1. `config/enums.py` 작성

* EnergyGridType, BoundaryPolicy, SplittingType, BackwardTransportPolicy

2. `config/defaults.py` 작성

* 반드시 기본값:

  * `DEFAULT_E_MIN = 1.0`
  * `DEFAULT_E_CUTOFF = 2.0`
* dtype 정책 상수화:

  * `PSI_DTYPE=float32`, `ACC_DTYPE=float64`

3. `config/simulation_config.py` 작성(SSOT)

* GridConfig / TransportConfig / NumericsConfig / SimulationConfig
* `__post_init__` 또는 `config/validation.py`에서 불변식 검사:

  * `E_cutoff > E_min`
  * `E_cutoff >= E_min + 1.0` (기본 강제 또는 warn+옵션)

4. 기존 분산 설정값 제거/단일화

* `grid.py`, `transport.py`, `config_resolver.py` 등에 있던 defaults를 SSOT로 연결
* “중복 기본값”을 삭제하거나 deprecated 처리

### 산출물

* `SimulationConfig`로 실행 파라미터가 결정되는 구조 완성
* 기존 테스트/샘플 실행이 설정 충돌 없이 동작

### 검증(acceptance)

* 기존 샘플 실행이 동작(결과 완전 일치까지는 요구하지 않음)
* `SimulationConfig.validate()`가 주요 불변식 위반을 차단

---

## Phase 1 — GPU accumulator 도입 + step loop sync 제거

**목표:** “GPU-resident”를 진짜로 만들고 성능 병목(동기화)을 제거.

### 작업 항목

1. `gpu/accumulators.py`

* `EscapesAccumulator`(GPU array float64) 및 reset 함수
* 채널 인덱스는 `core/accounting.py`에서 정의(단일 진실)

2. `core/accounting.py`

* 채널 정의를 **명확히 분리**:

  * `escape_weight[ch]` (psi weight)
  * `energy_deposited_total` (에너지/선량)
  * (선택) `escape_energy[ch]`를 별도로 둘지 결정
* 리포트 스키마(`ConservationReport`) 정의

3. `transport/simulation.py` (GPU-only loop) 구현

* step 루프 내 금지:

  * `cp.sum`, `.get()`, `float(...)`
* 허용:

  * kernel launch만 수행
* `sync_interval` 지원:

  * 0이면 종료 시 1회만 fetch
  * N>0이면 매 N step마다 중간 fetch/로그

4. 커널 API 변경

* `apply_gpu_resident(psi_gpu, dose_gpu, escapes_gpu, ...)` 형태로 통일
* escapes는 커널 내 `atomicAdd`로 누적

### 산출물

* 기본 모드에서 host sync가 0회인 run loop
* 종료 시에만 `psi/dose/escapes`를 CPU로 가져와 리포트 생성

### 검증(acceptance)

* 프로파일링(간단 측정)에서 step당 CPU 시간이 커널 런칭 위주로만 나타남
* `sync_interval=0` 모드에서 `.get()`이 호출되지 않음(코드 검사/grep)

---

## Phase 2 — Direct tracking 구현(Angular escape + Spatial leakage)

**목표:** 물리 채널을 수치 잔차와 분리하고 CPU/GPU 불일치를 구조적으로 제거.

### 작업 항목

#### 2.1 Angular Scattering: direct clip + escape

* 기존 “used_sum/full_sum 보정” 대신:

  * kernel support를 적용하며 `ith_target`이 밖이면 `THETA_BOUNDARY_WEIGHT += w`
* cutoff(예: forward-only)가 있으면:

  * `THETA_CUTOFF_WEIGHT` 채널로 분리 누적
* LUT 저장 정책:

  * constant memory 의존 금지
  * shared tile 캐싱(선택)

#### 2.2 Spatial Streaming: direct boundary leak

* streaming 단계에서 domain 밖으로 나가는 기여를 직접 계산:

  * `SPATIAL_LEAK_WEIGHT += w_outside`
* 차이 기반 `sum(in)-sum(out)` 제거

#### 2.3 Residual 분리

* 커널에서 직접 누적한 escape 합으로도 보존이 안 맞는 경우:

  * residual은 host-side에서 계산해 `RESIDUAL_WEIGHT`로 보고
  * residual이 tolerance 초과 시 warning/테스트 fail

### 산출물

* escape/leakage가 물리적으로 정의된 “경계 통과”로만 구성
* 수치적 오차는 residual로만 나타남

### 검증(acceptance)

* 동일 config에서 CPU/GPU 불일치 원인이었던 escape 항목이 구조적으로 동일해짐
* residual이 작은 값(정의한 tolerance 내)으로 유지

---

## Phase 3 — Validation 체계 구축(golden snapshots + NIST range)

**목표:** CPU reference에 의존하지 않아도 GPU-only에서 회귀를 자동 감지.

### 작업 항목

1. `validation/golden_snapshots/` 정의

* 최소 2~3개 스냅샷:

  * 작은 grid(빠른) 1개: Nx=32 Nz=32 Ntheta=90 Ne=64
  * 중간 grid 1개: Nx=64 Nz=64 Ntheta=180 Ne=100
  * (선택) 스트레스 1개

각 스냅샷 구성:

* `config.yaml`
* `expected.npz`:

  * `dose_final`
  * `escapes_weight` (채널별)
  * (선택) `psi_final`은 용량이 크면 생략 가능
  * (선택) step별 요약(mass history)은 디버그용

2. `validation/compare.py`

* 비교 metric 정의:

  * `dose`: L1/L2/relative max error
  * `escapes_weight`: 채널별 상대오차
  * tolerance는 항목별로 분리:

    * escapes는 보통 더 타이트(예: 1e-6~1e-4)
    * dose는 누적/원자적 영향 반영(예: 1e-4~1e-2)
* deterministic mode가 없다면 tolerance는 현실적으로 설정

3. `validation/nist_validation.py`

* 전 에너지 구간 range 적분:

  * `Range(E0) = ∫_{E_cutoff}^{E0} dE / S(E)`
* NIST range 테이블(프로젝트가 가지고 있는 데이터가 있으면 그걸 사용, 없으면 내부 LUT를 기준으로 “자체 일관성 + 단조성” 검증부터 시작)
* 오차 기준:

  * 초기: 2~5% 허용
  * LUT/단위 확정 후 1~2%로 강화

4. (선택) CPU reference 유지 여부 결정

* 유지한다면 `validation/reference_cpu/`로 이동하고, 실행은 테스트에서만

### 산출물

* `python -m validation.compare --snapshot <name>` 통과
* `python -m validation.nist_validation` 통과

### 검증(acceptance)

* PR마다 자동으로 golden/NIST 검증 수행 가능(로컬/CI)

---

## 4. 커널/누적기 API 설계(구체안)

### 4.1 Accumulator 채널(권장 최소)

`escape_weight[ch]` (float64):

* 0: THETA_BOUNDARY
* 1: THETA_CUTOFF
* 2: ENERGY_STOPPED  (E_cutoff 아래로 내려가 psi에서 제거된 weight)
* 3: SPATIAL_LEAK
* 4: RESIDUAL (host-side)

별도 scalar(float64):

* `energy_deposited_total` (또는 dose가 곧 deposited energy라면 생략 가능하나, 권장: 명시적으로 분리)

### 4.2 커널 호출 서명(권장)

* Angular:

  * inputs: `psi_in`
  * outputs: `psi_tmp`
  * side effects: `escape_weight[THETA_*] += ...`
* Energy:

  * inputs: `psi_tmp`
  * outputs: `psi_tmp2`
  * side effects: `dose += ...`, `escape_weight[ENERGY_STOPPED] += ...`, `energy_deposited_total += ...`
* Stream:

  * inputs: `psi_tmp2`
  * outputs: `psi_out`
  * side effects: `escape_weight[SPATIAL_LEAK] += ...`

---

## 5. 테스트/검증 체크리스트

### 5.1 정합성(물리/수치)

* [ ] `sum(psi_final) + sum(escape_weight_except_residual)`이 입력 대비 감소하지 않음(정책에 따라)
* [ ] residual이 tolerance 이하
* [ ] dose가 음수 없음
* [ ] E_cutoff 처리: cutoff 아래로 내려간 weight는 psi에서 사라지고, ENERGY_STOPPED로만 간다(또는 deposit 정책대로)

### 5.2 성능/동기화

* [ ] 기본 모드(sync_interval=0)에서 step loop 동기화 0회
* [ ] 대형 grid에서 CPU 사용률이 낮고 GPU가 주도(launch overhead만 존재)

---

## 6. 리스크와 의사결정 포인트(구현 중 반드시 확정할 항목)

1. **Escape 채널에 누적하는 단위**

* weight만? energy도? 혼재 금지.
* 권장: weight 채널 + deposited energy scalar 분리

2. **Streaming 구현 패턴(scatter vs gather)**

* leakage direct tracking 관점에서 scatter가 편한 경우가 많음
* 성능/정확도 기준으로 operator별로 선택

3. **determinism level 제공 여부**

* 필요하면 Phase 2~3 사이에 옵션으로 추가(테스트 안정화에 도움)

---

## 7. 작업 순서 요약(가장 실용적인 “오늘부터 하는” 순서)

1. Phase 0: `SimulationConfig` 도입 + 기존 defaults 제거/연결
2. Phase 1: accumulator GPU 도입 + run loop sync 제거
3. Phase 2: angular/streaming direct tracking 전환 + residual 보고
4. Phase 3: golden snapshot 생성 + compare 도구 + NIST range 검증 도입
5. (선택) CPU reference를 validation 전용으로 이동
