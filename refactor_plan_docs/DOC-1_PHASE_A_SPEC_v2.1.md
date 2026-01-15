# DOC-1: Phase A SPEC — Accounting & Baseline Profiling

**Document ID**: SMP-PH-A-2.1  
**Status**: Final / Binding  
**Depends on**: SMP-MASTER-2.1  
**Revision**: v2.0 → v2.1 (Config SSOT 통합, Baseline 정의 추가)

---

## A-0 Phase Context Summary

### 목적

Phase A의 목적은 "무엇이 계산되고 있는지 확실히 아는 상태"를 만드는 것이다.

1. 물리 정책의 ambiguity 제거
2. Weight/Energy 회계 분리 및 명확화
3. 이후 모든 Phase에서 비교 기준이 될 GPU 성능 baseline 고정
4. Configuration Single Source of Truth (SSOT) 확립

### 완료 조건

Phase A는 다음이 모두 충족될 때 완료된다:

- [ ] R-ACC-001 ~ R-ACC-003 구현 및 검증
- [ ] R-CFG-001 ~ R-CFG-004 구현 및 검증  
- [ ] R-PROF-001 ~ R-PROF-003 구현
- [ ] V-ACC-001, V-ACC-002, V-PROF-001 통과
- [ ] Config-S, Config-M에 대한 Golden snapshot 생성

---

## A-1 Configuration SSOT Requirements

### R-CFG-001 — Single Source of Truth 원칙

모든 기본값은 단일 파일에서만 정의되어야 한다.

**필수 사항:**
- `config/defaults.py`가 모든 기본값의 유일한 정의 위치
- 다른 모듈에서 기본값 정의 금지
- 기본값 사용 시 반드시 `defaults.py`에서 import

**현재 문제점:**
- `grid.py`의 `E_cutoff: float = 1.0` (defaults.py는 2.0)
- `initial_info.yaml`의 독립적 기본값
- `GridSpecsV2`와 `SimulationConfig`의 이중 구조

**해결 방안:**
- `GridSpecsV2`의 모든 기본값을 `defaults.py` import로 교체
- `initial_info.yaml`은 override 용도로만 사용 (기본값 정의 금지)

### R-CFG-002 — EscapeChannel 통합

Escape channel 정의가 단일 위치에만 존재해야 한다.

**현재 문제점:**
- `escape_accounting.py`: `EscapeChannel(Enum)` - 문자열 기반
- `accounting.py`: `EscapeChannel(IntEnum)` - 정수 기반
- GPU 커널: 하드코딩된 인덱스 (0, 1, 2, 3)

**해결 방안:**
- `accounting.py`의 `EscapeChannel(IntEnum)`을 canonical로 지정
- `escape_accounting.py`는 deprecated 처리 후 redirect
- GPU 커널 상수는 `kernel_constants.py`에서 정의하고 검증 함수 추가

### R-CFG-003 — Config 계층 통합

`GridSpecsV2`와 `SimulationConfig` 사이의 관계를 명확히 한다.

**계층 구조:**
```
SimulationConfig (최상위)
├── GridConfig (grid 관련)
├── TransportConfig (transport 관련)
├── NumericsConfig (수치 설정)
└── BoundaryConfig (경계 조건)

GridSpecsV2 → GridConfig로부터 생성
PhaseSpaceGridV2 → GridSpecsV2로부터 생성
```

**변환 규칙:**
- `SimulationConfig` → `GridSpecsV2`: factory function 제공
- 직접 `GridSpecsV2` 생성 시 `defaults.py` 기본값 사용

### R-CFG-004 — 설정 파일 검증

런타임에 로드되는 모든 설정 파일은 스키마 검증을 통과해야 한다.

**검증 항목:**
- E_cutoff > E_min (필수)
- E_cutoff < E_max (필수)
- E_cutoff - E_min >= 1.0 MeV (권장, 경고)
- delta_s <= min(delta_x, delta_z) (권장, 경고)
- 모든 차원 크기 > 0

---

## A-2 Accounting Requirements

### R-ACC-001 — Angular Kernel 정책 단일화

Angular scattering kernel은 아래 정책 중 하나만 선택하고 혼재를 금지한다.

**Policy-A: Normalized Kernel (권장)**
- Kernel sum = 1.0으로 정규화
- 경계 손실만 escape로 추적 (THETA_BOUNDARY)
- THETA_CUTOFF는 diagnostic으로만 기록 (balance에 미포함)
- Mass balance: `W_in = W_out + escape_boundary`

**Policy-B: Unnormalized Kernel**
- Kernel은 원래 Gaussian 값 유지
- 절단 손실과 경계 손실 모두 escape로 추적
- Mass balance: `W_in = W_out + escape_cutoff + escape_boundary`
- 주의: kernel_sum > 1.0인 경우 mass 생성 문제 발생

**선택 기준:**
- 현재 구현은 Policy-A를 따르고 있음
- Policy-A를 공식 정책으로 확정
- SPEC 문서의 "no implicit renormalization" 문구는 Policy-B를 의도했으나, 수치적 안정성을 위해 Policy-A 채택

### R-ACC-002 — Weight Accounting Closure

모든 step에서 다음 accumulator를 유지해야 한다.

**입력:**
- `W_in`: Step 시작 시 총 weight (`sum(psi)`)

**출력:**
- `W_out`: Step 종료 시 남은 weight (`sum(psi_new)`)

**Escape channels:**
- `W_escape[0]`: THETA_BOUNDARY - 각도 경계 이탈
- `W_escape[1]`: THETA_CUTOFF - kernel 절단 (diagnostic)
- `W_escape[2]`: ENERGY_STOPPED - 에너지 cutoff 도달
- `W_escape[3]`: SPATIAL_LEAK - 공간 경계 이탈

**Residual:**
- `W_residual = W_in - W_out - W_escape[0] - W_escape[2] - W_escape[3]`
- Diagnostic 용도, balance equation에 명시적으로 포함하지 않음

### R-ACC-003 — Energy Accounting Closure

Energy는 weight와 절대 혼용하지 않으며 별도로 추적한다.

**입력:**
- `E_in`: Step 시작 시 총 에너지 (`sum(psi * E_centers)`)

**출력:**
- `E_out`: Step 종료 시 남은 에너지
- `E_deposit`: 매질에 침적된 에너지 (dose)

**Escape energy:**
- `E_escape[2]`: ENERGY_STOPPED 입자의 잔여 에너지

**Balance equation:**
```
E_in = E_out + E_deposit + E_escape + E_residual
```

**주의사항:**
- ENERGY_STOPPED escape channel은 weight와 energy를 별도로 추적
- Weight: 정지된 입자 수 (또는 weight)
- Energy: 해당 입자들의 에너지 합계

---

## A-3 Baseline Profiling Requirements

### R-PROF-001 — 필수 성능 메트릭

다음 항목을 모든 transport step에서 계측 가능해야 한다.

**Kernel Timing:**
- `t_angular`: Angular scattering kernel 시간
- `t_energy`: Energy loss kernel 시간
- `t_spatial`: Spatial streaming kernel 시간
- `t_total`: 전체 step 시간 (overhead 포함)

**Memory Metrics:**
- `mem_psi`: Phase space tensor 메모리 (bytes)
- `mem_dose`: Dose accumulator 메모리 (bytes)
- `mem_escape`: Escape accumulator 메모리 (bytes)
- `mem_lut`: LUT 메모리 (bytes)

**Throughput:**
- `dram_read`: DRAM 읽기 throughput (GB/s)
- `dram_write`: DRAM 쓰기 throughput (GB/s)

**Occupancy:**
- `active_blocks`: 활성 thread block 수
- `theoretical_occupancy`: 이론적 occupancy (%)

### R-PROF-002 — Baseline Snapshot 생성

Config-S, Config-M에 대해 baseline snapshot을 생성하고 저장한다.

**Snapshot 내용:**
- Configuration 전체 (JSON 또는 YAML)
- Final dose distribution (Nz × Nx, float32)
- Final escape values (5 channels, float64)
- Final weight remaining (scalar, float64)
- Step-by-step conservation reports (optional)

**저장 위치:**
```
validation/golden_snapshots/
├── config_s/
│   ├── config.yaml
│   ├── dose.npy
│   ├── escapes.npy
│   └── metadata.json
├── config_m/
│   └── ...
└── config_l/
    └── ...
```

**생성 조건:**
- GPU: NVIDIA RTX 3080 또는 동급
- Driver: CUDA 12.x
- CuPy version: 명시
- 난수 시드: 해당 없음 (deterministic)

### R-PROF-003 — Regression Test Framework

Golden snapshot과 현재 실행 결과를 비교하는 테스트 프레임워크를 제공한다.

**비교 항목:**
- Dose L2 error: `||dose_test - dose_golden||_2 / ||dose_golden||_2`
- Dose max error: `max(|dose_test - dose_golden|)`
- Escape relative error: `|escape_test - escape_golden| / escape_golden`
- Weight relative error: 동일

**Pass/Fail 기준:**
- Dose L2 error < 1e-4
- Escape relative error < 1e-5
- Weight relative error < 1e-6

**Tolerance 조정:**
- 다른 GPU에서 실행 시: tolerance × 10
- Debug mode에서: tolerance / 10

---

## A-4 Validation Criteria

### V-ACC-001 — Weight Closure Test

**테스트 내용:**
각 step에서 weight conservation을 검증한다.

**수식:**
```
error = |W_in - W_out - sum(physical_escapes)| / W_in
```

**Pass 기준:**
- `error < 1e-6`

**Physical escapes:**
- THETA_BOUNDARY + ENERGY_STOPPED + SPATIAL_LEAK
- THETA_CUTOFF는 제외 (diagnostic)

### V-ACC-002 — Energy Closure Test

**테스트 내용:**
각 step에서 energy conservation을 검증한다.

**수식:**
```
error = |E_in - E_out - E_deposit - E_escape| / E_in
```

**Pass 기준:**
- `error < 1e-5`

### V-CFG-001 — Config SSOT Compliance

**테스트 내용:**
코드베이스 전체에서 기본값 정의가 SSOT 원칙을 따르는지 검증한다.

**검증 방법:**
1. `defaults.py` 외 파일에서 기본값 할당 검색
2. `GridSpecsV2`, `SimulationConfig` 등의 dataclass 기본값 검사
3. 불일치 발견 시 실패

### V-PROF-001 — Baseline Snapshot Validity

**테스트 내용:**
저장된 golden snapshot이 재현 가능한지 검증한다.

**검증 방법:**
1. 동일 GPU에서 동일 config로 재실행
2. Dose L2 error < 1e-10 (bitwise 근접)
3. Escape 값 bitwise 일치

---

## A-5 Performance Targets

### P-ACC-001 — Accounting Overhead

Accounting 및 profiling 추가로 인한 성능 저하를 제한한다.

**목표:**
- Step time 증가 ≤ 5% (accounting 활성화 vs 비활성화)

**측정 조건:**
- Config-M 사용
- 100 steps 평균
- Warm-up 10 steps 제외

### P-PROF-001 — Profiling Overhead

상세 profiling 활성화 시 성능 저하를 제한한다.

**목표:**
- Step time 증가 ≤ 20% (full profiling vs no profiling)

**측정 조건:**
- Config-M 사용
- 100 steps 평균
- CUDA event timing 사용

---

## A-6 Implementation Checklist

### Config SSOT 구현

- [ ] `defaults.py`에 모든 기본값 통합
- [ ] `GridSpecsV2` 기본값을 `defaults.py` import로 교체
- [ ] `SimulationConfig` ↔ `GridSpecsV2` 변환 factory 구현
- [ ] 기존 코드의 하드코딩된 기본값 제거

### EscapeChannel 통합

- [ ] `accounting.py`의 `EscapeChannel(IntEnum)`을 canonical로 확정
- [ ] `escape_accounting.py`에 deprecation warning 추가
- [ ] `escape_accounting.py` 사용처를 `accounting.py`로 마이그레이션
- [ ] GPU 커널 상수 동기화 검증 함수 추가

### Baseline Profiling

- [ ] Kernel timing 측정 코드 추가
- [ ] Memory 사용량 측정 코드 추가
- [ ] Golden snapshot 생성 스크립트 작성
- [ ] Regression test 프레임워크 구현

### 검증

- [ ] V-ACC-001 테스트 케이스 작성
- [ ] V-ACC-002 테스트 케이스 작성
- [ ] V-CFG-001 정적 분석 스크립트 작성
- [ ] V-PROF-001 재현성 테스트 작성

---

## A-7 Risk Assessment

### Risk 1: 기존 코드 호환성

**위험:** Config SSOT 적용 시 기존 테스트/사용 코드 파손
**완화:** 
- Deprecation warning 기간 (2주) 부여
- 마이그레이션 가이드 문서 제공

### Risk 2: GPU 커널 상수 불일치

**위험:** Python enum과 CUDA 상수 불일치로 인한 silent error
**완화:**
- 런타임 검증 함수 추가
- CI에서 상수 일치 검사 자동화

### Risk 3: Baseline 재현 불가

**위험:** GPU 드라이버 업데이트로 인한 bitwise 불일치
**완화:**
- D1 수준 tolerance 기반 비교 옵션 제공
- 드라이버 버전 metadata 기록

---

## A-8 Timeline Estimate

| Task | 예상 소요 시간 |
|------|----------------|
| Config SSOT 통합 | 1-2일 |
| EscapeChannel 통합 | 0.5일 |
| Baseline profiling 구현 | 1-2일 |
| Golden snapshot 생성 | 0.5일 |
| 검증 및 테스트 | 1일 |
| **Total** | **4-6일** |

---

*End of DOC-1*
