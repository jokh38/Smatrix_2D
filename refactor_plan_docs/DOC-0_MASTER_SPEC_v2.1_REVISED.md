# DOC-0: Smatrix_2D MASTER SPEC v2.1 (Revised)

**Document ID**: SMP-MASTER-2.1  
**Status**: Final / Binding  
**Revision**: v2.0 → v2.1 (명확화 및 누락 사항 보완)  
**Audience**: Core developers, GPU/Physics integrators

---

## 0.1 Revision Summary

v2.0에서 v2.1로의 주요 변경 사항:

1. Phase 간 의존성 명시적 정의
2. 검증 기준(V)의 구체화 - 에너지 범위, 측정 조건 명시
3. 성능 목표(P)의 측정 환경 정의
4. 대표 Configuration 3종 표준화
5. Runtime LUT 선택 정책 추가

---

## 0.2 Project Objective (Clarified)

Smatrix_2D는 다음을 동시에 만족하는 deterministic proton transport engine이다.

### Primary Goals

1. **Physics Validity**
   - CSDA + MCS 기반 transport에서 임상적 정확도 유지
   - 70 MeV 양성자의 Bragg peak 위치 오차 < 1mm (물 기준)
   - 횡방향 선량 분포 FWHM 오차 < 5%

2. **Numerical Accountability**
   - Weight/Energy 보존이 step-wise로 닫히는 회계 체계
   - 상대 오차 tolerance: 1e-6 (weight), 1e-5 (energy)

3. **Scalability with Resolution**
   - 1.0 mm 해상도: 현재 목표 (Block-sparse 불필요)
   - 0.5 mm 해상도: 향후 목표 (Block-sparse 필수)

4. **GPU-First Architecture**
   - 모든 설계 결정은 GPU 성능을 1차 제약조건으로 고려
   - CPU fallback은 검증 목적으로만 유지

---

## 0.3 Non-Goals (Explicit)

구현 범위에서 명시적으로 제외되는 항목:

- Multi-GPU / MPI 분산 처리
- Full nuclear reaction modeling (hook는 허용하되 구현하지 않음)
- Monte-Carlo stochastic sampling
- 시간 의존적 빔 전달 (time-resolved delivery)
- 환자 CT 기반 heterogeneous medium (Phase 1 scope 외)

---

## 0.4 Global Design Principles

### G-P1: Accounting Closure

모든 step에서 다음이 독립적으로 닫혀야 한다.

**Weight Balance:**
```
W_in = W_out + W_escape_theta_boundary + W_escape_theta_cutoff 
     + W_escape_energy_stopped + W_escape_spatial_leak ± ε
```
- ε ≤ 1e-6 × W_in (상대 오차)

**Energy Balance:**
```
E_in = E_out + E_deposit + E_escape + E_residual ± ε
```
- ε ≤ 1e-5 × E_in (상대 오차)

### G-P2: Determinism Levels

| Level | 명칭 | 조건 | 용도 |
|-------|------|------|------|
| D0 | Bitwise | 동일 HW + 동일 config → bitwise identical | 회귀 테스트 |
| D1 | Statistical | 다른 GPU → 통계/보존성 기준 만족 | 배포 환경 |

D0은 atomic operation 순서에 의존하므로, 동일 GPU에서만 보장됨.
Cross-GPU 비교는 D1 기준(tolerance 1e-5)을 적용.

### G-P3: Provenance First

모든 LUT, grid, physics 데이터는 다음을 명시해야 한다:

- 생성 경로 (source file, generation script)
- 버전 (semantic versioning)
- 생성 일시 (ISO 8601 format)
- 검증 상태 (validated/unvalidated)

---

## 0.5 Phase Structure (Revised)

### Phase Dependencies

```
Phase A ──────────────────────────────────────────────────┐
(Accounting & Baseline)                                   │
    │                                                     │
    ▼                                                     │
Phase B-1 ─────────────────────────────────────────────┐  │
(Tier-1 Scattering LUT)                                │  │
    │                                                  │  │
    ├──── [Optional] ──── Phase B-2                    │  │
    │                     (Tier-2 Geant4 LUT)          │  │
    │                                                  │  │
    ▼                                                  │  │
Phase C ───────────────────────────────────────────────│──┤
(Block-Sparse + Non-Uniform Grid)                      │  │
    │                                                  │  │
    │  ◄── 0.5mm 해상도 필요 시에만 진입              │  │
    │                                                  │  │
    ▼                                                  │  │
Phase D ◄──────────────────────────────────────────────┴──┘
(Optimization & Hardening)
```

### Phase Summary Table

| Phase | Name | 핵심 목적 | 선행 조건 | 필수 여부 |
|-------|------|-----------|-----------|-----------|
| A | Accounting & Baseline | 정책/회계 고정 + 성능 기준선 | 없음 | 필수 |
| B-1 | Tier-1 Scattering LUT | Highland 기반 σ LUT | Phase A | 필수 |
| B-2 | Tier-2 Geant4 LUT | Geant4 검증 데이터 | Phase B-1 | 선택 |
| C | Block-Sparse | 0.5mm 해상도 대응 | Phase B-1 | 조건부 |
| D | Optimization | 실측 기반 GPU 최적화 | Phase A, B-1 | 필수 |

---

## 0.6 Standard Configurations

모든 성능 측정, 검증, 회귀 테스트는 다음 3종 configuration을 기준으로 한다.

### Config-S (Small) - 빠른 검증용

| Parameter | Value | 비고 |
|-----------|-------|------|
| Nx, Nz | 32 | 약 3mm 해상도 |
| Ntheta | 45 | 4° 간격 |
| Ne | 35 | 2 MeV 간격 |
| E_beam | 70 MeV | |
| E_min | 1.0 MeV | |
| E_cutoff | 2.0 MeV | |
| delta_s | 1.0 mm | |
| Expected runtime | < 5 sec | RTX 3080 기준 |

### Config-M (Medium) - 기본 검증용

| Parameter | Value | 비고 |
|-----------|-------|------|
| Nx, Nz | 100 | 1mm 해상도 |
| Ntheta | 180 | 1° 간격 |
| Ne | 100 | 0.7 MeV 간격 |
| E_beam | 70 MeV | |
| E_min | 1.0 MeV | |
| E_cutoff | 2.0 MeV | |
| delta_s | 1.0 mm | |
| Expected runtime | < 30 sec | RTX 3080 기준 |

### Config-L (Large) - 고해상도 검증용

| Parameter | Value | 비고 |
|-----------|-------|------|
| Nx, Nz | 200 | 0.5mm 해상도 |
| Ntheta | 360 | 0.5° 간격 |
| Ne | 200 | 0.35 MeV 간격 |
| E_beam | 150 MeV | |
| E_min | 1.0 MeV | |
| E_cutoff | 2.0 MeV | |
| delta_s | 0.5 mm | |
| Expected runtime | < 5 min | RTX 3080 기준, Block-sparse 필요 |

---

## 0.7 ID Convention

### Requirement IDs

Format: `R-<AREA>-###`

| AREA | 의미 |
|------|------|
| ACC | Accounting (회계) |
| PROF | Profiling (성능 측정) |
| LUT | Lookup Table |
| SCAT | Scattering (산란) |
| MAT | Material (재료) |
| BSP | Block-Sparse |
| GRID | Grid Configuration |
| GPU | GPU Optimization |
| CFG | Configuration |

### Validation IDs

Format: `V-<AREA>-###`

검증 항목은 해당 Requirement와 동일한 AREA 코드를 사용.

### Performance IDs

Format: `P-<AREA>-###`

성능 목표는 측정 환경(Config-S/M/L)을 명시해야 함.

---

## 0.8 Escape Channel Definitions

### Channel Index Table (Canonical)

모든 코드에서 이 인덱스를 사용해야 함.

| Index | Name | 물리적 의미 | 회계 포함 |
|-------|------|-------------|-----------|
| 0 | THETA_BOUNDARY | 각도 도메인 경계(0°, 180°) 이탈 | Yes |
| 1 | THETA_CUTOFF | Gaussian kernel ±k·σ 절단 손실 | Diagnostic only |
| 2 | ENERGY_STOPPED | E < E_cutoff 정지 | Yes |
| 3 | SPATIAL_LEAK | 공간 도메인 이탈 | Yes |
| 4 | RESIDUAL | 수치 잔여 오차 | No (계산값) |

### Mass Balance Equation

```
W_in = W_out + escape[0] + escape[2] + escape[3] + residual
```

Note: `escape[1]` (THETA_CUTOFF)는 normalized kernel 사용 시 실제 손실이 아니므로 balance에서 제외.

---

## 0.9 Document References

| Document ID | Title | 내용 |
|-------------|-------|------|
| DOC-1 | Phase A SPEC | Accounting & Baseline Profiling |
| DOC-2 | Phase B SPEC | Physics Data Pipeline & LUT |
| DOC-3 | Phase C SPEC | Block-Sparse + Non-Uniform Grid |
| DOC-4 | Phase D SPEC | GPU Optimization & Hardening |

---

## 0.10 Change Log

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2025-01-10 | Initial release |
| 2.1 | 2025-01-15 | Phase dependencies, Standard configs, Escape channel clarification |

---

## 0.11 Glossary

| Term | Definition |
|------|------------|
| CSDA | Continuous Slowing Down Approximation |
| MCS | Multiple Coulomb Scattering |
| LUT | Lookup Table |
| SSOT | Single Source of Truth |
| Highland formula | σ_θ = (13.6/βp)·√(L/X0)·[1+0.038·ln(L/X0)] |
| Bragg peak | 양성자 깊이-선량 곡선의 최대점 |
| E_cutoff | 입자 정지로 간주하는 최소 에너지 |

---

*End of DOC-0*
