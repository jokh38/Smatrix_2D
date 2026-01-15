# DOC-2: Phase B-1 SPEC — Tier-1 Scattering LUT

**Document ID**: SMP-PH-B1-2.1  
**Status**: Final / Binding  
**Depends on**: SMP-PH-A-2.1 (Phase A 완료 필수)  
**Revision**: v2.0 Phase B를 B-1 (필수)과 B-2 (선택)로 분리

---

## B1-0 Phase Context Summary

### 목적

Phase B-1의 목적은 angular scattering의 "analytic 계산을 runtime에서 제거"하고, σ(E, material) 데이터를 LUT로 변환하는 것이다.

### 현재 상태 vs 목표

| 항목 | 현재 | 목표 |
|------|------|------|
| σ 계산 방식 | Highland formula 직접 계산 | LUT lookup |
| Material 지원 | Water only | 4종 (water, lung, bone, aluminum) |
| 계산 위치 | SigmaBuckets 생성자 | Offline generation + Runtime load |

### Phase B-2와의 관계

- **B-1 (Tier-1)**: Highland formula 기반 LUT 생성 - 필수
- **B-2 (Tier-2)**: Geant4 기반 검증 LUT - 선택적

B-1만으로도 완전한 기능을 제공하며, B-2는 물리적 정확도 검증 목적으로만 사용된다.

### 완료 조건

Phase B-1은 다음이 모두 충족될 때 완료된다:

- [ ] R-LUT-STOP-001 유지 (기존 Stopping power LUT)
- [ ] R-SCAT-T1-001 ~ R-SCAT-T1-003 구현
- [ ] R-MAT-001 ~ R-MAT-003 구현
- [ ] V-SCAT-T1-001, V-MAT-001 통과
- [ ] P-LUT-001 성능 목표 달성

---

## B1-1 Stopping Power LUT Requirements

### R-LUT-STOP-001 — NIST PSTAR 기반 S(E) LUT 유지

기존 Stopping power LUT 구현을 유지하고 문서화한다.

**현재 구현:**
- `core/lut.py`의 `StoppingPowerLUT` 클래스
- NIST PSTAR 데이터 (0.01 - 200 MeV)
- Linear interpolation

**요구 사항:**
- Energy grid: E_centers에서 정의
- 단위: MeV/mm (density 반영)
- GPU read-only 경로 사용 (texture memory 또는 constant memory)

**확장 계획:**
- Phase B-1: Water만 지원 (현재 상태 유지)
- 향후: Material별 S(E) LUT 추가

---

## B1-2 Tier-1 Scattering LUT Requirements

### R-SCAT-T1-001 — Highland 기반 σ_norm LUT 구조

Highland formula를 사용하여 정규화된 scattering power LUT를 생성한다.

**Highland Formula:**
```
σ_θ(E, L, mat) = (13.6 MeV / (β·p)) · √(L/X0) · [1 + 0.038·ln(L/X0)]
```

**정규화:**
```
σ_norm(E, mat) = σ_θ(E, L=1mm, mat) / √(1mm)
```

즉, 단위 경로 길이당 scattering angle (rad/√mm)

**Runtime 사용:**
```
σ(E, Δs, mat) = σ_norm(E, mat) · √(Δs)
```

**LUT 구조:**
- 차원: [N_materials, N_energies]
- N_energies: 100-200 points (E_min ~ E_max)
- 단위: rad/√mm

### R-SCAT-T1-002 — Energy Grid 정의

Scattering LUT의 energy grid를 정의한다.

**요구 사항:**
- E_min: 1.0 MeV (defaults.py의 DEFAULT_E_MIN)
- E_max: Configuration에 따름 (70-200 MeV)
- 간격: Uniform 또는 Logarithmic

**권장 설정:**
- Uniform: 0.5 MeV 간격 (70 MeV beam 기준 140 points)
- Logarithmic: Low energy 영역에서 더 촘촘하게

**Interpolation:**
- Linear interpolation (1D)
- 범위 외 요청: Edge clamping (extrapolation 금지)

### R-SCAT-T1-003 — Offline Generation Pipeline

LUT 생성을 offline 단계로 분리한다.

**Generation Script:**
```
scripts/generate_scattering_lut.py
```

**입력:**
- Material definitions (YAML 또는 JSON)
- Energy grid specification
- Output format (NPY, HDF5, or binary)

**출력:**
```
data/lut/
├── scattering_lut_water.npy
├── scattering_lut_lung.npy
├── scattering_lut_bone.npy
├── scattering_lut_aluminum.npy
└── metadata.json
```

**Metadata 내용:**
- Generation date (ISO 8601)
- Energy grid (min, max, N, type)
- Material properties used
- Formula version (Highland v1)
- Checksum (SHA-256)

**Runtime Loading:**
- Lazy loading: 필요 시 로드
- Caching: 동일 material 재요청 시 캐시 사용
- Validation: Checksum 검증 (optional)

---

## B1-3 Material System Requirements

### R-MAT-001 — Material Descriptor 구조

모든 재료는 표준화된 descriptor로 표현한다.

**필수 속성:**
- `name`: Material 식별자 (string)
- `rho`: 밀도 (g/cm³)
- `X0`: Radiation length (mm) - 직접 지정 또는 계산

**선택 속성:**
- `composition`: 원소 조성 (혼합물용)
  - `{symbol: str, Z: int, A: float, weight_fraction: float}`
- `I_mean`: Mean excitation energy (eV)

**계산 속성 (derived):**
- `X0_derived`: Composition으로부터 계산된 X0
- `rho_e`: Electron density (electrons/cm³)

### R-MAT-002 — X0 계산 규칙

Radiation length X0 계산 방법을 표준화한다.

**단일 원소:**
```
X0 [g/cm²] = 716.4 · A / (Z · (Z+1) · ln(287/√Z))
X0 [mm] = X0 [g/cm²] / rho [g/cm³] · 10
```

**혼합물 (Bragg additivity):**
```
1/X0_mix = Σ (wi / X0_i)
```
wi: i번째 성분의 weight fraction

**우선순위:**
1. 직접 지정된 X0 값 사용
2. 지정되지 않은 경우 composition으로부터 계산
3. Composition도 없으면 오류

### R-MAT-003 — 기본 Material Bundle

최소 4종의 material을 기본 제공한다.

**Water (H2O):**
- rho: 1.0 g/cm³
- X0: 36.08 cm → 360.8 mm
- Composition: H (11.19%), O (88.81%)
- 용도: 기본 phantom, 연조직 근사

**Lung Equivalent:**
- rho: 0.26 g/cm³
- X0: 138.8 cm → 1388 mm
- 용도: 폐 조직 근사

**Cortical Bone:**
- rho: 1.92 g/cm³
- X0: 16.6 cm → 166 mm
- 용도: 뼈 조직 근사

**Aluminum:**
- rho: 2.70 g/cm³
- X0: 8.90 cm → 89.0 mm
- Z: 13, A: 27.0
- 용도: 장비 모델링, 검증

### R-MAT-004 — Material Registry

Material 정의를 중앙 레지스트리에서 관리한다.

**Registry 구조:**
```
data/materials/
├── materials.yaml       # 모든 material 정의
├── water.yaml          # 개별 파일 (optional)
└── custom/             # 사용자 정의 material
```

**Runtime API:**
```
get_material("water") → MaterialDescriptor
list_materials() → ["water", "lung", "bone", "aluminum", ...]
register_material(descriptor) → None
```

**Validation:**
- 필수 속성 존재 확인
- 값 범위 검증 (rho > 0, X0 > 0)
- 중복 이름 검출

---

## B1-4 Runtime Integration Requirements

### R-SCAT-T1-004 — SigmaBuckets와 LUT 통합

기존 SigmaBuckets를 LUT 기반으로 전환한다.

**현재 (변경 전):**
```
SigmaBuckets.__init__():
    for iE in range(Ne):
        sigma = Highland_formula(E_centers[iE], delta_s, X0)
        # 직접 계산
```

**목표 (변경 후):**
```
SigmaBuckets.__init__():
    self.sigma_lut = load_scattering_lut(material)
    for iE in range(Ne):
        sigma_norm = self.sigma_lut.lookup(E_centers[iE])
        sigma = sigma_norm * sqrt(delta_s)
        # LUT lookup
```

**호환성:**
- 기존 인터페이스 유지 (get_bucket_id, get_kernel 등)
- LUT 로드 실패 시 fallback to Highland (warning 출력)

### R-SCAT-T1-005 — GPU Memory Layout

Scattering LUT를 GPU에서 효율적으로 접근할 수 있도록 한다.

**Option A: Constant Memory**
- 크기 제한: ~64KB
- 장점: 캐시 효율, broadcast 최적화
- 적합: Material 수 적고 energy grid 작을 때

**Option B: Texture Memory**
- 크기: 거의 무제한
- 장점: Hardware interpolation
- 적합: 고해상도 LUT, 다수 material

**Option C: Global Memory (현재)**
- 크기: 무제한
- 단점: 캐시 미스 가능
- 적합: 초기 구현, 프로토타이핑

**권장:**
- Phase B-1: Global memory로 구현 (단순성)
- Phase D: Profiling 결과에 따라 Texture로 전환 검토

---

## B1-5 Validation Criteria

### V-SCAT-T1-001 — LUT vs Direct Calculation 일치

LUT lookup 결과와 직접 계산 결과가 일치하는지 검증한다.

**테스트 범위:**
- Energy: E_min ~ E_max (10 MeV 간격)
- Material: 4종 전체

**Pass 기준:**
```
|σ_lut(E, mat) - σ_direct(E, mat)| / σ_direct < 1e-4
```

**주의:**
- Interpolation 오차 허용
- Grid edge에서 오차가 클 수 있음

### V-SCAT-T1-002 — Energy 범위 외 동작

LUT 범위 외 energy 요청 시 동작을 검증한다.

**테스트 케이스:**
- E < E_min: Clamp to E_min
- E > E_max: Clamp to E_max

**Pass 기준:**
- 예외 발생 없음
- 경고 로그 출력
- Edge 값 반환

### V-MAT-001 — Material Consistency

Material 정의의 일관성을 검증한다.

**테스트 내용:**
- X0 직접 지정 vs composition 계산 일치 (< 1%)
- rho × X0 [g/cm²] 가 합리적 범위 내
- 기본 4종 material에 대해 reference 값과 비교

**Reference Values (NIST):**

| Material | X0 [g/cm²] | X0 [mm] at given rho |
|----------|------------|---------------------|
| Water | 36.08 | 360.8 |
| Aluminum | 24.01 | 89.0 |

---

## B1-6 Performance Targets

### P-LUT-001 — LUT Lookup Speedup

LUT lookup이 analytic 계산 대비 유의미한 speedup을 제공해야 한다.

**목표:**
- Speedup ≥ 3× (analytic Highland 대비)

**측정 조건:**
- Config-M 사용
- SigmaBuckets 생성 시간 측정
- 100회 반복 평균

**예상 결과:**
- Analytic: Highland formula 계산 + sqrt, log 연산
- LUT: Memory read + linear interpolation
- LUT가 compute-bound에서 memory-bound로 전환

### P-LUT-002 — Memory Overhead

LUT로 인한 추가 메모리 사용을 제한한다.

**목표:**
- LUT memory < 1 MB (모든 material 포함)

**계산:**
```
4 materials × 200 energies × 4 bytes = 3.2 KB
```

실제로는 매우 작음, overhead 무시 가능

---

## B1-7 Implementation Checklist

### LUT Generation

- [ ] Highland formula 기반 σ_norm 계산 함수 구현
- [ ] Energy grid 생성 함수 구현
- [ ] LUT 저장/로드 함수 구현 (NPY format)
- [ ] Metadata 생성 및 저장
- [ ] Generation script 작성

### Material System

- [ ] MaterialDescriptor dataclass 정의
- [ ] X0 계산 함수 (단일 원소, 혼합물)
- [ ] 기본 4종 material 정의 파일 작성
- [ ] Material registry 구현
- [ ] Runtime loading API

### Integration

- [ ] SigmaBuckets LUT 기반 전환
- [ ] Fallback to Highland (LUT 없을 때)
- [ ] GPU memory upload 구현
- [ ] 기존 테스트 통과 확인

### Validation

- [ ] V-SCAT-T1-001 테스트 작성
- [ ] V-SCAT-T1-002 테스트 작성
- [ ] V-MAT-001 테스트 작성

---

## B1-8 Migration Guide

### 기존 코드 사용자

**변경 없이 동작:**
- `SigmaBuckets` 생성 인터페이스 동일
- `get_kernel`, `get_bucket_id` 동일

**새로운 기능:**
- `SigmaBuckets(material="lung")` - material 지정 가능
- LUT 경로 지정 가능 (환경 변수 또는 config)

### 개발자

**주요 변경:**
- `_compute_sigma_theta` → `_lookup_sigma_norm`
- Material 파라미터 추가

**Deprecation:**
- 없음 (기존 Highland 기반 코드는 fallback으로 유지)

---

## B1-9 Risk Assessment

### Risk 1: LUT Interpolation 오차

**위험:** Linear interpolation으로 인한 물리 정확도 저하
**완화:**
- Energy grid 해상도를 충분히 높게 설정 (0.5 MeV 이하)
- 저에너지 영역에서 logarithmic grid 검토

### Risk 2: Material 정의 오류

**위험:** 잘못된 X0, rho 값으로 인한 계산 오류
**완화:**
- Reference 값과 비교 검증
- NIST 데이터 기반 validation

### Risk 3: File I/O 오버헤드

**위험:** LUT 로드 시간이 startup time 증가
**완화:**
- Lazy loading (필요 시 로드)
- Binary format 사용 (NPY)
- Caching 적용

---

## B1-10 Timeline Estimate

| Task | 예상 소요 시간 |
|------|----------------|
| LUT generation 구현 | 1일 |
| Material system 구현 | 1-2일 |
| SigmaBuckets 통합 | 1일 |
| Validation 테스트 | 1일 |
| 문서화 | 0.5일 |
| **Total** | **4-6일** |

---

*End of DOC-2*
