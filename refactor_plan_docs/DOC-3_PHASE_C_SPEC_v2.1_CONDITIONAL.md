# DOC-3: Phase C SPEC — Block-Sparse + Non-Uniform Grid

**Document ID**: SMP-PH-C-2.1  
**Status**: Conditional / Deferred  
**Depends on**: SMP-PH-B1-2.1 (Phase B-1 완료 필수)  
**Activation Condition**: 0.5mm 해상도 필요 시점

---

## C-0 Phase Context Summary

### 목적

Phase C의 목적은 고해상도(0.5mm 이하) 시뮬레이션에서 발생하는 메모리 및 계산량 폭증 문제를 해결하는 것이다.

### 구현 조건

Phase C는 다음 조건 중 하나라도 충족될 때만 구현을 시작한다:

1. **해상도 요구**: delta_x 또는 delta_z < 1.0mm 필요
2. **메모리 제약**: 현재 GPU VRAM으로 Config-L 실행 불가
3. **성능 요구**: Config-L에서 step time > 1초

### 현재 상태

| 해상도 | 메모리 사용량 | Block-sparse 필요성 |
|--------|---------------|---------------------|
| 1.0 mm (Config-M) | ~720 MB | 불필요 |
| 0.5 mm (Config-L) | ~5.7 GB | 필요 |
| 0.25 mm | ~46 GB | 필수 |

**결론:** 현재 1.0mm 해상도에서는 Phase C가 불필요하다.

---

## C-1 문제 정의

### Memory Scaling Problem

Phase space tensor 크기:
```
Memory = Ne × Ntheta × Nz × Nx × sizeof(float32)
```

해상도에 따른 메모리 증가:
```
1.0mm: 100 × 180 × 100 × 100 × 4 = 720 MB
0.5mm: 200 × 360 × 200 × 200 × 4 = 11.5 GB (double buffer 포함)
0.25mm: 400 × 720 × 400 × 400 × 4 = 184 GB
```

### Computational Scaling Problem

계산량은 메모리 크기에 비례:
```
FLOPs ∝ Ne × Ntheta × Nz × Nx
```

0.5mm에서 1.0mm 대비 16배 계산량 증가.

### Sparsity Observation

실제 시뮬레이션에서 대부분의 phase space는 비어있다:
- 빔 경로 주변에만 weight 존재
- 에너지 분포는 narrow band
- 각도 분포는 forward-peaked

Typical active fraction: 5-15% (beam geometry에 따라)

---

## C-2 Block-Sparse Architecture

### Design Philosophy

전체 phase space를 처리하는 대신, weight가 존재하는 block만 처리한다.

**Block 정의:**
- 공간적으로 인접한 cell들의 묶음
- 각 block은 독립적으로 활성/비활성 상태를 가짐
- 비활성 block은 커널 실행에서 완전히 제외

### Block Size Selection

**고려 사항:**

1. **Angular kernel half-width와의 관계**
   - k_cutoff=5.0, sigma~10mrad, delta_theta=1°=17.5mrad 기준
   - half_width = ceil(5.0 × 10 / 17.5) = 3 bins
   - 0.5° 해상도에서: half_width = ceil(5.0 × 10 / 8.7) = 6 bins

2. **GPU warp size**
   - NVIDIA: 32 threads
   - Block 차원은 32의 배수가 효율적

3. **Halo region overhead**
   - Block 경계에서 인접 block 데이터 필요
   - Block이 작으면 halo 비율 증가

**권장 Block Size:**

| 해상도 | Angular half-width | 권장 Block | 비고 |
|--------|-------------------|------------|------|
| 1.0° | ~3 bins | 8×8 | 충분 |
| 0.5° | ~6 bins | 16×16 | 권장 |
| 0.2° | ~15 bins | 32×32 | 최소 |

### Active Block Mask

각 step에서 block의 활성 상태를 결정하는 mask:

**Update Policy:**
```
block_active[b] = (max(weight in block b) > threshold)
```

**Update Frequency:**
- 매 step: 정확하지만 overhead 발생
- N steps마다: 빠르지만 불필요한 계산 가능
- 권장: N=10 steps마다 + weight 급변 시 즉시 갱신

### Memory Management

**Static Allocation:**
- 전체 dense array 할당
- Block mask로 처리 범위만 제한
- 장점: 단순, 메모리 연속성
- 단점: 메모리 절약 효과 없음

**Dynamic Allocation:**
- Active block만 할당
- Block 추가/제거 시 재할당
- 장점: 실제 메모리 절약
- 단점: 복잡성, fragmentation

**Hybrid (권장):**
- Dense array 유지하되 GPU 커널은 active block만 실행
- 메모리 절약은 제한적이나 구현 단순
- Phase D에서 dynamic allocation 검토

---

## C-3 Non-Uniform Grid

### Motivation

물리적으로 중요한 영역에서 해상도를 높이고, 덜 중요한 영역에서 낮춤.

**Energy:**
- Bragg peak 근처: 높은 해상도 필요
- 고에너지 영역: 낮은 해상도 충분

**Angle:**
- Forward direction (90°): 높은 해상도
- Tail (60°, 120°): 낮은 해상도 충분

**Space:**
- Beam path: 높은 해상도
- 외곽: 낮은 해상도

### Energy Grid Specification

**SPEC 권장 (참고용):**
```
2–10 MeV: 0.2 MeV
10–30 MeV: 0.5 MeV
30–70 MeV: 2.0 MeV
```

**구현 방법:**
- Non-uniform E_centers, E_edges 배열 생성
- Interpolation에서 non-uniform 간격 고려
- Energy loss operator에서 bin splitting 로직 수정

### Angular Grid Specification

**SPEC 권장 (참고용):**
```
Core (85–95°): Δθ=0.2°
Wings (70–85°, 95–110°): Δθ=0.5°
Tails (60–70°, 110–120°): Δθ=1°
```

**구현 방법:**
- Non-uniform theta_centers 배열
- Angular kernel 크기가 각도에 따라 가변
- Kernel LUT가 각도별로 다름

### Implementation Complexity

Non-uniform grid는 다음 연산자에 영향:

| Operator | Uniform 가정 | Non-uniform 수정 필요 |
|----------|--------------|----------------------|
| Angular scattering | delta_theta 고정 | 가변 delta_theta |
| Energy loss | delta_E 고정 | 가변 delta_E |
| Spatial streaming | delta_x, delta_z 고정 | 가변 (복잡) |

**권장:**
- 공간 grid는 uniform 유지 (streaming 단순화)
- Energy, Angular만 non-uniform 검토

---

## C-4 Requirements (조건부)

### R-BSP-001 — Block 정의

**조건:** 0.5mm 해상도 구현 시

- Block size: 16×16 (공간 기준, x-z plane)
- Threshold: configurable, default 1e-10
- Block indexing: (iz // 16, ix // 16)

### R-BSP-002 — Active Block 실행 제한

**조건:** 0.5mm 해상도 구현 시

- Inactive block에서 커널 실행 금지
- Block당 thread block 할당
- Block mask update: 10 steps마다

### R-BSP-003 — Halo Management

**조건:** 0.5mm 해상도 구현 시

- Halo size: Angular kernel half-width에 따라 결정
- Halo 데이터 교환은 block 처리 전에 완료
- Boundary block은 domain boundary 조건 적용

### R-GRID-E-001 — Non-Uniform Energy Grid

**조건:** 정밀 Bragg peak 해석 필요 시

- Low energy (< 10 MeV): 0.2 MeV 간격
- Mid energy (10-30 MeV): 0.5 MeV 간격
- High energy (> 30 MeV): 2.0 MeV 간격

### R-GRID-T-001 — Non-Uniform Angular Grid

**조건:** 정밀 각도 분포 해석 필요 시

- Forward region (85-95°): 0.2° 간격
- Mid region: 0.5° 간격
- Tail region: 1° 간격

---

## C-5 Validation (조건부)

### V-BSP-001 — Dense Equivalence

Block-sparse 결과가 dense 결과와 일치하는지 검증.

**테스트 조건:**
- Config-M (dense 가능) 사용
- Dense vs Block-sparse 비교

**Pass 기준:**
- Dose map L2 error ≤ 1e-3
- Weight/energy closure identical (1e-6 이내)
- Escape values identical (1e-6 이내)

### V-BSP-002 — Threshold Sensitivity

Threshold 값에 따른 결과 변화 검증.

**테스트:**
- threshold = 1e-8, 1e-10, 1e-12 비교
- 결과 차이가 threshold 변화에 비례하는지 확인

### V-GRID-001 — Non-Uniform Conservation

Non-uniform grid에서 conservation 유지 검증.

**Pass 기준:**
- Weight closure < 1e-6
- Energy closure < 1e-5

---

## C-6 Performance Targets (조건부)

### P-BSP-001 — Speedup Target

**조건:** 0.5mm 해상도 구현 시

**목표:**
- Dense 대비 ≥ 3× speedup
- Active block ratio ~10% 가정

**측정:**
- Config-L 사용
- 100 steps 평균

### P-BSP-002 — Memory Target

**조건:** 0.5mm 해상도 구현 시

**목표:**
- Working memory < 2 GB (psi double buffer + accumulators)
- Config-L (100×130×100×100) 기준

---

## C-7 Implementation Strategy

### Phase C-1: Basic Block-Sparse

1. Block mask 데이터 구조 추가
2. 커널에 block 범위 검사 추가
3. Dense equivalence 검증

### Phase C-2: Optimized Block-Sparse

1. Block-level kernel launch
2. Dynamic block mask update
3. Halo exchange optimization

### Phase C-3: Non-Uniform Grid (선택)

1. Energy grid non-uniform 지원
2. Angular grid non-uniform 지원
3. Operator 수정 및 검증

---

## C-8 현재 권장 사항

### 1.0mm 해상도에서 (현재)

- **Phase C 구현 불필요**
- Config-M이 약 720MB로 대부분의 GPU에서 실행 가능
- Block-sparse 오버헤드가 이득보다 클 수 있음

### Phase C 진입 조건

다음 조건 충족 시 Phase C 구현 시작:

1. 사용자/연구 요구로 0.5mm 해상도 필요
2. Config-L 실행 시 GPU OOM 발생
3. 대규모 beam 시뮬레이션에서 성능 병목

### 대안 검토

Phase C 전에 다음 대안을 먼저 검토:

1. **GPU 업그레이드**: VRAM이 큰 GPU 사용
2. **Tiling**: Z-axis tiling으로 메모리 분할 (이미 구현됨)
3. **Mixed precision**: float16 사용 (정확도 트레이드오프)

---

## C-9 Risk Assessment

### Risk 1: Block Size와 Kernel 불일치

**위험:** Angular kernel이 block 크기를 초과하면 halo가 과도하게 커짐
**완화:**
- Block size를 충분히 크게 설정 (16×16 이상)
- Kernel 크기에 따른 adaptive block size

### Risk 2: Non-Uniform Grid 복잡성

**위험:** 모든 operator 수정 필요, 버그 가능성 증가
**완화:**
- Uniform grid 유지 (공간)
- Energy/Angular만 선택적으로 non-uniform
- 철저한 단위 테스트

### Risk 3: Block Mask Overhead

**위험:** Block mask update가 오히려 오버헤드
**완화:**
- Update frequency 조절
- Incremental update (전체 재계산 대신)

---

## C-10 Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-15 | Phase C 보류 | 현재 1.0mm 해상도에서 불필요 |
| - | Block size 16×16 권장 | Angular kernel half-width 고려 |
| - | Spatial grid uniform 유지 | Streaming 단순화 |

---

## C-11 Activation Checklist

Phase C 구현 시작 전 확인 사항:

- [ ] 0.5mm 해상도 필요성 문서화
- [ ] Config-L 실행 불가 확인 (OOM 또는 성능)
- [ ] Phase A, B-1 완전 완료
- [ ] Golden snapshot (Config-M) 확보
- [ ] Block-sparse 대안 검토 완료

---

*End of DOC-3*

**Note:** 이 문서는 조건부 구현 사양이다. 현재 1.0mm 해상도에서는 구현이 불필요하며, 0.5mm 해상도가 필요해지는 시점에 활성화된다.
