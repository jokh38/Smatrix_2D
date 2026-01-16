# 에너지 보존 및 상태 전파 분석 보고서

매우 중요한 물리적 관점을 지적해 주셨습니다. 양성자 수송에서는 **입자 수 보존이 아닌 에너지 보존**이 핵심입니다.

---

## 1. 현재 코드의 보존량 정의 문제

### 1.1 코드가 추적하는 것: "Weight" (입자 수)

**문제 위치**: `smatrix_2d/core/accounting.py`

```python
# 현재 구현된 Mass Balance (Policy-A)
"""
Mass Balance Equation (Policy-A):
    W_in = W_out + W_escapes + W_residual

Where:
    W_in: Initial total weight in domain
    W_out: Final total weight remaining in domain
    W_escapes: Sum of physical escapes
"""

def validate_conservation(mass_in, mass_out, escape_weights, tolerance=1e-6):
    """Validate mass conservation under Policy-A."""
    physical_escapes = sum(escape_weights[ch] for ch in PHYSICAL_ESCAPE_CHANNELS)
    expected_out = mass_in - physical_escapes
    relative_error = abs(expected_out - mass_out) / mass_in
    return relative_error <= tolerance
```

**물리적 오류**:
- 코드는 **weight(입자 수) 보존**을 검증
- 그러나 양성자는 **멈추면서 사라짐** → 입자 수는 감소해야 함
- **에너지만 보존**되어야 하는데, 에너지 보존 검증이 없음

---

### 1.2 올바른 물리: 에너지 보존

양성자 수송의 에너지 보존 방정식:

$$E_{total}(t) = E_{kinetic}(t) + E_{deposited}(t) = \text{const}$$

각 스텝에서:
$$\sum_{i} w_i \cdot E_i^{(n)} + D^{(n)} = \sum_{i} w_i \cdot E_i^{(n+1)} + D^{(n+1)}$$

여기서:
- $w_i$: 빈 $i$의 가중치 (입자 수)
- $E_i$: 빈 $i$의 에너지
- $D$: 누적 deposited dose

---

## 2. 현재 코드의 에너지 추적 분석

### 2.1 Energy Loss Operator 분석

**위치**: `smatrix_2d/operators/energy_loss.py`

```python
def apply(self, psi, delta_s, deposited_energy=None):
    for iE_in in range(Ne):
        E_in = self.grid.E_centers[iE_in]
        S = self.stopping_power_lut.get_stopping_power(E_in)
        deltaE = S * delta_s
        E_new = E_in - deltaE
        
        weight_slice = psi[iE_in]  # 이 에너지 빈의 입자들
        
        # Case 2: 에너지가 cutoff 이하로 떨어짐 → 입자 멈춤
        if E_new < self.E_cutoff:
            # 모든 에너지를 dose로 기록
            deposited_energy += total_weight * E_in  # ✓ 에너지 기록
            
            # 문제: escape_energy_stopped에 WEIGHT 기록 (이름과 불일치)
            escape_energy_stopped += np.sum(total_weight)  # ✗ 에너지가 아님!
            
            # 입자는 psi_out에 추가되지 않음 → 사라짐 ✓
            continue
        
        # Case 3: 정상 에너지 손실
        # Bin splitting으로 새 에너지 빈에 분배
        psi_out[iE_out] += w_lo * weight_slice
        psi_out[iE_out + 1] += w_hi * weight_slice
        
        # deltaE만 dose에 기록
        deposited_energy += deltaE * np.sum(weight_slice, axis=0)  # ✓
```

**분석 결과**:

| 항목 | 구현 상태 | 비고 |
|-----|----------|-----|
| 에너지 손실 → Dose 기록 | ✓ 구현됨 | `deltaE × weight` |
| 정지 입자 에너지 → Dose | ✓ 구현됨 | `E_in × weight` |
| 정지 입자 제거 | ✓ 구현됨 | `psi_out`에 미추가 |
| 에너지 보존 검증 | ✗ 미구현 | weight 보존만 검증 |

---

### 2.2 Conservation Report의 에너지 추적

**위치**: `smatrix_2d/core/accounting.py`

```python
@dataclass
class ConservationReport:
    # Weight 추적 (구현됨)
    mass_in: float = 0.0
    mass_out: float = 0.0
    escape_weights: dict[EscapeChannel, float] = field(default_factory=dict)
    
    # Energy 추적 (선언만 됨, 실제 사용 안 함)
    escape_energy: dict[EscapeChannel, float] = field(default_factory=dict)
    deposited_energy: float = 0.0
    
    def compute_energy_closure(self) -> dict[str, float]:
        """Energy closure 계산"""
        e_in = 0.0   # ✗ 실제로 계산되지 않음!
        e_out = 0.0  # ✗ 실제로 계산되지 않음!
        e_deposit = self.deposited_energy
        e_escape = self.total_escape_energy()
        
        # E_in = 0이면 항상 "closed"로 판정
        if e_in == 0.0:
            return True  # ✗ 검증 우회!
```

**핵심 문제**: `e_in`, `e_out`이 **0으로 초기화되어 실제 계산되지 않음**

---

## 3. 상태 벡터 전파 분석

### 3.1 시뮬레이션 루프 확인

**위치**: `smatrix_2d/transport/simulation.py`

```python
class TransportSimulation:
    def __init__(self, config, psi_init=None):
        # 초기 상태 설정
        if psi_init is not None:
            self.psi_gpu = cp.asarray(psi_init)
        else:
            self.psi_gpu = self._initialize_beam_gpu()  # 초기 빔 생성
    
    def step(self) -> ConservationReport:
        """한 스텝 실행"""
        mass_in = float(cp.sum(self.psi_gpu))
        
        # 핵심: psi_gpu가 갱신되어 다음 스텝에 사용됨
        self.psi_gpu = self.transport_step.apply(
            psi=self.psi_gpu,           # 현재 상태 입력
            accumulators=self.accumulators,
        )
        # self.psi_gpu는 이제 새로운 상태
        
        mass_out = float(cp.sum(self.psi_gpu))
        return report
    
    def run(self, n_steps):
        for step in range(n_steps):
            report = self.step()  # 매 스텝마다 psi_gpu 갱신
            self.reports.append(report)
```

**결론**: ✓ **상태 벡터는 올바르게 전파됨**

- `self.psi_gpu`가 클래스 멤버로 유지
- 매 `step()` 호출 시 `transport_step.apply()`의 출력으로 갱신
- 다음 스텝에서 갱신된 `self.psi_gpu` 사용

---

### 3.2 Transport Step 내부 연산자 순서

**위치**: GPU 커널 (추정, `smatrix_2d/gpu/kernels.py`)

```python
def apply(self, psi, accumulators):
    # 연산자 순서: A_s ∘ A_E ∘ A_θ (또는 다른 순서)
    
    # 1. Angular Scattering: θ 분포 확산
    psi = angular_scattering(psi)
    
    # 2. Energy Loss: E 감소, dose 누적
    psi = energy_loss(psi, accumulators.dose_gpu)
    
    # 3. Spatial Streaming: z 방향 이동
    psi = spatial_streaming(psi)
    
    return psi  # 갱신된 상태
```

**상태 전파 다이어그램**:

```
Step n:   ψ(E,θ,z,x)ⁿ
              ↓
         A_θ (scattering)
              ↓
         A_E (energy loss) → Dose 누적
              ↓
         A_s (streaming)
              ↓
Step n+1: ψ(E,θ,z,x)ⁿ⁺¹
```

---

## 4. 에너지 보존이 구현되지 않은 증거

### 4.1 run_simulation.py의 보존 검증

```python
# 현재 코드가 검증하는 것
print(f"  Conservation valid: {last.is_valid}")  # Weight 보존만!
print(f"  Relative error: {last.relative_error:.2e}")

# Mass balance 출력
print(f"  Final weight: {final_weight:.6f}")
print(f"  Mass balance: {final_weight + total_escape:.6f}")  # Weight!
```

### 4.2 에너지 보존 검증 코드 부재

```python
# 있어야 하지만 없는 코드:
E_kinetic_initial = compute_total_kinetic_energy(psi_initial, E_centers)
E_kinetic_final = compute_total_kinetic_energy(psi_final, E_centers)
E_deposited = np.sum(deposited_dose)

energy_conservation_error = abs(
    E_kinetic_initial - (E_kinetic_final + E_deposited)
) / E_kinetic_initial

print(f"  Energy conservation error: {energy_conservation_error:.2e}")
```

---

## 5. 핵심 문제 요약

### 5.1 현재 코드의 보존량 추적

| 보존량 | 물리적 의미 | 구현 상태 | 검증 상태 |
|-------|-----------|----------|----------|
| Weight (W) | 입자 수 | ✓ 추적됨 | ✓ 검증됨 |
| Kinetic Energy (E_k) | 운동 에너지 | ✗ 미추적 | ✗ 미검증 |
| Deposited Energy (D) | 흡수 선량 | ✓ 추적됨 | ✗ 미검증 |
| Total Energy (E_k + D) | **보존량** | ✗ 미계산 | ✗ 미검증 |

### 5.2 물리적 모순

```
현재 코드 가정: W_in = W_out + W_escaped (입자 수 보존)

실제 물리:
- 양성자는 에너지를 잃으며 전진
- E < E_cutoff 되면 멈춤 (입자 "소멸")
- 입자 수는 감소함 (W_out < W_in)
- 에너지만 보존: E_kinetic + E_deposited = const
```

---

## 6. 권장 수정사항

### 6.1 에너지 보존 계산 함수 추가

```python
def compute_total_kinetic_energy(psi: np.ndarray, E_centers: np.ndarray) -> float:
    """
    Total kinetic energy in phase space.
    
    E_total = Σ_i Σ_j Σ_k Σ_l  ψ[i,j,k,l] × E_centers[i]
    """
    # E_centers를 4D로 브로드캐스트
    E_4d = E_centers[:, np.newaxis, np.newaxis, np.newaxis]
    return np.sum(psi * E_4d)
```

### 6.2 Conservation Report 수정

```python
@dataclass
class ConservationReport:
    # 기존 weight 추적 유지 (진단용)
    mass_in: float = 0.0
    mass_out: float = 0.0
    
    # 에너지 추적 추가 (물리적 보존량)
    kinetic_energy_in: float = 0.0   # 새로 추가
    kinetic_energy_out: float = 0.0  # 새로 추가
    deposited_energy: float = 0.0
    
    def validate_energy_conservation(self, tol=1e-6) -> bool:
        """실제 물리적 보존량 검증"""
        E_total_in = self.kinetic_energy_in
        E_total_out = self.kinetic_energy_out + self.deposited_energy
        
        error = abs(E_total_in - E_total_out) / E_total_in
        return error < tol
```

### 6.3 Step 함수 수정

```python
def step(self) -> ConservationReport:
    # Weight (진단용)
    mass_in = float(cp.sum(self.psi_gpu))
    
    # Kinetic Energy (물리적 보존량)
    E_kinetic_in = self._compute_kinetic_energy_gpu(self.psi_gpu)
    
    # Transport step
    self.psi_gpu = self.transport_step.apply(...)
    
    # After step
    mass_out = float(cp.sum(self.psi_gpu))
    E_kinetic_out = self._compute_kinetic_energy_gpu(self.psi_gpu)
    E_deposited = float(cp.sum(self.accumulators.dose_gpu))
    
    report = ConservationReport(
        mass_in=mass_in,
        mass_out=mass_out,
        kinetic_energy_in=E_kinetic_in,
        kinetic_energy_out=E_kinetic_out,
        deposited_energy=E_deposited,
    )
    
    # 에너지 보존 검증
    report.is_valid = report.validate_energy_conservation()
    
    return report
```

---

## 7. 결론

| 질문 | 답변 |
|-----|-----|
| 에너지 보존이 올바르게 구현되었나? | ✗ **아니오** - Weight 보존만 검증, 에너지 보존 미검증 |
| 상태 벡터가 매 스텝 갱신되는가? | ✓ **예** - `self.psi_gpu`가 올바르게 전파됨 |
| 정지 입자 처리가 올바른가? | △ **부분적** - Dose 기록은 됨, 보존 검증은 안 됨 |

**핵심 수정 필요사항**:
1. `compute_total_kinetic_energy()` 함수 구현
2. 매 스텝 에너지 보존 검증 추가
3. Weight 보존 검증을 "진단용"으로 재분류
4. Conservation Report에 에너지 항목 추가