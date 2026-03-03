# 5. Calibration Analysis

본 문서는 단백질-리간드 결합 예측을 수행하는 생성형 모델(DiffDock)의 rank1 confidence score에 대한 **확률적 신뢰도(Calibration)**를 분석한다. 

DiffDock과 같은 확산(Diffusion) 기반 모델은 역확산 과정을 통해 포즈를 생성한 후, 별도의 보조 모델(Auxiliary confidence model)을 통해 점수를 산출한다. 이 과정에서 모델이 출력하는 confidence 값과 실제 결합 성공 확률 간의 괴리(특히 Overconfidence)가 발생하기 쉬우므로, 이를 정량적으로 평가하고 교정하는 과정이 필수적이다.

---

## 5.1 Motivation & Definition

DiffDock는 각 ligand pose에 대해 confidence score $s$를 출력한다. 이 값이 잘 보정(calibrated)되었다면, 높은 confidence는 실제 active 또는 성공 pose일 확률이 높음을 의미하며 다음이 성립해야 한다.

$$
P(\text{success} \mid \text{confidence} = s) \approx s
$$

이를 정량화하기 위해, 전체 예측을 confidence 구간(bin) $B_k$로 나누어 다음과 같이 평균 신뢰도와 실제 성공률을 정의한다.

**평균 신뢰도 (Mean Confidence):**

$$
\text{Conf}_{\text{mean}, k} = \frac{1}{|B_k|} \sum_{i \in B_k} \text{confidence}_i
$$

**실제 성공률 (Empirical Rate):**

$$
\text{Rate}_{\text{emp}, k} = \frac{\text{Number of positives in } B_k}{|B_k|}
$$

각 구간에서의 Calibration 오차는 $| \text{Conf}_{\text{mean}, k} - \text{Rate}_{\text{emp}, k} |$ 로 계산된다.

---

## 5.2 Calibration Metrics & Types

전체적인 보정 오차를 평가하기 위해 **ECE (Expected Calibration Error)** 등을 사용한다. ECE가 작을수록 모델의 예측 신뢰도가 실제 분포와 잘 일치함을 의미한다.

$$
\text{ECE} = \sum_{k=1}^{K} \frac{n_k}{N} | \text{Conf}_{\text{mean}, k} - \text{Rate}_{\text{emp}, k} |
$$

프로젝트의 목적에 따라 두 가지 기준(Type)으로 Calibration을 분리하여 평가한다.

1. **Label-based calibration (`label_*`):** `label = 1` (Active ligand)을 positive로 정의. Confidence가 실제 활성 물질일 확률을 제대로 반영하는지 평가.
2. **Pose-based calibration (`pose_*`):** 지정된 거리 기준(예: 2.0 Å) 이내의 결합을 positive로 정의. Confidence가 생성된 포즈의 구조적 정확도를 반영하는지 평가.

---

## 5.3 Result Data Structure

분석 결과는 타겟(Target)별 독립적인 연산을 거쳐 다음 두 파일에 저장된다.

**1. 구간별 상세 분석 (`calibration_table_diffdock_2.csv`)**
각 타겟 및 평가 기준에 따른 Confidence Bin별 상세 지표를 담고 있다.

| column | 설명 |
|---|---|
| `target` | 타겟 단백질 이름 |
| `calib_type` | calibration 기준 (`label` 또는 `pose`) |
| `bin_id` / `bin_low` / `bin_high` | confidence 구간 정보 (index, 하한, 상한) |
| `n` | 해당 bin의 총 샘플 수 |
| `mean_conf` | bin 내 모델의 평균 신뢰도 |
| `empirical_rate`| bin 내 실제 positive 비율 |
| `abs_gap` | calibration 오차 ($| \text{mean\_conf} - \text{empirical\_rate} |$) |

**2. 타겟별 요약 분석 (`calibration_summary_diffdock_2.csv`)**
타겟별 전반적인 Calibration 성능을 Label과 Pose 기준으로 나누어 요약한다.

| column | 설명 |
|---|---|
| `target` | 타겟 단백질 이름 |
| `n_total_success` | 전체 성공 샘플 수 (추론 성공 기준) |
| `label_n`, `label_ece`, `label_mce`, `label_brier` | **Label 기반 평가 지표:** 샘플 수, ECE, MCE(최대 오차), Brier Score |
| `pose_n`, `pose_ece`, `pose_mce`, `pose_brier` | **Pose 기반 평가 지표:** 샘플 수, ECE, MCE(최대 오차), Brier Score |
| `n_bins` / `binning` | Calibration 계산에 사용된 구간(bin) 수 및 방식 |
| `pose_cutoff_A` | Pose 성공 기준으로 사용된 거리 임계값 (Å) |
| `require_success` | 성공 포즈를 필터링 조건으로 요구하는지 여부 |

---

## 5.4 Interpretation & Analysis

그래프나 시각화 자료 없이도 제공된 `.csv` 파일의 수치를 통해 모델의 Calibration 상태를 직접 진단할 수 있다.

**1. 구간별 신뢰도 상태 진단 (Reliability Check)**
`calibration_table_diffdock_2.csv`의 데이터를 확인하여 모델의 과적합 여부를 판단한다.
* **Well-calibrated:** `mean_conf` $\approx$ `empirical_rate` 이며, 요약본의 `ECE` 수치가 낮다.
* **Overconfident:** `mean_conf` > `empirical_rate` (모델이 스스로의 예측을 과대평가함. 확산 모델에서 흔히 발생).
* **Underconfident:** `mean_conf` < `empirical_rate` (모델의 예측이 실제 정답률보다 보수적임).

**2. 평가 지표의 종합적 해석 (ECE, MCE, Brier)**
`calibration_summary_diffdock_2.csv`를 통해 타겟별 전반적 성능을 평가한다.
* **ECE (Expected Calibration Error):** 전반적인 보정 오차의 평균을 보여준다.
* **MCE (Maximum Calibration Error):** 특정 구간에서 발생한 최악의 오차를 보여준다. ECE가 낮더라도 MCE가 높다면, 특정 Confidence 구간(예: 0.9~1.0)에서 모델이 크게 착각하고 있을 수 있으므로 주의해야 한다.
* **Brier Score:** 예측 확률과 실제 결과 간의 평균 제곱 오차로, 전반적인 예측의 정확성과 캘리브레이션을 동시에 평가할 수 있다.

**3. Enrichment 평가와의 독립성**
Enrichment Factor (EF)나 ROC-AUC가 높아 모델이 랭킹(Ranking)을 잘 수행하더라도, 출력된 Confidence score 자체는 확률적으로 붕괴되어 있을 수 있다. 랭킹 품질(Ranking quality)과 확률 품질(Probability quality)은 완전히 분리하여 수치적으로 접근해야 한다.

---

## 5.5 Actionable Insights (Post-hoc & Usage)

**1. 실무적 활용 (Practical Implication)**
신뢰할 수 있는 Calibration 평가는 Downstream Wet-lab 실험의 비용을 최소화하고, Confidence 임계치(Threshold)를 설정하여 효과적으로 가짜 양성(False Positive)을 걸러내는 데 핵심적인 역할을 한다.

**2. 사후 보정 (Optional Post-hoc Calibration)**
평가 결과 특정 타겟 군에서 ECE나 MCE가 일관되게 높게 나타난다면, 단순 임계값 조정을 넘어 다음과 같은 사후 확률 보정을 검토할 수 있다.
* **Temperature Scaling:** 모델의 출력 로짓(Logits)을 온도 매개변수 $T$로 나누어 분포를 부드럽게 조정한다.
* **Platt Scaling:** 모델의 출력 점수를 로지스틱 회귀에 적합시켜 확률값으로 보정한다.

모든 캘리브레이션 지표 확인 및 보정은 타겟별로 독립 수행하는 것을 권장한다.
