---

# 5. Calibration Analysis

본 문서는 DiffDock rank1 confidence score의 **확률적 신뢰도(calibration)**를 분석한다.
모델이 출력하는 confidence 값이 실제 성공 확률과 얼마나 일치하는지 정량적으로 평가한다.

---

# 5.1 Motivation

DiffDock는 각 ligand pose에 대해 confidence score `s`를 출력한다.

이 값이 잘 보정(calibrated)되었다면:

* 높은 confidence → 실제 active 또는 성공 pose일 확률 높음
* 낮은 confidence → 실패 확률 높음

즉,

```
P(success | confidence = s) ≈ s
```

가 성립해야 한다.

---

# 5.2 Calibration Table 구조

파일:

```
calibration_table_diffdock_2.csv
```

## Column 정의

| column         | 설명                                       | 비고                        |   |
| -------------- | ---------------------------------------- | -------------------------- | - |
| target         | 타겟 이름                                    |                            |   |
| calib_type     | calibration 기준 (label / pose)            |                            |   |
| bin_id         | confidence 구간 index                      |                            |   |
| bin_low        | bin 하한                                   |                            |   |
| bin_high       | bin 상한                                   |                            |   |
| n              | 해당 bin 샘플 수                              |                            |   |
| mean_conf      | bin 내 평균 confidence                      |                            |   |
| empirical_rate | 실제 positive 비율                           |                            |   |
| abs_gap        |                                          | mean_conf - empirical_rate |   |

---

# 5.3 Calibration 정의

각 confidence bin B_k 에 대해:

```
mean_conf_k = average(confidence_i in B_k)
```

```
empirical_rate_k = (# positives in B_k) / n_k
```

calibration 오차:

```
abs_gap_k = | mean_conf_k - empirical_rate_k |
```

---

# 5.4 Expected Calibration Error (ECE)

전체 calibration 오차는 다음과 같이 정의한다.

$$
\text{ECE} = \sum_{k=1}^{K} \frac{n_k}{N} \left| \text{Conf}_{\text{mean}, k} - \text{Rate}_{\text{emp}, k} \right|
$$

* n_k : bin k 샘플 수
* N   : 전체 샘플 수

ECE가 작을수록 calibration이 우수하다.

---

# 5.5 Calibration Type

## 1. Label-based calibration

positive 정의:

```
label = 1 (active)
```

즉, confidence가 active 확률을 반영하는지 평가한다.

---

## 2. Pose-based calibration

positive 정의:

```
COMdist ≤ 2.0 Å
```

즉, confidence가 실제 pose 정확도를 반영하는지 평가한다.

---

# 5.6 Calibration Summary 파일

파일:

```
calibration_summary_diffdock_2.csv
```

## 주요 column

| column     | 설명                         |
| ---------- | -------------------------- |
| target     | 타겟                         |
| calib_type | calibration 기준             |
| ECE        | expected calibration error |
| mean_gap   | 평균 abs_gap                 |
| max_gap    | 최대 abs_gap                 |
| n_total    | 전체 샘플 수                    |

---

# 5.7 Interpretation Guide

## 1. Well-calibrated model

특징:

* mean_conf ≈ empirical_rate
* abs_gap 작음
* ECE 낮음

---

## 2. Overconfident model

```
mean_conf > empirical_rate
```

confidence가 실제 성공 확률보다 과대평가됨.

---

## 3. Underconfident model

```
mean_conf < empirical_rate
```

confidence가 실제보다 보수적임.

---

# 5.8 Target-wise Calibration Variability

각 target별로:

* confidence 분포
* 성공률 분포
* retry rate

가 다르다.

특히 다음 조건에서 calibration 왜곡 가능:

```
retry_ratio > 0.2
```

또는

* active 비율 매우 낮음
* inference 실패 편향 존재

---

# 5.9 Relationship to Enrichment

높은 EF가 반드시 좋은 calibration을 의미하지 않는다.

예:

* 모델이 ranking은 잘함 (high EF)
* 그러나 confidence 값 자체는 확률적으로 왜곡됨

따라서 다음을 분리 평가해야 한다:

* Ranking quality (EF, ROC-AUC)
* Probability quality (Calibration)

---

# 5.10 Practical Implication

Calibration이 중요한 이유:

1. Confidence threshold 기반 필터링
2. Active 후보 우선순위 결정
3. Downstream wet-lab validation 비용 최소화

---

# 5.11 Recommended Usage

실무에서는:

1. Target별 ECE 확인
2. High-ECE target 재학습 검토
3. 필요 시 temperature scaling 적용

---

# 5.12 Optional Post-hoc Calibration

간단한 보정 방법:

```
s_calibrated = sigmoid(a * s + b)
```

또는

```
temperature scaling
```

---

# 5.13 Reproducibility

Calibration 계산은 다음 파일을 기반으로 한다:

* master_table.csv
* calibration_table_diffdock_2.csv
* calibration_summary_diffdock_2.csv

모든 계산은 target 단위 독립 수행한다.

---

# 5.14 Summary

본 분석을 통해:

* DiffDock confidence score의 확률적 신뢰도 평가
* Target별 calibration 편차 정량화
* Ranking metric과의 독립성 검증

을 수행한다.

---
