# 03. Metrics Definition

본 문서는 이번 벤치마크 평가에서 사용된 모든 평가 지표의 공식적인 수학적 정의를 다룹니다.
모든 정의는 최종 결과물인 `metrics_summary_all_diffdock_2.csv`의 데이터 구조 및 `src/metrics/dude_metrics.py`의 구현과 정확히 일치합니다.

**기본 변수 정의:**
* $N$: 전체 리간드의 수
* $n$: 활성 물질(Active)의 수
* $y_i \in \{0, 1\}$: $i$번째 리간드의 실제 라벨 (1=Active, 0=Decoy)
* $s_i$: 모델이 출력한 $i$번째 리간드의 Confidence 점수

---

## 1. 추론 안정성 지표 (Inference Stability Metrics)

추론 과정에서 파이프라인이 정상적으로 작동했는지를 평가합니다.

* **Fail Rate (`fail_rate`):** 그래프 생성 실패 등으로 유효한 포즈를 얻지 못한 비율
* **Skip Rate (`skip_rate`):** 데이터 누락 등의 이유로 처리가 건너뛰어진 비율
* **Retry Rate (`retry_rate`):** 전체 샘플 중 재시도가 필요한(실패+건너뜀) 샘플의 비율

$$
\text{retry\_rate} = \frac{\text{Fail Count} + \text{Skip Count}}{N}
$$

* **Coverage Rate (`coverage_rate`):** 성공적으로 추론을 완료한 샘플의 비율 ($1 - \text{retry\_rate}$)

---

## 2. 분류 지표 (Classification Metrics)

### 2.1 ROC-AUC
전체적인 랭킹 성능을 순위 기반 통계(Rank-based statistics)로 계산합니다.
$R_i$를 각 Active 리간드의 순위(1이 가장 높은 점수)라 할 때, ROC-AUC는 다음과 같이 정의됩니다 (Mann-Whitney U 통계량과 동일).

$$
\text{ROC-AUC} = \frac{\sum R_i - \frac{n(n+1)}{2}}{n \times (N-n)}
$$
*(1.0은 완벽한 정렬, 0.5는 무작위 정렬, 0.5 미만은 역정렬을 의미합니다.)*

### 2.2 LogAUC
ROC 커브의 초기 부분을 로그 스케일로 적분하여 초기 탐색(Early enrichment) 성능에 가중치를 부여한 지표입니다. FPR(False Positive Rate)과 TPR(True Positive Rate)을 기준으로 계산합니다.

$$
\text{LogAUC} = \int \text{TPR} \, d(\log_{10}(\text{FPR}))
$$

### 2.3 BEDROC
BEDROC (Boltzmann-Enhanced Discrimination of ROC) 역시 초기 랭킹에 강한 가중치($\alpha$)를 부여하는 지표입니다. $\alpha=20$을 기본값으로 사용합니다.

$$
\text{BEDROC} = \text{Normalization} \times \sum_{i \in \text{Actives}} e^{-\frac{\alpha R_i}{N}}
$$

---

## 3. 가상 탐색 집중 지표 (Early Enrichment Metrics)

### 3.1 Enrichment Factor (EF@k%)
상위 $k\%$ (예: 1%, 5%, 10%) 추출 시, 무작위 추출 대비 Active를 얼마나 더 많이 찾아냈는지 나타냅니다.
상위 $k\%$ 내의 Active 개수를 $\text{TP}_k$, 상위 $k\%$의 전체 샘플 수를 $N_k$라 할 때:

$$
\text{EF@k\%} = \frac{\text{TP}_k / N_k}{n / N}
$$

### 3.2 Normalized Enrichment Factor (nEF@k%)
해당 $k\%$ 구간에서 달성할 수 있는 이론적 '최대 EF' 대비 현재 달성한 EF의 비율입니다.

$$
\text{EF}_{\text{max}} = \frac{\min(n, N_k) / N_k}{n / N}
$$

$$
\text{nEF@k\%} = \frac{\text{EF@k\%}}{\text{EF}_{\text{max}}}
$$
*(0은 무작위 수준, 1은 이론적 최대치를 의미합니다.)*

---

## 4. 포즈 및 포켓 품질 지표 (Pose & QC Metrics)

### 4.1 COM 거리 및 성공률 (COM Distance & COMdist2rate)
정답 리간드($C_{\text{ref}}$)와 예측 리간드($C_{\text{pred}}$)의 질량 중심(Center of Mass) 간 거리를 측정합니다.

$$
\text{COM} = | C_{\text{pred}} - C_{\text{ref}} |
$$

* **COMdist2rate:** 전체 추론 성공 샘플 중 $\text{COM} \le 2.0 \text{ \AA}$ 을 만족하는 비율 (성공률)
* **mean_COMdist:** 전체 리간드의 평균 COM 거리
* **active_mean_COMdist:** Active 물질들만의 평균 COM 거리
* **top1_mean_COMdist:** Confidence 점수가 가장 높은 1순위 포즈의 평균 COM 거리

### 4.2 포켓 품질 지표 (Pocket QC)
생성된 3D 구조의 물리적 타당성을 검증합니다.
* **Clash Rate (`clash_rate_all`, `clash_rate_top1pct`):** 단백질과 리간드 간 비정상적인 입체적 충돌(Steric clash)이 발생한 포즈의 비율
* **Pocket-in Rate (`pocket_in_rate_all`, `pocket_in_rate_top1pct`):** 리간드가 지정된 타겟 포켓 영역 내부에 안정적으로 자리 잡은 비율

---

## 5. 결측치 처리 정책 (Missing Policy Impact)

추론에 실패했거나 건너뛴(Skip/Fail) 리간드의 처리 방침입니다.
본 벤치마크는 인위적인 성능 부풀리기를 방지하기 위해 `missing_policy = bottom`을 채택합니다.

* 누락된 리간드의 Confidence 점수는 $-\infty$ 로 처리됩니다.
* 전체 랭킹의 최하단에 배치되어 EF 및 ROC 지표가 최대한 보수적으로 계산되도록 강제합니다.
