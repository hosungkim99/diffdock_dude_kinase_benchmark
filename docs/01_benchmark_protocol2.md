# 01. Benchmark Protocol

본 문서는 DUD-E kinase 하위 데이터셋에서 DiffDock을 평가하기 위해 사용된 벤치마크 프로토콜을 설명합니다. 
재현성을 보장하기 위해 모든 정의, 평가 기준, 그리고 데이터 집계(aggregation) 정책을 명시적으로 규정합니다.

---

## 1. 벤치마크 데이터셋 (Benchmark Dataset)

### 1.1 DUD-E Kinase 서브셋

- 데이터셋 출처: Directory of Useful Decoys, Enhanced (DUD-E)
- 타겟 클래스: Kinase
- 타겟 개수: 26개
- 타겟 목록:
  - abl1
  - akt1
  - akt2
  - braf
  - cdk2
  - csf1r
  - egfr
  - fak1
  - fgfr1
  - igf1r
  - jak2
  - kit
  - kpcb
  - lck
  - mapk2
  - met
  - mk01
  - mk10
  - mk14
  - mp2k1
  - plk1
  - rock1
  - src
  - tgfr1
  - vgfr2
  - wee1
 
- 데이터셋 버전 / 다운로드 출처:
  - https://dude.docking.org/
- 수용체(Receptor) 전처리:
  - 잔기(Residue) 명명 정규화 적용 여부: ㄴ
  - 사용된 스크립트: `standardize_pdb_resnames.py`
  - 수정된 잔기 타입 (해당하는 경우):
    - [e.g., HID/HIE/HIP → HIS]

---

### 1.2 리간드 구성 (Ligand Composition)

<img width="577" height="1216" alt="image" src="https://github.com/user-attachments/assets/9ec3e927-f201-41ad-9d22-b5008dbada7c" />

정의:

- Actives (활성 물질): DUD-E에서 제공하는, 실험적으로 결합이 검증된 물질.
- Decoys (디코이): 물리화학적 특성은 일치하지만 구조적(topologically)으로 다른 화합물.

데이터 분할(Split) 처리:

- 추론(Inference) 시 Active와 Decoy는 분리되어 처리됨.
- 타겟 및 분할별 CSV 파일 생성에 사용된 스크립트:
  - `create_csv_with_sdf_normal.py` (해당하는 경우)

---

## 2. 모델 추론 설정 (Model Inference Settings)

### 2.1 DiffDock 버전

도킹 추론은 DiffDock(GitHub: gcorso/DiffDock)을 사용하여 수행됨.

- Git commit hash: `85c49b60d3e0b0182a59ee43a34a6d7036981284`
- Git describe: `v1.1.3-2-g85c49b6`
- 추론 설정: `default_inference_args.yaml`
- Confidence 모델 경로: `./workdir/v1.1/confidence_model`
- 복합체당 샘플링 수: `samples_per_complex = 10`
- 확산 추론 스텝 수: `inference_steps = 20`
- 배치 크기: `batch_size = 16`
- 확률성(Stochasticity) 설정:
  - `no_random = false`
  - `no_random_pocket = false`
- 구동 환경:
  - 파이썬 버전: 3.9.18
  - CUDA 버전: 13.0
  - GPU 타입: a10

---

### 2.2 포즈 선택 전략 (Pose Selection Strategy)

- 포즈 선택: **rank1 만 사용 (rank1 only)**
- 정의:
  - 각 리간드에 대해 예측된 가장 높은 confidence 점수를 가진 포즈를 사용함.
  - DiffDock은 확률적 샘플링을 통해 각 리간드당 여러 개의 포즈 가설을 생성함. 학습된 confidence 모델이 각 포즈에 confidence 점수를 부여하며, 이 중 가장 높은 점수를 받은 포즈가 rank1 포즈로 선택됨.
- 근거 (왜 Rank1만 사용하는가?):
  - 도킹 랭킹과 가상 탐색(Virtual screening) 평가 지표 간의 일관성을 보장하기 위함.
  - 도킹 예측과 가상 탐색 랭킹의 일관성을 확보하고자, 리간드당 가장 높은 confidence를 가진 포즈 하나만 사용함.

---

### 2.3 Confidence 점수 정의

- Confidence 점수 추출 출처:
  - `make_diffdock_score_table.py` 스크립트를 통해 rank1 포즈 출력을 기반으로 생성된 `diffdock_scores_rank1.csv`
- 랭킹 기준:
  - 리간드들은 confidence 점수의 내림차순으로 정렬됨.
- 평가 지표 계산 시 역할 (Used for):
  - 추출된 Confidence 점수는 리간드의 랭킹 기준으로 사용되며, 다음 지표들을 계산하는 핵심 입력값으로 활용됨.
    - ROC-AUC
    - EF@k%
    - nEF
    - Calibration 분석

---

## 3. 추론 안정성 정의 (Inference Stability Definition)

### 3.1 상태 카테고리

각 리간드는 다음 상태 중 하나로 분류됨:

- **Success (성공)**
- **Skip (건너뜀)**
- **Fail (실패)**

정의:

- Success:
  - Rank1 포즈 파일이 존재함
  - Confidence 점수가 성공적으로 계산됨
  - Rank1 포즈 및 confidence 점수 생성에 성공함
- Fail:
  - 그래프 생성 실패로 인해 추론이 종료됨
  - 조건: ["No edges and no nodes..."]
  - 유효한 포즈가 생성되지 않음
- Skip:
  - 테스트 또는 confidence 데이터셋에서 리간드가 처리되지 않음
  - 조건 1: ["test dataset does not contain..."]
  - 조건 2: ["confidence dataset does not contain..."]

파싱 방법:

- 사용된 스크립트: `parse_inference_err.py`
- 출처: 각 타겟별 `.err` 로그 파일

---

### 3.2 재시도(Retry) 정의

- Retry = Fail + Skip
- 타겟별 재시도 비율:
  - `retry_rate = (fail + skip) / total_ligands`

---

### 3.3 누락 정책 (Missing Policy)

- missing_policy = **bottom**
- 정의:
  - 누락된 리간드(skip/fail)는 부여 가능한 가장 낮은 confidence 점수를 할당받고 전체 랭킹의 맨 아래에 위치함.
  - artificial inflation 방지를 위해 missing ligand는 ranking에서 가장 마지막에 위치하도록 하는 방침.
- 근거:
  - 보수적인 평가 전략
  - 인위적인 성능 부풀리기(inflation) 방지

---

## 4. 구조적 유효성 기준 (Structural Validity Criteria)

### 4.1 COM 거리 기준 (COM Distance Criterion)

- 평가 지표: 질량 중심(Center-of-Mass, COM) 거리
- 정의:

$$
\text{COM} = | C_{\text{predicted}} - C_{\text{reference}} |
$$

- COM 거리를 사용하는 이유:
  - RMSD보다 계산이 안정적
  - atomic ordering mismatch 문제 회피
  - coarse pose correctness 판단에 충분
- 임계값: **COM ≤ 2 Å**
- 해석 및 연계 지표:
  - 정답(reference) 대비 COM 거리가 2 Å 이하일 경우 포즈가 구조적으로 유효한 것(Hit)으로 간주하며, 전체 샘플 중 이 임계값을 통과한 비율은 최종 요약본에서 `COMdist2rate` 지표로 집계됨.
  - 추가적으로 생성 품질 확인을 위해 `mean_COMdist` (전체 평균), `active_mean_COMdist` (Active 평균), `top1_mean_COMdist` (Rank1 평균) 거리를 추적함.
- 사용된 스크립트:
  - `compute_comdist2.py`

---

### 4.2 포켓 품질 관리 (Pocket QC)

**QC 필터링은 기하학적으로 불가능한 포즈로 인해 발생하는 인위적인 성능 부풀려짐(Enrichment)을 방지합니다.**

수행 스크립트:

- `pocket_qc_rank1.py`
- `qc_rank1_bundle.py`
- `sanity_check.py`

QC 플래그(Flag) 및 요약 지표 매핑:

- **Steric clash (입체적 충돌):** 단백질 원자와 리간드 원자가 비정상적으로 겹치는지 검사함. 최종 요약본에서 `clash_rate_all` (전체 충돌 비율) 및 `clash_rate_top1pct` (상위 1% 내 충돌 비율)로 집계됨.
- **Outside pocket (포켓 이탈):** 생성된 포즈가 지정된 결합 포켓 영역 내부에 정상적으로 위치하는지 검사함. 최종 요약본에서 정상 진입 비율인 `pocket_in_rate_all` 및 `pocket_in_rate_top1pct`로 집계됨.
- Geometric inconsistency (기하학적 불일치 - 해당하는 경우)

출력 결과물:

- `flagged_clash.csv`
- `flagged_outside.csv`
- `pocket_qc_summary.json`
- `qc_summary.json`

---

## 5. 평가 지표 (Evaluation Metrics)
**상세한 평가 지표 정의는 `docs/03_metrics_definition.md`에 제공됩니다.**

### 5.1 분류 지표 (Classification Metrics)

전체적인 활성 물질(Active) 분류 성능을 평가합니다.

- ROC-AUC
- LogAUC
- BEDROC (초기 탐색 성능에 가중치를 부여한 ROC 지표)

계산 스크립트:

- `eval_dude_metrics.py`

---

### 5.2 초기 탐색 지표 (Early Enrichment Metrics)

가상 탐색 환경에서 상위 최상위권의 타격률(Hit rate)을 평가합니다.

- EF@1%, EF@5%, EF@10%
- nEF@1%, nEF@5%, nEF@10%

정의: 상세한 평가 지표 정의는 `docs/03_metrics_definition.md`에 제공됩니다.

---

### 5.3 포즈 기반 지표 (Pose-based Metrics)

생성된 포즈(Pose)의 구조적 정확도와 거리를 평가합니다.

- **COMdist2rate:** 전체 샘플 중 COM ≤ 2 Å 기준을 통과한 성공 비율
- **평균 COM 거리 지표:**
  - `mean_COMdist`: 전체 리간드의 평균 COM 거리
  - `active_mean_COMdist`: 활성 물질(Active)들의 평균 COM 거리
  - `top1_mean_COMdist`: 상위 1순위(Confidence 최고점) 포즈의 평균 COM 거리

---

### 5.4 포켓 품질 지표 (Pocket QC Metrics)

생성된 포즈가 단백질 포켓 내에 물리적으로 타당하게 위치하는지 평가합니다.

- **Clash Rate (입체적 충돌 비율):**
  - `clash_rate_all`: 전체 포즈 중 충돌이 발생한 비율
  - `clash_rate_top1pct`: 상위 1% 포즈 중 충돌이 발생한 비율
- **Pocket-in Rate (포켓 내부 위치 비율):**
  - `pocket_in_rate_all`: 전체 포즈 중 포켓 내부에 정상적으로 생성된 비율
  - `pocket_in_rate_top1pct`: 상위 1% 포즈 중 포켓 내부에 정상적으로 생성된 비율
---

## 6. 데이터 집계 전략 (Aggregation Strategy)

### 6.1 타겟별 평가 (Per-target Evaluation)

- 평가 지표는 각 타겟별로 독립적으로 계산됨.
- 출력 결과물:
  - `metrics.json`
  - `ranking.csv`
  - `roc.csv`

---

### 6.2 타겟 간 집계 (Cross-target Aggregation)

집계에 사용된 스크립트:

- `build_master_table.py`
- `build_metrics_summary2.py`
- `build_calibration.py`

집계 방법:

- 타겟 간 매크로 평균(Macro-average) 적용
- [평균(Mean) / 중앙값(Median) — 명시할 것]

---

### 6.3 캘리브레이션 분석 (Calibration Analysis)

- Confidence 구간(binning) 설정 방법: [   ]
- 평가 지표:
  - Expected Calibration Error (ECE) [사용되는 경우]
- 출력 파일:
  - `calibration_table.csv`
  - `calibration_summary.csv`

---

## 7. 전체 평가 워크플로우 (Overall Evaluation Workflow)

1. 수용체(Receptor) 정규화
2. 타겟별 CSV 생성
3. DiffDock 추론
4. 후처리(Postprocessing)
5. 점수 테이블(Score table) 구축
6. 추론 로그 파싱
7. COM 거리 평가
8. 평가 지표 계산
9. QC 필터링
10. 데이터 집계 및 캘리브레이션 분석

전체 파이프라인 스크립트 경로:

- `scripts_2/run/`
- `scripts_2/postprocess/`
- `scripts_2/eval/`
- `src/aggregate/`

---

## 재현성 참고사항 (Reproducibility Notes)

- 모든 스크립트는 본 레포지토리에 포함되어 있습니다.
- 동일한 추론 결과(outputs)가 주어질 경우, 모든 평가 지표는 결정론적(deterministically)으로 동일하게 계산됩니다.
- 설정 파일:
  - `config.json`
  - `master_table.config.json`
