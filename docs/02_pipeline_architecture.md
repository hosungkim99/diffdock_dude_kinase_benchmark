---
# 2. Pipeline Architecture

본 문서는 DiffDock 기반 DUD-E Kinase 벤치마크의 전체 파이프라인 구조를 기술한다.
파이프라인은 다음의 6개 Stage로 구성된다.

Preprocess → Inference → Postprocess → Quality Control (QC) → Evaluation → Aggregation

각 Stage는 독립 실행 가능하며, target 단위로 병렬 처리된다.

---

# 2.1 Overall Data Flow

```
DUD-E Raw
  ├── receptor.pdb
  ├── actives_final.sdf
  └── decoys_final.sdf
        ↓
[Preprocess]
        ↓
DiffDock-ready CSV
        ↓
[Inference]
        ↓
rank1.sdf + confidence score
        ↓
[Postprocess]
        ↓
master_table.csv
        ↓
[QC]
        ↓
COMdist / clash / pocket-in metrics
        ↓
[Evaluation]
        ↓
EF, nEF, ROC-AUC, LogAUC
        ↓
[Aggregation]
        ↓
metrics_summary_all.csv
calibration_table.csv
```

---

# 2.2 Stage A — Preprocessing

## 목적

DiffDock inference에 적합한 입력 형식으로 DUD-E 데이터를 정제한다.

## 입력

* `receptor.pdb`
* `actives_final.sdf`
* `decoys_final.sdf`

## 주요 처리

### 1. PDB 표준화

비정형 residue 명칭을 canonical residue name으로 변환.

문제:
HIS → HID / HIE
와 같은 비표준 residue가 inference 실패를 유발함.

### 2. Ligand CSV 생성

DiffDock inference 입력 형식:

```
complex_name, protein_path, ligand_path
```

각 ligand에 대해:

x_i = (protein_t, ligand_{t,i})

---

# 2.3 Stage B — DiffDock Inference

## 목적

각 target–ligand pair에 대해 pose distribution을 생성하고 confidence score를 계산한다.

## 설정

* samples_per_complex = 10
* inference_steps = 20
* rank1 pose 사용
* confidence model 적용

## 수학적 해석

DiffDock는 다음 확산 과정을 학습한다:

p_theta(x_0 | x_T, t)

여기서

* x_0 : 최종 ligand pose
* x_T : noise 상태
* t : diffusion time

최종 score:

s_i = confidence(rank1_pose_i)

---

# 2.4 Stage C — Postprocessing

## 목적

Inference 결과를 정리하여 master table 생성.

## 출력

`master_table.csv`

### 주요 column

| column     | 의미                         |
| ---------- | -------------------------- |
| ligand_id  | ligand identifier          |
| label      | 1=active, 0=decoy          |
| success    | inference 성공 여부            |
| status     | success / failed / skipped |
| confidence | rank1 confidence           |
| pocket_in  | pocket 내부 여부               |
| COMdist_A  | center-of-mass distance    |

---

# 2.5 Stage D — Quality Control (QC)

## 1. COM Distance

d_COM = || c_ligand - c_pocket ||^2

cutoff:

d_COM ≤ 2.0 Å

## 2. Clash Count

van der Waals overlap 계산.

## 3. Retry 정의

retry = failed + skipped

retry ratio > 20% 이면 해당 target inference 미완료로 판단.

---

# 2.6 Stage E — Evaluation Metrics

모든 metric은 confidence score 기반 ranking으로 계산한다.

## 1. ROC-AUC

[
\text{AUC} = \int_0^1 TPR(FPR) dFPR
]
AUC = 

## 2. Enrichment Factor

selection fraction ( \chi )

EF_chi = (TP_chi / N_chi) / (A / N)

* TP_\chi: 상위 χ%에서 active 수
* A: 전체 active 수

---

## 3. Normalized EF (nEF)

nEF_chi = EF_chi / EF_chi_max

EF_chi_max = 1 / chi

---

## 4. LogAUC

early enrichment 강조:

LogAUC = ∫ TPR(FPR) d(log FPR)

---

## 5. BEDROC

[
BEDROC_\alpha =
\frac{
\sum_i e^{-\alpha r_i}
}{
R_\alpha
}
]

---

# 2.7 Stage F — Aggregation

## Target-level → Global Summary

### 1. metrics_summary_all.csv

각 target:

* EF@1%
* EF@5%
* EF@10%
* nEF
* AUC
* coverage
* retry rate

### 2. calibration_table.csv

confidence calibration:

[
\text{abs gap} =
| \text{mean_confidence} - \text{empirical_rate} |
]

---

# 2.8 Failure Taxonomy

Inference 실패는 다음으로 분류:

1. No edges and no nodes
2. test dataset does not contain
3. confidence dataset does not contain

---

# 2.9 Design Principles

## 1. Target-wise isolation

모든 계산은 target 단위 독립 수행.

## 2. Deterministic rerun 가능

* 모든 입력은 CSV 기반
* 모든 결과는 재계산 가능

## 3. Leakage 최소화

* target split 독립
* actives/decoys 분리

---

# 2.10 HPC Parallelization Strategy

각 target은 독립 작업:

[
T = {t_1, t_2, ..., t_{26}}
]

총 추론 수:

[
\sum_{t=1}^{26} (N_{actives}^{(t)} + N_{decoys}^{(t)})
]

SLURM job array 구조 사용.

---

# 2.11 Reproducibility Checklist

* DiffDock version 명시
* Conda env export 포함
* random seed 고정
* commit hash 기록

---

# 2.12 Summary

본 파이프라인은 다음을 달성한다:

1. 26 kinase target 대량 inference
2. pose-level QC 정량화
3. early enrichment 정밀 분석
4. calibration 분석
