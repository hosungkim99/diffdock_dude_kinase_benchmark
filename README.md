---

# DiffDock DUD-E Kinase Benchmark Pipeline

## Overview

This repository provides a fully automated benchmarking pipeline for evaluating **DiffDock** on the **DUD-E kinase subset (26 targets)**.

The pipeline covers:

* Receptor preprocessing
* Target-wise ligand CSV generation
* DiffDock inference
* Post-processing & QC
* Metric computation (EF, nEF, LogAUC, ROC-AUC)
* Pose quality evaluation (COM distance)
* Target-level aggregation
* Calibration analysis
* Final benchmark summary table generation

The objective is to provide a **reproducible, large-scale, structure-aware evaluation framework** for generative docking models.

---

## Key Contributions

* 🔁 Fully automated multi-stage benchmarking pipeline
* 📊 Integrated enrichment + pose-quality evaluation
* 🧪 Explicit skip/fail taxonomy from inference logs
* 🧮 Target-level master table aggregation
* 📈 Calibration analysis (confidence vs correctness)
* 🏗 Modular structure (run / postprocess / eval / qc / aggregate)

---

# Pipeline Overview

The complete pipeline consists of the following ordered stages:

---

## Stage 1 — Receptor Preprocessing

### 1.1 Residue Naming Normalization

Normalize non-canonical residue names in `receptor.pdb`:

```
HID / HIE / HIP → HIS  
TPO → THR  
SEP → SER  
...
```

Script:

```
scripts_2/run/standardize_pdb_resnames.py
```

Purpose:

* Prevent DiffDock inference failure due to residue naming inconsistencies.

---

## Stage 2 — Ligand CSV Generation

For each target and each split:

* `actives`
* `decoys`

Create a CSV file compatible with DiffDock inference.

Script:

```
scripts_2/run/create_csv_with_sdf_normal.py
```

Output:

```
{target}_actives.csv
{target}_decoys.csv
```

---

## Stage 3 — DiffDock Inference

Run DiffDock per target and split.

Script:

```
scripts_2/run/run_diffdock_target_simple.sh
```

Output structure:

```
results/
  └── {target}/
        ├── actives/
        ├── decoys/
        └── logs/
```

Rank-1 pose and confidence score are used for evaluation.

---

## Stage 4 — Postprocessing

Parse inference outputs and extract:

* rank1 pose
* confidence score
* missing / skipped / failed cases

Core logic:

```
src/inference/parse_logs.py
src/qc/postprocess.py
```

---

## Stage 5 — Score Table Construction

Generate per-target score table:

Script:

```
make_diffdock_score_table.py
```

Output:

```
diffdock_scores_rank1.csv
```

Columns:

* ligand_id
* label (1=active, 0=decoy)
* confidence_score
* inference_status

---

## Stage 6 — Metric Computation

Compute enrichment-based metrics:

Script:

```
eval_dude_metrics.py
```

Metrics:

* EF@1%
* EF@5%
* EF@10%
* nEF@k%
* ROC-AUC
* LogAUC

Output:

```
metrics_summary.csv
```

---

## Stage 7 — Pose Quality Evaluation (COM Distance)

Evaluate structural correctness:

Script:

```
src/geometry/comdist.py
```

Criterion:

```
COM distance ≤ 2 Å  → Pose hit
```

Outputs:

* COM distance per ligand
* Hit@K
* SR@p%

---

## Stage 8 — Target-Level Aggregation

### 8.1 Master Table

Script:

```
build_master_table.py
```

Aggregates:

* enrichment metrics
* pose metrics
* coverage
* skip/fail statistics

Output:

```
master_table.csv
```

---

### 8.2 Calibration Table

Script:

```
build_calibration.py
```

Analyzes:

* confidence vs correctness
* Expected Calibration Error (ECE)
* reliability diagram components

Output:

```
calibration_table.csv
```

---

### 8.3 Final Metric Summary

Script:

```
build_metrics_summary2.py
```

Produces:

* global summary across 26 kinase targets
* performance distribution
* difficulty categorization

---

# Repository Structure

```
src/
  aggregate/
  geometry/
  inference/
  io/
  metrics/
  qc/

scripts_2/
  run/
  eval/
  aggregate/
  qc/

configs/
```

---

# Metric Definitions

## Enrichment Factor (EF@p%)

Let:

* ( N ) = total ligands
* ( A ) = total actives
* ( n_p = \lfloor p \cdot N \rfloor )
* ( a_p ) = actives in top ( n_p )

[
EF@p% = \frac{a_p / n_p}{A / N}
]

---

## Normalized EF (nEF)

[
nEF = \frac{EF - 1}{EF_{max} - 1}
]

---

## LogAUC

Area under semi-log enrichment curve.

---

## COM Distance

For predicted pose center of mass ( c_{pred} ) and crystal pose ( c_{true} ):

[
d_{COM} = | c_{pred} - c_{true} |_2
]

Pose hit condition:

[
d_{COM} \le 2\ \text{Å}
]

---

# Failure Taxonomy

Inference failures categorized as:

* **Failed** → e.g., "No edges and no nodes"
* **Skipped** → dataset mismatch / confidence dataset missing
* **Missing** → no output file generated

Retry rate:

[
\text{retry rate} = \frac{\text{failed} + \text{skipped}}{\text{total ligands}}
]

---

# Reproducibility

* Fixed rank1 pose
* Confidence-based ranking
* Deterministic evaluation scripts
* Explicit log parsing
* Target-wise isolation

---

# Data

This repository does **not** include DUD-E raw datasets.

Users must download DUD-E separately and configure:

```
DUDE_ROOT=/path/to/dude_raw
```

---

# Limitations

* Receptor treated as rigid (DiffDock default)
* No side-chain flexibility modeling
* COM distance used instead of full RMSD for pose evaluation
* Benchmark limited to kinase subset

---

# Citation

If this pipeline is used, please cite:

```
@misc{diffdock_dude_kinase_benchmark,
  author = {Hosung Kim},
  title  = {DiffDock DUD-E Kinase Benchmark Pipeline},
  year   = {2026},
  url    = {https://github.com/hosungkim99/diffdock_dude_kinase_benchmark}
}
```

---

# Future Extensions

* Flexible receptor evaluation
* Cross-dataset validation (LIT-PCBA)
* Multi-model comparison (AutoDock-GPU, DiffDock-Pocket)
* Structural difficulty stratification

---

---

# 다음 단계 제안

이제 공개용으로 한 단계 더 가려면:

1. environment.yml 생성
2. example config 파일 정리
3. figures/ 폴더에 실제 결과 그림 추가
4. repo public 전환

---

원하면 다음으로:

* 🔥 environment.yml 자동 생성
* 🔥 공개용 논문화 Abstract 작성
* 🔥 포트폴리오용 한 페이지 요약 PDF 구조 설계

중 어떤 걸 진행할지 선택해라.
