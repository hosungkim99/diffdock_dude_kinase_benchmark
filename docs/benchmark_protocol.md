# 01. Benchmark Protocol

This document describes the benchmark protocol used for evaluating DiffDock on the DUD-E kinase subset.  
All definitions, evaluation criteria, and aggregation policies are explicitly specified to ensure reproducibility.

---

## 1. Benchmark Dataset

### 1.1 DUD-E Kinase Subset

- Dataset source: Directory of Useful Decoys, Enhanced (DUD-E)
- Target class: Kinase
- Number of targets: 26
- Target list:
  - [ABL1]
  - [AKT2]
  - [...]
- Dataset version / download source:
  -  https://dude.docking.org/
- Receptor preprocessing:
  - Residue naming normalization applied: ㄴ
  - Script used: `standardize_pdb_resnames.py`
  - Modified residue types (if applicable):
    - [e.g., HID/HIE/HIP → HIS]

---

### 1.2 Ligand Composition

For each target:

- Number of actives: [   ]
- Number of decoys: [   ]
- Total ligands per target: [   ]

Definitions:

- Actives: Experimentally validated binders provided in DUD-E.
- Decoys: Property-matched but topologically distinct compounds.

Split handling:

- Actives and decoys are processed separately during inference.
- CSV files generated per target and per split using:
  - `create_csv_with_sdf_normal.py` (if applicable)

---

## 2. Model Inference Settings

### 2.1 DiffDock Version

- DiffDock version: [   ]
- Commit hash (if applicable): [   ]
- Environment:
  - Python version: [   ]
  - CUDA version: [   ]
  - GPU type: [   ]

---

### 2.2 Pose Selection Strategy

- Pose selection: **rank1 only**
- Definition:
  - The highest-confidence predicted pose per ligand is used.
- Rationale:
  - Ensures consistency between docking ranking and virtual screening metrics.

---

### 2.3 Confidence Score Definition

- Confidence score extracted from:
  - [rank1 SDF metadata / output JSON / score table]
- Ranking criterion:
  - Ligands sorted by descending confidence score.
- Used for:
  - ROC-AUC
  - EF@k%
  - nEF
  - Calibration analysis

---

## 3. Inference Stability Definition

### 3.1 Status Categories

Each ligand is categorized as:

- **Success**
- **Skip**
- **Fail**

Definitions:

- Fail:
  - Condition: ["No edges and no nodes..." or equivalent]
- Skip:
  - Condition 1: ["test dataset does not contain..."]
  - Condition 2: ["confidence dataset does not contain..."]
- Success:
  - Successfully generated rank1 pose and confidence score

Parsing method:

- Script used: `parse_inference_err.py`
- Source: `.err` log files per target

---

### 3.2 Retry Definition

- Retry = Fail + Skip
- Retry rate per target:
  - `retry_rate = (fail + skip) / total_ligands`

---

### 3.3 Missing Policy

- missing_policy = **bottom**
- Definition:
  - Missing ligands (skip/fail) are assigned the lowest possible ranking.
- Rationale:
  - Conservative evaluation strategy
  - Avoids artificial performance inflation

---

## 4. Structural Validity Criteria

### 4.1 COM Distance Criterion

- Metric: Center-of-Mass (COM) distance
- Threshold: **COM ≤ 2 Å**
- Interpretation:
  - Pose considered structurally valid if COM ≤ 2 Å relative to reference
- Script used:
  - `compute_comdist2.py`

---

### 4.2 Pocket Quality Control (QC)

Performed using:

- `pocket_qc_rank1.py`
- `qc_rank1_bundle.py`
- `sanity_check.py`

QC flags include:

- Steric clash
- Outside pocket
- Geometric inconsistency (if applicable)

Output artifacts:

- `flagged_clash.csv`
- `flagged_outside.csv`
- `pocket_qc_summary.json`
- `qc_summary.json`

---

## 5. Evaluation Metrics

### 5.1 Classification Metrics

- ROC-AUC
- Precision-Recall (if applicable)
- LogAUC (if applicable)

Computed using:

- `eval_dude_metrics.py`

---

### 5.2 Early Enrichment Metrics

- EF@1%
- EF@5%
- EF@10%
- nEF@k%

Definition:

\[
EF@k\% = \frac{\text{Hit rate in top k\%}}{\text{Overall hit rate}}
\]

nEF:

\[
nEF = \frac{EF - 1}{EF_{max} - 1}
\]

---

### 5.3 Pose-based Metrics

- Hit definition:
  - COM ≤ 2 Å
- Hit@K:
  - [If used, define K]

---

## 6. Aggregation Strategy

### 6.1 Per-target Evaluation

- Metrics computed independently per target.
- Outputs:
  - `metrics.json`
  - `ranking.csv`
  - `roc.csv`

---

### 6.2 Cross-target Aggregation

Aggregated using:

- `build_master_table.py`
- `build_metrics_summary2.py`
- `build_calibration.py`

Aggregation method:

- Macro-average across targets
- [Mean / Median — specify]

---

### 6.3 Calibration Analysis

- Confidence binning method: [   ]
- Metric:
  - Expected Calibration Error (ECE) [if used]
- Output files:
  - `calibration_table.csv`
  - `calibration_summary.csv`

---

## 7. Overall Evaluation Workflow

1. Receptor normalization
2. CSV generation per target
3. DiffDock inference
4. Postprocessing
5. Score table construction
6. Inference log parsing
7. COM distance evaluation
8. Metric computation
9. QC filtering
10. Aggregation and calibration analysis

Full pipeline scripts:

- `scripts_2/run/`
- `scripts_2/postprocess/`
- `scripts_2/eval/`
- `src/aggregate/`

---

## Reproducibility Notes

- All scripts are included in this repository.
- All metrics are computed deterministically given the same inference outputs.
- Configuration files:
  - `config.json`
  - `master_table.config.json`
