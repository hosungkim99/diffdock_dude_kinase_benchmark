
---

# DiffDock DUD-E Kinase Benchmark Pipeline

A fully reproducible benchmarking framework for evaluating **DiffDock** on the **DUD-E kinase subset (26 targets)**.

This repository implements a structured, multi-stage evaluation pipeline including:

* Inference stability tracking
* Multi-layer structural QC
* COM-based pose validation
* DUD-E enrichment metrics (EF / nEF / ROC / LogAUC)
* Cross-target aggregation and calibration analysis

This project is designed not merely to run DiffDock, but to provide a **rigorous evaluation protocol** for structure-aware virtual screening.

---

# 1. Project Overview

Virtual screening benchmarks often report only enrichment metrics.
This framework extends evaluation by integrating:

1. **Inference Stability Analysis** (fail / skip / retry taxonomy)
2. **Structural Validity Checks** (pocket containment, clash detection)
3. **Pose Accuracy Proxy** (COM ≤ 2Å)
4. **Early Enrichment Metrics**
5. **Confidence Calibration Analysis**
6. **Cross-target aggregation**

Pipeline:

```text
receptor normalization
→ CSV generation (actives / decoys)
→ DiffDock inference
→ postprocess (retry + score table)
→ QC (multi-layer)
→ COM distance evaluation
→ DUD-E metrics
→ aggregate (master table + calibration)
```

---

# 2. Installation

## 2.1 Requirements

* Python ≥ 3.9
* RDKit
* NumPy
* Pandas
* SciPy
* DiffDock (installed separately)

Optional:

* SLURM (for cluster execution)

---

## 2.2 Clone Repository

```bash
git clone https://github.com/<your-id>/diffdock-dude-benchmark.git
cd diffdock-dude-benchmark
```

---

## 2.3 Environment Setup (Example)

```bash
conda create -n diffdock_bench python=3.9
conda activate diffdock_bench
pip install -r requirements.txt
```

---

# 3. Repository Structure

```text
.
├── README.md
├── docs/
│   ├── benchmark_protocol.md
│   ├── metrics_definition.md
│   ├── qc_design.md
│   └── calibration_analysis.md
│
├── scripts_2/
│   ├── run/
│   ├── postprocess/
│   ├── eval/
│   └── aggregate/
│
├── src/
│   ├── io/
│   ├── postprocess/
│   ├── eval/
│   ├── qc/
│   └── aggregate/
│
└── outputs/
    └── <target>/
```

### Directory Roles

* **scripts_2/** → executable pipeline scripts
* **src/** → core logic modules
* **docs/** → detailed technical specification
* **outputs/** → per-target results

---

# 4. Pipeline Diagram

```text
┌─────────────────────────────┐
│ Receptor Normalization      │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ CSV Generation              │
│ (actives / decoys)          │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ DiffDock Inference          │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Postprocess                 │
│ - retry analysis            │
│ - score table generation    │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Multi-layer QC              │
│ - pocket containment        │
│ - clash detection           │
│ - sanity checks             │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ COM Distance Evaluation     │
│ (Hit: COM ≤ 2Å)             │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ DUD-E Metrics               │
│ EF / nEF / ROC / LogAUC     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Aggregate                   │
│ master table / calibration  │
└─────────────────────────────┘
```

---

# 5. Execution Example (Single Target: ABL1)

### 5.1 Run Inference

```bash
bash scripts_2/run/run_diffdock_target_simple.sh abl1
```

---

### 5.2 Postprocess

```bash
python scripts_2/postprocess/postprocess_diffdock_results.py --target abl1
python scripts_2/postprocess/parse_inference_err.py --target abl1
python scripts_2/postprocess/make_diffdock_score_table.py --target abl1
```

---

### 5.3 QC

```bash
python scripts_2/eval/pocket_qc_rank1.py --target abl1
python scripts_2/eval/qc_rank1_bundle.py --target abl1
python scripts_2/qc/sanity_check.py --target abl1
```

---

### 5.4 COM Distance

```bash
python scripts_2/eval/compute_comdist2.py --target abl1
```

---

### 5.5 DUD-E Metrics

```bash
python scripts_2/eval/eval_dude_metrics.py --target abl1
```

---

### 5.6 Aggregate

```bash
python scripts_2/aggregate/build_master_table.py
python scripts_2/aggregate/build_metrics_summary2.py
python scripts_2/aggregate/build_calibration.py
```

---

# 6. Example Result Snapshot (ABL1)

### Inference Stability

* Actives: 295
* Decoys: 10,885
* Retry tracked explicitly (fail + skip)

### Enrichment

* ROC-AUC ≈ 0.78
* EF@1% ≈ 7–8
* Early enrichment significantly above random baseline

### Structural QC

* Pocket containment evaluated
* Clash detection applied
* Rank1 pose COM ≤ 2Å used as hit proxy

### Calibration

* Confidence binning analysis
* Reliability assessment across targets

---

# 7. Reproducibility Principles

* All evaluation parameters stored in config files
* No silent filtering of failures
* Retry explicitly logged
* QC separated from metric computation
* Calibration analyzed post-hoc

This framework enables transparent and reproducible benchmarking of generative docking models.

---

# 8. Documentation

Detailed protocol and metric definitions are provided in:

```text
docs/
```

* `benchmark_protocol.md`
* `metrics_definition.md`
* `qc_design.md`
* `calibration_analysis.md`

---

# 9. Intended Use

This repository is suitable for:

* Research benchmarking
* Model comparison studies
* Structure-aware evaluation development
* Reproducible AI-for-drug-discovery pipelines
* Technical portfolio demonstration

---

---


