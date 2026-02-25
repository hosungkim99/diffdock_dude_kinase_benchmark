# 03. Metrics Definition

This document formally defines all evaluation metrics used in this benchmark.
All definitions correspond exactly to the implementation in
`src/metrics/dude_metrics.py`.

Let:

- N = total number of ligands
- n = number of actives
- y_i ∈ {0,1} be the binary label
- s_i be the confidence score
- y_sorted : After sorting by descending score

---

## 1. Base Rate

The base rate (active ratio) is defined as:
base_rate = n / N

This represents the fraction of actives in the dataset.

---

## 2. ROC-AUC

ROC-AUC is computed using rank-based statistics.

Let:

- n_pos = number of positives
- n_neg = number of negatives
- R_i = rank of each active ligand (1 = highest score)

Then:

AUC =
( sum(R_i) - n_pos*(n_pos+1)/2 )
/ ( n_pos * n_neg )

Interpretation:
  - 1.0 → perfect ranking
  - 0.5 → random ranking
  - <0.5 → inverted ranking

Ties are handled by assigning average ranks.

This implementation corresponds to the Mann–Whitney U statistic.

---

## 3. Enrichment Factor (EF@k%)

For a top fraction \( k \) (e.g., 1%, 5%, 10%):

Let:

- k%          = top fraction of ranked ligands
- N_k         = floor(k% × N)
- TP_k        = number of actives within top N_k

Then:

EF@k% = (TP_k / N_k) / (n / N)

Interpretation:

- EF@k% > 1 → enrichment over random
- EF@k% = 1 → random performance
- EF@k% < 1 → worse than random

---

## 4. Normalized Enrichment Factor (nEF)

Maximum possible enrichment at k% is:

EF_max = min(n, N_k) / N_k ÷ (n / N)

Normalized EF:

nEF@k% = EF@k% / EF_max

0 ≤ nEF ≤ 1.

Interpretation:

- 0 → random
- 1 → theoretical maximum enrichment

---

## 5. LogAUC (Rank-based)

Defined over early portion of ranking.
LogAUC emphasizes early enrichment by integrating TPR over a logarithmic FPR scale.

Let:
  - FPR = false positive rate
  - TPR = true positive rate

For cutoff \( \alpha \) (default 0.1):

LogAUC = ∫ TPR d(log10(FPR))

Computed via trapezoidal rule over log-scaled x-axis.

This emphasizes early ranking performance.

---

## 6. Adjusted LogAUC (DUD-E style)

DUD-E-style logAUC is computed in FPR space.

Let:

- FPR_min = 0.001
- Random baseline logAUC = 14.462 (%)

Raw logAUC:

\[
\text{raw} =
\frac{\int_{FPR_{min}}^{1} TPR(FPR)\, d\log_{10}(FPR)}
{\log_{10}(1/FPR_{min})}
\]

Converted to percentage:

\[
\text{raw\_pct} = 100 \times \text{raw}
\]

Adjusted LogAUC:

\[
\text{Adjusted LogAUC} =
\text{raw\_pct} - \text{random\_baseline}
\]

This centers random performance at 0.

---

## 7. BEDROC

BEDROC (Boltzmann-Enhanced Discrimination of ROC) emphasizes early ranking.

Let:

- ranks of actives = r_i
- α = early weighting parameter (default 20)

RIE:

\[
RIE = \frac{1}{n}
\sum_i e^{-\alpha r_i / N}
\times
\frac{1 - e^{-\alpha}}
{1 - e^{-\alpha / N}}
\]

BEDROC:

BEDROC = normalization ×
         sum_i exp(-alpha * r_i / N) for actives

Clipped to [0,1].

Interpretation:

- 0 → random
- 1 → perfect early concentration

---

## 8. ROC-based EF@FPR

For FPR threshold τ (default 0.01):

\[
\text{ROC-EF@τ} =
100 \times TPR(FPR = τ)
\]

Used as DUD-style early enrichment metric.

---

## 9. Missing Policy Impact

If `missing_policy = bottom`:

- Missing ligands are assigned score = -∞
- They appear at the end of ranking
- EF and ROC metrics are computed conservatively

If `missing_policy = drop`:

- Missing ligands are excluded from evaluation

---

## 10. Ranking Procedure (Implementation Order)

All metrics follow this pipeline:

1. Load score table
2. Assign missing scores (bottom policy)
3. Sort by descending confidence score
4. Compute cumulative statistics
5. Compute metrics


## Summary of Metrics Used

Standard metrics:
- ROC-AUC
- EF@1%, EF@5%, EF@10%
- nEF@1%, nEF@5%, nEF@10%

Early-emphasis metrics:
- LogAUC
- Adjusted LogAUC
- BEDROC
- ROC-EF@1%

All implementations correspond to `src/metrics/dude_metrics.py`.
