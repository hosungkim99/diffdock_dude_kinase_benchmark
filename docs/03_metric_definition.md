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
- R_i = rank of the i-th positive sample

Then:

\[
\text{AUC} =
\frac{\sum R_i - \frac{n_{pos}(n_{pos}+1)}{2}}
{n_{pos} \cdot n_{neg}}
\]

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

Maximum possible enrichment at fraction p:

\[
EF_{max}(p) = \min\left(\frac{1}{p}, \frac{1}{\text{base\_rate}}\right)
\]

Normalized EF:

\[
nEF = \frac{EF - 1}{EF_{max} - 1}
\]

Clipped to [0, 1].

Interpretation:

- 0 → random
- 1 → theoretical maximum enrichment

---

## 5. LogAUC (Rank-based)

Defined over early portion of ranking.

Let:

- x_i = i / N
- TPR(x) = cumulative true positive rate

For cutoff \( \alpha \) (default 0.1):

\[
\text{LogAUC} =
\frac{\int_{0}^{\alpha} TPR(x)\, d\log_{10}(x)}
{\log_{10}(\alpha) - \log_{10}(1/N)}
\]

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

BEDROC emphasizes early retrieval using exponential weighting.

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

\[
BEDROC =
\frac{RIE - RIE_{min}}
{RIE_{max} - RIE_{min}}
\]

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
