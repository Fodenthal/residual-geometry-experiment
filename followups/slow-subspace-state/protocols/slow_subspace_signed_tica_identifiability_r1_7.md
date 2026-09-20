# R1.7 — Faithful Signed-TICA Split-Fit Identifiability

**Status:** Proposed / freeze before execution  
**Purpose:** Test whether independently fitting the *original canonical signed TICA estimator* on different document samples recovers the same slow geometry.

---

## 0. Motivation

The canonical slow-subspace result has now passed a provenance audit.

The exact frozen residual-space basis

- SHA-256: `966736…0cce6c`
- rank: 31
- composition: 30 signed time-lagged generalized-eigenproblem directions + 1 PCA direction

reproduces the original random-in-span result exactly:

- validation median signed timescale: **24**
- test median signed timescale: **25**
- 512 random directions in canonical \(Q_{31}\)
- sampling seed 31

R1.5/R1.6 do **not** bear on the identifiability of this object because they studied a different estimator and observable: unsigned/squared activity with independently optimized nonlinear directions.

The unresolved question is therefore narrow:

> If the original signed TICA pipeline is independently fit on different samples from the same C4 training distribution, does it recover the same slow subspace?

This experiment should answer that question and nothing else.

---

# 1. Scientific question

Let \(D_A\) and \(D_B\) be disjoint halves of the original 4,000-document training corpus.

Run the exact canonical signed-TICA fitting and rank-selection pipeline independently:

\[
Q_A = \mathcal A_{\mathrm{TICA}}(D_A),
\qquad
Q_B = \mathcal A_{\mathrm{TICA}}(D_B).
\]

Ask:

1. **Geometric identifiability:** How similar are \(Q_A\) and \(Q_B\)?
2. **Functional persistence:** Are generic directions inside each independently fitted span still slow on common held-out data?
3. **Cross-sample transfer:** Does a span fit on one half retain signed persistence on the other half?
4. **Nested-core question:** If rank 31 is unstable, are smaller nested ranks \(k\in\{8,13,21\}\) more reproducible?

No semantics, SAE analysis, causal intervention, alternative estimator, optimizer restart, or corpus comparison is authorized in this experiment.

---

# 2. Frozen object and provenance

## 2.1 Canonical estimator

The authoritative estimator is the exact code/configuration that generated the original signed TICA probe bank and canonical residual-space \(Q_{31}\).

Required properties:

- Model: `google/gemma-2-2b`
- Hook: `blocks.12.hook_resid_post`
- Residual dimension: 2304
- Signed residual projection observable
- Lag set: \(\{8,16,32,64,128\}\)
- Original lagged-covariance construction
- Original covariance regularization
- Original generalized-eigenproblem solver
- Original PCA candidate construction
- Original candidate deduplication
- Original validation-timescale ranking
- Original top-\(k\) selection rule
- Original signed timescale evaluator

Do **not** substitute:

- squared or absolute projections;
- the R1.5/R1.6 nonlinear optimizer;
- a PCA-512 approximation;
- a rewritten estimator based only on this prose;
- a new whitening convention;
- a new rank-selection rule.

The canonical code path is the specification. This document describes the experiment around it.

## 2.2 Canonical reference

Frozen reference basis:

`canonical_Q31_sha256 = 966736…0cce6c`

Frozen canonical random-in-span reference:

| Split | N dirs | Median \(\tau\) |
|---|---:|---:|
| validation | 512 | 24 |
| test | 512 | 25 |

Canonical test distribution:

- min 5
- Q05 10
- Q25 18
- median 25
- Q75 40.25
- Q90 60
- Q95 82.45
- max 256

These values are provenance references, not targets to optimize toward.

---

# 3. Data

Use the exact original corpus and document ordering.

- Original training documents: 4,000
- Validation documents: 500
- Test documents: 500
- Sequence length: 1024

## 3.1 Split-fit partition

Reuse the exact deterministic 2,000 / 2,000 train partition already used in R1.5 if its document indices are available and verified.

\[
D_{\mathrm{train}} = D_A \sqcup D_B,
\qquad |D_A|=|D_B|=2000.
\]

Reason: this makes the comparison maximally clean. The document split is held fixed while the estimator is corrected back to the canonical signed TICA construction.

If the historical A/B index artifact cannot be recovered unambiguously, create one deterministic split with a frozen seed before fitting and save the indices.

Do not search over splits.

## 3.2 Ranking and evaluation data

Use the original pipeline's existing validation split for candidate ranking / selection exactly as originally done.

Use the original 500-document test split as the primary common held-out functional evaluation set.

The test split must not affect fitting, rank selection, hyperparameters, or any branch decision before final evaluation.

---

# 4. Implementation plan

## Phase P0 — Provenance lock

Before expensive computation:

1. Locate the exact canonical TICA fitting code/config.
2. Record:
   - git commit;
   - script paths;
   - model revision;
   - lag set;
   - regularization parameters;
   - covariance convention;
   - eigensolver;
   - candidate counts/families;
   - deduplication;
   - validation ranking;
   - timescale implementation.
3. Verify the canonical Q31 artifact hash.
4. Verify the audited signed-timescale evaluator is the one used in R1.6A.

**Fail closed** if the canonical estimator cannot be identified unambiguously.

Do not reconstruct it from memory.

---

## Phase P1 — Exact sufficient-statistic capture

Prefer reuse of existing exact residual-space sufficient statistics if they are already available and provenance-valid.

If they do not exist, authorize **one** model pass over the 4,000 training documents, with documents assigned to frozen halves A/B.

Accumulate only the exact residual-space sufficient statistics required by the canonical TICA/PCA fitting code.

Requirements:

- operate on the 2304-dimensional residual stream;
- no PCA-512 approximation;
- no raw full residual tensor needs to be retained if exact sufficient statistics can be accumulated online;
- accumulate A and B separately;
- derive full-train statistics from A+B where mathematically exact;
- preserve the canonical centering / normalization / lag weighting conventions.

The objective is one exact capture reusable for:

- full-data provenance refit;
- A fit;
- B fit.

No second model capture is authorized unless P1 fails a technical integrity check.

---

# 5. Mandatory full-data reproduction gate

Before interpreting A/B split fits, run the canonical estimator on the combined A+B sufficient statistics.

Call the result \(Q_{\mathrm{full}}^{\mathrm{refit}}\).

Compare it against the frozen canonical \(Q_{31}\).

Primary equivalence metric:

\[
S_{31}(Q_{\mathrm{full}}^{\mathrm{refit}}, Q_{\mathrm{canonical}})
=
\frac{1}{31}
\left\|
(Q_{\mathrm{full}}^{\mathrm{refit}})^\top
Q_{\mathrm{canonical}}
\right\|_F^2.
\]

Also verify:

- selected family composition is consistent with the canonical fit;
- selected directions have the expected signed held-out timescale profile;
- a 512-direction random-in-span assay on validation reproduces the fat-span phenomenon.

### Gate

Proceed to scientific adjudication only if:

\[
S_{31}(Q_{\mathrm{full}}^{\mathrm{refit}},Q_{\mathrm{canonical}})\ge 0.99
\]

**or** an explicitly documented numerical-equivalence criterion demonstrates that any discrepancy is only sign / basis rotation inside a numerically degenerate eigenspace while preserving the same rank-31 span.

Additionally require validation random-in-span median \(\tau\) to lie in:

\[
[20,28].
\]

If the full-data refit fails this gate:

\[
\boxed{\texttt{PROVENANCE\_REFIT\_FAILED}}
\]

Stop. Do not interpret split-fit instability.

This gate is specifically intended to prevent another R1.5/R1.6 estimator mismatch.

---

# 6. Independent split fits

Run the exact canonical estimator independently on:

\[
D_A
\quad\text{and}\quad
D_B.
\]

Call the selected rank-31 spans:

\[
Q_A,\qquad Q_B.
\]

There is no optimizer seed sweep. Conditional on each dataset and the canonical generalized-eigenproblem implementation, the fit is treated as deterministic.

Save:

- all candidate probe metadata;
- generalized eigenvalues;
- validation timescales;
- selected probe IDs/families;
- selected rank order;
- orthonormalized \(Q_A,Q_B\);
- hashes of all fit artifacts.

---

# 7. Primary endpoint: geometric recovery

For

\[
k\in\{8,13,21,31\},
\]

form the nested selected spans \(Q_A^{(k)}\) and \(Q_B^{(k)}\) using the canonical rank ordering.

Compute

\[
S_k(A,B)
=
\frac{1}{k}
\left\|
(Q_A^{(k)})^\top Q_B^{(k)}
\right\|_F^2.
\]

Interpretation:

- \(S_k=1\): identical \(k\)-dimensional subspaces;
- \(S_k=0\): orthogonal;
- ambient random baseline is approximately \(k/2304\).

Also report the canonical correlations / principal angles, but do not create additional decision rules from them.

Primary rank is:

\[
\boxed{k=31}.
\]

Nested ranks are included because the original experiment already showed substantially stronger generic persistence at smaller \(k\), and they can tell us whether a stable core exists even if the full selected Q31 is less reproducible.

---

# 8. Functional endpoint: random-in-span signed persistence

For each fit and each

\[
k\in\{8,13,21,31\},
\]

sample 512 random unit directions uniformly inside the orthonormal span.

For span \(Q\):

\[
g_j\sim\mathcal N(0,I_k),
\qquad
u_j=\frac{Qg_j}{\|Qg_j\|},
\qquad j=1,\dots,512.
\]

Use the canonical row-wise random-in-span procedure and frozen sampling seed 31.

Evaluate each \(u_j\) with the **original signed timescale metric**.

Primary evaluation set:

- untouched 500-document test split.

Also report validation for continuity with the original artifact.

For each span/split report:

- min;
- Q05;
- Q25;
- median;
- Q75;
- Q90;
- Q95;
- max;
- mean;
- right-censoring fraction.

The primary functional statistics are:

\[
m_A = \operatorname{median}\tau(Q_A^{(31)};\mathrm{test}),
\]

\[
m_B = \operatorname{median}\tau(Q_B^{(31)};\mathrm{test}).
\]

Canonical benchmark:

\[
m_{\mathrm{canonical}}=25.
\]

---

# 9. Selected-axis sanity table

For the 31 selected basis directions from each fit, separately report their signed timescale distributions on validation and test:

- Q25;
- median;
- Q75;
- Q90;
- max.

This is diagnostic only.

Purpose: distinguish

> “the estimator failed to find slow directions”

from

> “it found individually slow directions whose entire span is not generically slow.”

Do not interpret individual axes semantically.

---

# 10. Cross-fit signed-timescale transfer

This is secondary to the common-test endpoint but directly tests sample dependence.

For each independently fitted span:

- evaluate its 512 frozen random-in-span directions on \(D_A\);
- evaluate the same directions on \(D_B\).

Thus obtain:

\[
m(Q_A;A),\quad m(Q_A;B),
\]

\[
m(Q_B;A),\quad m(Q_B;B).
\]

Define:

\[
R_{A\rightarrow B}
=
\frac{m(Q_A;B)}{m(Q_A;A)},
\]

\[
R_{B\rightarrow A}
=
\frac{m(Q_B;A)}{m(Q_B;B)}.
\]

Because the denominator uses the fit sample, these ratios may contain in-sample optimism. Therefore:

- report them;
- use them only as supporting evidence;
- do **not** let them override the common held-out test result.

The clean functional question remains whether both independently fitted spans are fat and slow on the same untouched test distribution.

---

# 11. Decision rules

## 11.1 Geometry

At \(k=31\):

### `GEOMETRY_HIGH`

\[
S_{31}\ge0.70.
\]

### `GEOMETRY_LOW`

\[
S_{31}<0.40.
\]

### `GEOMETRY_INTERMEDIATE`

otherwise.

These are intentionally coarse. The goal is to distinguish a clearly recoverable common object from a strongly sample-sensitive one, not to over-interpret small metric differences.

---

## 11.2 Functional persistence

Using the canonical test median \(25\):

### `FUNCTION_PRESERVED`

Both split fits satisfy:

\[
m_A\ge15
\quad\text{and}\quad
m_B\ge15.
\]

This retains at least 60% of the canonical median and remains far above the ordinary random/PCA scale.

### `FUNCTION_COLLAPSED`

Either split fit satisfies:

\[
m\le5.
\]

### `FUNCTION_INTERMEDIATE`

otherwise.

---

# 12. Frozen scientific outcomes

## A. `SIGNED_TICA_IDENTIFIABLE`

Criteria:

- `GEOMETRY_HIGH`
- `FUNCTION_PRESERVED`

Interpretation:

> Independent samples recover substantially the same rank-31 slow geometry, and generic directions in both estimates remain strongly persistent on common held-out data.

This would license treating the signed TICA slow region as an approximately identifiable population-level subspace for this model/layer/corpus distribution.

---

## B. `NONCANONICAL_FAT_SLOW_FAMILY`

Criteria:

- `GEOMETRY_LOW`
- `FUNCTION_PRESERVED`

Interpretation:

> Independent samples recover geometrically different spans, but both spans are generically slow on the same held-out distribution.

This would reject a unique canonical Q31 while preserving a strong scientific object: a broader family of distinct, functionally equivalent persistent subspaces.

If this occurs, the next estimand should be basis/subspace-consensus or operator-level structure rather than semantics of one fitted Q31.

---

## C. `SIGNED_TICA_SAMPLE_DEPENDENT`

Criteria:

- `GEOMETRY_LOW`
- `FUNCTION_COLLAPSED`

Interpretation:

> At 2,000 documents the fitted signed-TICA geometry is both geometrically unstable and not robustly fat on common held-out data.

This would make finite-sample/distribution dependence the leading explanation.

Do **not** immediately infer that the full-data canonical Q31 is spurious; it is already directly validated. The next question would be sample-size scaling of the faithful estimator.

---

## D. `SIGNED_TICA_IDENTIFIABILITY_MIXED`

Any other combination.

Interpretation must follow the observed geometry, nested-rank curve, selected-axis sanity table, and common-test random-in-span distributions without forcing a binary story.

---

# 13. Nested-core interpretation

Regardless of the rank-31 decision, report:

\[
S_8,\ S_{13},\ S_{21},\ S_{31}
\]

alongside the corresponding test random-in-span medians.

This is not a separate experiment and requires no new capture.

A pattern such as

\[
S_8,S_{13}\gg S_{31}
\]

with strong random-in-span persistence at small \(k\) would support:

> a reproducible slow core with a less-identifiable outer shell.

A flat low-stability curve would instead support more globally sample-sensitive geometry.

Do not add more ranks unless the registered four are insufficient to interpret a threshold crossing.

---

# 14. Minimal controls

Only the following controls are authorized.

### C1. Full-data canonical refit gate

Purpose: verifies we are actually running the original estimator.

### C2. Ambient random-subspace overlap baseline

Analytical expectation:

\[
E[S_k]\approx k/2304.
\]

Purpose: establishes whether observed overlap is above chance.

No Monte Carlo is needed unless implementation debugging requires it.

### C3. Original PCA/random signed-timescale references

Reuse existing frozen references where available.

Purpose: contextualize whether split-fit random-in-span medians remain meaningfully slow.

Do not rerun broad control batteries.

---

# 15. Explicit non-goals

Do **not** perform in R1.7:

- unsigned/squared persistence;
- nonlinear direction optimization;
- optimizer restart studies;
- semantic labeling;
- SAE projection/overlap;
- causal interventions;
- refit-after-deletion;
- code/Wikipedia corpus shifts;
- layer sweeps;
- model sweeps;
- new lag choices;
- new regularization;
- sample-size sweeps;
- alternative TICA formulations;
- operator-consensus construction.

Those are downstream questions conditional on the result.

---

# 16. Compute / storage strategy

Priority order:

1. Reuse exact canonical residual-space sufficient statistics if provenance-valid.
2. Otherwise do one exact train capture and accumulate A/B sufficient statistics online.
3. Reuse the existing validation/test data or projected traces where they are exactly compatible with the canonical signed evaluator.
4. If new held-out projection traces are required, capture only projections onto the fitted spans rather than full 2304D residual tensors.

Do not store raw full residuals unless the canonical fitting implementation genuinely requires them and no exact sufficient-statistic route is available.

The full-data, A, and B fits should all use the same captured sufficient statistics so that estimator fidelity, not representation recapture, is the only varying factor.

---

# 17. Required outputs

Create one experiment directory containing at minimum:

```text
r1_7_config.json
provenance.json
split_indices.json

full_refit/
    q31.npy
    selected_probes.json
    fit_metadata.json

split_A/
    q31.npy
    selected_probes.json
    fit_metadata.json

split_B/
    q31.npy
    selected_probes.json
    fit_metadata.json

geometry/
    nested_overlap.csv
    canonical_correlations.npz

timescales/
    random_in_span_validation.parquet
    random_in_span_test.parquet
    selected_axis_summary.csv
    crossfit_summary.csv

decision.json
r1_7_report.md
```

`decision.json` must contain:

- provenance gate result;
- \(S_8,S_{13},S_{21},S_{31}\);
- \(m_A,m_B\) on test;
- functional classification;
- geometry classification;
- final frozen scientific outcome;
- any fail-closed reason.

---

# 18. Stopping rule

Stop after R1.7 once one of the frozen outcomes is assigned.

Do not automatically launch a sample-size experiment.

The only prospective follow-up justified directly by each outcome is:

- `SIGNED_TICA_IDENTIFIABLE`  
  → semantics / basis-invariant state characterization becomes more defensible.

- `NONCANONICAL_FAT_SLOW_FAMILY`  
  → define a consensus/operator-level estimand before more semantics.

- `SIGNED_TICA_SAMPLE_DEPENDENT`  
  → only then consider a minimal faithful sample-size scaling experiment.

- `SIGNED_TICA_IDENTIFIABILITY_MIXED`  
  → inspect the already-produced nested and transfer diagnostics; specify one targeted follow-up only if they identify a concrete ambiguity.

No further computation should be queued automatically.

---

# 19. One-sentence interpretation target

The experiment should end with one of the following forms:

> “Independent signed-TICA fits on disjoint C4 samples recover substantially the same fat slow subspace.”

or

> “Independent signed-TICA fits recover different geometry but both remain generically slow on held-out data.”

or

> “The canonical full-data slow span is real, but faithful split fits at 2,000 documents are sample-sensitive and do not recover an equally strong fat slow region.”

The purpose of R1.7 is to decide among those statements with the smallest faithful experiment.
