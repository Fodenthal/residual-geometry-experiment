# R1.8 — Slow-Subspace Semantic and Variance Characterization

**Status:** Proposed / freeze before execution  
**Purpose:** Determine what kind of information the now-identifiable signed-TICA slow subspace carries, while avoiding premature axis-level or SAE interpretation.

## 0. Motivation

R1.7 established that the canonical signed-TICA slow region is a reproducible geometric object:

- Full refit vs canonical \(Q_{31}\): \(S_{31}=0.999125\)
- Independent split fits: \(S_{31}(Q_A,Q_B)=0.864059\)
- Chance baseline: \(0.01345\)
- Test random-in-span median signed timescale:
  - \(Q_A\): 24.5
  - \(Q_B\): 25.0
- Cross-fit median retention:
  - \(Q_A\): \(24\rightarrow24\)
  - \(Q_B\): \(24\rightarrow24\)

Thus the main unresolved question is no longer whether the slow region exists. It is:

> **What information does this reproducible persistent region carry?**

R1.8 answers three narrow questions, in order of scientific priority:

1. Is slow-space variation primarily **between documents** or **within documents**?
2. Does the previously observed coarse-topic signal reproduce in independent signed-TICA fits?
3. Are individual TICA axes themselves reproducible enough to justify axis-level interpretation?

SAE analysis is explicitly gated on these results and is not part of the primary experiment.

## 1. Scientific questions

### Q1. Between-document vs within-document variation

For slow coordinates

\[
z_{d,t}=Q^\top h_{d,t},
\]

how much variance is due to persistent differences between documents versus changes within documents?

This distinguishes:

- **document-level persistent state**: topic, source/style, genre, document identity, long-lived subject matter
- **within-document dynamic state**: changing subject, current entity/event, discourse state, or other semantic variables that update over time

### Q2. Semantic reproducibility

The previous audit found that canonical \(Q_{31}\) carried more coarse-topic information than PCA/random controls.

Does the same semantic signal appear in independently recovered slow spans \(Q_A\) and \(Q_B\)?

The key claim is not:

> canonical Q31 happened to decode topic.

It is:

> independent estimates of the persistent subspace recover the same semantic information.

### Q3. Axis identifiability

Does the reproducible 31D slow span also contain reproducible individual TICA directions?

If yes, leading-axis interpretation may be meaningful.

If no, semantic work must remain basis-invariant at the subspace/readout level.

## 2. Inputs and frozen artifacts

Reuse existing artifacts only.

Required:

- Canonical signed-TICA \(Q_{31}\)
- Independent R1.7 fits \(Q_A,Q_B\)
- R1.7 generalized eigenvalues / selected probe metadata
- Existing semantic-audit residual tensor:
  - 1,200 C4 documents
  - 12 sampled positions per document
  - layer-12 Gemma-2-2B residuals
  - shape approximately \(1200\times12\times2304\)
- Existing semantic labels
- Existing semantic train/validation/sealed-test split: 800 / 200 / 200 documents
- Existing nuisance features
- Existing PCA-31 and random-31 controls
- Existing semantic classifier / CE-gain implementation

Do not generate new labels.

Do not capture new model activations unless the existing semantic tensor cannot be located or fails integrity checks.

## 3. Phase A — Basis-invariant variance decomposition

### 3.1 Primary quantity

For an orthonormal basis \(Q\in\mathbb R^{2304\times k}\), define:

\[
z_{d,t}=Q^\top h_{d,t}.
\]

For each document:

\[
\bar z_d = \frac1{T_d}\sum_t z_{d,t}.
\]

Define total covariance:

\[
\Sigma_{\text{total}}=\operatorname{Cov}_{d,t}(z_{d,t}),
\]

between-document covariance:

\[
\Sigma_{\text{between}}=\operatorname{Cov}_d(\bar z_d),
\]

and within-document covariance:

\[
\Sigma_{\text{within}}=
E_d[\operatorname{Cov}_t(z_{d,t}\mid d)].
\]

Using a consistent finite-sample convention,

\[
\Sigma_{\text{total}}\approx \Sigma_{\text{between}}+\Sigma_{\text{within}}.
\]

Primary scalar:

\[
R_{\text{between}}(Q)
=
\frac{\operatorname{tr}(\Sigma_{\text{between}})}
{\operatorname{tr}(\Sigma_{\text{total}})}.
\]

Also report

\[
R_{\text{within}}=1-R_{\text{between}}.
\]

This quantity is invariant to rotations within \(Q\).

### 3.2 Subspaces

Compute \(R_{\text{between}}\) for:

- canonical slow \(Q_{31}\)
- \(Q_A\)
- \(Q_B\)
- PCA-31
- existing random-31 controls

Primary slow summary:

\[
R_{\text{between}}^{\text{slow}}
=
\operatorname{median}
\left(
R_{\text{between}}(Q_{\text{canonical}}),
R_{\text{between}}(Q_A),
R_{\text{between}}(Q_B)
\right).
\]

Use the existing random-control distribution to contextualize the slow value.

### 3.3 Bootstrap

Use document bootstrap only.

- 200 bootstrap replicates
- sample documents with replacement
- preserve all sampled positions within each selected document

Report 95% percentile intervals for:

- canonical slow
- \(Q_A\)
- \(Q_B\)
- PCA-31
- slow minus random-median difference

No token-level bootstrap.

### 3.4 Interpretation

Do not preregister a hard biological threshold.

Interpret qualitatively:

- high between-doc fraction: persistent information is dominated by document-level state
- substantial within-doc fraction: dynamic semantic-state interpretation remains plausible

The decision-relevant comparison is against PCA/random controls, not whether the ratio exceeds an arbitrary number.

## 4. Phase B — Semantic reproducibility gate

Reuse the exact previous coarse-topic task and classifier.

No ontology changes. No relabeling. No hyperparameter search.

### 4.1 Frozen semantic endpoint

For each subspace \(Q\), fit:

- nuisance-only classifier
- nuisance + \(Q^\top h\) classifier

Define:

\[
\Delta CE(Q)
=
CE(\text{nuisance})
-
CE(\text{nuisance}+Q^\top h).
\]

Positive values indicate additional linearly accessible topic information beyond nuisance features.

Compute on sealed test for:

- canonical \(Q_{31}\)
- \(Q_A\)
- \(Q_B\)
- PCA-31
- existing random-31 controls

Use the exact prior implementation.

### 4.2 Primary semantic reproducibility criterion

Let

\[
\Delta_A=\Delta CE(Q_A),\qquad
\Delta_B=\Delta CE(Q_B).
\]

Let \(q_{.95}^{\text{rand}}\) be the sealed-test Q95 across the existing random-31 controls.

Define `SEMANTICS_REPRODUCED` if both

\[
\Delta_A>q_{.95}^{\text{rand}}
\]

and

\[
\Delta_B>q_{.95}^{\text{rand}}.
\]

Otherwise define `SEMANTICS_NOT_REPRODUCED`.

PCA-31 is reported but does not determine this binary gate.

### 4.3 Ambient semantic readout reproducibility

Map semantic classifier coefficients back to residual space before geometric comparison.

If \(B_A\) is the semantic coefficient matrix in \(Q_A\) coordinates, define:

\[
W_A=Q_A B_A.
\]

Similarly,

\[
W_B=Q_B B_B.
\]

Orthonormalize the non-null discriminant column spaces and compute:

\[
S_{\text{sem}}
=
\frac1r
\left\|
\hat W_A^\top \hat W_B
\right\|_F^2,
\]

where \(r\) is the effective discriminant rank after removing the softmax-common direction.

This comparison is basis-invariant with respect to rotations inside \(Q_A,Q_B\).

### 4.4 Label-shuffle null

Construct a semantic-readout reproducibility null by independently shuffling training labels.

Use:

- 100 shuffle replicates
- same train/val/test documents
- same nuisance features
- same classifier code
- independently fit semantic readouts inside \(Q_A\) and \(Q_B\)

For each replicate compute \(S_{\text{sem}}^{\text{shuffle}}\).

Define `SEMANTIC_GEOMETRY_REPRODUCED` if

\[
S_{\text{sem}}
>
Q95\left(S_{\text{sem}}^{\text{shuffle}}\right).
\]

Use plus-one empirical p-value:

\[
p=
\frac{1+\#\{S^{\text{shuffle}}\ge S_{\text{real}}\}}
{1+N_{\text{shuffle}}}.
\]

No additional semantic nulls are authorized.

## 5. Phase C — Individual TICA-axis stability gate

This phase determines whether axis-level interpretation is licensable.

Use only \(Q_A,Q_B\) and their associated ordered generalized-eigenproblem directions.

### 5.1 Matching

For the first 10 selected TICA directions from A and B:

1. compute absolute cosine matrix
   \[
   C_{ij}=|v_i^{A\top}v_j^B|;
   \]
2. solve one-to-one maximum-weight matching;
3. report matched absolute cosine for each axis;
4. report rank displacement between matched axes.

Do not force same-rank matching.

### 5.2 Eigenvalue separation

For each matched leading axis, report the local generalized-eigenvalue gap:

\[
g_i
=
\min(
|\lambda_i-\lambda_{i-1}|,
|\lambda_i-\lambda_{i+1}|
)
\]

with endpoint handling.

Purpose: distinguish unstable axes because the operator is poorly separated from unstable axes despite clear eigengaps.

### 5.3 Axis-interpretability gate

Define a leading axis as stable if its matched A/B absolute cosine is at least 0.80.

Define `AXES_INTERPRETABLE` if at least 3 of the top 5 matched axes satisfy

\[
|\cos|\ge0.80.
\]

Otherwise define `AXES_NOT_INTERPRETABLE`.

If `AXES_NOT_INTERPRETABLE`, do not inspect activation-max examples, assign semantic labels to axes, or perform SAE-to-axis matching.

## 6. Frozen overall outcomes

### A. `DOCUMENT_LEVEL_SEMANTIC_STATE`

Conditions:

- `SEMANTICS_REPRODUCED`
- `SEMANTIC_GEOMETRY_REPRODUCED`
- slow space shows clearly elevated between-document fraction relative to random controls

Interpretation:

> The reproducible persistent region preferentially carries reproducible document-level semantic information.

Natural next step:

- distinguish content/topic from source/style/document identity with one compact conditional-label experiment.

### B. `DYNAMIC_SEMANTIC_STATE_PLAUSIBLE`

Conditions:

- `SEMANTICS_REPRODUCED`
- `SEMANTIC_GEOMETRY_REPRODUCED`
- substantial within-document variance remains relative to controls

Interpretation:

> The persistent region carries reproducible semantics and also varies meaningfully within documents, motivating one dynamic-state labeling experiment.

Natural next step:

- design one within-document semantic variable whose value actually changes over token position.

### C. `SEMANTICS_NOT_STABLE`

Condition:

- semantic function or semantic readout geometry fails to reproduce

Interpretation:

> The slow subspace is reproducible, but the current coarse-topic semantic result is not itself stable enough to support a semantic-state claim.

Natural next step:

- stop semantic extrapolation from the current topic audit;
- inspect whether failure comes from label noise / task imbalance before designing new semantics.

### D. Axis status

If semantics reproduce, report separately:

- `AXES_INTERPRETABLE`
- or `AXES_NOT_INTERPRETABLE`

Axis status does not alter the subspace-level semantic conclusion.

## 7. SAE policy

Do **not** run SAE overlap analysis in R1.8.

SAE analysis becomes authorized only if:

1. semantics reproduce across \(Q_A,Q_B\), and
2. there is a specific semantic readout/subspace worth decomposing.

If authorized later, the target should be the semantic readout \(W\), not blind overlap with all of \(Q_{31}\).

Preferred later question:

> Is the reproducible semantic variable concentrated in a small set of SAE features or distributed across many?

## 8. Minimal controls

Only:

### C1. PCA-31

Purpose: determine whether variance and semantic effects are specific to temporally selected geometry versus generic high-variance structure.

### C2. Existing random-31 controls

Purpose: calibrate expected equal-rank behavior.

### C3. Label-shuffle semantic-geometry null

Purpose: test whether agreement between independently fitted semantic readouts exceeds chance.

No new random basis bank is needed unless the existing bank is unavailable.

## 9. Explicit non-goals

Do not perform:

- new corpus capture
- new model/layer
- new semantic ontology
- source/style labeling
- dynamic-state labeling
- SAE overlap
- causal patching
- refit-after-deletion
- axis activation-max inspection before stability gate
- alternative TICA estimator
- sample-size scaling
- extra rank sweeps beyond the existing 31D object

These are downstream branches conditional on R1.8.

## 10. Required outputs

```text
r1_8_config.json
provenance.json

variance/
    variance_decomposition.csv
    variance_bootstrap.parquet

semantics/
    semantic_gain_summary.csv
    semantic_readout_overlap.json
    semantic_shuffle_null.parquet

axes/
    axis_matching.csv
    eigengap_summary.csv
    axis_decision.json

decision.json
r1_8_report.md
```

`decision.json` must include:

- between-document variance fractions for canonical / A / B / PCA / random
- semantic gains for canonical / A / B / PCA / random
- random Q95 threshold
- semantic readout overlap
- shuffle Q95 and plus-one p-value
- top-5 axis matched cosines
- axis gate
- overall scientific outcome
- authorized next branch

## 11. Stopping rule

Stop after the three registered analyses above.

Do not automatically launch downstream semantic labeling or SAE work.

Next branch:

- reproducible semantics + mostly between-document variance
  -> one compact content-vs-source/style disentangling experiment

- reproducible semantics + meaningful within-document variance
  -> one dynamic semantic-state experiment

- semantics fail to reproduce
  -> stop semantic interpretation and diagnose the task/labels

- stable axes
  -> axis-level examples may be added to the relevant semantic branch

- unstable axes
  -> continue only with basis-invariant subspace/readout analyses

## 12. One-sentence target

R1.8 should allow a statement of the form:

> “The independently reproducible slow subspace carries reproducible semantic information, and its variation is primarily [between / within] documents.”

with the separate qualifier:

> “Individual TICA axes are [stable enough / not stable enough] to interpret.”

That is the minimum sufficient next step before designing a richer semantic-variable experiment.
