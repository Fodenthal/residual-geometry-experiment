# Residual Sequence-Memory Geometry — Predictive Carrier Resolution
## Arm D-R2.4 Q-Resolution Specification

## 0. Status and precedence

```text
protocol_revision:
arm_d_r2_4_predictive_carrier_resolution

status:
NEXT EXPERIMENT
```

This specification governs the next Arm D stage after the R2.2/R2.3 carrier analyses and the halted R2.3b common-target comparison.

R2.4 does not alter any frozen earlier verdict. In particular:

- R1 remains evidence for a strong low-dimensional fixed-coordinate predictive phenomenon in paired remote-history differences;
- R2.2 remains evidence that the inherited rank-16 carrier is not supported as a finite-order autonomous state under the tested composition rule;
- R2.3 remains exploratory evidence that carrier choice changes both predictive retention and composition behavior;
- R2.3b remains halted before its real-data comparison because the proposed common-target comparison could not distinguish a true carrier effect from an energy-only rival during qualification.

The purpose of R2.4 is therefore narrower and upstream of another autonomy test:

\[
\boxed{
\text{What compact representation of the current remote-history difference best summarizes its linearly predictable future?}
}
\]

R2.4 estimates predictive carriers **without optimizing them for compositionality, autonomy, causality, or generalization**.

The output of R2.4 is a frozen carrier object, or a frozen statement that no unique compact carrier is identified. Only after that freeze may the autonomy branch resume.

---

# 1. Why carrier resolution comes before another autonomy test

Let

\[
x_t\in\mathbb R^{256}
\]

be the frozen R1 aperture coordinate of the paired remote-history residual difference at the source position.

For a rank-\(r\) carrier

\[
Q\in\mathbb R^{256\times r},
\qquad Q^\top Q=I_r,
\]

define

\[
z_t=Q^\top x_t.
\]

Previous Arm D stages mixed two distinct questions:

1. **predictive sufficiency:** how much of the future of \(x_t\) can be predicted from \(z_t\)?
2. **dynamical closure:** does a one-step law fitted in \(z\)-coordinates generate the correct longer-lag law?

R2.4 resolves the first question before returning to the second.

The central methodological rule is:

\[
\boxed{
Q\text{ may be fit to predict the future, but may not be fit to look autonomous.}
}
\]

Forbidden during R2.4 are objectives of the form

\[
\min_{Q,A}
\sum_k
\left\|Q^\top x_{t+k}-A^kQ^\top x_t\right\|^2,
\]

or any equivalent objective that rewards composition, semigroup consistency, spectral simplicity, or causal transport while estimating \(Q\).

This separation is load-bearing. If \(Q\) is optimized to compose, a later positive autonomy test is partly built into the estimator.

---

# 2. Scientific questions

R2.4 asks five ordered questions.

## Q0 — available predictive ceiling

How much of the future aperture difference is linearly predictable from the full current aperture, and how much can an unconstrained rank-\(r\) source bottleneck retain?

## Q1 — compact predictive carrier

Can a fixed rank-\(r\) source subspace summarize most of the linearly predictable future trajectory across several lags?

## Q2 — metric dependence

Does the recovered carrier remain similar when prediction is scored in:

1. the ordinary energy-weighted Euclidean metric; and
2. a covariance-whitened metric that does not automatically privilege high-variance destination directions?

## Q3 — stability and rank concentration

Is the carrier reproducible across independent document halves, and how many dimensions are required before predictive performance saturates?

## Q4 — redundancy / load-bearingness

After deleting the recovered carrier and fully refitting the same estimator, does a comparably predictive orthogonal carrier remain? Is any deletion effect larger than expected from the amount of variance removed?

R2.4 does **not** ask whether one candidate carrier is statistically "better" than PCA, the inherited R1 carrier, or another named estimator unless a separately qualified comparison is added before confirmatory access.

---

# 3. Scientific object and common target

The source variable is always the current aperture difference

\[
x_t\in\mathbb R^p,
\qquad p=256.
\]

For each lag

\[
k\in K,
\]

the common destination target is

\[
y_{t,k}=x_{t+k}\in\mathbb R^{256}.
\]

Primary lag set:

```text
K_Q = [1, 2, 4, 8, 16, 32, 64]
```

Lag 128 is a frozen descriptive sensitivity.

Every cross-carrier predictive score uses the **same 256-dimensional destination target**. A carrier is never rewarded merely for having an easy-to-predict low-dimensional target of its own.

For candidate source carrier \(Q\), fit on train only

\[
\widehat y_{t,k}
=
B_{Q,k}Q^\top x_t.
\]

The decoder \(B_{Q,k}\) is unrestricted and fit separately for each lag unless explicitly stated otherwise.

The primary common-target score is the held-out predictive gain over the frozen persistence baseline used in the carrier report:

\[
\Gamma_Q(k)
=
R^2\!\left(y_{t,k},B_{Q,k}Q^\top x_t\right)
-
R^2\!\left(y_{t,k},\widehat y^{\rm persistence}_{t,k}\right).
\]

All carrier comparisons use identical documents, destination targets, baselines, centering, and scoring conventions.

---

# 4. Data boundary: development versus confirmation

The existing 2,000-document carrier pool is **spent**.

Its former sealed 300 documents were opened by R2.2. Therefore no R2.4 result computed on that pool is confirmatory.

Define two pools.

## 4.1 DEV-Q pool

```text
source:
existing R2.2/R2.3 capture

size:
2000 documents

role:
method development only
```

DEV-Q may be used for:

- the matched-deletion audit;
- the direction-versus-magnitude diagnostic;
- implementation debugging;
- numerical conditioning choices;
- estimator learning curves;
- choice of confirmatory sample size;
- frozen lag weights;
- frozen rank grid;
- deciding whether an optional comparative qualification campaign is worth running.

Nothing computed on DEV-Q can promote a carrier to the reference state handed to the next autonomy stage.

## 4.2 CONFIRM-Q pool

A fresh document-disjoint C4/en capture is required after the estimator, metrics, rank grid, lag set, regularization, access order, and decision rules are frozen.

No document may have appeared in any prior Arm D development, confirmatory, test, R2.2, R2.3, or R2.3b pool.

The confirmatory pool is partitioned at the document level into:

```text
fit-half-A: 35%
fit-half-B: 35%
validation: 15%
test:       15%
```

The test split is read exactly once after all carrier identities, ranks, complement-refit rules, and result labels are frozen on fit/validation.

## 4.3 Prospective sizing

The confirmatory document count is selected on DEV-Q before new capture.

Candidate totals:

```text
[2000, 4000, 8000]
```

Use document-resampling learning curves to estimate uncertainty in:

- common-target predictive gain;
- rank curves;
- split-half MSC;
- principal-angle profiles;
- complement-refit remaining gain.

Freeze the smallest count at which the main estimates have reached a clear variance plateau under the prespecified learning-curve rule. More bootstrap replicates do not substitute for independent documents.

---

# 5. Preflight P0 — variance-deletion audit

This audit is run on DEV-Q **before interpreting the old PCA-deletion result as evidence of a privileged carrier**.

## 5.1 Motivation

The top-variance rank-16 subspace captures approximately

\[
V_{\rm PCA16}\approx0.6722
\]

of aperture variance, whereas rank-16 Haar subspaces in the prior carrier run captured approximately \(0.063\) to \(0.071\).

Deleting PCA16 therefore removes far more activation energy than deleting a generic rank-16 subspace.

A large loss of predictive gain after PCA16 deletion can mean either:

1. the deleted directions are specially load-bearing for the remote-history trace; or
2. deleting a sufficiently large fraction of aperture variance destroys prediction regardless of the directions' special temporal role.

R2.4 must not assume the first interpretation.

## 5.2 Deleted variance

For orthonormal deletion basis \(D\in\mathbb R^{256\times r_D}\), define

\[
V(D)
=
\frac{\operatorname{tr}(D^\top\Sigma_x D)}{\operatorname{tr}(\Sigma_x)}.
\]

Project the source as

\[
x_t^{(-D)}=(I-DD^\top)x_t.
\]

Then **fully refit** the predictive model from the remaining source coordinates to the unchanged 256-dimensional future target.

Define

\[
\Gamma_{-D}(k)
\]

and surviving fraction

\[
S_D(k)
=
\frac{\Gamma_{-D}(k)}{\Gamma_{\rm full}(k)}
\]

whenever the denominator clears the frozen positive-gain floor.

## 5.3 Deletion families

At minimum evaluate:

```text
A. PCA1:16
B. PCA17:32
C. rank-16 Haar deletions
D. mixed top-PC / complement-Haar deletions
   m top PCs + (16-m) orthogonal complement directions
   m in [0, 2, 4, 8, 12, 16]
E. small-angle rotations of PCA1:16 toward lower-PC directions
F. energy-matched rank-varying deletions where feasible
```

Important mathematical limitation:

> PCA1:16 maximizes captured variance among all rank-16 subspaces.

Therefore an exact control that simultaneously matches both rank 16 and \(V_{\rm PCA16}\) is generally impossible unless it is nearly the same PCA subspace. The audit is consequently a **variance-dose-response analysis**, not a perfect same-rank/same-energy null at the PCA endpoint.

R2.4 must state this limitation explicitly.

## 5.4 Interpretation

Plot

\[
S_D(k)
\quad\text{against}\quad
V(D)
\]

for every deletion family and lag.

The old statement

```text
"the top-variance region is load-bearing"
```

is retained only if PCA1:16 is materially more destructive than the local variance-dose response would predict.

If surviving gain is explained smoothly by deleted variance, use:

```text
LOAD_BEARINGNESS_ENERGY_EXPLAINED
```

If PCA1:16 is more destructive than the variance-dose response, use on DEV only:

```text
LOAD_BEARINGNESS_GEOMETRY_SUGGESTIVE
```

A confirmatory geometry-specific load-bearing claim requires replay on CONFIRM-Q with the deletion family frozen.

This preflight does **not** gate whether Q-resolution itself runs.

---

# 6. Preflight P1 — direction versus magnitude in the old composition failure

This is a DEV-only diagnostic using the frozen inherited carrier and frozen earlier transition fits.

Its purpose is to determine whether the old composition failure is mainly:

1. a directional/action failure; or
2. a scale/gain failure with partially preserved direction.

For each lag, let

\[
\widehat z_{t+k}^{\rm comp}=M^k s_t
\]

or its frozen top-block prediction, and let

\[
\widehat z_{t+k}^{\rm direct}=D_k s_t.
\]

Report separately:

### Directional agreement

\[
C_{\rm dir}(k)
=
\mathbb E_d
\left[
\frac{
\langle \widehat z_{d,t+k}^{\rm comp},\widehat z_{d,t+k}^{\rm direct}\rangle
}{
\|\widehat z_{d,t+k}^{\rm comp}\|\,\|\widehat z_{d,t+k}^{\rm direct}\|
}
\right].
\]

### Radial error

\[
E_{\rm radial}(k)
=
\mathbb E_d
\left[
\left|
\log
\frac{\|\widehat z_{d,t+k}^{\rm comp}\|}
{\|\widehat z_{d,t+k}^{\rm direct}\|}
\right|
\right].
\]

### Best global scale-corrected action error

Fit on train only

\[
a_k^*
=
\arg\min_a
\sum_d
\left\|
\widehat z_{d,t+k}^{\rm direct}
-a\widehat z_{d,t+k}^{\rm comp}
\right\|^2,
\]

then evaluate

\[
E_{\rm shape}(k)
=
\mathbb E_d
\left[
\left\|
\widehat z_{d,t+k}^{\rm direct}
-a_k^*\widehat z_{d,t+k}^{\rm comp}
\right\|^2
\right].
\]

Bootstrap all three quantities over documents and compare the scale-corrected action error against the same estimation-noise envelope logic used in the earlier composition work.

Interpretation is descriptive only:

```text
DIRECTION_PRESERVED_SCALE_FAILS
FULL_ACTION_FAILS
MIXED_DIRECTION_SCALE_FAILURE
```

No autonomy claim is created here.

The outcome may change the later **M-stage model class** — for example motivating a direction-plus-gain model — but it does not change the primary Q estimator in this specification.

---

# 7. Q0 — predictive ceilings

Ceilings are estimated before fitting a new scientific carrier.

## 7.1 Full-aperture ceiling

For every lag \(k\), fit the full linear predictor

\[
\widehat y_{t,k}
=
B_{{\rm full},k}x_t
\]

with document-grouped ridge selection.

Record

\[
\Gamma_{\rm full}(k).
\]

This is the available linear predictive ceiling from the entire frozen 256-dimensional aperture under the chosen score.

## 7.2 Unrestricted rank-r transport ceiling

For

\[
r\in\{1,2,4,8,16,32,64\},
\]

fit the best reduced-rank map

\[
\widehat y_{t,k}
=
B_{r,k}U_{r,k}^\top x_t,
\]

where the source and destination predictive spaces are not required to be equal.

This gives

\[
\Gamma_{\rm RRR}(r,k).
\]

This is the correct ceiling for asking how much performance is lost by insisting on an \(r\)-dimensional source bottleneck at all.

## 7.3 Two efficiency ratios

For any candidate carrier \(Q_r\), report

\[
\eta_{\rm global}(Q_r,k)
=
\frac{\Gamma_{Q_r}(k)}{\Gamma_{\rm full}(k)},
\]

and

\[
\eta_{\rm rank}(Q_r,k)
=
\frac{\Gamma_{Q_r}(k)}{\Gamma_{\rm RRR}(r,k)}.
\]

The first asks:

> how much of all available aperture predictability does this carrier preserve?

The second asks:

> how close is this carrier to the best rank-\(r\) source bottleneck?

Both are required because a carrier can be near the rank-\(r\) ceiling while rank \(r\) itself is far below the full-aperture ceiling.

---

# 8. Q1 — dual multi-horizon carrier estimators

R2.4 deliberately fits **two predictive metrics**.

This prevents a high-variance objective from silently defining "the state" by construction.

Let

\[
C_k
=
\mathbb E[x_t x_{t+k}^\top],
\]

\[
\Sigma_0
=
\mathbb E[x_tx_t^\top],
\qquad
\Sigma_k
=
\mathbb E[x_{t+k}x_{t+k}^\top].
\]

Use train-only means and equal document weighting.

Regularize source and destination covariance matrices separately using the existing Arm D conditioning contract:

\[
W_0=(\Sigma_0+\epsilon_0I)^{-1/2},
\qquad
W_k=(\Sigma_k+\epsilon_kI)^{-1/2}.
\]

Ridge scales are resolved without validation/test access and recorded.

## 8.1 Q-E — energy-weighted multi-horizon predictive carrier

Define

\[
G_E
=
\sum_{k\in K_Q}
w_k
W_0 C_k C_k^\top W_0.
\]

Let \(V_E^{(r)}\) contain the top \(r\) eigenvectors of \(G_E\).

Map to source-aperture coordinates:

\[
Q_E(r)
=
\operatorname{orth}\!\left(W_0V_E^{(r)}\right).
\]

This estimator finds source directions whose linear information predicts large Euclidean future-block variance.

It is intentionally labeled **energy-weighted**. It is not treated as metric-neutral.

## 8.2 Q-W — covariance-whitened multi-horizon predictive carrier

Define

\[
G_W
=
\sum_{k\in K_Q}
w_k
W_0
C_k
(\Sigma_k+\epsilon_kI)^{-1}
C_k^\top
W_0.
\]

Equivalently,

\[
G_W
=
\sum_{k\in K_Q}
w_k
\left(W_0C_kW_k\right)
\left(W_0C_kW_k\right)^\top.
\]

Let \(V_W^{(r)}\) contain the top \(r\) eigenvectors of \(G_W\), and define

\[
Q_W(r)
=
\operatorname{orth}\!\left(W_0V_W^{(r)}\right).
\]

This is a multi-horizon CCA/VAMP-like source subspace: destination directions are variance-normalized before their predictability contributes to the objective.

Q-W is the **primary scientific carrier estimator** for R2.4 because the immediate question is whether a compact future-predictive state exists beyond the trivial preference for high-energy directions.

Q-E is a required co-reported estimator because Euclidean residual energy is still architecturally meaningful and may identify a different useful carrier.

## 8.3 Lag weights

Primary:

\[
w_k=1/|K_Q|.
\]

No lag weight is selected using confirmatory performance.

Required sensitivity:

```text
long-horizon K = [8, 16, 32, 64]
```

Lag 128 is descriptive only.

---

# 9. Q2 — rank curves and predictive sufficiency

Evaluate

\[
r\in\{1,2,4,8,16,32,64\}
\]

for both Q-E and Q-W.

For every rank report:

- \(\Gamma_Q(k)\) on the raw common target;
- whitened-target predictive score;
- \(\eta_{\rm global}(Q,k)\);
- \(\eta_{\rm rank}(Q,k)\);
- multi-horizon weighted mean of each quantity;
- rank-specific train/validation uncertainty;
- residual variance captured by the carrier.

Define descriptive compression ranks:

\[
r_{90}^{E}
=
\min\left\{r:
J_E(r)\ge0.9\max_{r'\le64}J_E(r')
\right\},
\]

\[
r_{90}^{W}
=
\min\left\{r:
J_W(r)\ge0.9\max_{r'\le64}J_W(r')
\right\}.
\]

Also report the fraction of the full 256-dimensional ceiling reached at those ranks.

The 90% number is a compression summary, not a significance threshold.

Do not choose rank by maximizing a noisy held-out score. Choose the **smallest** rank reaching the frozen fraction of the nested rank-64 curve on validation, then freeze that rank before test.

---

# 10. Q3 — split stability and metric dependence

## 10.1 Independent-half fits

Fit each estimator separately on fit-half-A and fit-half-B:

\[
Q_E^{A}(r),\quad Q_E^{B}(r),
\]

\[
Q_W^{A}(r),\quad Q_W^{B}(r).
\]

For equal-rank subspaces \(U,V\), report

\[
\operatorname{MSC}(U,V)
=
\frac1r\|U^\top V\|_F^2,
\]

plus:

- full principal-angle spectrum;
- directional containment both ways;
- held-out score-space canonical correlation.

## 10.2 Stability nulls

Zero is not the stability null.

For two independent Haar rank-\(r\) subspaces in 256 dimensions,

\[
\mathbb E[\operatorname{MSC}]
=
\frac{r}{256}.
\]

At rank 16 this is

\[
0.0625.
\]

This analytic value is only a sanity check.

The inferential stability reference is a **full-refit pairing-destruction null** that preserves source and destination marginal geometry while destroying the true source-future association.

Run at least 200 frozen null refits per estimator/rank using document-coherent future derangement or another qualification-verified association-destruction construction.

Also report a positive estimator-noise reference from bootstrap refits within one fit half.

A stable carrier must exceed the full-refit null distribution; bootstrap self-overlap is reported as the attainable sampling ceiling.

## 10.3 Cross-metric geometry

At the frozen ranks, compare

\[
Q_E
\quad\text{and}\quad
Q_W
\]

using the same MSC, angle, and containment metrics.

R2.4 does not force them into a single averaged basis.

If they align strongly, Q-W remains the primary reference and Q-E is recorded as metric-concordant support.

If they disagree materially, emit:

```text
METRIC_DEPENDENT_PREDICTIVE_CARRIERS
```

and freeze both as separate predictive representations. Do not manufacture a consensus carrier by taking their union or midpoint after seeing the data.

---

# 11. Q4 — existing carriers are descriptive comparators, not a new hypothesis test

Evaluate on the same held-out common targets:

```text
inherited R1 carrier
PCA top-variance carrier
R2.3 predictive refit
R2.2 complement-refit carrier
single-lag VAMP / predictive carrier where provenance permits
Q-E
Q-W
```

Report:

- common-target raw predictive gain;
- whitened-target predictive score;
- captured source variance;
- rank;
- overlap with Q-E and Q-W;
- split stability where independently refittable.

No label such as

```text
Q_W_BEATS_PCA
```

or

```text
SPECIAL_CARRIER
```

may be produced from these rankings alone.

If a formal comparative claim is desired, it becomes a separately locked branch `Q4X` with known-answer qualification before confirmatory access. Its qualification suite must include at minimum:

```text
true predictive carrier world
energy-only rival world
multiple redundant carrier world
isotropic null
high-variance but nonpredictive world
low-variance strongly predictive world
```

The energy-only rival that invalidated R2.3b must be included. Failure to distinguish it blocks the comparative claim but does not block the estimation program.

---

# 12. Q5 — complement refit and redundant carrier family

After Q-W's estimator and rank are frozen on validation, project it out of the source aperture:

\[
x_t^\perp
=
(I-Q_WQ_W^\top)x_t.
\]

Then rerun the **entire Q-W estimator from scratch** in the complement:

```text
centering
source/destination covariance estimation
regularization
multi-horizon operator construction
rank curve
rank selection
split stability
common-target decoder fitting
held-out scoring
```

Call the complement carrier

\[
Q_{W,2}.
\]

Report remaining predictive fractions

\[
C_{\perp}^{\rm global}(k)
=
\frac{\Gamma_{Q_{W,2}}(k)}{\Gamma_{\rm full}(k)},
\]

and

\[
C_{\perp}^{\rm relative}(k)
=
\frac{\Gamma_{Q_{W,2}}(k)}{\Gamma_{Q_W}(k)}.
\]

## 12.1 Deletion controls for Q5

For the frozen \(Q_W\), compare its deletion against:

- rank-matched Haar deletions;
- same-rank deletions approximately matched in captured variance where feasible;
- the frozen variance-dose-response family from P0.

Unlike PCA16, a generic recovered Q-W need not sit at the maximum-variance endpoint, so a closer same-rank energy match may be possible.

## 12.2 Interpretation

Use the continuous results first. Descriptive ontology:

```text
DOMINANT_COMPACT_CHANNEL
    deleting Q-W and refitting leaves little predictive gain relative to
    rank/energy-matched deletions

REDUNDANT_PREDICTIVE_CHANNELS
    an approximately orthogonal refit recovers a similarly predictive,
    independently stable carrier

STABLE_CORE_BROAD_SHOULDER
    Q-W is stable and efficient but substantial weaker predictive structure
    remains outside it

NONIDENTIFIED_CARRIER
    predictive gain exists but the fitted subspace is not stable beyond null
```

Optional iterative deflation runs only if the first complement refit is substantial and stable. Its stopping rule is frozen before the deflation curve is inspected.

---

# 13. Independent result axes

R2.4 reports four axes rather than forcing one ladder.

## 13.1 P — predictive compression

```text
P0_NO_COMPACT_PREDICTIVE_COMPRESSION
P1_COMPACT_PREDICTIVE_COMPRESSION
P_TECHNICAL_FAILURE
```

The report must always include the full rank curve and the continuous \(r_{90}\) summaries. The label does not replace them.

## 13.2 G — metric geometry

```text
G0_METRIC_DEPENDENT_CARRIERS
G1_METRIC_CONCORDANT_CARRIER
G_UNDERRESOLVED
```

## 13.3 I — identifiability / redundancy

```text
I0_NONIDENTIFIED_CARRIER
I1_STABLE_CARRIER
I2_STABLE_CORE_BROAD_SHOULDER
I3_REDUNDANT_CARRIER_FAMILY
I4_DOMINANT_CHANNEL
```

## 13.4 L — deletion/load-bearingness

```text
L0_ENERGY_EXPLAINS_DELETION
L1_GEOMETRY_SPECIFIC_DELETION
L_UNDERRESOLVED
```

The L-axis is explicitly separate from predictive sufficiency. A carrier can be highly predictive yet redundant and therefore weakly load-bearing under deletion.

---

# 14. What qualifies a carrier for the next autonomy stage

R2.4 does not require a unique privileged basis before any future work can continue.

The handoff depends on the observed ontology.

## Case A — one stable metric-concordant compact carrier

Freeze

\[
\mathcal Q_0=Q_W(r_*).
\]

This becomes the sole primary carrier for the next autonomy test.

## Case B — stable but metric-dependent carriers

Freeze

\[
\mathcal Q_0=
\{Q_W(r_W^*),Q_E(r_E^*)\}.
\]

The next autonomy stage must test both under a multiplicity-controlled contract or explicitly choose one metric for a scientific reason fixed before looking at autonomy results.

## Case C — redundant carrier family

Freeze the first two independently recovered stable carriers and their common-target predictive scores.

The next dynamics question becomes:

\[
\boxed{
\text{Are the dynamics equivalent across redundant predictive realizations?}
}
\]

rather than asking whether one arbitrarily chosen basis is "the" state.

## Case D — no compact stable carrier

Do not run another fixed-low-rank autonomy search.

The program must reconsider one of:

- higher rank;
- changing-coordinate state;
- nonlinear state compression;
- conditional source representation;
- the possibility that remote-history prediction is intrinsically distributed in this aperture.

---

# 15. Confirmatory promotion rule

A carrier estimated on DEV-Q is never promoted directly into the autonomy stage.

Promotion requires all of:

1. estimator frozen before CONFIRM-Q capture;
2. rank rule frozen before test;
3. independent-half stability on fresh documents;
4. validation selection only under the locked metric;
5. one untouched test evaluation reproducing predictive sufficiency and geometry in sign and material magnitude;
6. all artifact hashes frozen before test;
7. no use of composition, causal response, or future generalization performance to choose the carrier.

The confirmatory stage establishes a **predictive representation**, not a causal state mechanism.

---

# 16. Multiplicity and inference policy

R2.4 is primarily an estimation experiment.

Continuous rank curves, ceilings, subspace angles, deletion curves, and predictive efficiencies are reported with document-bootstrap uncertainty rather than converted unnecessarily into many binary tests.

Formal null inference is required for:

- split-half carrier stability;
- any geometry-specific deletion claim;
- any optional named-carrier superiority claim in Q4X.

The Q-E and Q-W metrics answer different prespecified scientific questions and are co-reported rather than treated as two attempts to win one hypothesis test.

Any optional Q4X comparison controls family-wise error over all named candidate carriers and ranks included in its frozen comparison family.

---

# 17. Numerical contract

All fitting uses float64 statistics.

Required audits:

- covariance symmetry;
- ridge-resolution trace;
- source and destination condition numbers;
- exact orthonormality of every reported carrier;
- invariance of predictions under within-subspace orthogonal rotation;
- equivalence of direct future-block implementation and summed per-lag implementation;
- reduced-rank regression dense-reference check in a small synthetic aperture;
- Q-E eigensystem replay;
- Q-W eigensystem replay;
- no validation/test data in means, covariance regularization, or carrier fitting;
- document-weighted rather than token-weighted scoring.

If covariance whitening requires normalized ridge above the existing Arm D heavy-regularization band, mark the corresponding metric `NUMERICALLY_UNSTABLE` and do not turn its carrier into a reference state.

---

# 18. Required synthetic / implementation worlds

Because Q0-Q3 are estimation tasks, they do not need a large hypothesis-power qualification campaign. They do require known-answer implementation checks.

At minimum:

### Q-S0 — isotropic no-prediction null

No stable predictive subspace should be recovered beyond the null overlap distribution.

### Q-S1 — low-rank predictive source

A known rank-\(r\) source bottleneck predicts several future lags. Q-E and Q-W should recover the planted source span when source/destination covariance is isotropic.

### Q-S2 — energy-only nuisance

High-variance directions carry no extra predictive relation. Q-E may move toward them according to its metric; Q-W should not falsely interpret variance alone as normalized predictive structure.

### Q-S3 — low-variance predictive carrier

A low-energy subspace carries strong normalized future information. Q-W must recover it.

### Q-S4 — two redundant predictive carriers

Two orthogonal source subspaces each predict the same future latent variable. Complement refit should recover the second carrier.

### Q-S5 — metric disagreement world

One subspace dominates raw energy-weighted prediction while another dominates covariance-normalized prediction. The pipeline should report metric dependence rather than manufacture a consensus basis.

These worlds validate implementation and interpretation. They do not authorize a post-hoc candidate-superiority test.

---

# 19. Execution order

```text
R2.4-P0   lock this spec and existing-run provenance
R2.4-P1   DEV matched-deletion variance-dose audit
R2.4-P2   DEV direction-versus-magnitude diagnostic
R2.4-P3   implement and unit-test full / reduced-rank ceilings
R2.4-P4   implement and unit-test Q-E / Q-W
R2.4-P5   DEV rank curves, stability learning curves, sample sizing
R2.4-P6   freeze lags, weights, ridges, rank rule, nulls, sample size

---------- NO METHOD CHANGES AFTER THIS LINE ----------

R2.4-C0   capture fresh CONFIRM-Q documents
R2.4-C1   fit full and rank-r predictive ceilings on fit documents
R2.4-C2   fit Q-E and Q-W rank curves
R2.4-C3   independent-half stability and null refits
R2.4-C4   validation rank freeze and metric-geometry classification
R2.4-C5   descriptive existing-carrier comparison
R2.4-C6   Q-W complement refit and deletion controls
R2.4-C7   freeze P/G/I/L labels and all carrier artifacts
R2.4-C8   exactly one test evaluation
R2.4-C9   freeze reference carrier object for the next M stage
```

If an implementation error is found after C0 but before test, repair requires a new versioned specification and an explicit contamination audit. No threshold or estimator may be changed because of an unfavorable confirmatory result.

---

# 20. Required artifacts

```text
arm_d/r2_4_q_resolution/
  spec/
    arm_d_r2_4_spec.md
    arm_d_r2_4_spec.sha256
    run_manifest.json
    source_hashes.json
    software_commit.txt
    environment_lock.txt

  preflight/
    deletion_basis_manifest.json
    deletion_variance.parquet
    deletion_surviving_gain.parquet
    deletion_dose_response.parquet
    direction_magnitude_by_lag.parquet
    scale_corrected_action_error.parquet

  ceilings/
    full_aperture_gain.parquet
    reduced_rank_gain.parquet
    reduced_rank_bases.npz

  carriers/
    q_energy_rank_curve.npz
    q_whitened_rank_curve.npz
    q_energy_metrics.parquet
    q_whitened_metrics.parquet
    covariance_conditioning.json

  stability/
    split_half_subspaces.npz
    msc_principal_angles.parquet
    pairing_destroyed_null.parquet
    within_half_bootstrap_overlap.parquet
    cross_metric_overlap.parquet

  comparators/
    inherited_r1_metrics.parquet
    pca_metrics.parquet
    predictive_refit_metrics.parquet
    complement_refit_metrics.parquet
    vamp_metrics.parquet

  redundancy/
    q_primary_deleted_basis.npz
    complement_refit_basis.npz
    complement_rank_curve.parquet
    deletion_controls.parquet
    remaining_gain.parquet

  confirmation/
    validation_freeze.json
    test_access_counter.json
    test_metrics.parquet
    final_labels.json
    reference_carrier_manifest.json
    final_report.md
```

---

# 21. Required tests

The implementation test suite must cover:

- exact document disjointness between DEV-Q and CONFIRM-Q;
- test-access counter starts at zero;
- no-op identity of the common-target scoring code under a full-rank source basis;
- reduced-rank regression recovery on a planted low-rank linear system;
- Q-E recovery under isotropic covariance;
- Q-W recovery under isotropic covariance;
- Q-W resistance to the energy-only world;
- low-variance predictive-carrier recovery;
- redundant-carrier complement recovery;
- metric-disagreement classification;
- MSC analytic Haar expectation \(r/256\);
- full-refit association-destruction null replay;
- subspace rotation invariance;
- deletion projection idempotence;
- deleted-variance calculation against explicit PCA sums;
- full refit after every deletion rather than rescoring a frozen estimator;
- train-only rank and ridge selection;
- validation-only rank freeze;
- one-time test seal.

---

# 22. Reporting contract

The final report must lead with continuous quantities rather than only labels.

Required headline figures:

1. **Full and reduced-rank predictive ceilings versus rank.**
2. **Q-E and Q-W predictive sufficiency versus rank.**
3. **Q-E/Q-W split-half stability and principal-angle spectra.**
4. **Cross-metric Q-E versus Q-W subspace overlap.**
5. **Surviving gain versus deleted variance.**
6. **Complement-refit remaining gain after deleting the frozen Q-W carrier.**
7. **Descriptive common-target comparison to inherited R1, PCA, predictive refit, and complement refit.**
8. **DEV-only direction-versus-magnitude decomposition of the old composition failure.**

The report must explicitly distinguish:

```text
predictive sufficiency
metric dependence
subspace stability
load-bearingness
redundancy
composition
causality
```

No one of these is allowed to stand in for another.

---

# 23. Permitted interpretations

## Strongest R2.4 outcome

If a low-rank Q-W carrier:

- captures most available normalized future predictability;
- retains substantial raw common-target predictability;
- is stable across independent halves beyond full-refit nulls;
- agrees substantially with Q-E or has a clearly reported metric relation;
- reproduces on the untouched test;

then R2.4 may conclude:

> A compact, reproducible predictive representation of the remote-history state has been identified without assuming autonomous dynamics.

If complement refit is weak relative to rank/energy-matched deletions, it may additionally say that this is a dominant carrier under the tested linear aperture.

## Metric-dependent outcome

If Q-E and Q-W are each stable and predictive but geometrically distinct:

> The compact predictive state is metric-dependent: energy-weighted and variance-normalized future prediction select different stable residual carriers.

That is a substantive result, not a failed estimator.

## Redundant outcome

If deletion and refit recover another comparably predictive stable orthogonal carrier:

> The remote-history state admits multiple redundant low-dimensional linear realizations in the aperture; no unique privileged carrier is established.

## Diffuse outcome

If the full aperture substantially outperforms every rank up to 64 or split-half geometry is unstable:

> The present data do not support a compact stable fixed-subspace representation of the remote-history predictive state at the tested ranks.

---

# 24. Explicit non-claims

R2.4 does not establish that:

- the recovered carrier is autonomous;
- the recovered carrier composes;
- the carrier is causal or load-bearing in the model's computation merely because it predicts;
- PCA geometry is or is not mechanistically privileged unless the variance-deletion audit supports that claim;
- one basis is uniquely "the state" when multiple metrics or complement refits disagree;
- the representation generalizes across layers, corpora, tasks, or models;
- the state has a clean semantic dictionary.

Those are later questions.

---

# 25. Program handoff

The program order after R2.4 is:

\[
\boxed{
\text{predictive ceilings}
\rightarrow
\text{carrier estimation}
\rightarrow
\text{stability / metric dependence / redundancy}
\rightarrow
\text{fresh carrier freeze}
\rightarrow
\text{autonomy}
\rightarrow
\text{conditional dynamics if required}
\rightarrow
\text{causality}
\rightarrow
\text{generalization}.
}
\]

The next autonomy experiment must take the R2.4 reference object as fixed input. It may not search over new carriers using composition performance.

The central R2.4 principle is therefore:

\[
\boxed{
\text{First identify what state is being represented; only then ask for its law of motion.}
}
