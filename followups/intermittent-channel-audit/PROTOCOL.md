# Residual Sequence-Memory Geometry
## Intermittent-Channel Audit
### Artifact-Only Locked Implementation Specification

---

## 0. Status and Purpose

This document specifies a bounded, artifact-only estimator audit of the completed C4 residual-geometry pilot.

The completed experiment measured whether a residual projection remains similar across token positions:

\[
P_{d,t,j}=v_j^\top r_{d,t},
\qquad
\operatorname{corr}(P_{d,t,j},P_{d,t+k,j}).
\]

That estimator is well matched to information that remains continuously expressed along one direction. It can miss a distinct case in which the same residual direction is used intermittently but carries different payload values on different activations.

The motivating model is:

\[
P_{d,t,j}=g_{d,t,j}z_{d,t,j}+\varepsilon_{d,t,j},
\qquad
g_{d,t,j}\in\{0,1\},
\]

where:

- \(g_{d,t,j}\) indicates whether direction \(j\) is active;
- \(z_{d,t,j}\) is the signed payload carried while active;
- \(\varepsilon_{d,t,j}\) is background activity.

Ordinary signed autocorrelation can be small because activity is rare, because active moments do not cluster, or because different active moments carry different payloads. This audit separates those factors without fitting any new residual directions and without running the model again.

The stage is an estimator audit, not a replacement for the original result. It does not modify any completed persistence, subspace, attention-alignment, or causal verdict.

---

## 1. One-Sentence Summary

Using the frozen C4 residual projection tensors, separately measure signed payload persistence, activity-envelope persistence, binary co-activation structure, natural duty cycle, and active-pair payload consistency, then test whether there is a held-out-generalizing population of residual directions with weak signed persistence but strong sequence-dependent activity persistence.

---

## 2. Scientific Object

For document \(d\), token position \(t\), and frozen residual probe direction \(v_j\), let:

\[
P_{d,t,j}=v_j^\top r_{d,t}.
\]

The original primary signal is within-document signed persistence. Define:

\[
\widetilde P_{d,t,j}
=
P_{d,t,j}
-
\frac{1}{T}\sum_{u=0}^{T-1}P_{d,u,j}.
\]

The current audit adds three distinct temporal objects.

### 2.1 Signed payload

\[
x_{d,t,j}=\frac{\widetilde P_{d,t,j}}{s_j},
\]

where \(s_j\) is a train-fitted probe scale.

This retains sign and magnitude and asks whether the carried value remains similar.

### 2.2 Activity envelope

\[
q_{d,t,j}=x_{d,t,j}^2.
\]

This discards sign and asks whether projection energy is temporally structured.

### 2.3 Binary activity gate

\[
g_{d,t,j}=\mathbf 1\{|x_{d,t,j}|>\theta_j\}.
\]

This asks when the direction is active, separately from what value it carries.

The audit therefore estimates:

```text
signed persistence:
Does the same payload remain present?

envelope persistence:
Does the direction remain energetically active in temporally structured episodes?

co-activation persistence:
Are active moments more likely to occur near other active moments than chance predicts?

active-pair payload consistency:
When the direction is active at both endpoints, are the payload values related?

duty cycle:
How often does the direction naturally become active?
```

---

## 3. Primary Scientific Questions

### Q1. Missing quadrant

Do any frozen residual directions have:

\[
\tau_{\mathrm{signed}}\text{ low}
\qquad\text{and}\qquad
\tau_{\mathrm{activity}}\text{ high}?
\]

### Q2. Order dependence

Does the activity structure collapse when token positions are permuted within documents while preserving each direction's projection-value multiset?

### Q3. Global-amplitude robustness

Does the activity structure survive controlling for tokenwise global residual magnitude, estimated from the frozen random-direction projection bank?

### Q4. Sparse versus dense expression

Are activity-persistent directions naturally intermittent, or are they simply densely active directions whose sign changes?

### Q5. Payload behavior

When activity recurs, does the direction carry a related payload, an alternating payload, or effectively unrelated payload values?

### Q6. Geometric novelty

Do validation-selected intermittent candidates span geometry distinct from the previously identified slow subspace, or are they a different characterization of the same region?

### Q7. Probe-family dependence

Are candidates enriched among time-lagged probes, PCA directions, random directions, or low-lifetime time-lagged controls?

### Q8. Estimator handoff

Is the signal strong and stable enough to justify fitting a new gated or fourth-order direction estimator?

---

## 4. Claim Boundary

A positive result establishes only:

> Some frozen residual directions have temporally structured activity envelopes or co-activation patterns even though their signed projections have weak ordinary persistence.

A positive result does not by itself establish:

- that the directions carry memory rather than temporally clustered corpus features;
- that attention writes or retrieves the activity;
- that the same semantic variable is present across activations;
- that the activity is causally used;
- that the direction is a dedicated communication bus;
- that an optimized intermittent subspace exists;
- that the result transfers across corpora, layers, or models;
- that the activity survives document-splice controls requiring new forward passes.

Use the terms:

```text
activity-persistent direction
intermittent-channel candidate
low-signed/high-activity direction
```

Do not use the terms:

```text
confirmed memory bus
causal memory channel
dedicated state carrier
```

without later attention/KV and intervention evidence.

---

## 5. Relation to Existing Estimators

The program now separates two axes.

### Temporal occupancy

```text
dense expression
intermittent expression
```

### Coordinate stability

```text
same residual coordinates
changing residual coordinates
```

This yields:

| | Same coordinates | Changing coordinates |
|---|---|---|
| Dense | original slow estimator | asymmetric Stage 06C estimator |
| Intermittent | this audit | deferred event-conditioned asymmetric estimator |

This audit targets only the lower-left cell. It does not duplicate Stage 06C, which asks whether information remains linearly predictable while changing source and destination coordinates.

---

## 6. Scope and Non-Goals

### In scope

- Artifact-only reanalysis of frozen C4 residual projections.
- Exact reproduction of the original signed autocorrelation estimator.
- Squared-envelope autocorrelation.
- Log-envelope sensitivity analysis.
- Matched-occupancy binary gates.
- Train-fitted natural-duty gates.
- Co-activation lift and normalized co-activation timescale.
- Active-pair payload correlation.
- Burst duration and inter-burst gap summaries.
- Token-permutation and block-permutation controls.
- Random-gate controls.
- Global residual-amplitude proxy from random projections.
- Position-profile sensitivity.
- Probe-family comparisons.
- Validation selection and frozen test characterization.
- Candidate-span overlap with the existing slow and PCA spans.

### Deferred

- Fitting gated SFA directions.
- Optimizing a fourth-order objective.
- New model forward passes.
- Attention-pattern or per-source attention-output decomposition.
- Distance-filtered long-range attention writes.
- KV-cache probes.
- Spliced-document forward passes.
- Semantic claims about payload identity.
- Causal interventions.
- Nonlinear or event-conditioned asymmetric estimators.

The stage must remain an audit. A positive audit is the prerequisite for fitting a new intermittent estimator.

---

## 7. Frozen Run Identity

Primary source run:

```text
model:
google/gemma-2-2b

hook:
blocks.12.hook_resid_post

layer:
12

residual width:
2304

corpus:
allenai/c4, en

context length:
1024

context split:
train 80%, validation 10%, test 10%

probe families:
512 random
256 residual PCA
256 time-lagged
```

The audit must record the exact run root, configuration hash, model revision, tokenizer revision, context manifest hashes, direction artifact hashes, projection-chunk hashes, and original timescale artifact hashes.

---

## 8. Required Inputs

Required frozen artifacts:

```text
residual_probes/projections/train/*.npz
residual_probes/projections/val/*.npz
residual_probes/projections/test/*.npz

residual_probes/directions/random_residual.npz
residual_probes/directions/residual_pca.npz
residual_probes/directions/time_lagged_residual.npz

residual_probes/autocorr/probe_timescales.parquet
residual_probes/autocorr/profiles_*.npz

subspace/projection_bases/residual_first_kstar.npz
residual_probes/directions/residual_pca.npz
```

Each projection chunk must contain:

```text
context_ids
shape: (B,)

tokens
shape: (B, 1024)

probe_ids
shape: (P,)

probe_family
shape: (P,)

projections
shape: (B, 1024, P)
```

The full projection tensors are required. Saved autocorrelation profiles alone are insufficient because the new envelope, gate, permutation, and conditional statistics cannot be reconstructed from them.

Optional artifacts:

```text
tokenwise residual norms
position metadata beyond token index
precomputed token-string classes
attention-alignment summaries
semantic dossiers
```

Optional artifacts may enrich controls or interpretation but may not be required for the primary artifact-only result.

---

## 9. Repository Placement

Implement inside the existing residual-geometry namespace.

```text
src/persistent_state/
  intermittent/
    __init__.py
    preprocessing.py
    norm_proxy.py
    gates.py
    estimators.py
    nulls.py
    candidates.py
    geometry.py
    reporting.py

scripts/persistent_state/residual_geometry/
  08_validate_intermittent_inputs.py
  09_compute_intermittent_profiles.py
  10_compute_intermittent_nulls.py
  11_select_intermittent_candidates.py
  12_make_intermittent_report.py

configs/persistent_state/residual_geometry/intermittent/
  smoke.yaml
  pilot.yaml
  full.yaml

tests/persistent_state/residual_geometry/intermittent/
```

Reuse the existing artifact store, streaming chunk reader, autocorrelation utilities, split manifests, bootstrap implementation, logging, and report conventions.

Do not duplicate projection loading or document weighting logic.

---

## 10. Frozen Analysis Lock

Before any validation metric is inspected, write:

```text
analysis/intermittent_channel_audit/frozen_analysis_lock.json
```

The lock must contain:

- git commit and working-tree status;
- source run identity and all input hashes;
- probe families and exact probe IDs;
- split identities;
- maximum lag;
- scale estimator;
- envelope transforms;
- gate definitions;
- natural thresholds;
- matched-occupancy rates;
- valid-pair thresholds;
- smoothing rule;
- timescale crossing rule;
- permutation seeds and replicate counts;
- block-shuffle sizes;
- norm-proxy construction;
- position-control construction;
- candidate-selection criteria;
- deduplication threshold;
- bootstrap seed and replicates;
- test-use rule;
- decision-label thresholds.

No threshold may be changed after validation inspection without creating a new versioned protocol and labeling the revised run exploratory.

---

## 11. Split Hygiene

Use the original document split.

### Train

Use train documents only to fit:

- probe scales \(s_j\);
- natural activity thresholds;
- optional position profiles;
- optional regression of envelope activity on the global-amplitude proxy;
- token-class nuisance summaries when enabled.

The frozen residual directions are not refit.

### Validation

Use validation documents to:

- estimate primary audit metrics;
- compare probe families;
- select the candidate set;
- select candidate rank by a predeclared rule;
- decide whether a new intermittent estimator is warranted.

### Test

Freeze all transforms, candidate IDs, rankings, and decision rules before evaluating test.

The original pilot test split has already been inspected in prior analyses. Therefore test results from this audit are post-pilot descriptive rather than prospectively confirmatory. They still must not be used for method selection.

A future full run or new held-out corpus is required for a prospective claim.

### Bootstrap unit

Bootstrap over documents, never tokens or token pairs.

---

## 12. Input Validation and Reproduction Gate

Before computing any new statistic:

1. Verify every projection chunk against its manifest hash.
2. Verify identical probe ordering across chunks and splits.
3. Verify every document has exactly 1024 token positions.
4. Cast stored projections to float32 before statistics.
5. Verify finite values.
6. Verify probe direction norms are one within tolerance `1e-5`.
7. Verify family counts.
8. Verify no context ID appears in multiple splits.
9. Recompute the original within-document signed autocorrelation and timescale.
10. Compare recomputed values against saved artifacts.

Required reproduction tolerance:

```text
maximum absolute profile difference at reported lags:
1e-6 for float32 source projections
5e-4 for float16 stored projections

maximum timescale mismatch rate:
0.5% of probes

maximum absolute timescale difference for matched probes:
1 token, except threshold-tie cases
```

If the reproduction gate fails, stop. Do not interpret envelope results until the existing estimator is reproduced.

---

## 13. Common Preprocessing

### 13.1 Within-document demeaning

For every document and probe:

\[
\widetilde P_{d,t,j}
=
P_{d,t,j}
-
\frac{1}{T}\sum_{u=0}^{T-1}P_{d,u,j}.
\]

This exactly matches the original primary estimator.

### 13.2 Train-fitted probe scale

Fit on train documents:

\[
s_j
=
\sqrt{
\mathbb E_{d,t\in\mathrm{train}}
\left[
\widetilde P_{d,t,j}^2
\right]
}.
\]

Require:

\[
s_j^2>10^{-12}.
\]

Define standardized signed projection:

\[
x_{d,t,j}=\frac{\widetilde P_{d,t,j}}{s_j}.
\]

Apply the frozen train scale unchanged to validation and test.

The signed Pearson autocorrelation is invariant to this scalar rescaling; the scale is needed for natural thresholds and cross-probe amplitude summaries.

### 13.3 Primary envelope

Define:

\[
q_{d,t,j}=x_{d,t,j}^2.
\]

Then demean the envelope within each document:

\[
\widetilde q_{d,t,j}
=
q_{d,t,j}
-
\frac{1}{T}\sum_{u=0}^{T-1}q_{d,u,j}.
\]

### 13.4 Log-envelope sensitivity

To reduce domination by isolated extreme spikes, define:

\[
q^{\log}_{d,t,j}=\log(1+x_{d,t,j}^2),
\]

and demean it within document.

The squared envelope is primary. The log envelope is a required sensitivity analysis.

---

## 14. Artifact-Only Global Residual-Norm Proxy

Large global residual-amplitude changes can make every direction appear jointly active.

The random-direction bank permits an artifact-only proxy. Let \(u_a\), \(a=1,\ldots,m\), be the frozen random unit directions. For token \((d,t)\):

\[
\widehat n^2_{d,t}
=
\frac{d_{\mathrm{model}}}{m}
\sum_{a=1}^{m}
\left(u_a^\top r_{d,t}\right)^2.
\]

For an isotropic random direction:

\[
\mathbb E_u[(u^\top r)^2]
=
\frac{\|r\|_2^2}{d_{\mathrm{model}}},
\]

so \(\widehat n^2_{d,t}\) estimates tokenwise residual norm squared.

### 14.1 Leave-one-out rule

When evaluating random probe \(j\), exclude that probe from its own norm proxy:

\[
\widehat n^2_{d,t,-j}
=
\frac{d_{\mathrm{model}}}{m-1}
\sum_{a\neq j}
P_{d,t,a}^2.
\]

For PCA and time-lagged probes, use the full random bank.

### 14.2 Norm-normalized sensitivity

Define:

\[
P^{\mathrm{norm}}_{d,t,j}
=
\frac{P_{d,t,j}}
{\sqrt{\widehat n^2_{d,t}+\epsilon_n}},
\]

where:

\[
\epsilon_n
=
10^{-8}\cdot
\operatorname{median}_{d,t\in\mathrm{train}}
\widehat n^2_{d,t}.
\]

Then repeat within-document demeaning, scaling, envelope construction, gating, and temporal statistics using \(P^{\mathrm{norm}}\).

The raw projection analysis remains primary because norm normalization can remove real amplitude information. The norm-controlled analysis is required for interpretation.

### 14.3 Norm-proxy health

Record:

- number of random directions;
- effective rank of the random direction bank;
- median and tail of \(\widehat n^2\);
- correlation between proxy estimates from two random half-banks;
- sensitivity to using 128, 256, and 512 random directions;
- correlation with true residual norms if an optional norm artifact exists.

Required health gate:

```text
split-half tokenwise norm-proxy correlation on validation >= 0.95
```

If this fails, norm-controlled results are invalid and the overall audit is `GLOBAL_AMPLITUDE_CONTROL_INCONCLUSIVE` rather than positive.

---

## 15. Signed Autocorrelation

For lag \(k\), compute per-document signed correlation:

\[
r^{\mathrm{signed}}_{j,d}(k)
=
\operatorname{corr}
\left(
 x_{d,0:T-k,j},
 x_{d,k:T,j}
\right).
\]

If either lag slice has variance below \(10^{-12}\), mark missing.

Aggregate with equal document weight:

\[
R^{\mathrm{signed}}_j(k)
=
\frac{1}{|\mathcal D_{j,k}|}
\sum_{d\in\mathcal D_{j,k}}
 r^{\mathrm{signed}}_{j,d}(k).
\]

This must reproduce the original within-document profile.

---

## 16. Envelope Autocorrelation

For the primary squared envelope:

\[
r^{\mathrm{env}}_{j,d}(k)
=
\operatorname{corr}
\left(
 \widetilde q_{d,0:T-k,j},
 \widetilde q_{d,k:T,j}
\right).
\]

Aggregate with equal document weight:

\[
R^{\mathrm{env}}_j(k)
=
\frac{1}{|\mathcal D^{\mathrm{env}}_{j,k}|}
\sum_{d\in\mathcal D^{\mathrm{env}}_{j,k}}
 r^{\mathrm{env}}_{j,d}(k).
\]

Repeat for the log envelope:

\[
R^{\log\mathrm{env}}_j(k).
\]

Interpretation:

```text
high signed and high envelope:
payload and activity both persist

low signed and high envelope:
activity persists while signed payload does not

low signed and low envelope:
no detected persistent payload or activity

high signed and low envelope:
possible smooth low-amplitude drift, offset artifact, or estimator mismatch
```

Envelope autocorrelation is not interpreted as direct co-activation lift because it also depends on payload magnitudes and heavy tails.

---

## 17. Activity Gates

Run matched-occupancy and natural-threshold gates separately.

### 17.1 Matched 20% occupancy gate

For each document and probe, let \(\theta^{20}_{d,j}\) be the within-document 80th percentile of \(|x_{d,t,j}|\). Define:

\[
g^{20}_{d,t,j}
=
\mathbf 1\{|x_{d,t,j}|>\theta^{20}_{d,j}\}.
\]

This gate is primary for active-pair payload correlation because it supplies enough active pairs.

### 17.2 Matched 5% occupancy gate

Let \(\theta^{5}_{d,j}\) be the within-document 95th percentile. Define:

\[
g^{5}_{d,t,j}
=
\mathbf 1\{|x_{d,t,j}|>\theta^{5}_{d,j}\}.
\]

This gate is primary for high-amplitude co-activation specificity.

Quantile ties must be broken deterministically by stable token index so occupancy is exactly the requested count per document.

Matched-occupancy gates are comparison tools. They do not estimate natural duty cycle.

### 17.3 Natural-duty gates

Fit fixed train-scale thresholds:

\[
g^{\mathrm{nat},c}_{d,t,j}
=
\mathbf 1\{|x_{d,t,j}|>c\},
\qquad
c\in\{2.5,3.0\}.
\]

Primary natural threshold:

```text
c = 3.0
```

Sensitivity:

```text
c = 2.5
```

Define held-out natural duty cycle:

\[
d^{\mathrm{nat},c}_j
=
\mathbb E_{d,t}[g^{\mathrm{nat},c}_{d,t,j}].
\]

Report duty cycle by split and document.

---

## 18. Binary Co-Activation Lift

For gate family \(a\in\{5,20,\mathrm{nat}2.5,\mathrm{nat}3.0\}\), define lag-slice marginals within each document:

\[
p^{(0)}_{j,d,a}(k)
=
\frac{1}{T-k}
\sum_{t=0}^{T-k-1}g^a_{d,t,j},
\]

\[
p^{(k)}_{j,d,a}(k)
=
\frac{1}{T-k}
\sum_{t=0}^{T-k-1}g^a_{d,t+k,j},
\]

and joint activation probability:

\[
p^{(0,k)}_{j,d,a}(k)
=
\frac{1}{T-k}
\sum_{t=0}^{T-k-1}
 g^a_{d,t,j}g^a_{d,t+k,j}.
\]

Define per-document lift:

\[
L_{j,d,a}(k)
=
\frac{
 p^{(0,k)}_{j,d,a}(k)
}{
 p^{(0)}_{j,d,a}(k)
 p^{(k)}_{j,d,a}(k)
}.
\]

If either marginal is zero, mark missing.

Aggregate the log lift with equal document weight for numerical stability:

\[
\log L_{j,a}(k)
=
\frac{1}{|\mathcal D_{j,a,k}|}
\sum_{d\in\mathcal D_{j,a,k}}
\log L_{j,d,a}(k).
\]

Report:

\[
L_{j,a}(k)=\exp(\log L_{j,a}(k)).
\]

Interpretation:

```text
L = 1:
active endpoints co-occur at chance rate

L > 1:
activity clusters at lag k

L < 1:
activity is anti-clustered at lag k
```

Require at least 100 valid documents for pilot/full probe-level estimates.

---

## 19. Normalized Co-Activation Curve

Lift has baseline 1 rather than 0. Define excess lift:

\[
E_{j,a}(k)=L_{j,a}(k)-1.
\]

Define the zero-lag excess analytically from the gate duty cycle:

\[
E_{j,a}(0)
=
\frac{1}{d_{j,a}}-1.
\]

Define normalized co-activation:

\[
A_{j,a}(k)
=
\frac{E_{j,a}(k)}{E_{j,a}(0)}.
\]

Set:

\[
A_{j,a}(0)=1.
\]

This produces a decay curve with chance baseline 0, analogous to an autocorrelation curve.

The primary co-activation curve uses the matched 5% gate. The matched 20% and natural gates are required sensitivities.

---

## 20. Active-Pair Payload Correlation

For a gate \(g^a\), define active lag pairs:

\[
\mathcal I_{d,j,a,k}
=
\{t:g^a_{d,t,j}=1,\ g^a_{d,t+k,j}=1\}.
\]

If:

\[
|\mathcal I_{d,j,a,k}|<8,
\]

mark the document-probe-lag entry missing.

Otherwise compute:

\[
r^{\mathrm{active}}_{j,d,a}(k)
=
\operatorname{corr}
\left(
\{x_{d,t,j}\}_{t\in\mathcal I},
\{x_{d,t+k,j}\}_{t\in\mathcal I}
\right).
\]

Aggregate over documents with equal weight.

Primary gate:

```text
matched 20% occupancy
```

Sensitivity:

```text
matched 5% occupancy
natural 2.5-sigma
natural 3-sigma
```

Also report absolute payload correlation:

\[
R^{\mathrm{active,abs}}_{j,a}(k)
=
\operatorname{corr}
\left(
|x_t|,
|x_{t+k}|
\mid g_t=g_{t+k}=1
\right).
\]

Interpretation:

```text
high signed active-pair correlation:
related signed payload recurs

low signed but high absolute correlation:
payload magnitude recurs but sign changes

low signed and low absolute correlation:
activity recurs with changing or unrelated payload
```

Low active-pair correlation is not itself evidence of a communication bus. It only shows that activity persistence is not explained by a stable signed payload under this statistic.

---

## 21. Burst and Occupancy Summaries

For every gate family, identify maximal contiguous active runs within each document.

Report per probe:

- natural duty cycle;
- number of active runs per document;
- run-length median, Q90, and maximum;
- inter-run gap median and Q90;
- fraction of active tokens in runs of length at least 2, 4, 8, and 16;
- fraction of documents with at least one run;
- fraction of documents with at least two separated runs;
- active-position entropy over 32-token position bins.

These are characterization metrics, not primary decision statistics.

A direction that fires once per document can have high kurtosis but is not a repeatedly reused channel. The report must distinguish:

```text
one-shot sparse feature
single long episode
multiple recurring bursts
broad dense activity
```

---

## 22. Marginal Distribution Diagnostics

For each probe and split, report:

- mean;
- variance;
- skewness;
- excess kurtosis;
- median absolute deviation;
- Q95, Q99, and Q99.9 of \(|x|\);
- fraction of total squared energy in the top 1%, 5%, and 20% of positions;
- Gini coefficient of \(q=x^2\).

These diagnostics identify whether envelope effects are driven by isolated extreme spikes.

A high-envelope candidate must be reported alongside kurtosis and energy concentration. High kurtosis alone is not evidence of temporal memory.

---

## 23. Timescale Extraction

Maximum lag:

| Mode | Maximum lag |
|---|---:|
| smoke | 128 |
| pilot | 512 |
| full | 512 primary, 768 sensitivity |

For signed, squared-envelope, and log-envelope profiles:

1. Set the zero-lag value to 1.
2. Apply a centered moving average of width 5 for \(k\geq1\).
3. Clip signed and envelope correlation profiles to `[-1, 1]`.
4. Define the first \(1/e\) crossing.

\[
\tau^{e}_j
=
\min\{k\geq1:R^e_j(k)<1/e\}.
\]

For normalized co-activation:

\[
\tau^{\mathrm{coact},a}_j
=
\min\{k\geq1:A_{j,a}(k)<1/e\}.
\]

If no crossing occurs:

```text
tau = K + 1
right_censored = true
```

If fewer than 80% of lag points are valid:

```text
tau_valid = false
```

Do not force a timescale for active-pair payload correlation when support is sparse. Report the full valid-lag curve and a timescale only when at least 80% of lags through the crossing are valid.

---

## 24. Integrated Long-Lag Scores

Crossing times can be unstable for non-monotone profiles. Also compute integrated scores over frozen lags:

\[
\mathcal K
=
\{8,16,32,64,128\}.
\]

### Envelope long-lag mass

\[
S^{\mathrm{env}}_j
=
\sum_{k\in\mathcal K}
\max(R^{\mathrm{env}}_j(k),0).
\]

### Co-activation long-lag mass

\[
S^{\mathrm{coact},a}_j
=
\sum_{k\in\mathcal K}
\max(A_{j,a}(k),0).
\]

### Signed long-lag mass

\[
S^{\mathrm{signed}}_j
=
\sum_{k\in\mathcal K}
\max(R^{\mathrm{signed}}_j(k),0).
\]

Primary candidate selection uses both crossing times and integrated long-lag mass. No claim may depend on one unstable scalar alone.

---

## 25. Temporal Nulls and Controls

### 25.1 Full within-document token permutation

For every document, generate one permutation of token positions and apply it identically across all probes in that document.

This preserves:

- every projection value;
- every probe's marginal distribution;
- cross-probe values at a token only if the full token row is permuted jointly;
- document identity;
- token multiset.

It destroys sequential order.

Replicates:

| Mode | Screen | Candidate confirmation |
|---|---:|---:|
| smoke | 5 | 20 |
| pilot | 20 | 100 |
| full | 20 | 100 |

Run all signed, envelope, gate, co-activation, burst, and active-pair statistics on permuted sequences.

### 25.2 Block permutation

Partition each document into contiguous blocks and permute block order while preserving within-block token order.

Required block sizes:

```text
8
32
128
```

Interpretation:

```text
full permutation collapse:
activity depends on some sequential order

8-block survival but 32/128 collapse:
mostly local burst structure

128-block survival with full-permutation collapse:
long episodes survive local preservation
```

Block permutation is a sensitivity analysis, not a replacement for full permutation.

### 25.3 Matched random-gate null

For every document/probe/gate, sample random active positions with exactly the same active count as the real gate.

This preserves occupancy and destroys temporal placement.

Use the same random seeds across probes where feasible.

Report co-activation and burst statistics against this exact matched-sparsity null.

### 25.4 Episode sign-randomization diagnostic

Preserve the observed activity gate and envelope magnitudes but multiply each maximal active episode by an independent random sign.

This should:

- preserve envelope and co-activation statistics;
- reduce signed persistence when it is carried by consistent episode signs.

This is an implementation and interpretation diagnostic. It is not a corpus null.

### 25.5 Position-profile sensitivity

Using train documents, estimate the mean envelope by absolute token-position bin.

Primary bin width:

```text
32 tokens
```

Define train-fitted position residual:

\[
q^{\perp\mathrm{pos}}_{d,t,j}
=
q_{d,t,j}
-
\widehat{\mathbb E}_{\mathrm{train}}
[q_{d,t,j}\mid\operatorname{bin}(t)].
\]

Apply unchanged to validation and test and recompute envelope statistics.

Position control is a sensitivity because position structure can be genuine model behavior. A candidate that disappears entirely is labeled position-driven rather than intermittent-channel positive.

### 25.6 Token-class diagnostics

Using saved token IDs and the frozen tokenizer, classify tokens into coarse classes:

```text
alphabetic or word-like
numeric
punctuation-only
whitespace or newline
markup or code delimiter
URL-like
other
```

Report candidate activity enrichment by token class and the fraction of active positions dominated by one class.

Do not automatically exclude lexical candidates. Label them explicitly.

---

## 26. Probe Families and Controls

Evaluate all frozen probes.

Primary families:

```text
random residual directions
residual PCA directions
time-lagged residual directions
```

Required derived groups:

```text
top-31 previously persistent directions
low-lifetime time-lagged directions
random directions sampled inside the existing slow span, if projection artifacts exist
random directions inside matched PCA spans, if projection artifacts exist
```

The low-lifetime time-lagged group controls for artifacts of the temporal direction-fitting procedure.

The empirical random-direction distribution is the primary geometric null. All statistics, gates, norm controls, and permutations must be computed identically for random probes.

---

## 27. Primary Headline Estimands

The primary report must include, for every probe:

```text
tau_signed
S_signed

tau_env
S_env

tau_log_env
S_log_env

tau_coact_q95
S_coact_q95

tau_coact_q80
S_coact_q80

natural_duty_3sigma
natural_duty_2.5sigma

active_payload_r at lags 8, 16, 32, 64, 128
active_abs_payload_r at the same lags

permutation-collapse fractions
norm-controlled counterparts
kurtosis and burst summaries
```

The central visualization is:

\[
(\tau_{\mathrm{signed}},\tau_{\mathrm{env}})
\]

with:

- point color by probe family;
- point size by natural duty cycle;
- marker outline by permutation significance;
- separate raw and norm-controlled panels.

A second required visualization is:

\[
(S_{\mathrm{signed}},S_{\mathrm{coact},q95}).
\]

---

## 28. Validation Candidate Definition

A probe is a validation `LOW_SIGNED_HIGH_ACTIVITY_CANDIDATE` only if all conditions hold.

### 28.1 Weak signed persistence

At least one of:

\[
\tau^{\mathrm{signed}}_j
\leq
Q_{95}^{\mathrm{random}}
(\tau^{\mathrm{signed}}),
\]

or:

\[
S^{\mathrm{signed}}_j
\leq
Q_{95}^{\mathrm{random}}
(S^{\mathrm{signed}}).
\]

The report must state which branch admitted the candidate.

### 28.2 Strong activity persistence

Both:

\[
\tau^{\mathrm{env}}_j
>
Q_{95}^{\mathrm{random}}
(\tau^{\mathrm{env}}),
\]

and:

\[
S^{\mathrm{coact},q95}_j
>
Q_{95}^{\mathrm{random}}
(S^{\mathrm{coact},q95}).
\]

### 28.3 Sequence-order dependence

Candidate-confirmation permutation p-value:

\[
p_j
=
\frac{1+
\#\{b:S^{\mathrm{coact},q95}_{j,b}
\geq
S^{\mathrm{coact},q95}_{j,\mathrm{real}}\}}
{1+B}
\]

must pass Benjamini-Hochberg FDR:

```text
q <= 0.05
```

across all tested probes.

Also require median collapse across envelope and co-activation long-lag mass:

\[
C_j
=
1-
\frac{S_{j,\mathrm{perm\ median}}}
{S_{j,\mathrm{real}}}
\geq0.50.
\]

### 28.4 Global-amplitude robustness

Under norm-normalized projections, require either:

- the candidate remains above the random Q95 activity threshold; or
- at least 50% of its real-minus-random activity excess remains.

### 28.5 Nontrivial temporal extent

Require at least one positive norm-controlled excess at lag:

```text
32, 64, or 128
```

This prevents purely adjacent-token burst features from entering the headline long-range candidate set.

### 28.6 Intermittent occupancy label

A candidate receives the stronger `INTERMITTENT` label only when primary natural duty cycle satisfies:

\[
0.005
\leq
d^{\mathrm{nat},3.0}_j
\leq0.25.
\]

Candidates outside this band remain activity-persistent but are labeled:

```text
ultra-rare
or
dense/sign-changing
```

rather than intermittent.

---

## 29. Candidate Deduplication and Ranking

Deduplicate validation candidates with absolute cosine similarity threshold:

```text
0.95
```

Priority order:

1. larger norm-controlled co-activation long-lag excess;
2. larger raw co-activation long-lag excess;
3. larger envelope long-lag excess;
4. lower signed long-lag mass;
5. stable probe ID tie-break.

For candidate \(j\), define validation activity excess:

\[
e_j
=
\max\left(
S^{\mathrm{coact},q95}_{j,\mathrm{norm}}
-
\operatorname{median}_{b\in\mathrm{perm}}
S^{\mathrm{coact},q95}_{j,b,\mathrm{norm}},
0
\right).
\]

Rank candidates by \(e_j\).

Define:

\[
k^{\mathrm{int}}_{80}
\]

as the smallest \(k\) whose cumulative \(e_j\) reaches 80% of total positive candidate excess.

Also report activity-excess participation ratio:

\[
r_{\mathrm{int}}
=
\frac{(\sum_j e_j)^2}{\sum_j e_j^2}.
\]

These are probe-relative summaries, not intrinsic dimensions.

---

## 30. Frozen Test Evaluation

After validation selection, freeze:

- candidate probe IDs;
- candidate ordering;
- deduplication choices;
- \(k^{\mathrm{int}}_{80}\);
- all train-fitted transforms;
- all thresholds;
- random and permutation seeds used for test.

Evaluate the same probes on test.

For each validation candidate, report whether test retains:

1. weak signed persistence;
2. strong envelope persistence;
3. strong co-activation long-lag mass;
4. permutation FDR significance;
5. at least 50% permutation collapse;
6. norm robustness;
7. long-range excess at lag 32, 64, or 128;
8. occupancy class.

Primary stability summaries:

```text
fraction of validation candidates retaining all criteria
fraction retaining activity criteria regardless of signed threshold
validation-test Spearman for all headline metrics
median absolute metric change
candidate-set Jaccard when the rule is independently applied on test
```

The independent test-selected set is descriptive only and must not replace the frozen validation-selected set.

---

## 31. Candidate Geometry

Construct the Euclidean-orthonormal validation-selected candidate basis:

\[
Q_{\mathrm{int}}
=
\operatorname{orth}
([v_1,\ldots,v_{k^{\mathrm{int}}_{80}}]).
\]

Compare against:

```text
Q_slow_kstar
PCA-31
PCA-128
PCA-256
random matched-rank subspaces
```

Report:

- projection overlap;
- effective shared dimensions;
- principal-angle spectrum;
- median and worst principal angle;
- candidate-family composition;
- geometric participation ratio;
- activity-excess participation ratio.

Interpretation:

```text
high overlap with slow:
intermittent audit mainly recharacterizes the existing slow region

low matched-rank overlap but containment in broad PCA:
new temporal selection inside the broader high-variance substrate

low overlap with slow and broad PCA:
possible genuinely distinct residual region, requiring strong artifact checks
```

Do not run projection-collapse or causal claims in this audit.

---

## 32. Group-Level Statistical Tests

### 32.1 Candidate-rate enrichment

Compare the fraction satisfying the full validation candidate rule among:

```text
time-lagged
PCA
random
low-lifetime time-lagged
```

Use document bootstrap with the candidate rule recomputed for fixed probes within each bootstrap replicate.

Report rate differences and ratios with 95% confidence intervals.

### 32.2 Distributional enrichment

Compare families on:

- \(\tau_{\mathrm{env}}\);
- \(S_{\mathrm{env}}\);
- \(\tau_{\mathrm{coact},q95}\);
- \(S_{\mathrm{coact},q95}\);
- norm-controlled activity excess;
- natural duty cycle;
- active-pair payload correlation.

Use paired probe identity where the same probe has raw and norm-controlled variants. Do not treat probe directions as IID model samples; family CIs are descriptive over the frozen probe bank.

### 32.3 Random-direction calibration

The empirical random bank defines the ambient null. Report Q50, Q90, Q95, Q99, and maximum for every headline metric.

Do not convert random-direction z-scores into Gaussian p-values.

---

## 33. Bootstrap Confidence Intervals

Bootstrap over documents.

Replicates:

| Mode | Replicates |
|---|---:|
| smoke | 20 |
| pilot | 200 |
| full | 500 |

Primary seed:

```text
47
```

For every probe, compute bootstrap intervals for:

- signed profile at lags `[1,2,4,8,16,32,64,128,256,512]`;
- envelope profile at the same lags;
- co-activation profiles for q95 and q80 gates;
- signed, envelope, and co-activation timescales;
- integrated long-lag masses;
- natural duty cycle;
- active-pair payload correlations;
- permutation-collapse fraction.

For the frozen candidate set, bootstrap:

- median headline metrics;
- candidate retention rate;
- candidate-family composition;
- \(k^{\mathrm{int}}_{80}\);
- \(r_{\mathrm{int}}\);
- candidate-span overlap with slow and PCA bases.

Directions remain fixed inside the document bootstrap.

---

## 34. Primary Decision Labels

### `INTERMITTENT_CHANNEL_SIGNAL_POSITIVE`

Require all:

1. Input and signed-reproduction gates pass.
2. Norm-proxy health passes.
3. At least five deduplicated nonrandom validation candidates satisfy the full rule.
4. Their geometric effective rank is at least 3.
5. Nonrandom candidate rate exceeds random candidate rate, with document-bootstrap 95% CI for the difference above 0.
6. Median candidate permutation collapse is at least 0.50.
7. At least 60% of frozen validation candidates retain the activity-persistence, permutation, norm-robustness, and long-range criteria on test.
8. The test median norm-controlled activity excess is positive with document-bootstrap 95% CI above 0.

Interpretation:

> The existing residual probe bank contains a stable population of low-signed/high-activity directions whose activity is temporally clustered beyond random, permutation, and global-amplitude controls.

This label still does not establish memory content or causal use.

### `INTERMITTENT_CHANNEL_SIGNAL_SUGGESTIVE`

Use when:

- family-level activity enrichment is positive;
- some candidates survive controls;
- but count, rank, test retention, or norm robustness misses the positive gate.

### `ACTIVITY_PERSISTENCE_REPARAMETERIZES_SLOW`

Use in addition to positive or suggestive when candidate-span projection overlap with the existing slow span is high:

```text
mean squared containment >= 0.75
```

### `ACTIVITY_PERSISTENCE_GEOMETRICALLY_DISTINCT`

Use in addition to positive when:

- candidate-span containment in the slow span is below 0.25;
- overlap exceeds matched random controls;
- candidate test retention passes;
- broad PCA containment is reported.

This label means distinct from the measured slow span, not independent of all existing residual geometry.

### `TOKEN_LOCAL_ACTIVITY_ONLY`

Use when raw activity effects are positive but no norm-controlled excess remains at lags 32, 64, or 128, or block permutation shows all signal is explained inside 8-token blocks.

### `GLOBAL_AMPLITUDE_DRIVEN`

Use when raw activity effects are positive but norm normalization removes most family and candidate excess and the norm proxy is healthy.

### `POSITION_DRIVEN_ACTIVITY`

Use when activity persistence disappears under frozen position-profile residualization and candidate activity is strongly concentrated by absolute position.

### `NO_INTERMITTENT_SIGNAL_DETECTED`

Use when:

- implementation and power checks pass;
- no stable nonrandom candidate enrichment is observed;
- candidate rate is consistent with random controls;
- no held-out long-range activity excess survives permutation and norm controls.

This means no signal under the frozen probe bank and estimator. It does not show that optimized intermittent directions do not exist.

### `INTERMITTENT_AUDIT_INCONCLUSIVE`

Use when:

- projection tensors are incomplete;
- signed reproduction fails;
- random-bank norm proxy fails;
- too few documents support the statistics;
- threshold ties or missing values are widespread;
- confidence intervals are too broad.

---

## 35. Scientific Interpretation Matrix

| Signed persistence | Activity persistence | Natural duty | Active-pair payload | Interpretation |
|---|---|---|---|---|
| High | High | Dense or moderate | High | Broadcast or continuously readable state candidate |
| Low | High | Intermittent | Low | Reused intermittent-channel candidate with changing payload |
| Low | High | Intermittent | High | Sparse recurrence of a related payload |
| Low | High | Dense | Low | Sign-changing or magnitude-modulated dense channel |
| Low | Low | Ultra-rare | Undefined | One-shot sparse feature or insufficient support |
| Low | Low | Moderate | Low | Token-local feature or no temporal organization |
| High | Low | Any | Any | Drift, low-amplitude offset, oscillation, or estimator artifact; inspect profile |
| Any | High raw, low norm-controlled | Any | Any | Global residual-amplitude confound |
| Any | High only within 8-token blocks | Any | Any | Local burst feature rather than long-range channel |

Mechanistic language must follow the full row, not one scalar.

---

## 36. Required Tables and Figures

### Figures

1. Signed-timescale versus envelope-timescale scatter.
2. Signed long-lag mass versus co-activation long-lag mass.
3. Raw versus norm-controlled co-activation score.
4. Real versus permutation score for all probes.
5. Candidate profile cards showing signed, envelope, co-activation, and active-pair curves.
6. Duty-cycle distribution by probe family.
7. Burst-length and inter-burst-gap distributions.
8. Candidate activity heatmaps over example documents.
9. Block-permutation survival by block size.
10. Candidate-span principal-angle spectrum against slow and PCA spans.
11. Validation-test metric stability.
12. Random-direction null bands for every headline profile.

### Tables

1. Input and reproduction health.
2. Norm-proxy health.
3. Probe-family metric summaries.
4. Candidate-rate enrichment.
5. Validation candidate list and test retention.
6. Marginal distribution and burst diagnostics.
7. Candidate geometry overlaps.
8. Decision-rule checklist.
9. Limitations and unresolved confounds.

---

## 37. Required Artifacts

```text
analysis/intermittent_channel_audit/
  frozen_analysis_lock.json
  input_manifest.json
  input_validation.json
  signed_reproduction_summary.json

  preprocessing/
    probe_scales.npy
    natural_thresholds.json
    random_bank_norm_proxy_summary.json
    position_profiles.npz

  profiles/
    signed_profiles_{split}.npz
    envelope_profiles_{split}.npz
    log_envelope_profiles_{split}.npz
    coactivation_profiles_q95_{split}.npz
    coactivation_profiles_q80_{split}.npz
    coactivation_profiles_natural_{split}.npz
    active_payload_profiles_{split}.npz

  summaries/
    probe_summary_{split}.parquet
    burst_summary_{split}.parquet
    token_class_summary_{split}.parquet
    family_summary_{split}.parquet

  nulls/
    permutation_manifest.json
    permutation_probe_summary.parquet
    block_permutation_summary.parquet
    random_gate_summary.parquet
    episode_sign_randomization_summary.parquet

  candidates/
    validation_candidate_manifest.json
    validation_candidate_summary.parquet
    frozen_candidate_basis.npz
    test_candidate_retention.parquet
    candidate_geometry_summary.json
    candidate_principal_angles.npz

  uncertainty/
    bootstrap_probe_metrics.parquet
    bootstrap_candidate_metrics.parquet
    fdr_summary.json

  decision/
    intermittent_audit_decision.json
    estimator_handoff_manifest.json

  report/
    intermittent_channel_audit_report.md
```

Every artifact must include source hashes, config hash, code identity, split, estimator variant, gate variant, norm-control status, and random seed.

---

## 38. Unit and Integration Tests

Required tests:

### Preprocessing

- within-document demeaning yields zero mean within tolerance;
- train scale is fit only on train;
- held-out transforms do not refit;
- quantile gate occupancy is exact under ties;
- natural gate thresholds are frozen;
- random-probe norm proxy uses leave-one-out construction.

### Estimators

- independent Bernoulli gates return lift near 1;
- Markov-clustered gates return positive co-activation decay;
- reused bus with independent payload gives low signed and high activity persistence;
- constant payload with clustered gate gives high signed and high activity persistence;
- isolated spikes give high kurtosis but weak long-lag co-activation;
- alternating-sign bursts preserve envelope and reduce signed correlation;
- smoothing and crossing rules match the original estimator.

### Nulls

- full permutation preserves projection-value multisets exactly;
- joint token-row permutation preserves cross-probe token rows;
- block permutation preserves within-block order;
- matched random gates preserve active counts exactly;
- episode sign randomization preserves envelopes exactly.

### Geometry

- candidate basis is orthonormal;
- principal angles match direct SVD;
- random-subspace overlap matches expected \(k/d\) scale;
- deduplication is deterministic.

### End-to-end

- smoke run is deterministic under fixed seeds;
- rerunning with the same lock yields identical hashes;
- validation candidate IDs are unchanged when test artifacts are absent;
- test evaluation cannot alter candidate selection.

---

## 39. Synthetic Validation Suite

Before real-data interpretation, run synthetic projection generators with matched document length and probe count.

### Null A: IID Gaussian

\[
x_t\sim\mathcal N(0,1).
\]

Expected:

```text
low signed
low envelope
coactivation near chance
```

### Null B: Global amplitude modulation

\[
x_{t,j}=a_t\epsilon_{t,j}.
\]

Expected:

```text
raw envelope positive across many directions
norm-controlled envelope collapses
```

### Positive C: Broadcast state

\[
x_t=z_t+\epsilon_t,
\]

with slowly varying \(z_t\).

Expected:

```text
high signed
high envelope
high coactivation
```

### Positive D: Intermittent reused bus

\[
x_t=g_tz_t+\epsilon_t,
\]

where \(g_t\) is temporally clustered and \(z_t\) is independent across episodes.

Expected:

```text
low signed
high envelope
high coactivation
low active-pair payload correlation across episodes
```

### Positive E: Sparse recurring payload

Clustered \(g_t\) and episode-stable \(z_t\).

Expected:

```text
moderate or high signed conditional correlation
high envelope
high coactivation
```

### Null F: One-shot spikes

One random extreme spike per document.

Expected:

```text
high kurtosis
no robust long-range coactivation
poor test stability
```

The synthetic suite must pass before real candidate labels are emitted.

---

## 40. Execution Order

```text
1. Commit or record the current code identity.
2. Validate and hash all frozen projection and direction artifacts.
3. Write the frozen intermittent-audit lock.
4. Reproduce the original signed autocorrelation and timescales.
5. Fit train probe scales and natural thresholds.
6. Build and validate the random-bank residual-norm proxy.
7. Run the synthetic validation suite.
8. Compute validation signed, envelope, log-envelope, gate, co-activation, active-pair, and burst metrics.
9. Run screen permutations and block permutations.
10. Compute norm-controlled and position-controlled sensitivities.
11. Apply the frozen validation candidate rule.
12. Run 100-replicate candidate-confirmation permutations.
13. Freeze candidate IDs, ordering, k80, and candidate basis.
14. Compute candidate geometry against slow, PCA, and random spans.
15. Freeze the audit decision before test.
16. Evaluate the frozen pipeline and candidate set on test.
17. Bootstrap document-level uncertainty.
18. Produce the final decision artifact and report.
19. Freeze an estimator-handoff manifest only if the result is positive or suggestive.
```

---

## 41. Completion Criteria

The audit is complete when:

- all required projection inputs are validated;
- the original signed estimator is reproduced;
- the random-bank norm proxy passes or the report is marked inconclusive;
- synthetic null and positive controls pass;
- squared and log-envelope profiles are complete;
- matched-occupancy and natural gates are complete;
- co-activation and active-pair profiles are complete;
- full and block permutation controls are complete;
- norm-controlled sensitivity is complete;
- validation candidates are selected under the frozen rule;
- candidate confirmation permutations are complete;
- candidate geometry is compared to slow and PCA spans;
- frozen test characterization is complete;
- bootstrap intervals and FDR corrections are complete;
- one formal decision label is emitted;
- limitations are stated explicitly.

A null result is a valid completed result. Do not fit gated SFA merely because the audit was implemented.

---

## 42. Estimator Handoff Rule

Create:

```text
analysis/intermittent_channel_audit/decision/estimator_handoff_manifest.json
```

with:

```text
fit_new_intermittent_estimator:
true or false
```

Set `true` only for:

```text
INTERMITTENT_CHANNEL_SIGNAL_POSITIVE
or
INTERMITTENT_CHANNEL_SIGNAL_SUGGESTIVE
```

and only if at least one candidate family retains norm-controlled long-range activity excess on test.

The handoff must freeze:

- preferred envelope transform;
- preferred gate family;
- target lag set;
- matched-sparsity null;
- candidate occupancy regime;
- whether gated SFA, fourth-order optimization, or both should be implemented;
- the validation/test benchmark that a fitted estimator must beat;
- the existing slow-basis overlap comparator.

The next estimator must be fit on train documents only and evaluated against the frozen audit benchmark. It may not redefine activity after seeing held-out data.

---

## 43. Final Research Role

This audit determines whether the current program is missing a specific, architecture-plausible type of sequence-time structure:

\[
\boxed{
\text{the same residual direction is used in temporally structured episodes}
}
\]

without requiring:

\[
\boxed{
\text{the same signed payload to remain present across those episodes.}
}
\]

The original slow estimator asks:

\[
\text{Does the value stay similar?}
\]

This audit adds:

\[
\text{Does the channel stay or recur as active?}
\]

The two are complementary. A positive result would justify fitting for intermittent geometry. A negative result would narrow the immediate program toward asymmetric transport and direct attention/KV analyses rather than adding a more complicated sparse temporal optimizer.
