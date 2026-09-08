# State-Lifetime Geometry in the Residual Stream
## Experiment 1 Locked Implementation Specification

This document is the locked implementation contract for the residual-geometry experiment as currently implemented. If pilot results expose an evaluation flaw or required anti-confound, update this document first and treat the update as the new locked version before rerunning the stage. The current locked version includes the fixed held-out projection-collapse test, the fat-subspace diagnostic battery, test-split confirmation, attention-alignment residual-PCA robustness controls, and the ordered mechanistic follow-up plan.

---

## 0. One-Sentence Summary

This experiment tests whether the residual stream has a state-lifetime geometry: a structured distribution of temporal persistence across residual-stream directions, possibly concentrated in a coherent subspace and organized by attention-mediated context routing.

---

## 1. Scientific Object

The object of study is the residual stream \(r_{d,t}^{(\ell)}\).

For a residual probe direction \(v_j \in \mathbb{R}^{d_{\mathrm{model}}}\), define the scalar projection:

\[
P_{d,t,j} = v_j^\top r_{d,t}^{(\ell)}.
\]

The primary estimand is the distribution-relative residual projection timescale:

\[
\tau_{\mathrm{res},j}(D).
\]

For the primary run:

\[
\tau_{\mathrm{res},j}(D_{\mathrm{C4}})
\]

where \(D_{\mathrm{C4}}\) is broad English web text from C4.

The descriptive claim is:

\[
\{\tau_{\mathrm{res},j}(D_{\mathrm{C4}})\}_j
\text{ is structured and heavy-tailed across residual directions.}
\]

The stronger geometric claim is:

\[
\text{high-}\tau_{\mathrm{res}}\text{ directions concentrate in a finite or structured subspace.}
\]

The mechanistic alignment claim is:

\[
\text{persistent residual directions align preferentially with stable attention-output directions.}
\]

This experiment makes claims about residual-stream geometry in the model. The primary analysis uses residual probes: random directions, residual PCA directions, time-lagged directions, and attention-output PCA directions when available. Decoder-dictionary coverage can be added later as an optional comparison, but it is not an input to the residual-first basis.

---

## 2. Scope Boundary

The residual-first persistent subspace is constrxucted from residual-stream probes only. Optional decoder-dictionary coverage can be reported only after that subspace has been estimated. It answers:

\[
\text{does an external dictionary cover the residual-first persistent subspace?}
\]

It does not define the subspace.

---

## 3. Model, Layer, and Dataset

Primary run:

- model: `google/gemma-2-2b`
- model variant: non-instruct
- hook: `blocks.12.hook_resid_post`
- layer index: 12
- residual width: 2304

### 3.1 Architecture Verification

Before running smoke, `00_validate_env.py` must inspect the loaded `google/gemma-2-2b` model under the active runtime and record:

- number of layers;
- hidden size;
- number of attention heads;
- GQA/query-group count if available;
- max position/context length;
- sliding-window size;
- attention logit soft-capping or final-logit soft-capping settings when exposed;
- per-layer attention type when exposed by the implementation;
- whether the target hook layer `blocks.12.hook_resid_post` follows a local/sliding-window or global attention block under the active runtime.

Save:

- `configs/resolved_model_architecture.json`

The report must state whether layer 12 is local/sliding-window or global under the actual implementation used. For 1024-token contexts, if the sliding-window size exceeds or equals 1024, local layers still have access to the full experimental window. In that case, the local/global distinction affects architectural interpretation more than the primary autocorrelation estimator. Do not assume parity from documentation or layer numbering alone; record what the loaded model exposes.

Primary corpus:

- dataset: `allenai/c4`
- config: `en`
- split: `validation`
- streaming: true

Document processing:

1. Remove empty or whitespace-only examples.
2. Tokenize with the Gemma 2 tokenizer.
3. Keep examples with at least 1024 tokens before truncation.
4. Truncate to exactly 1024 tokens.
5. Do not include BOS. Use `add_special_tokens=False`.
6. Save ordered contexts with stable `context_id = sha256(raw_text)`.

Dataset sizes:

| Mode | Documents | Max scan rows | Purpose |
|---|---:|---:|---|
| smoke | 128 | 10,000 | shape tests and artifact checks |
| pilot | 5,000 | 500,000 | first residual-geometry result |
| full | 50,000 | 2,000,000 | final main run |

Primary window is 1024 tokens. Do not run a separate 512-token robustness comparison for B1. The probe directions are fit on 1024-token train documents, so refitting or re-evaluating the pipeline on truncated 512-token contexts is not a clean window-length comparison and requires a full extra forward pass. If lag-window sensitivity is needed, compute it from the existing autocorrelation profiles by re-extracting \(\tau_{\mathrm{within}}\) under smaller maximum-lag cutoffs or by comparing decay curves over shared lag ranges. This uses the same residual projection artifacts and does not change B1 gates.

Context split:

- train: 80%
- validation: 10%
- test: 10%
- split seed: 2

Train plus validation are used for pilot decisions. Test is held for final confirmation after the pipeline is frozen.

---

## 4. Repository Placement

This public artifact repository contains the residual-geometry implementation
extracted from the larger research workspace.

Required residual-geometry layout:

```text
src/persistent_state/
  __init__.py
  config/schema.py
  config/loader.py
  data/contexts.py
  data/artifact_store.py
  residuals/probes.py
  residuals/pca.py
  residuals/time_lagged.py
  residuals/projections.py
  autocorr/estimators.py
  subspace/projection.py
  subspace/attention_alignment.py
  reporting/residual_report.py

scripts/persistent_state/residual_geometry/
  00_validate_env.py
  01_build_context_pool.py
  02_compute_residual_probes.py
  03_compute_residual_autocorr.py
  04_residual_subspace_pilot.py
  05_projection_collapse.py
  06_attention_alignment.py
  07_make_residual_report.py

configs/persistent_state/residual_geometry/
  smoke.yaml
  pilot.yaml
  full.yaml

tests/persistent_state/residual_geometry/
```

Reuse existing model loading, tokenizer handling, context-pool style, artifact conventions, logging, and device helpers where possible.

---

## 5. Residual Probe Families

### 5.1 Random Residual Directions

Sample random unit directions:

\[
u_j \sim \mathcal{N}(0, I), \quad u_j \leftarrow u_j / \|u_j\|_2.
\]

Counts:

| Mode | Directions |
|---|---:|
| smoke | 64 |
| pilot | 512 |
| full | 2,048 |

Save:

- `residual_probes/directions/random_residual.npz`

### 5.2 Residual PCA Directions

Fit randomized PCA on sampled residual stream positions.

Sample caps:

| Mode | Token positions | Components |
|---|---:|---:|
| smoke | 100,000 | 64 |
| pilot | 1,000,000 | 256 |
| full | 5,000,000 | 512 |

Seed: 4.

Save:

- `residual_probes/directions/residual_pca.npz`
- `residual_probes/pca_fit_summary.json`

### 5.3 Attention-Output PCA Directions

This is not required for the first B1 pilot. Run it only after random, PCA, or time-lagged residual probes show a positive or suggestive signal.

For each layer/head \(h\), collect attention output vectors at the residual stream write point and fit PCA. Keep the top \(m=8\) directions per head.

Save:

- `residual_probes/directions/attention_head_pca.npz`
- `subspace/head_output_pca.parquet`

### 5.4 Time-Lagged Residual Directions

This is a primary Experiment 1 probe family. Random probes ask whether persistence is broad. PCA probes ask whether persistence is visible in high-variance residual directions. Time-lagged probes directly ask for directions whose residual projections are stable across token offsets.

PCA is not guaranteed to find temporal directions because PCA optimizes variance:

\[
\max_{\|v\|=1} \operatorname{Var}(v^\top r_t),
\]

while the residual-geometry target is closer to:

\[
\max_{\|v\|=1} \operatorname{corr}(v^\top r_t, v^\top r_{t+k}).
\]

A real persistent direction can be moderate- or low-variance and therefore rank poorly under PCA. Time-lagged probes close this loophole.

Fit time-lagged directions on the train split only. Evaluate their autocorrelation on validation and test splits exactly like random and PCA probes. Do not fit and evaluate time-lagged directions on the same documents for headline results.

Let centered residuals be:

\[
\bar r_t = r_t - \mu.
\]

Use the global train-token mean:

\[
\mu =
\mathbb{E}_{d,t \in \mathrm{train}}
\left[
r_{d,t}^{(\ell)}
\right].
\]

Do not use validation or test tokens to estimate \(\mu\) for PCA or time-lagged fitting. Position-wise centering is a sensitivity analysis because position effects can be real residual-stream structure; if run, report whether it changes the probe-family timescale rankings.

Estimate covariance:

\[
\Sigma_0 =
\mathbb{E}
\left[
\bar r_t \bar r_t^\top
\right]
\]

and lagged covariance:

\[
\Sigma_k =
\mathbb{E}
\left[
\bar r_t \bar r_{t+k}^\top
\right].
\]

Use a symmetrized multi-lag covariance:

\[
\Sigma_{\mathrm{lag}}
=
\sum_{k \in K_{\mathrm{lag}}}
w_k
\frac{\Sigma_k + \Sigma_k^\top}{2}.
\]

Solve the regularized generalized eigenproblem:

\[
\Sigma_{\mathrm{lag}} v
=
\lambda
(\Sigma_0 + \epsilon I)v.
\]

Use ridge:

\[
\epsilon = 10^{-4} \cdot \frac{\operatorname{tr}(\Sigma_0)}{d_{\mathrm{model}}}
\]

Compute the condition number of \(\Sigma_0 + \epsilon I\) after applying the default \(\epsilon\). If the condition number exceeds \(10^4\), increase \(\epsilon\) by doubling until the condition number falls below \(10^4\) or until:

\[
\epsilon > 10^{-1} \cdot \frac{\operatorname{tr}(\Sigma_0)}{d_{\mathrm{model}}}.
\]

Record the resolved \(\epsilon\) and the final condition number in `residual_probes/time_lagged_fit_summary.json`. If the condition number cannot be brought below \(10^4\) within this range, proceed with the smallest \(\epsilon\) that was attempted and add:

```json
{"condition_number_warning": true}
```

to the fit summary. Report the resolved \(\epsilon\) and condition number in the final report.

Ridge-stability diagnostics. For the time-lagged generalized eigenproblem, record the full ridge-resolution trace:

- initial \(\epsilon\);
- final \(\epsilon\);
- number of doublings;
- initial condition number;
- final condition number;
- final \(\epsilon\) as a fraction of \(\operatorname{tr}(\Sigma_0) / d_{\mathrm{model}}\);
- top generalized eigenvalues;
- fraction of output directions passing the positive-validation-persistence filter.

Define:

```text
epsilon_scale = epsilon_final / (trace(Sigma0) / d_model)
```

Interpretation:

- `epsilon_scale <= 1e-3`: well-conditioned or lightly regularized fit.
- `1e-3 < epsilon_scale <= 1e-2`: moderately regularized; report normally with caveat.
- `1e-2 < epsilon_scale <= 1e-1`: heavily regularized; time-lagged probe results are valid as stabilized probes but should not be overinterpreted as precise generalized eigen-directions.
- `epsilon_scale > 1e-1` or condition number remains above threshold: mark `time_lagged_fit_unstable = true`; do not make a level-3 autocorrelation-optimized-direction claim unless the time-lagged probes still beat random/PCA on held-out validation and pass a ridge sensitivity check.

Residual covariance spectrum diagnostics. For PCA and time-lagged fitting, record diagnostics of the train-token residual covariance used for \(\Sigma_0\):

- top 128 eigenvalues of \(\Sigma_0\), or all available eigenvalues if fewer than 128 are fit;
- explained-variance curve for the fitted PCA/whitening subspace;
- effective rank:

\[
r_{\mathrm{eff}}
=
\frac{
(\operatorname{tr}\Sigma_0)^2
}{
\operatorname{tr}(\Sigma_0^2)
};
\]

- participation ratio within the fitted whitening subspace;
- fraction of variance explained by the top 1, 5, 10, 50, and 100 PCs;
- condition number before ridge within the fitted/whitened subspace;
- condition number after ridge;
- cosine overlap between top time-lagged directions and top residual PCA directions.

Save these diagnostics in:

- `residual_probes/residual_covariance_spectrum.json`
- `residual_probes/residual_covariance_eigenvalues.npy`

Smoke uses these only as sanity checks. Pilot uses them for interpretation. Do not use smoke spectrum diagnostics to tune scientific thresholds unless there is an obvious numerical failure.

Run a ridge sensitivity check on pilot if `epsilon_scale > 1e-2`:

\[
\epsilon
\in
\{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}
\cdot
\frac{\operatorname{tr}(\Sigma_0)}{d_{\mathrm{model}}}.
\]

Compare validation upper-tail timescale rankings and subspace overlap between top time-lagged directions across \(\epsilon\) values. Save the sensitivity results in `residual_probes/time_lagged_ridge_sensitivity.parquet` and summarize them in `residual_probes/time_lagged_fit_summary.json`.

For numerical stability, fit the generalized eigenproblem in a PCA-whitened residual subspace:

| Mode | Whitening PCs | Output time-lagged directions |
|---|---:|---:|
| smoke | 128 or all available | 32 |
| pilot | 512 | 256 |
| full | 1,024 | 512 |

Map the resulting directions back to residual space and normalize to unit norm.

Lag sets:

| Mode | \(K_{\mathrm{lag}}\) |
|---|---|
| smoke | `[8, 16, 32]` |
| pilot/full | `[8, 16, 32, 64, 128]` |

Default weights:

\[
w_k = 1.
\]

Long-lag-weighted variants are sensitivity analyses, not the primary pilot.

Record generalized eigenvalues in `residual_probes/time_lagged_fit_summary.json`. Time-lagged directions with strong negative or oscillatory validation autocorrelation are not eligible for \(G_k\). A time-lagged probe is eligible only if:

\[
\operatorname{median}_{k \in \{8,16,32\}}
R^{\mathrm{within}}_j(k)
>
0.
\]

This filter is applied on validation autocorrelation, not on train fit statistics. Anti-persistent or sign-changing modes may be scientifically interesting, but they are not counted as persistent-state directions in the first implementation.

Save:

- `residual_probes/directions/time_lagged_residual.npz`
- `residual_probes/time_lagged_fit_summary.json`
- `residual_probes/time_lagged_ridge_sensitivity.parquet`, when required

## 6. Residual Projection Artifacts

For each split and probe family, compute:

\[
P_{d,t,j}=v_j^\top r_{d,t}.
\]

Store projections as chunked arrays:

```text
residual_probes/projections/
  train/
    chunk_00000.npz
  val/
  test/
```

Each chunk contains:

- `context_ids`: string array, shape `(B,)`
- `tokens`: int array, shape `(B, 1024)`
- `probe_ids`: string array, shape `(P,)`
- `probe_family`: string array, shape `(P,)`
- `projections`: float16 or float32 array, shape `(B, 1024, P)`

Use float32 for PCA fitting and projection accumulation when feasible. Stored projections may be float16 if the autocorrelation implementation is numerically stable after casting to float32 during statistics.

All metrics must be computed streaming over chunks.

---

## 7. Residual Autocorrelation Estimators

Let \(P_{d,t,j}\) be a residual projection for document \(d\), token position \(t\), and probe \(j\), with \(T=1024\).

Maximum lag:

| Mode | \(K\) |
|---|---:|
| smoke | 128 |
| pilot | 512 |
| full | 512 primary, 768 sensitivity |

### 7.1 Raw Per-Document Autocorrelation

\[
r^{\mathrm{raw}}_{j,d}(k)
=
\mathrm{corr}
\left(
P_{d,0:T-k,j},
P_{d,k:T,j}
\right).
\]

Edge cases:

- If either lagged vector has variance below \(10^{-12}\), mark the document-probe-lag value missing.
- Aggregate with equal document weight over non-missing values.
- Require at least 100 valid documents for pilot/full probe-level estimates.

### 7.2 Within-Document Demeaned Autocorrelation

For each document and probe:

\[
\tilde P_{d,t,j}
=
P_{d,t,j}
-
\frac{1}{T}\sum_{u=0}^{T-1} P_{d,u,j}.
\]

Compute the same lag correlation using \(\tilde P\).

This is the primary residual-probe autocorrelation estimator because it removes document-level offsets.

### 7.3 Document-Permutation Control

For each document and probe, randomly permute token positions within the document. Preserve the projection value multiset exactly.

Replicates:

| Mode | Replicates |
|---|---:|
| smoke | 5 |
| pilot | 20 |
| full | 20 |

What this control rules out and does not rule out. For dense residual projections, within-document permutation destroys token-sequential order. A reduction in `tau_within` under permutation is therefore evidence that the measured persistence depends on ordered context rather than only on the document-level multiset of projection values. It does not distinguish between:

1. the model actively maintaining a residual direction over token positions;
2. the corpus having locally coherent semantic structure that the residual stream passively reflects.

Interpret the permutation control as follows:

- If `tau_within` does not drop under permutation: persistence may be mostly document-level offset or token multiset structure. This is a strong negative signal for ordered-context persistence.
- If `tau_within` drops moderately or strongly: persistence depends on ordered local context. This is necessary for the phenomenon of interest but not sufficient to distinguish corpus coherence from model-maintained state.
- If `tau_within` drops by more than 50%: persistence is strongly sequence-order-dependent. Report this explicitly, but do not interpret it as evidence against model-maintained state by itself.

The permutation-control result must be stated explicitly in the report alongside the primary timescale results. Do not report timescale distributions without their permutation-control reduction fraction.

---

## 8. Residual Timescale Extraction

For estimator \(e \in \{\mathrm{raw}, \mathrm{within}\}\):

\[
R^e_j(k)=\mathbb{E}_d[r^e_{j,d}(k)].
\]

Smooth \(R_j(k)\):

- centered moving average of width 5 for \(k \geq 1\);
- leave \(R_j(0)=1\);
- clip to `[-1, 1]`.

Define:

\[
\tau^e_{\mathrm{res},j}
=
\min\{k \geq 1: R^e_j(k) < 1/e\}.
\]

If no lag crosses \(1/e\) within the primary pilot/full maximum lag \(K=512\):

```text
tau = K + 1
right_censored = true
```

If fewer than 80% of lags have valid estimates:

```text
tau_valid = false
```

High-censoring check. After computing probe timescales on the pilot, record the right-censoring rate separately for each probe family. If more than 30% of non-random probes are right-censored at \(K=512\), promote the 768-lag sensitivity run from optional to required before any subspace or dimensionality claim is made, and label all dimensionality estimates as lower bounds in the report. If more than 30% of non-random probes remain right-censored at \(K=768\), report the experiment as window-limited at this layer and corpus and stop before projection collapse.

The expectation is that 1024-token documents with \(K=512\) lags will resolve most persistence in the range of scientific interest. Substantial right-censoring at this window would itself be a notable finding about the timescale distribution.

Save:

- `residual_probes/autocorr/profiles_{split}_{probe_family}_{estimator}.npz`
- `residual_probes/autocorr/probe_timescales.parquet`
- `residual_probes/autocorr/residual_probe_summary.json`

---

## 9. Bootstrap Confidence Intervals

Bootstrap over documents, not tokens.

Replicates:

| Mode | Replicates |
|---|---:|
| smoke | 20 |
| pilot | 200 |
| full | 200 |

Seed: 13.

For each probe and estimator, compute bootstrap CIs for:

- \(R_j(k)\) at lags `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768]` where available;
- \(\tau_{\mathrm{res},j}\);
- right-censoring rate.

Clip the lag list to the mode-specific maximum \(K\). For example, smoke mode reports only lags up to 128.

Save:

- `residual_probes/autocorr/bootstrap_probe_timescales.parquet`

---

## 10. B1 Residual-Probe Pilot Decision

This is the first important decision point. It asks whether the residual stream has visible temporal geometry under the residual-probe families.

Provisional positive criteria:

1. At least 90% of attempted random, PCA, and time-lagged probes have valid `tau_within`.
2. \(Q_{90}(\tau_{\mathrm{within}})\) over PCA probes exceeds random-direction \(Q_{90}\) by at least 20%, or the combined eligible residual-probe distribution has \(Q_{95} / Q_{50} \geq 2\).
3. \(Q_{90}(\tau_{\mathrm{within}})\) over time-lagged probes exceeds random-direction \(Q_{90}\) by at least 30% on validation data.
4. The cleanest positive pattern is:

\[
\tau_{\mathrm{time\mbox{-}lagged}}
>
\tau_{\mathrm{PCA}}
>
\tau_{\mathrm{random}}
\]

in upper-tail quantiles such as \(Q_{90}\) or \(Q_{95}\).
5. Top persistent probe decay curves remain above the random median decay curve for at least two nonzero lags among `[16, 32, 64, 128]`.
6. Within-document demeaned persistence remains visible.
7. Document-permutation controls reduce the top-probe median `tau_within`, showing that persistence depends on ordered context rather than only document-level projection multisets. A reduction above 50% is recorded as strong sequence-order dependence, not as a negative result by itself.

Interpretation table:

| Residual result | Interpretation |
|---|---|
| random, PCA, and time-lagged probes all weak | no visible residual state-lifetime structure at this layer/corpus/probe budget |
| PCA upper tail exceeds random upper tail | structured high-variance residual directions carry longer-lived information |
| time-lagged upper tail exceeds PCA and random | persistent directions exist and are best found by optimizing temporal stability |
| time-lagged strong but PCA weak | persistent directions may be moderate- or low-variance and not PCA-visible |
| random probes also have a heavy upper tail | persistence may be broad or diffuse rather than PCA-concentrated |
| within-document signal does not drop under permutation | apparent persistence is likely document-level multiset structure |

This pilot can be positive, negative, or suggestive. A suggestive result is enough to run the residual-first subspace pilot.

---

## 11. Residual-First Persistent Subspace

Construct the residual-first persistent probe set \(G_k\) from eligible residual probes.

Probe families eligible for \(G_k\):

- random residual directions;
- residual PCA directions;
- time-lagged residual directions;
- attention-output PCA directions, when available.

Minimum residual-probe coverage:

| Mode | Random directions | PCA directions | Time-lagged directions | Minimum valid eligible probes after dedup |
|---|---:|---:|---:|---:|
| smoke | 64 | optional 64 | optional 32 | 64 |
| pilot | 512 | 256 | 256 | 640 |
| full | 2,048 | 512 | 512 | 1,280 |

Deduplicate directions with absolute cosine similarity above 0.95. Deduplication is performed in a fixed priority order:

1. time-lagged residual directions;
2. residual PCA directions;
3. attention-output PCA directions, when available;
4. random residual directions.

Within each family, process probes in descending \(\tau^{\mathrm{within}}_{\mathrm{res},j}\). This keeps autocorrelation-optimized directions when they rediscover a high-variance or generic direction, and makes the \(G_k\) family composition and dimensionality breakdown deterministic.

Define \(G_k\) as the top \(k\) valid eligible probes by \(\tau^{\mathrm{within}}_{\mathrm{res},j}\) after deduplication. Orthonormalize with QR or SVD:

\[
Q_k^{\mathrm{res}} = \mathrm{orth}(G_k).
\]

Run \(k\) grid:

| Mode | k values |
|---|---|
| smoke | `[8, 16, 32]` |
| pilot | `[8, 16, 32, 64, 128, 256]` |
| full | `[8, 16, 32, 64, 128, 256, 512, 1024]` |

Save:

- `subspace/projection_bases/residual_first_k{k}.npz`
- `subspace/residual_probe_coverage.json`

Every saved residual-first basis artifact must include:

- the orthonormal basis;
- the exact integer \(k\);
- the ordered source probe IDs and probe families;
- the validation ranking rule and deduplication threshold;
- the fit/evaluation provenance for each source probe family.

The source directions are not necessarily generalized eigenvectors. \(G_k\) may contain
time-lagged, PCA, attention-output PCA, and random residual probes. Use the neutral term
`source probes` unless referring specifically to time-lagged generalized eigenvectors.

---

## 12. Dimensionality Estimate

Define lifetime excess over the random-direction median:

\[
e_j =
\max
\left(
\tau^{\mathrm{within}}_{\mathrm{res},j}
-
\mathrm{median}_{u \in \mathrm{random}}
\tau^{\mathrm{within}}_{\mathrm{res},u},
0
\right).
\]

Sort eligible residual probes by \(\tau^{\mathrm{within}}_{\mathrm{res},j}\). Define `k_80pct_lifetime_excess` as the smallest \(k\) whose cumulative \(e_j\) reaches 80% of the cumulative excess at \(k_{\max}\).

Probe-family dependence. The dimensionality estimate depends on which probe families dominate the top of the \(\tau\) ranking. If time-lagged probes account for more than 60% of the top-\(k\) probes at `k_80pct_lifetime_excess`, report the dimensionality estimate as `time_lagged_probe_dominated` and also compute the estimate separately for each probe family:

- `k_80pct_random_only`: cumulative lifetime excess over random probes only.
- `k_80pct_pca_only`: cumulative lifetime excess over PCA probes only.
- `k_80pct_time_lagged_only`: cumulative lifetime excess over time-lagged probes only.

The family-specific estimates answer different questions. The random-only estimate asks how diffuse persistence is in generic directions. The time-lagged estimate asks how many autocorrelation-optimized directions are needed to account for most persistent variation. The PCA estimate asks how many high-variance directions contribute. Report all three when available. The primary headline dimensionality is the combined estimate, but the family breakdown must appear in the report.

Also estimate an elbow with a two-line piecewise linear fit to cumulative lifetime excess.

For the ordered top-\(k_\star\) source-probe pool, also report a slowness-concentration
participation ratio:

\[
r_{\mathrm{slow}}
=
\frac{
\left(\sum_{j=1}^{k_\star} e_j\right)^2
}{
\sum_{j=1}^{k_\star} e_j^2
}.
\]

Report this next to the geometric participation-ratio rank of the same ordered
source-probe pool:

\[
r_{\mathrm{geom}}
=
\frac{
\left(\sum_i \sigma_i(G_{k_\star})^2\right)^2
}{
\sum_i \sigma_i(G_{k_\star})^4
}.
\]

These answer different questions. \(r_{\mathrm{geom}}\) measures how many distinct
residual-space directions the selected source probes span. \(r_{\mathrm{slow}}\)
measures how many equally weighted directions would produce the observed
concentration of lifetime excess. A source pool may be geometrically high-rank while
its slowness is concentrated in a much smaller effective core.

Report:

- `k_80pct_lifetime_excess`;
- `k_80pct_random_only`, when available;
- `k_80pct_pca_only`, when available;
- `k_80pct_time_lagged_only`, when available;
- `k_elbow`;
- `top_kstar_geometric_participation_ratio`;
- `top_kstar_slowness_participation_ratio`;
- top-\(k\) probe-family composition at `k_80pct_lifetime_excess`;
- valid probe count;
- deduplicated probe count;
- right-censored probe fraction.

Do not overinterpret any exact integer. `k_80pct_lifetime_excess`,
`top_kstar_geometric_participation_ratio`, and
`top_kstar_slowness_participation_ratio` are not intrinsic dimensions of the model.
They are probe-relative summaries exposed by the chosen probe families, scoring
rule, corpus, layer, and lag window. The scientific object is whether persistence
appears concentrated in a small core, tens of directions, hundreds of directions,
or diffusely spread across many directions under this measurement procedure.

Save:

- `subspace/dimensionality_summary.json`

Also construct and save the exact headline basis:

\[
k_\star = k_{\mathrm{80pct\_lifetime\_excess}},
\qquad
Q_{k_\star}^{\mathrm{res}} = \operatorname{orth}(G_{k_\star}).
\]

Save:

- `subspace/projection_bases/residual_first_kstar.npz`

This artifact must include `k_star` and the ordered source probe IDs and families. Do
not silently substitute the nearest configured projection-collapse grid value. For
example, if \(k_\star=31\), a saved `residual_first_k32.npz` artifact is not an
acceptable substitute for `residual_first_kstar.npz`.

---

## 13. Projection-Collapse Test

For each residual vector \(r_{d,t}\), project out the residual-first basis:

\[
r^{\perp k}_{d,t}
=
r_{d,t}
-
Q_k^{\mathrm{res}}(Q_k^{\mathrm{res}})^\top r_{d,t}.
\]

Then recompute dense residual-probe autocorrelation before and after projection.

Before constructing \(G_k\), draw held-out evaluation probe sets:

1. \(J_{\mathrm{random}}\): fresh random unit directions with a seed distinct from the main random-probe seed. Use 64 directions in smoke, 128 directions in pilot, and 256 directions in full mode by default. These directions are independent of the random probes used for \(G_k\) construction in each mode. They are never ranked or selected on \(\tau\).
2. \(J_{\mathrm{lag\text{-}heldout}}\): held-out time-lagged directions fit on disjoint train shards or disjoint train documents not used to fit the time-lagged directions eligible for \(G_k\). Use 64 directions in smoke, 128 in pilot, and 256 in full when enough documents are available. These directions are evaluated on the same validation/test splits as other probes and are never eligible for \(G_k\).

Fix both held-out sets at this point and use them for all \(C(k)\) evaluations across all basis choices: residual-first, PCA, and random control. Neither held-out set may overlap with the \(G_k\) candidate pool. Enforce this by drawing or fitting held-out evaluation probes before any \(G_k\)-eligible probe is sampled or fit, using separate RNG streams and disjoint fit-document assignments.

Rationale: \(J_{\mathrm{random}}\) avoids circularity and tests whether projection collapse has a global or diffuse effect on generic residual directions. But if the true persistent subspace is compact, random directions may have little mass in that subspace. \(J_{\mathrm{lag\text{-}heldout}}\) tests whether the discovered residual-first basis generalizes to newly fit persistent directions that were not selected into \(G_k\). Report both. Treat random-\(J\) collapse as evidence about global coverage and held-out-time-lagged collapse as the sharper test of compact persistent-subspace recovery.

Primary projection-collapse metric:

\[
C(k)
=
1
-
\frac{
\mathrm{median}_{j \in J_\star}
\left[
\tau^{\mathrm{within}}_{\mathrm{res},j}(r^{\perp k})
\right]
}{
\mathrm{median}_{j \in J_\star}
\left[
\tau^{\mathrm{within}}_{\mathrm{res},j}(r)
\right]
},
\]

where \(J_\star\) is either \(J_{\mathrm{random}}\) or \(J_{\mathrm{lag\text{-}heldout}}\). Report \(C_{\mathrm{random}}(k)\) and \(C_{\mathrm{lag\text{-}heldout}}(k)\) separately.

Controls:

- project out top \(k\) PCA directions by variance;
- project out \(k\) random orthonormal directions.

Positive projection-collapse evidence:

- held-out time-lagged collapse is nontrivial: \(C_{\mathrm{lag\text{-}heldout}}(k_{\max}) \geq 0.20\);
- residual-first projection collapse exceeds random-basis collapse by at least 0.10 absolute on \(J_{\mathrm{lag\text{-}heldout}}\);
- random-direction collapse \(C_{\mathrm{random}}(k)\) is reported as a global/diffuse coverage statistic, not required to exceed 0.20 for a compact-subspace claim;
- residual-first projection selectively reduces high-lifetime probe timescales more than low-lifetime probe timescales within the held-out evaluation sets.

Save:

- `subspace/projection_autocorr.parquet`
- `subspace/projection_collapse_summary.json`
- `subspace/heldout_projection_eval_probes.npz`
- `subspace/projection_collapse_candidate_time_lagged.npz`
- `subspace/projection_collapse_bases/residual_first_k{k}.npz`

The saved projection-collapse basis artifacts must include the ordered source probe
IDs and families used to construct each basis. These artifacts record the bases used
for the projection-collapse result. They are distinct from the exact headline
\(Q_{k_\star}^{\mathrm{res}}\) artifact used by the fat-subspace diagnostic below.

### 13.1 Test-Split Projection-Collapse Confirmation

After a positive or suggestive validation projection-collapse result, rerun projection collapse on the held-out test split while reusing the already fixed candidate basis probes and held-out evaluation probes. Do not refit \(G_k\)-candidate probes or held-out evaluation probes for the test confirmation. The purpose is:

\[
\text{directions selected/fixed before test}
\rightarrow
\text{same projection-collapse behavior on test}.
\]

The implementation must distinguish output overwrite from probe refitting. Recomputing `projection_autocorr.parquet` for `--splits test` may overwrite the output tables, but it must load existing:

- `subspace/projection_collapse_candidate_time_lagged.npz`;
- `subspace/heldout_projection_eval_probes.npz`;

unless an explicit refit flag is passed. In the script interface, `--overwrite` may recompute output tables, while `--refit-probes` is required to refit candidate or held-out probe directions.

---

### 13.2 Fat-Subspace Diagnostic Battery

Projection collapse establishes that removing a recovered basis reduces the
timescales of independently fit held-out persistent probes. It does not establish
that a generic direction inside the headline span is slow. Pairwise nonredundancy and
high effective rank also do not answer that question.

This diagnostic distinguishes:

1. a fat full candidate pool, where generic uniformly sampled directions inside
   \(Q_{k_\star}^{\mathrm{res}}\) remain slow;
2. a hierarchical region, where generic directions are slower than ambient random
   directions but preferred slow axes remain;
3. a thin frame-dependent result, where the selected source probes are slow but
   generic rotations inside their span are not;
4. shrinkage or selection overfit, where the selected source probes themselves do
   not remain slow on untouched test documents;
5. a smaller effective slow core, where the random-in-span curve has a stable
   plateau over a nested range and held-out marginal source-probe timescales enter
   the ambient-random null band below that range.

The diagnostic calibrates the geometric interpretation of Stage C. It is not a
mechanism test and does not block Stage D attention-alignment analysis.

#### 13.2.1 Canonical Inputs and Split Policy

Use the exact saved:

```text
subspace/projection_bases/residual_first_kstar.npz
```

with:

\[
k_\star = k_{\mathrm{80pct\_lifetime\_excess}}.
\]

Treat \(Q_{k_\star}^{\mathrm{res}}\) as the candidate pool, not as an assumed
intrinsic dimension and not as the only random-in-span evaluation point. Let
\(g_1,\ldots,g_{k_\star}\) be its ordered, ranked, deduplicated source probes. Define
the nested truncations:

\[
Q_k^{\mathrm{res}}
=
\operatorname{orth}(g_1,\ldots,g_k),
\qquad
\operatorname{span}(Q_1^{\mathrm{res}})
\subseteq
\cdots
\subseteq
\operatorname{span}(Q_{k_\star}^{\mathrm{res}}).
\]

The source probes are not necessarily all time-lagged generalized eigenvectors.
They are the exact ordered residual-first source probes recorded in the headline
artifact. Do not refit a smaller source-probe set separately for each \(k\): that
would change both the fitted probe family and the dimensionality, destroying the
interpretability of the nested curve.

Use the locked sweep:

\[
\mathcal{K}_{\mathrm{fat}}
=
\operatorname{unique}
\left(
\{1,2,3,5,8,13,21,k_\star\}
\cap
\{1,\ldots,k_\star\}
\right).
\]

The script may accept an explicit sweep override for smoke tests or exploratory
pilot comparisons, but must record it. It must construct every requested
\(Q_k^{\mathrm{res}}\) from the same ordered source-probe pool and must not
substitute nearby projection-collapse grid artifacts such as
`residual_first_k32.npz` when \(k_\star=31\).

Run the battery on validation first. Lock all seeds, direction sets, source probe
IDs, nested sweep values, metrics, null-band definition, and plots after inspecting
validation. Then evaluate the same fixed directions on test alongside Stage C2
without refitting, reselection, or choosing a different \(k\). If test has already
been inspected for the current pilot, label the pilot battery exploratory and use
this split policy prospectively for the full run.

The complete nested curve is the primary result. A smaller effective slow-core
dimension may be reported descriptively from the exploratory pilot. A confirmatory
claim about a particular cutoff requires a validation-locked rule or interval that
is applied unchanged to untouched test data. Do not pick the most attractive
test-split \(k\) after viewing the curve.

#### 13.2.2 Shared Evaluation

For every diagnostic direction \(u_j\), compute the same within-document
autocorrelation profile and threshold-crossing timescale used by the main analysis:

\[
P_{d,t,j} = u_j^\top r_{d,t}^{(\ell)},
\qquad
\tau_j^{\mathrm{within}}
=
\min\{k \geq 1 : R_j^{\mathrm{within}}(k) < 1/e\}.
\]

Use the same document weighting, smoothing, validity filters, maximum lag, and
right-censoring rule as the main analysis. In addition to \(\tau\), report the full
autocorrelation profiles and the supplementary positive-profile area:

\[
A_j^+
=
\sum_{k=1}^{K}
\max\left(R_j^{\mathrm{within}}(k), 0\right).
\]

\(A_j^+\) is a supplementary stability readout, not a replacement for the primary
timescale metric. It reduces sensitivity to integer threshold-crossing quantization.

Bootstrap uncertainty over documents, not tokens. Distinguish document-bootstrap
uncertainty from uncertainty due to the sampled diagnostic directions.

#### 13.2.3 FS1: Candidate-Pool Random-in-Span Calibration

The full candidate-pool endpoint remains
\(\operatorname{span}(Q_{k_\star}^{\mathrm{res}})\). Evaluate 512 random unit
directions inside this span as the endpoint of the nested FS2 sweep. For each sample:

\[
z_i \sim \mathcal{N}(0,I_{k_\star}),
\qquad
u_{i,k_\star}
=
Q_{k_\star}^{\mathrm{res}}
\frac{z_i}{\|z_i\|_2}.
\]

Use seed:

```text
fat_subspace_random_in_span_seed = 31
```

Compare held-out \(\tau\), \(A^+\), and mean autocorrelation profiles for:

- random directions inside the full candidate pool
  \(Q_{k_\star}^{\mathrm{res}}\);
- ambient random residual directions;
- individual residual PCA axes;
- the original ordered source probes \(g_1,\ldots,g_{k_\star}\);
- random directions inside top-PCA spans \(U_{31}^{\mathrm{PCA}}\),
  \(U_{128}^{\mathrm{PCA}}\), and \(U_{256}^{\mathrm{PCA}}\), when available.

For each family, report \(Q_{50},Q_{75},Q_{90},Q_{95}\) for \(\tau\) and \(A^+\).
The direct random-in-PCA-span controls test whether elevated timescale is special to
the recovered span or generic to broad high-variance PCA geometry.

Interpretation:

- if generic uniformly sampled directions inside \(Q_{k_\star}^{\mathrm{res}}\)
  remain much slower than ambient random and direct PCA-span controls on test, the
  recovered span supports a coordinate-free slow-region claim;
- if they remain above ambient random but below the selected source probes, report a
  hierarchical slow region;
- if they resemble ambient random or PCA-axis probes, report a thin,
  frame-dependent set of selected slow directions.

Do not replace this wording with “every direction” or “the entire subspace is slow.”
FS1 samples generic directions; it does not establish a universal statement over all
vectors in the span.

#### 13.2.4 FS2: Nested Dimension Sweep and Lower-Band Support

Let \(g_1,\ldots,g_{k_\star}\) be the ordered source probes used to construct
\(Q_{k_\star}^{\mathrm{res}}\). Evaluate the complete locked nested sweep:

\[
k \in \mathcal{K}_{\mathrm{fat}}
=
\operatorname{unique}
\left(
\{1,2,3,5,8,13,21,k_\star\}
\cap
\{1,\ldots,k_\star\}
\right).
\]

Use one fixed coefficient matrix:

\[
Z \in \mathbb{R}^{512 \times k_\star},
\qquad
Z_{i,:}
\sim
\mathcal{N}(0,I_{k_\star}),
\]

drawn once with `fat_subspace_random_in_span_seed = 31`. For every requested \(k\),
reuse its leading coordinates:

\[
u_{i,k}
=
Q_k^{\mathrm{res}}
\frac{Z_{i,1:k}}{\|Z_{i,1:k}\|_2}.
\]

This produces a uniform random direction inside every nested span while coupling
the Monte Carlo samples across \(k\). The coupled design makes changes in the curve
easier to attribute to dimensionality rather than independent sampling noise.

For every \(k \in \mathcal{K}_{\mathrm{fat}}\), report:

- \(Q_{50},Q_{75},Q_{90},Q_{95}\) of \(\tau\) and \(A^+\);
- the mean autocorrelation profile;
- document-bootstrap intervals;
- the full random-in-span median-\(\tau\) curve as a function of \(k\).

Also report the ordered source-probe marginal curve:

\[
k
\mapsto
\tau^{\mathrm{within}}(g_k; D)
\]

for every source-probe rank \(k=1,\ldots,k_\star\) on available train, validation,
and test splits. Define the ambient-random null band on each evaluation split:

\[
T_{\mathrm{null}}(D)
=
Q_{95}
\left(
\left\{
\tau^{\mathrm{within}}(u;D):
u \in J_{\mathrm{ambient\ random}}
\right\}
\right).
\]

Report the first ranks at which source-probe held-out \(\tau\) enters this null band,
but do not use test to move a validation-locked cutoff.

Retain lower-ranked-band controls. Sample 128 directions from lower-ranked bands
when those bands exist:

\[
\operatorname{span}(g_9,\ldots,g_{k_\star}),
\qquad
\operatorname{span}(g_{17},\ldots,g_{k_\star}).
\]

Use seeds:

```text
fat_subspace_mixing_band_seed = 33
```

Report \(\tau\), \(A^+\), and mean autocorrelation-profile summaries by nested
prefix and lower-ranked band. Interpret the random-in-span and marginal curves
together:

- a plateau over a range of \(k\) supports a stable nested slow core;
- a peak followed by decline indicates dilution as lower-ranked directions enter;
- source probes whose held-out \(\tau\) enters the ambient null band do not earn a
  confirmatory place in a named effective slow core;
- a random-in-span median curve alone does not license a dimensional cutoff,
  because a small number of strong members can support elevated mixtures.

The full curve is primary. If the exploratory pilot suggests a smaller effective
core, record a candidate interval and prospectively lock the cutoff rule before
using untouched full-run test data for confirmation.

#### 13.2.5 FS3: PCA Embedding Geometry

Measure how \(Q_{k_\star}^{\mathrm{res}}\) sits inside residual PCA geometry. For
available PCA dimensions:

\[
p \in \{1,2,4,8,16,31,64,128,256,512\},
\]

compute:

\[
M_p = (Q_{k_\star}^{\mathrm{res}})^\top U_p^{\mathrm{PCA}},
\qquad
\operatorname{containment}(Q,U_p)
=
\frac{1}{k_\star}\|Q^\top U_p\|_F^2.
\]

For \(p \geq k_\star\), also report the principal-angle spectrum:

\[
\theta_i = \arccos(\sigma_i(M_p)).
\]

For \(p < k_\star\), report containment and the available singular values only. Do
not pad missing singular values with zeros and summarize the padded principal-angle
spectrum: that mechanically inserts \(90^\circ\) angles and makes median-angle
comparisons misleading.

Plot the mean squared loading of the source span on each PCA axis and cumulative PCA
energy. Use the direct random-in-PCA-span direction controls from FS1 as the primary
timescale calibration. Random 31-dimensional subspaces sampled inside PCA-256 are
optional, not required: for the marginal generic-direction comparison, direct random
vectors inside PCA-256 answer the relevant question more simply.

#### 13.2.6 FS4: Source-Probe Stability

For each ordered source probe \(g_1,\ldots,g_{k_\star}\), compute \(\tau\), \(A^+\),
and autocorrelation profiles on train, validation, and test documents. Report:

- \(Q_{50},Q_{75},Q_{90},Q_{95}\) by split;
- train-to-validation and train-to-test shrinkage ratios;
- Spearman rank correlations across splits;
- right-censoring rates;
- source probe family and fit/evaluation provenance.

The source probes were ranked using validation, so validation summaries are
selection-conditioned. Untouched test is the confirmation split. If source probes
are slow on train or validation but approach ambient random on test, classify the
result as `shrinkage_dominated` and do not make a stable slow-region claim.

#### 13.2.7 Artifacts and Report Classification

Recommended script:

```text
scripts/persistent_state/residual_geometry/05b_fat_subspace_diagnostics.py
```

The script is evaluation-only. It must load fixed source probes and bases and must
not refit residual directions.

Save:

```text
subspace/fat_subspace/random_in_span_directions.npz
subspace/fat_subspace/nested_random_in_span_directions.npz
subspace/fat_subspace/mixing_directions.npz
subspace/fat_subspace/direct_random_pca_span_directions.npz
subspace/fat_subspace/direction_timescales.parquet
subspace/fat_subspace/nested_dimension_sweep.parquet
subspace/fat_subspace/source_probe_rank_curve.parquet
subspace/fat_subspace/autocorr_profiles.npz
subspace/fat_subspace/mean_autocorrelation_profiles.parquet
subspace/fat_subspace/pca_embedding_geometry.parquet
subspace/fat_subspace/source_probe_split_stability.parquet
subspace/fat_subspace/bootstrap_quantiles.parquet
subspace/fat_subspace/fat_subspace_summary.json
```

The final report must state whether the result supports:

```text
fat
hierarchical
thin_frame_dependent
generic_pca_span_slow
shrinkage_dominated
nested_slow_core_candidate
nested_slow_core_confirmed
```

Multiple labels may apply. `nested_slow_core_candidate` is descriptive.
`nested_slow_core_confirmed` requires a validation-locked cutoff rule or interval
that survives unchanged on untouched test. The report must separately state whether
the evidence supports a coordinate-free generic slow-region claim or only a
selected slow-direction claim. It must also state:

- the complete locked nested sweep values;
- the random-in-span median-\(\tau\) curve across nested \(k\);
- the held-out marginal \(\tau(g_k)\) curve and ambient-random null band;
- `top_kstar_geometric_participation_ratio`;
- `top_kstar_slowness_participation_ratio`;
- any exploratory effective slow-core candidate interval;
- whether any named effective slow-core cutoff was validation-locked and confirmed
  unchanged on untouched test, or remains descriptive only.

### 13.3 Exploratory Semantic Readout

This is a post-hoc, validation-only interpretation stage. It is not part of the
locked geometry gate and does not support a claim that the recovered directions are
validated semantic concepts, clean discourse variables, or causal features. Its
purpose is narrower: characterize whether the recovered slow region behaves more
like durable document state than a collection of token-local triggers, and generate
hypotheses for a cleaner follow-up corpus.

Recommended scripts:

```text
scripts/persistent_state/residual_geometry/semantic_axis_comparison.py
scripts/persistent_state/residual_geometry/analyze_semantic_document_dedup.py
```

The primary controlled comparison uses saved Stage-B validation projection chunks;
it must not refit probes or use test documents. For sampled directions that are
linear combinations of saved parent axes, reconstruct their projection traces from
the saved parent-axis traces and verify ambient-direction reconstruction error below
`1e-4`.

#### 13.3.1 Direction Families

Use fixed samples from the following families:

1. the top 10 time-lagged residual directions by validation
   \(\tau_{\mathrm{within}}\), labeled `persistent_reference`;
2. 10 ambient-random residual directions sampled with a recorded seed;
3. the first 10 fixed random directions from the saved
   `random_in_q{k_star}` artifact;
4. the first 10 fixed random directions from each saved matched comparator:
   `random_in_pca31`, `random_in_pca128`, and `random_in_pca256`.

PCA axes selected by variance or validation lifetime may be included for qualitative
navigation, but the load-bearing matched comparison is generic sampled directions
inside \(Q_{k_\star}^{\mathrm{res}}\) versus generic sampled directions inside
matched PCA spans and ambient-random directions. Record the exact probe IDs,
selection reason, sample count, and seed.

#### 13.3.2 Sustained High-Projection Spans

For each direction \(v_j\), use its scalar validation projections:

\[
P_{d,t,j} = v_j^\top r_{d,t}^{(\ell)}.
\]

Compute the threshold separately for each direction as the global token-level
validation quantile:

\[
q_{95,j}
=
\operatorname{quantile}_{d,t \in \mathrm{validation}}
\left(P_{d,t,j}, 0.95\right).
\]

Retain every maximal contiguous run within a validation document satisfying:

\[
P_{d,t,j} \geq q_{95,j}
\]

for at least 8 tokens. Call these `retained_q95_spans`. They are sustained
high-projection spans. Do not call them semantically coherent spans unless semantic
coherence is established by a separate annotation procedure.

For the full document-deduplicated recount, compute metrics from the complete set of
retained spans before truncating any readable export:

- number of retained q95 spans;
- number and fraction of validation documents with at least one retained q95 span;
- retained spans per covered document;
- per-document maximum retained-span length;
- median per-document maximum span length among covered documents;
- mean per-document maximum span length across all validation documents.

When reporting document coverage, each validation document counts at most once.
This prevents repeated runs from one page from inflating the document-level
comparison.

#### 13.3.3 Top-Window Concentration Check

Separately, select the top 12 and bottom 12 token positions by scalar projection for
each direction. For each polarity, report:

- number of distinct source documents among the 12 windows;
- fraction of windows from distinct source documents;
- largest single-document share.

This is a top-window concentration check, not an independence test. A durable
document-state direction may legitimately produce several extreme windows inside
one page. Use the full retained-span document coverage from section 13.3.2 as the
safeguard against mistaking one unusual page for a corpus-wide pattern.

#### 13.3.4 Qualitative Labels

Decode readable windows around the upper- and lower-tail token positions. Shallow
regex and density labels may summarize recurring differences between the two tails,
including register, domain, source-template, formatting, and scraped-text patterns.
These labels are navigation aids only. They are not trained classifiers, validated
concept labels, causal variables, or evidence that a direction has one settled
semantic meaning.

#### 13.3.5 Required Artifacts and Reporting Boundary

Save the full recount artifacts:

```text
subspace/semantic_axis_document_dedup/document_dedup_direction_summary.csv
subspace/semantic_axis_document_dedup/document_dedup_group_summary.csv
subspace/semantic_axis_document_dedup/summary.json
```

Readable top-window and top-span exports may also be saved for qualitative
inspection, but they do not substitute for the all-retained-span recount. If the
full recount artifacts are absent, report only the representative exported-top-12
sanity check and do not report all-span document-coverage aggregates.

Safe interpretation:

```text
The exploratory semantic readout is consistent with durable web-document state,
register, formatting, and source-template structure.
```

Do not upgrade this to abstract maintained-state or reasoning-variable claims
without a cleaner corpus, predeclared labels or running variables, and appropriate
source-template and local-lexical-clustering controls.

---

## 14. Attention Alignment

Run after the residual-first subspace pilot is positive or suggestive.

For each layer/head \(h\), collect per-head attention output vectors at the residual stream write point:

\[
o_{h,d,t} = W_O^h z_{h,d,t},
\]

where \(z_{h,d,t}\) is the head output before the output projection and \(W_O^h\) is that head's slice of the attention output matrix. Use the per-head result after \(W_O\) and before addition to the residual stream, at the same token position \(t\). In TransformerLens-style hook terms, prefer the per-head attention result hook when available rather than the summed attention output.

Fit attention-output PCA on train documents only. Center each head's vectors by that head's global train-token mean before PCA. Keep \(m=8\) principal directions per head. Evaluate alignment on validation/test residual directions; do not use validation/test vectors to fit head PCA.

\[
U_h = [u_{h,1}, \ldots, u_{h,m}].
\]

For each residual-first persistent direction \(g_j\), compute:

\[
A_{j,h}^{\mathrm{res}}
=
\max_{i \leq m}
|\langle \hat g_j, u_{h,i} \rangle|.
\]

Also compute the full head-subspace projection norm:

\[
B_{j,h}^{\mathrm{res}}
=
\|U_h^\top \hat g_j\|_2
=
\sqrt{\sum_{i=1}^m \langle \hat g_j, u_{h,i}\rangle^2}.
\]

The max-single-PC metric \(A\) is conservative and asks whether a direction resembles one dominant head-output PC. The projection-norm metric \(B\) asks whether the direction lies in the top head-output subspace, even when alignment is distributed across PCs.

### 14.1 Attention-Alignment Robustness Controls

Before interpreting attention alignment mechanistically, control for generic residual anisotropy using residual PCA. For each residual direction \(v\) and residual PCA basis \(P_k\) with:

\[
k \in \{16,32,64,128\},
\]

compute:

\[
v_{\perp,k} = (I - P_kP_k^\top)v.
\]

Record:

\[
\mathrm{fraction\_removed}_k(v)
=
1 - \frac{\|v_{\perp,k}\|_2^2}{\|v\|_2^2}.
\]

If \(\|v_{\perp,k}\|_2\) is below a small threshold, mark the residualized alignment as invalid rather than normalizing numerical noise. Otherwise evaluate both \(A\) and \(B\) on:

\[
\tilde v_{\perp,k}
=
\frac{v_{\perp,k}}{\|v_{\perp,k}\|_2}.
\]

Report, by direction group and \(k\):

- raw max-PC alignment;
- raw head-subspace projection norm;
- residual-PCA-controlled max-PC alignment;
- residual-PCA-controlled head-subspace projection norm;
- fraction removed by residual PCA.

This separates attention-specific alignment from alignment caused by both persistent directions and attention outputs living in the same generic high-variance residual subspace.

Primary comparison:

- residual-first persistent directions vs random residual directions;
- high-lifetime residual probes vs low-lifetime residual probes;
- upstream heads vs downstream or same-layer heads.

Positive attention-alignment evidence:

- persistent residual directions have higher max-over-head alignment than random residual directions;
- persistent residual directions have higher max-over-head-subspace projection norm than random residual directions;
- bootstrap 95% CI for the median alignment difference excludes 0;
- the alignment gap survives at least one residual-PCA control setting, or else the result is explicitly reported as likely explained by generic residual anisotropy;
- aligned heads are stable across bootstrap samples of documents;
- aligned heads are in layers plausibly upstream of the residual hook.

This does not prove those heads create the persistent direction. It supports the architectural story:

\[
\text{attention-mediated accumulation}
\rightarrow
\text{persistent residual directions}.
\]

Interpret attention alignment as geometric alignment with attention write directions, not as an estimate of attention routing strength or attention magnitude. Attention/logit soft-capping and other runtime-specific attention details must be treated as architecture metadata, not as quantities inferred from PCA alignment alone.

Save:

- `subspace/head_output_pca.parquet`
- `subspace/residual_direction_head_alignment.parquet`
- `subspace/attention_alignment_summary.json`

### 14.2 Mechanistic Follow-Up Gate

Run mechanistic follow-ups only after Stage C2 and Stage D are positive or scientifically informative. These stages are not required for the residual-geometry pilot claim. They are used to decide whether the persistent directions are written, routed, refreshed, or read by attention and/or MLP blocks.

Locked Stage M execution order:

1. M0: block-output subspace comparison.
2. M3: attention transport test.
3. M1: GQA-safe top-head ablation.
4. M4: MLP writer check, only if M0 suggests MLP involvement or M3/M1 are weak despite strong residual geometry.

Direction ablation is not part of locked Stage M. It requires a separate intervention-design document before implementation. A naive deletion intervention:

\[
r_t \leftarrow r_t - (r_t^\top v)v
\]

must not be treated as a valid locked intervention until that document resolves:

- patching versus deletion;
- which positions are intervened on;
- whether the \(v\)-coordinate is patched from another context or zeroed;
- how residual norm and residual distribution statistics are preserved;
- whether attention patterns are frozen or allowed to change;
- which downstream quantity is measured;
- matched random-direction and norm-matched controls.

### 14.3 M0: Block-Output Subspace Comparison

This stage was previously referred to as D2 during pilot planning. The locked stage name is M0.

Question:

\[
\text{Are high-}\tau\text{ directions preferentially coupled to attention block output geometry, MLP output geometry, or generic residual PCA geometry?}
\]

This stage must be run before interpreting the attention transport test. Even if the transport computation is cheaper, M0 decides whether a positive transport correlation is attention-specific or likely to reflect generic residual geometry.

Default pilot scope:

- layers: `[7, 8, 9, 10, 11, 12]`;
- full or extended scope: layers `[0, ..., 12]` when compute permits;
- primary residual hook remains `blocks.12.hook_resid_post`, so all tested directions \(v\) are layer-12 residual directions even when block-output subspaces come from earlier layers.
- default pilot `components_per_subspace = 32`;
- use the same component count for attention block, MLP output, and residual subspaces;
- when feasible, run sensitivity checks at `components_per_subspace = 8` and `64`.

For each layer \(\ell\), fit PCA subspaces on train documents:

\[
U_{\mathrm{attn},\ell}
=
\text{top PCs of total attention block output}
\]

\[
U_{\mathrm{MLP},\ell}
=
\text{top PCs of MLP output}
\]

\[
U_{\mathrm{resid},\ell}
=
\text{top PCs of residual stream}.
\]

For each direction \(v\), compute:

\[
B_{\mathrm{attn},\ell}(v)
=
\|U_{\mathrm{attn},\ell}^{\top}v\|_2
\]

\[
B_{\mathrm{MLP},\ell}(v)
=
\|U_{\mathrm{MLP},\ell}^{\top}v\|_2
\]

\[
B_{\mathrm{resid},\ell}(v)
=
\|U_{\mathrm{resid},\ell}^{\top}v\|_2.
\]

Compare direction groups:

- persistent residual directions;
- random residual directions;
- low-lifetime time-lagged directions.

Run residual-PCA-controlled versions using the same residualization procedure as Stage 14.1. Report raw and residualized scores.

Anti-triviality rule. Raw attention overlap can be high simply because attention writes into the residual stream. Therefore M0 must report attention and MLP overlap relative to residual PCA overlap:

\[
E_{\mathrm{attn},\ell}(v)
=
B_{\mathrm{attn},\ell}(v)
-
B_{\mathrm{resid},\ell}(v)
\]

\[
E_{\mathrm{MLP},\ell}(v)
=
B_{\mathrm{MLP},\ell}(v)
-
B_{\mathrm{resid},\ell}(v).
\]

Primary comparisons:

\[
\Delta E_{\mathrm{attn},\ell}
=
\operatorname{median}_{v \in \mathrm{persistent}}
E_{\mathrm{attn},\ell}(v)
-
\operatorname{median}_{v \in \mathrm{random}}
E_{\mathrm{attn},\ell}(v)
\]

\[
\Delta E_{\mathrm{MLP},\ell}
=
\operatorname{median}_{v \in \mathrm{persistent}}
E_{\mathrm{MLP},\ell}(v)
-
\operatorname{median}_{v \in \mathrm{random}}
E_{\mathrm{MLP},\ell}(v).
\]

Interpretation:

| M0 result | Interpretation |
|---|---|
| attention \(>\) MLP beyond residual PCA | high-\(\tau\) state is preferentially coupled to attention write/routing geometry |
| MLP \(>\) attention | persistent state may be MLP-written; Stage 06 attention alignment is secondary |
| both high | MLPs write or update state; attention routes, refreshes, or reads it |
| attention \(\approx\) residual PCA for all groups | M0 is uninformative about attention specificity |
| neither survives residual-PCA control | mechanism is not localized by tested subspaces |

Default uninformative criterion. Treat M0 as uninformative about attention specificity if raw \(B_{\mathrm{attn}}\) is positive but \(|\Delta E_{\mathrm{attn}}| \leq 0.01\) or its document-bootstrap confidence interval overlaps zero. Apply the same rule to MLP specificity using \(\Delta E_{\mathrm{MLP}}\). Report the raw \(B\) scores anyway, but do not interpret raw block-output overlap as mechanism-specific evidence without positive excess over residual PCA geometry.

Save:

- `subspace/block_output_subspace_overlap.parquet`
- `subspace/block_output_subspace_summary.json`
- `subspace/block_output_pca_directions.npz`

### 14.4 M3: Attention Transport Test

Question:

\[
\text{Do aligned heads route scalar persistent state across positions?}
\]

For a persistent direction \(v\), define source-position state:

\[
s_s = (r_s^{\mathrm{source}})^\top v.
\]

The source residual \(r_s^{\mathrm{source}}\) must be the residual stream available to the queried head at layer \(\ell\), preferably the layer-\(\ell\) attention input residual before attention normalization. If the implementation instead uses normalized attention input, `resid_pre`, or another hook-specific tensor, record the exact hook and normalization state. Do not use layer-12 post-residual states as the source for earlier-layer heads unless explicitly running a separate retrospective-state analysis; that would weaken the transport interpretation.

For head \((\ell,h)\), compute:

\[
\widehat{s}^{\mathrm{transport}}_t
=
\sum_s A^{\ell,h}_{t,s}((r_s^{\mathrm{source}})^\top v).
\]

Compare it to the actual post-\(W_O\) head write into \(v\):

\[
o_{\ell,h,t}^\top v.
\]

Primary metric:

\[
\operatorname{corr}
\left(
\sum_s A^{\ell,h}_{t,s}((r_s^{\mathrm{source}})^\top v),
o_{\ell,h,t}^\top v
\right).
\]

Compute the correlation pooled over `(document, destination_position)` pairs. Exclude early destination positions where the model is still building context; pilot default is `min_destination_position = 64`. Report bootstrap confidence intervals over documents, not tokens.

Head sets:

- raw-top heads from Stage 06, for example `L12H7`, `L11H0`, `L10H6`, `L7H2`, `L12H6` in the first pilot;
- residualized-top heads from Stage 06 residual-PCA controls, for example `L12H3`, `L11H0`, `L9H3`, `L8H0`, `L10H6` in the first pilot;
- same-layer low-alignment control heads;
- same-layer random control heads;
- high-variance attention-output control heads.

Positive transport evidence:

\[
\operatorname{corr}_{\mathrm{aligned\ heads}}
>
\operatorname{corr}_{\mathrm{matched\ controls}}.
\]

The locked M3 gate is a signed group-level test. It must compare aligned-head sets against all matched control sets, including same-layer random controls and high-variance attention-output controls. A result that beats only one control family is not positive for claim level 7b.

Interpretation depends on M0:

- if M0 shows attention-specific geometry, positive M3 supports attention routing of persistent state;
- if M0 shows attention overlap approximately equal to residual PCA overlap, positive M3 is weaker and may reflect generic residual structure.

Save:

- `subspace/attention_transport.parquet`
- `subspace/attention_transport_summary.json`

### 14.4.1 Exploratory M3b Transport Diagnostics

If locked M3 is not positive, optional exploratory diagnostics may be reported to understand why the group-level gate failed. These diagnostics are not a substitute for locked M3 and do not support claim level 7b unless prospectively added to the locked design and rerun.

Report:

- per-head signed persistent transport:

\[
\operatorname{median}_{v \in \mathrm{persistent}} r_{\ell,h,v}
\]

- per-head absolute persistent transport:

\[
\operatorname{median}_{v \in \mathrm{persistent}} |r_{\ell,h,v}|
\]

- within-head persistent-minus-random transport:

\[
\operatorname{median}_{v \in \mathrm{persistent}} r_{\ell,h,v}
-
\operatorname{median}_{v \in \mathrm{random}} r_{\ell,h,v}
\]

- within-head persistent-minus-random absolute transport:

\[
\operatorname{median}_{v \in \mathrm{persistent}} |r_{\ell,h,v}|
-
\operatorname{median}_{v \in \mathrm{random}} |r_{\ell,h,v}|.
\]

Use these diagnostics to distinguish:

- aligned heads that transport persistent directions more than random directions;
- heads with strong generic transport for many directions;
- suppressive or anti-transport heads with large negative signed correlations.

Large negative signed correlations must not be converted into positive locked evidence post hoc. They may be reported as exploratory suppressive or anti-copy signals, with the sign preserved.

### 14.5 M1: GQA-Safe Top-Head Ablation

Question:

\[
\text{Do aligned heads causally support high-}\tau\text{ persistence?}
\]

Define head ablation as post-\(W_O\) residual-write ablation. For a query/output head:

\[
o_{\ell,h,t}=z_{\ell,h,t}W_O^h.
\]

Ablate by setting:

\[
o_{\ell,h,t}\leftarrow 0.
\]

In GQA models, head ablation means ablating a per-query-head post-\(W_O\) residual write vector. It does not mean deleting or modifying a shared KV group.

Stage M1 must load:

- `configs/resolved_model_architecture.json`

and record:

- number of query heads;
- number of KV heads or query groups;
- per-layer attention type;
- sliding-window size;
- whether layer 12's window covers the 1024-token experimental context.

Evaluation probes:

- use the same \(J_{\mathrm{lag\text{-}heldout}}\) probe set from Stage C2;
- do not introduce a new persistent evaluation set.

For a head set \(S\), define:

\[
C_{\mathrm{head}}(S)
=
1-
\frac{
\operatorname{median}_{j\in J_{\mathrm{lag\text{-}heldout}}}
\tau_j(\text{ablate heads }S)
}{
\operatorname{median}_{j\in J_{\mathrm{lag\text{-}heldout}}}
\tau_j(\text{clean})
}.
\]

Compare:

- raw-top heads;
- residualized-top heads;
- same-layer low-alignment controls;
- same-layer random controls;
- high-variance attention-output controls.

Positive causal head-support evidence:

\[
C_{\mathrm{aligned\ heads}}
>
C_{\mathrm{matched\ controls}}.
\]

Save:

- `subspace/head_ablation_autocorr.parquet`
- `subspace/head_ablation_summary.json`

### 14.5.1 Exploratory M1-lite After Failed M3

If locked M3 fails but M3b identifies head-specific transport candidates, an optional exploratory M1-lite may be run. This is post-hoc and validation-informed if the head sets are chosen from M3 diagnostics. It must be labeled as exploratory in artifacts and reports and must not be used to claim level 8.

Exploratory M1-lite should still use the fixed \(J_{\mathrm{lag\text{-}heldout}}\) probes from Stage C2. This keeps the evaluation directions fixed, but it does not remove the post-hoc nature of validation-selected head sets.

Candidate head-set families:

- positive transport candidates: heads with large positive persistent-minus-random signed transport;
- negative transport candidates: heads with large negative signed persistent transport or large positive persistent-minus-random absolute transport caused by sign reversal;
- transport-positive controls: same-layer or nearby heads with large transport correlations that were not selected by Stage 06 alignment;
- low or negative controls: heads with weak or negative transport diagnostics.

For negative transport candidates, report the direction of the ablation effect separately. If ablating such a head increases held-out persistence, interpret that as exploratory suppressive-head evidence rather than as support for the usual positive \(C_{\mathrm{head}}\) story.

### 14.6 M4: MLP Writer Check

Run this stage if M0 suggests MLP involvement, or if M3/M1 are weak despite a strong residual geometry result.

Question:

\[
\text{Are MLPs writing or updating persistent directions?}
\]

Compute:

\[
\mathrm{MLP}_{\ell,t}^\top v
\]

for:

- persistent residual directions;
- random residual directions;
- low-lifetime time-lagged directions.

Positive MLP-writer evidence requires persistent-direction MLP writes to be systematically larger, more structured, or more predictive of future \(v\)-state than matched controls.

Possible interpretation:

\[
\text{MLPs write/update persistent state}
\rightarrow
\text{attention routes/refreshes/reads it}.
\]

Save:

- `subspace/mlp_writer_alignment.parquet`
- `subspace/mlp_writer_summary.json`

---

## 15. Optional Dictionary Coverage Check

This is optional and secondary. It belongs in the report only after the residual-first subspace has been constructed from eligible residual probes.

Let \(Q_k^{\mathrm{res}}\) be the residual-first persistent basis and \(Q_m^{\mathrm{dict}}\) be an orthonormalized set of external dictionary directions, such as decoder directions if available. Define:

\[
\mathrm{coverage}(G_k,S_m)
=
\frac{1}{k}
\left\|
(Q_k^{\mathrm{res}})^\top Q_m^{\mathrm{dict}}
\right\|_F^2.
\]

Compare:

- slow dictionary directions;
- matched fast dictionary directions;
- random eligible dictionary directions.

This tests external-dictionary recovery of residual geometry. It must not be used to define the residual geometry.

Save:

- `subspace/dictionary_coverage.parquet`

---

## 16. Success Criteria

### Minimal Positive Residual Result

The result is minimally positive if:

1. residual probes meet mode-specific valid-probe coverage;
2. within-document residual-probe timescales have a nontrivial upper tail;
3. the signal survives document permutation controls;
4. time-lagged probes have a stronger upper tail than random probes on validation data, or PCA/attention-output probes have a stronger upper tail than random probes, or the random-probe distribution itself is broadly heavy-tailed.

### Strong Residual Geometry Result

The result is strong if:

1. \(k_{\mathrm{80pct\_lifetime\_excess}}\) is at most one third of the valid eligible residual-probe count, or at most 256 in pilot mode;
2. projection collapse is nontrivial on held-out time-lagged probes: \(C_{\mathrm{lag\text{-}heldout}}(k_{\max}) \geq 0.20\);
3. residual-first projection collapse exceeds random-basis collapse by at least 0.10 absolute on \(J_{\mathrm{lag\text{-}heldout}}\);
4. time-lagged probes have higher upper-tail timescales than PCA probes, or PCA probes have higher upper-tail timescales than random probes with time-lagged probes confirming the same directions. If `time_lagged_fit_unstable=true`, this criterion requires the ridge sensitivity check to show stable validation upper-tail rankings and stable top-direction subspace overlap across \(\epsilon\) values;
5. persistent residual directions align more with stable attention-head output directions than random directions, and the alignment gap survives at least one residual-PCA control setting;
6. the signal replicates on at least one additional layer or robustness corpus.

### Coordinate-Free Slow-Region Result

The stronger coordinate-free interpretation is supported only if the locked
fat-subspace diagnostic shows that generic uniformly sampled directions inside a
validation-locked nested truncation \(Q_k^{\mathrm{res}}\) remain slower than ambient
random directions and direct random-in-PCA-span controls on untouched test
documents. Report the complete nested curve, including the full
\(Q_{k_\star}^{\mathrm{res}}\) candidate-pool endpoint, rather than only the most
favorable \(k\).

A named smaller effective slow core requires agreement between:

1. a stable random-in-span median-\(\tau\) plateau over a nested range of \(k\);
2. held-out marginal source-probe \(\tau(g_k)\) staying above the ambient-random
   null band through the claimed cutoff;
3. a validation-locked cutoff rule or interval applied unchanged on untouched test.

The top-\(k_\star\) geometric and slowness-concentration participation ratios must be
reported side by side. A strong residual-geometry result does not by itself imply
that every direction, or even a generic direction, inside the full recovered
candidate pool is slow.

### Clean Negative Result

The result is cleanly negative if:

- raw residual-probe autocorrelation disappears under within-document demeaning; or
- within-document persistence does not exceed document-permutation controls; or
- random, PCA, and time-lagged residual probes all show narrow, short timescale distributions on validation data; or
- projection collapse is indistinguishable from random-basis projection collapse.

A clean negative should be reported as evidence against a compact, easily detectable residual-stream state-lifetime geometry at the tested layer, corpus, and probe budget. It does not rule out more diffuse persistence or persistence at other layers.

---

## 17. Compute Plan

### Stage A: Smoke

- 128 docs
- random probes: 64
- optional PCA probes: 64
- optional time-lagged probes: 32
- document length: 1024 tokens
- \(K=128\)
- no attention alignment
- no projection-collapse grid beyond `[8, 16, 32]`

Purpose: validate shapes, hooks, artifacts, and estimators.

### Stage B1: Residual-Probe Autocorrelation Pilot

- 5,000 docs
- random probes: 512
- PCA probes: 256
- time-lagged probes: 256, fit on train and evaluated on validation/test
- document length: 1024 tokens
- \(K=512\)
- raw and within-document autocorrelation
- document permutation controls
- B1 decision table

Forward pass and storage cost are approximately 2x the 512-token equivalent. Budget accordingly. Do not run an additional 512-token robustness pass for B1; use already-computed autocorrelation profiles for lag-window sensitivity if needed.

Purpose: decide whether residual lifetime structure is visible.

### Stage C: Residual-First Subspace Pilot

Run if B1 is positive or suggestive.

- construct \(G_k\);
- estimate dimensionality;
- run fixed-heldout projection collapse on validation with \(J_{\mathrm{random}}\) and \(J_{\mathrm{lag\text{-}heldout}}\);
- compare residual-first, PCA, and random-control bases at matched \(k\);
- optional dictionary coverage only if external directions are already available;
- do not make attention-organization claims at this stage.

Purpose: decide whether persistence is concentrated enough to justify full structural analysis.

### Stage C1: Fat-Subspace Diagnostics and Validation Lock

Run if Stage C projection collapse is positive or strongly suggestive.

- construct the exact \(Q_{k_\star}^{\mathrm{res}}\) artifact with ordered source probe IDs;
- compute geometric and slowness-concentration participation ratios for the
  top-\(k_\star\) source-probe pool;
- run FS1 full-candidate-pool random-in-span calibration on validation;
- run the coupled FS2 nested sweep over
  \(\{1,2,3,5,8,13,21,k_\star\}\), omitting values above \(k_\star\);
- report per-rank held-out source-probe \(\tau(g_k)\), the ambient-random null band,
  and lower-band mixing controls;
- run FS3 PCA containment and direct random-in-PCA-span controls on validation;
- run FS4 source-probe stability summaries for train and validation;
- lock seeds, coupled coefficient matrix, direction sets, sweep values, null-band
  definition, metrics, plots, and interpretation rules before test.

Purpose: determine whether the recovered span supports a coordinate-free generic
slow-region interpretation or only a selected slow-direction interpretation.

### Stage C2: Test-Split Geometry Confirmation

Run if Stage C projection collapse is positive or strongly suggestive.

- reuse the fixed validation-stage candidate and held-out projection-collapse probes;
- evaluate projection collapse on the test split without refitting probes;
- report whether \(q90_{\mathrm{before}}\) is invariant across basis families and \(k\);
- report whether residual-first and PCA collapse held-out time-lagged probes more than random-control bases.
- if Stage C1 was run, reuse its fixed diagnostic directions and coupled coefficient
  matrix and evaluate FS1-FS4 on test without refitting, reselection, sweep changes,
  cutoff changes, or seed changes.

Purpose: confirm that validation-selected/fixed persistent geometry and its locked
interpretation predict held-out test behavior.

### Stage D: Attention Alignment and Robustness

Run if Stage C is positive or strongly suggestive.

- fit attention-head output PCA on train documents;
- compute both max-single-PC alignment and top-\(m\) head-subspace projection norm;
- run residual-PCA controls at \(k \in \{16,32,64,128\}\);
- report residual-PCA containment of persistent, random, and low-lifetime directions;
- distinguish attention-specific alignment from generic residual anisotropy.

Purpose: test whether persistent directions align with attention write geometry after anti-confound controls.

Stage D does not wait on a fat or hierarchical Stage C1 classification. Stage C1
calibrates the subspace wording; Stage D asks a separate geometric-alignment question.

### Stage M: Mechanistic Follow-Ups

Run only after Stage D remains positive or scientifically informative.

- M0: block-output subspace comparison, with attention, MLP, and residual PCA subspaces at matched component count;
- M3: attention transport test using \(\sum_s A_{t,s}(r_s^{\mathrm{source}} \cdot v)\), pooled over document and destination-position pairs and bootstrapped over documents;
- M1: GQA-safe post-\(W_O\) top-head ablation using the fixed \(J_{\mathrm{lag\text{-}heldout}}\) probes from Stage C2;
- M4: MLP writer check using \(\mathrm{MLP}_{\ell,t}\cdot v\), only when M0 suggests MLP involvement or when M3/M1 are weak despite strong residual geometry;
- do not run direction ablation until a separate intervention-design document is written and locked.

Purpose: move from geometric alignment to causal or routing evidence.

### Stage F: Full Residual Subspace and Attention Alignment

Run only after pilot Stage C2 and Stage D clarify the confounds and expected claim.

- full projection-collapse grid;
- locked fat-subspace diagnostics with test confirmation;
- attention-head output PCA and residual-PCA-controlled attention alignment;
- optional dictionary coverage;
- final residual report.

Purpose: test the strong residual-stream geometry claim.

### Full Run

Run only after pilot results are positive or highly suggestive.

- 50,000 docs;
- random probes: 2,048;
- PCA probes: 512;
- time-lagged probes: 512;
- document length: 1024 tokens;
- \(K=512\) primary, \(K=768\) sensitivity;
- full projection and attention alignment analyses.

---

## 18. Required Reports

`07_make_residual_report.py` must produce:

- `reports/residual_geometry_report.md`
- `reports/residual_geometry_summary.json`

Required plots:

- random residual direction timescale histogram;
- residual PCA direction timescale histogram;
- time-lagged residual direction timescale histogram;
- PCA vs random upper-tail comparison;
- time-lagged vs PCA vs random upper-tail comparison;
- residual-probe decay curves for top persistent directions;
- raw vs within residual tau scatter;
- real vs document-permutation residual tau distributions;
- cumulative lifetime-excess curve;
- lag-window sensitivity from existing autocorrelation profiles when run;
- projection-collapse curves by basis and held-out evaluation family;
- fat-subspace random-in-span timescale and positive-profile-area distributions when run;
- fat-subspace nested random-in-span median-\(\tau\) and positive-profile-area curves
  across \(k\) when run;
- held-out marginal source-probe \(\tau(g_k)\) by ordered rank with the
  ambient-random null band when run;
- fat-subspace lower-band curves when run;
- fat-subspace PCA-containment and direct random-in-PCA-span controls when run;
- source-probe split-stability summaries when run;
- attention-alignment summary when run;
- M0 block-output subspace comparison summary when run;
- attention transport summary when run;
- head-ablation summary when run;
- MLP writer summary when run;
- optional dictionary coverage comparison when run.

The report must explicitly state:

- resolved model, layer, hook, and architecture metadata from `configs/resolved_model_architecture.json`;
- whether layer 12 is local/sliding-window or global under the active runtime;
- document count;
- residual probe count by family;
- valid and deduplicated probe counts;
- invalid and right-censored probe counts;
- \(Q_{50}, Q_{75}, Q_{90}, Q_{95}\) of \(\tau_{\mathrm{res}}\) by probe family;
- time-lagged fit split, lag set, whitening dimension, ridge parameter, ridge-resolution trace, `epsilon_scale`, condition number, eigenvalues, anti-persistent mode filtering counts, and ridge sensitivity results when required;
- document-permutation control results;
- lag-window sensitivity results from existing autocorrelation profiles when run;
- `k_80pct_lifetime_excess` and `k_elbow`;
- top-\(k_\star\) geometric and slowness-concentration participation ratios;
- probe-relative dimensionality caveat and probe-family-specific dimensionality estimates;
- held-out projection-collapse evaluation set sizes and seeds for \(J_{\mathrm{random}}\) and \(J_{\mathrm{lag\text{-}heldout}}\);
- whether projection-collapse test confirmation reused fixed candidate and held-out probes or refit them;
- whether projection collapse was positive;
- exact \(k_\star\), canonical fat-subspace source probe IDs and families, and whether
  the exact \(Q_{k_\star}^{\mathrm{res}}\) candidate pool or an explicitly recorded
  override was used;
- locked nested sweep values, coupled coefficient-matrix seed, random-in-span
  median-\(\tau\) curve, held-out marginal source-probe \(\tau(g_k)\) curve, and
  ambient-random null band;
- any exploratory effective slow-core candidate interval and whether a named cutoff
  is descriptive only or validation-locked and confirmed unchanged on untouched
  test;
- fat-subspace classification and whether the evidence supports a coordinate-free
  generic slow-region claim or only a selected slow-direction claim;
- fat-subspace validation/test split policy, including whether the current run is
  prospectively locked or exploratory because test had already been inspected;
- whether attention alignment was positive under raw and residual-PCA-controlled metrics;
- attention-alignment head-subspace projection norm summaries;
- residual-PCA containment summaries by direction group;
- M0 block-output subspace overlap summaries, including attention, MLP, residual, and excess-over-residual-PCA scores, when run;
- M3 transport correlations and document-bootstrap confidence intervals when run;
- M1 GQA-safe ablation head sets, matched controls, and \(C_{\mathrm{head}}\) results when run;
- M4 MLP writer alignment or prediction summaries when run;
- whether claims are descriptive, geometric, or attention-alignment claims.

---

## 19. Implementation Invariants

1. Every artifact includes config hash and git commit hash when available.
2. Every random sample records seed and source population.
3. Test split is not used for decisions before final reporting.
4. Position indices are zero-based.
5. Autocorrelation aggregates documents with equal weight.
6. All reported timescales name their document distribution.
7. Residual-first persistent subspaces must be constructed from eligible residual probes.
8. Dictionary coverage, if run, is secondary and cannot define residual geometry.
9. Projection-collapse analyses must use fixed held-out evaluation sets \(J_{\mathrm{random}}\) and \(J_{\mathrm{lag\text{-}heldout}}\), not probes used to define \(G_k\) or any projected basis.
10. Held-out projection-collapse evaluation sets are drawn or fit before any \(G_k\)-eligible probe is sampled or fit and are never modified after initial creation. Their seeds, sizes, and fit-document assignments are recorded in `subspace/projection_collapse_summary.json`.
11. Test-split projection-collapse confirmation must reuse existing `subspace/projection_collapse_candidate_time_lagged.npz` and `subspace/heldout_projection_eval_probes.npz` unless an explicit refit flag is provided.
12. `00_validate_env.py` must write `configs/resolved_model_architecture.json` before smoke artifacts are produced.
13. Time-lagged residual directions must be fit on train and evaluated on validation/test for headline claims.
14. Time-lagged directions are eligible for \(G_k\) only if their validation autocorrelation is positive at early/mid lags under the section 5.4 rule.
15. If `time_lagged_fit_unstable=true`, level-3 autocorrelation-optimized-direction claims require held-out validation superiority over random/PCA and a passing ridge sensitivity check.
16. Attention alignment must report both max-single-PC alignment and head-subspace projection norm.
17. Attention alignment must report residual-PCA-controlled alignment and fraction removed by residual PCA before making a mechanistic attention-organization interpretation.
18. M0 must compare attention and MLP output subspace overlap against residual PCA overlap at matched component count before claiming attention-specific or MLP-specific geometry.
19. M3 transport correlations must use a recorded source-residual hook, be pooled over `(document, destination_position)` pairs, exclude early destination positions by a recorded threshold, and use bootstrap confidence intervals over documents.
20. The locked M3 gate is the signed group-level aligned-versus-controls comparison. Absolute transport and within-head persistent-minus-random diagnostics are exploratory unless prospectively locked and rerun.
21. M1 head ablation must ablate post-\(W_O\) per-query-head residual write vectors and must record GQA/query-group architecture metadata.
22. M1 must use the fixed \(J_{\mathrm{lag\text{-}heldout}}\) probe set from Stage C2 rather than creating a new persistent-probe evaluation set.
23. If M1-lite is run after failed locked M3 using heads selected from M3 diagnostics, artifacts must record `post_hoc_head_selection=true`, the source diagnostic used for selection, and that the result is exploratory.
24. Direction ablation is not part of locked Stage M and must not be implemented or reported as a locked causal intervention without a separate intervention-design document.
25. Every stage is resumable and skips complete valid artifacts unless `--overwrite` is passed.
26. The fat-subspace diagnostic must use an exact saved
    \(Q_{k_\star}^{\mathrm{res}}\) candidate-pool basis with ordered source probe IDs
    and families. It must not silently substitute a nearby projection-collapse grid
    value.
27. Fat-subspace test confirmation must reuse validation-locked source probes,
    generated directions, coupled coefficient matrix, sweep values, seeds, metrics,
    null-band definition, and plots without refitting, reselection, or choosing a new
    cutoff.
28. Fat-subspace reports must distinguish selected source probes from generic
    uniformly sampled directions inside their span. Do not infer that every direction
    in a span is slow from a finite random-in-span sample.
29. For PCA dimensions below \(k_\star\), report containment and available singular
    values without padding missing principal angles with \(90^\circ\) values.
30. Every nested fat-subspace truncation must be constructed from the same ordered,
    deduplicated source-probe pool:
    \(Q_k^{\mathrm{res}}=\operatorname{orth}(g_1,\ldots,g_k)\). Do not refit source
    probes independently for each \(k\).
31. The nested random-in-span sweep must draw one fixed Gaussian coefficient matrix
    with a recorded seed and reuse leading coordinate prefixes across \(k\). This
    couples Monte Carlo noise across the nested curve while preserving uniform
    sampling inside each span.
32. A named smaller effective slow-core dimension is descriptive unless its cutoff
    rule or interval was locked on validation and applied unchanged to untouched
    test. Report the complete nested curve even when a smaller core is named.

---

## 20. Claim Hierarchy

Use this hierarchy when writing results:

1. **Residual temporal-structure claim:** residual projections have nontrivial temporal autocorrelation.
2. **Residual lifetime-distribution claim:** lifetimes across residual directions are heavy-tailed or otherwise structured.
3. **Autocorrelation-optimized direction claim:** time-lagged residual probes reveal longer-lived directions than random probes, and ideally longer-lived directions than PCA probes.
4. **Residual subspace claim:** high-lifetime directions concentrate in a finite or structured subspace.
5. **Projection-collapse claim:** removing the residual-first persistent basis reduces held-out residual-probe timescales more than random-control bases, with \(J_{\mathrm{lag\text{-}heldout}}\) as the primary compact-subspace recovery test.
5a. **Coordinate-free generic slow-region claim:** generic uniformly sampled
directions inside a validation-locked nested truncation of the recovered candidate
pool remain long-timescale on untouched test documents relative to ambient random
and direct random-in-PCA-span controls. The complete nested curve and full
candidate-pool endpoint are reported.
6. **Attention-specific geometric organization claim:** persistent residual directions align preferentially with stable attention-output subspaces, and this survives residual-PCA controls.
7a. **Attention-vs-MLP write-subspace claim:** high-\(\tau\) directions are preferentially coupled to attention-block output geometry, MLP-output geometry, both, or neither after comparison against residual PCA geometry at matched subspace dimensionality.
7b. **Attention transport/routing claim:** attention patterns route scalar persistent state across positions:

\[
\sum_s A_{t,s}((r_s^{\mathrm{source}})^\top v)
\rightarrow
o_{\ell,h,t}^\top v.
\]

8. **Causal head-support claim:** ablating aligned post-\(W_O\) head write vectors reduces high-\(\tau\) persistence more than matched control heads.
9. **Model-geometry claim:** the residual stream has state-lifetime geometry as a model-level property.

Only claim level 3 after time-lagged directions are fit on train and evaluated on held-out documents. Only claim level 4 after residual-first dimensionality is estimated. Only claim level 5 after fixed-heldout projection collapse is positive and test confirmation has either passed or is explicitly pending. Only claim level 5a after the locked fat-subspace battery shows that generic random-in-span directions remain slow on untouched test documents relative to ambient random and direct random-in-PCA-span controls for a validation-locked nested truncation, while reporting the complete nested curve and full candidate-pool endpoint. A named smaller effective slow-core dimension remains descriptive unless its cutoff rule or interval was locked on validation and confirmed unchanged on untouched test. Only claim level 6 after attention alignment is positive and residual-PCA controls are reported. Only claim level 7a after M0 compares attention, MLP, and residual subspace overlap. Only claim level 7b after locked M3 shows signed transport correlations above all matched control groups, with document-bootstrap confidence intervals. Exploratory M3b diagnostics using absolute transport or post-hoc within-head persistent-minus-random comparisons do not support claim level 7b unless prospectively locked and rerun. Only claim level 8 after GQA-safe post-\(W_O\) head ablation reduces held-out \(J_{\mathrm{lag\text{-}heldout}}\) timescales more than matched controls under a prospectively specified head set. Exploratory M1-lite after failed M3 does not support claim level 8.

Mechanistic decision table:

| C2 | M0 | M3 | M1 | Interpretation |
|---|---|---|---|---|
| positive | attention \(>\) MLP beyond residual PCA | positive | positive | strong attention-routing and causal-support story |
| positive | attention \(>\) MLP beyond residual PCA | positive | weak | attention transports state, but ablation may be redundant, noisy, or compensated |
| positive | both attention and MLP high | positive | positive | MLPs write or update state; attention routes or refreshes it |
| positive | MLP \(>\) attention | weak | weak | persistent state may be MLP-written; per-head Stage 06 alignment is secondary |
| positive | attention \(\approx\) residual PCA for all groups | any | any | M0 is uninformative about attention specificity |
| positive | neither survives residual-PCA control | weak | weak | compact persistent geometry exists, but mechanism is not localized by tested subspaces |
| weak | attention \(>\) MLP beyond residual PCA | positive | positive | attention transport or causal evidence may be real, but compact persistent-subspace recovery is not established |
| weak | attention \(>\) MLP beyond residual PCA | positive | weak/not run | attention-specific routing evidence exists, but projection-collapse failure weakens the persistent-subspace story |
| weak | MLP \(>\) attention | weak | weak | likely not an attention-mediated compact-subspace story; investigate an MLP/state-update framing |
| weak | any | any | any | mechanism evidence can be reported only with claims that do not rely on compact-subspace recovery |

---

## 21. Final Locked Thesis

This experiment tests whether the residual stream has state-lifetime geometry.

The primary chain under test is:

\[
\text{document distribution } D
\rightarrow
\text{residual projection timescales } \tau_{\mathrm{res},j}(D)
\rightarrow
\text{time-lagged persistent directions}
\rightarrow
\text{persistent residual subspace}
\rightarrow
\text{fixed-heldout projection collapse}
\rightarrow
\text{fat-subspace interpretation calibration}
\rightarrow
\text{attention-output alignment with residual-PCA controls}
\rightarrow
\text{attention/MLP mechanism tests}.
\]

For the primary run:

\[
\text{C4 broad web text}
\rightarrow
\tau_{\mathrm{res},j}(D_{\mathrm{C4}})
\rightarrow
\text{time-lagged residual probes}
\rightarrow
\text{residual-first persistent subspace}
\rightarrow
\text{held-out projection-collapse confirmation}
\rightarrow
\text{fat, hierarchical, or thin interpretation}
\rightarrow
\text{attention alignment and robustness controls}
\rightarrow
\text{ordered mechanistic follow-ups when justified}.
\]

A positive result supports the model-level claim that transformer residual streams are temporally organized: some directions carry information across longer token horizons than others, and those directions may form a coherent state-lifetime geometry. A negative result is still useful because it constrains where persistent state is not easily detectable: at the tested layer, corpus, probe families, and sample budget.
