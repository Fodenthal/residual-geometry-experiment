# Residual Geometry Pilot Run Summary

This document summarizes the current `persistent_state_residual_geometry` pilot run for discussion and follow-up analysis. It reflects the robust Stage 05 validation/test runs, the exploratory post-test FS1-FS4 fat-subspace battery, artifact-only fat-subspace robustness checks, the residual-PCA-controlled Stage 06 attention alignment run, the completed M0 block-output subspace comparison, exploratory M3/M1-lite mechanistic follow-ups, and post-hoc analyses of persistent directions, PCA-vs-persistent geometry, and semantic dossiers.

## Run Identity

- Repository: public residual-geometry artifact repository
- Mode: `pilot`
- Run name: `residual_geometry_pilot`
- Config hash: `9fa0bed92af4b2da3ca0e91c3f3056594a8c95d21dccf5bfe7e5bca08b97e3dd`
- Model: `google/gemma-2-2b`
- Hook: `blocks.12.hook_resid_post`
- Target layer: `12`
- Residual width: `2304`
- Dataset: `allenai/c4`, English validation split
- Context length: `1024`
- Context split: train `80%`, validation `10%`, test `10%`

Architecture metadata from the run:

- Transformer layers exposed by TransformerLens config: `26`
- Attention heads: `8`
- Query/key-value groups: `4`
- Context length: `8192`
- Window size: `4096`
- Layer 12 attention type: `global`
- For 1024-token contexts, the active attention window covers the whole experimental context.

## One-Line Result

The pilot found a real residual-stream persistence signal. Time-lagged residual directions have a heavy upper tail of within-document autocorrelation lifetimes; the signal collapses under document permutation; the top signal is heavy-tailed but spread over tens of nonduplicate directions; fixed held-out time-lagged probes collapse on both validation and test under residual-first/PCA projection but not random-control projection; generic random rotations inside the exact top-31 candidate pool remain slow on validation and test in the exploratory FS battery; and persistent directions show attention-specific block-output geometry beyond residual PCA controls, especially in late layers.

The main caveats are provenance and interpretation. The persistent subspace appears strongly PCA-contained: individual PCA axes are mostly not persistent, but PCA bases collapse held-out persistent directions nearly as well as residual-first bases. The completed FS battery is exploratory because the pilot test split had already been inspected before FS1-FS4 were run, so it does not support a prospective claim-level-5a confirmation. Qualitative dossiers and controlled semantic rotation samples suggest the slow region tracks web-document state, register, formatting, and source-template structure rather than clean single-topic semantics.

## Stage B1: Residual-Probe Autocorrelation

Validation probe counts and within-document timescales:

| Probe family | Count | Valid within | Right-censored | Q50 tau | Q75 tau | Q90 tau | Q95 tau |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 512 | 512 | 0 | 1.0 | 1.0 | 1.0 | 1.0 |
| PCA | 256 | 256 | 0 | 1.0 | 1.0 | 1.0 | 2.0 |
| time-lagged | 256 | 256 | 0 | 1.0 | 4.0 | 11.5 | 17.0 |

B1 decision status:

```text
minimally_positive_or_suggestive
```

Important B1 facts:

- Valid probe coverage was `100%`.
- Time-lagged Q90 over random Q90: `11.5x`.
- PCA Q90 over random Q90: `1.0x`.
- Non-random right-censored fraction: `0.0`.
- Top persistent probes stayed above the random median decay at `3` checked lags among `[16, 32, 64, 128]`.

Document permutation control:

| Probe family | Top-decile real median tau | Top-decile permuted median tau | Reduction |
|---|---:|---:|---:|
| random | 1.0 | 1.0 | 0.0 |
| PCA | 1.0 | 1.0 | 0.0 |
| time-lagged | 17.0 | 1.0 | 0.941 |

Interpretation: persistence is not broad across random directions, individual PCA axes mostly do not have long lifetimes, and the time-lagged signal depends strongly on ordered document structure.

## Time-Lagged Fit Health

- Lag set: `[8, 16, 32, 64, 128]`
- Whitening PCs requested/used: `512 / 512`
- Output directions: `256`
- Lagged pair count: `19,488,000`
- Train token count: `4,096,000`
- Initial/final epsilon: `0.00034823549316734574`
- Ridge doublings: `0`
- Final condition number: `147.7826604778887`
- Epsilon scale: `1e-4`
- Condition warning: `false`
- Fit unstable: `false`
- Positive-validation persistence fraction: `1.0`
- Anti-persistent or sign-changing count: `0`

Top generalized eigenvalues:

```text
0.9082, 0.8654, 0.8534, 0.8218, 0.7620, 0.7357,
0.7230, 0.6835, 0.6419, 0.6071, 0.5948, 0.5680
```

Interpretation: the time-lagged fit is numerically healthy and not obviously a ridge/conditioning artifact.

## Stage 04: Residual-First Subspace Pilot

Dimensionality summary:

```json
{
  "valid_probe_count": 1024,
  "random_tau_within_median": 1.0,
  "k_80pct_lifetime_excess": 31,
  "right_censored_probe_fraction": 0.0,
  "top_k_family_composition_at_k80": {
    "time_lagged": 30,
    "pca": 1
  },
  "k_80pct_random_only": 18,
  "k_80pct_pca_only": 8,
  "k_80pct_time_lagged_only": 25
}
```

Deduplication summary:

```json
{
  "valid_probe_count": 1024,
  "deduplicated_probe_count": 1024,
  "dedup_abs_cosine_threshold": 0.95,
  "family_counts_after_dedup": {
    "random": 512,
    "time_lagged": 256,
    "pca": 256
  }
}
```

Interpretation: the B1 signal is not one duplicated slow direction. `k_80pct_lifetime_excess = 31` suggests a compact but nontrivial set of tens of directions, dominated by time-lagged probes.

### Persistent Direction Contribution and Redundancy

A post-hoc contribution analysis recomputed `k_80pct_lifetime_excess = 31` and found that
the top-31 directions account for `0.803` of global lifetime excess. The distribution is
heavy-tailed: the top direction contributes `0.442` of top-31 excess, the top 5 contribute
`0.637`, and the top 10 contribute `0.751`.

This does not collapse to one duplicated direction. Within the top-31 set, pairwise
absolute cosine has median `0.035`, Q90 `0.092`, and max `0.237`; nearest-neighbor
absolute cosine has median `0.131`; and the top-31 participation-ratio effective rank is
`28.22`. The top-31 family composition is `30` time-lagged directions and `1` PCA
direction. Interpretation: the persistence signal has a dominant lead direction, but the
selected persistent set remains high-rank and nonredundant.

The geometric rank does not answer how many directions carry most of the slowness.
From the saved top-31 excess vector, the slowness-concentration participation ratio is
`4.71`, substantially smaller than the geometric rank `28.22`. The selected source
probes point in roughly 28 distinct geometric directions, while their lifetime excess
is concentrated like roughly 4.7 equally weighted directions. The completed FS1-FS4
battery therefore reports both ranks and sweeps nested truncations of the same ordered
top-31 candidate pool rather than assuming that `31` is an intrinsic slow-core
dimension.

## Stage 05: Projection Collapse

The robust Stage 05 implementation uses fixed held-out evaluation sets:

- `J_random`: 128 fresh held-out random directions.
- `J_lag-heldout`: 128 independently fit held-out time-lagged directions.
- Candidate time-lagged basis probes are fit on train shard A.
- Held-out time-lagged eval probes are fit on disjoint train shard B.
- `--overwrite` recomputes output tables but does not refit probes.
- `--refit-probes` is required to refit candidate or held-out probes.

The JSON status currently says:

```text
not_positive_or_not_evaluable
```

This is stale/overconservative because the saturated-`k` high-vs-low gate is too strict. The actual validation and test tables are strongly positive versus random-control projection.

### Validation Collapse

Validation `lag_heldout` invariant:

- `q90_tau_before = 18.6` for every basis and every `k`.

Validation collapse at selected `k`:

| k | PCA collapse | Random-control collapse | Residual-first collapse |
|---:|---:|---:|---:|
| 8 | 0.522 | 0.005 | 0.556 |
| 16 | 0.641 | 0.006 | 0.664 |
| 32 | 0.764 | 0.012 | 0.764 |
| 64 | 0.842 | 0.022 | 0.842 |
| 128 | 0.922 | 0.046 | 0.922 |
| 256 | 0.998 | 0.110 | 0.998 |

At `k=256`, residual-first minus random-control was `0.8889`.

### Test Collapse, C2 Confirmation

Test `lag_heldout` invariant:

- `q90_tau_before = 20.3` for every basis and every `k`.
- `random_heldout q90_tau_before = 1.0` for every basis and every `k`.

Test `lag_heldout` collapse:

| k | PCA collapse | Random-control collapse | Residual-first collapse |
|---:|---:|---:|---:|
| 8 | 0.514 | 0.002 | 0.382 |
| 16 | 0.632 | 0.005 | 0.430 |
| 32 | 0.756 | 0.012 | 0.584 |
| 64 | 0.841 | 0.021 | 0.821 |
| 128 | 0.923 | 0.049 | 0.989 |
| 256 | 0.998 | 0.097 | 1.000 |

Test C2 interpretation:

- Fixed validation-stage directions/probes predict held-out test behavior.
- Random-control projection barely collapses held-out lag probes.
- Residual-first projection strongly collapses held-out lag probes.
- PCA also strongly collapses held-out lag probes, and at small `k` PCA beats residual-first.

Refined projection-collapse story:

> We recovered a held-out-generalizing persistent residual subspace, but it is largely contained in high-variance residual PCA geometry. Individual PCA axes are not persistent, but persistent rotations live inside the PCA span.

## Stage C1: Exploratory Fat-Subspace Diagnostic Battery

The FS1-FS4 battery used the exact saved headline basis
`subspace/projection_bases/residual_first_kstar.npz` with `k_star = 31`, the ordered
source probes recorded in that artifact, and the locked nested sweep:

```text
[1, 2, 3, 5, 8, 13, 21, 31]
```

The coupled random-in-span coefficient seed was `31`. The full battery was evaluated
on validation and test without refitting source probes or changing directions between
splits. Because the pilot test split had already been inspected during Stage C2 before
FS1-FS4 were added, these results are exploratory. They are strong evidence for the
full-run design, but they do not prospectively establish claim level 5a.

### FS1: Generic Slow Directions Inside `Q31`

The locked classifier labels the full candidate-pool endpoint:

```text
fat
```

At the endpoint, 512 uniformly sampled directions inside `Q31` are slow on both
splits:

| Split | Family | Count | Q05 tau | Q50 tau | Q90 tau | Q95 tau |
|---|---|---:|---:|---:|---:|---:|
| validation | random inside `Q31` | 512 | 10.0 | 24.0 | 59.0 | 81.0 |
| test | random inside `Q31` | 512 | 10.0 | 25.0 | 60.0 | 82.45 |
| test | ambient random | 512 | 1.0 | 1.0 | 1.0 | 1.45 |
| test | individual PCA axes | 256 | 1.0 | 1.0 | 1.0 | 2.0 |

All 512 sampled test `Q31` directions exceed the ambient-random Q95 null threshold
`1.45`. Direct random-in-PCA-span controls remain much shorter lived: their test
median tau is `1.5` inside PCA-31 and `1.0` inside PCA-128 or PCA-256. This rules out
the simple explanation that generic directions inside any broad high-variance PCA
span are slow.

### FS2: Nested Sweep and Lower-Ranked Bands

The random-in-span median-tau curve declines as lower-ranked directions enter the
nested candidate pool:

| Nested k | Validation Q50 tau | Test Q50 tau |
|---:|---:|---:|
| 1 | 493.0 | 492.0 |
| 2 | 289.5 | 298.0 |
| 3 | 201.0 | 201.0 |
| 5 | 114.0 | 116.5 |
| 8 | 83.0 | 87.0 |
| 13 | 29.0 | 30.0 |
| 21 | 26.0 | 27.5 |
| 31 | 24.0 | 25.0 |

The full `Q31` endpoint remains slow rather than collapsing to ambient behavior.
Lower-ranked-band controls also remain slow on test:

| Band | Directions | Q05 tau | Q50 tau | Fraction above ambient-random Q95 |
|---|---:|---:|---:|---:|
| `g_9..g_31` | 128 | 12.0 | 18.0 | 1.0 |
| `g_17..g_31` | 128 | 10.0 | 14.0 | 1.0 |

Interpretation: the slow region has preferred high-lifetime axes and gradual
dilution, but it is not a thin frame-dependent set held up only by the first few
directions. The current pilot does not name a smaller effective slow-core cutoff:
there was no validation-locked cutoff rule, and the full nested curve is the primary
result.

### FS3-FS4: PCA Embedding and Split Stability

The `Q31` candidate pool is substantially PCA-contained without being equivalent to
the first 31 PCA directions:

| Residual PCA span | `Q31` containment |
|---:|---:|
| 31 | 0.090 |
| 64 | 0.161 |
| 128 | 0.287 |
| 256 | 0.521 |

This matches the Stage 05 story: persistent rotations live partly inside broad PCA
geometry, while individual PCA axes and generic PCA-span directions are mostly
short-lived.

Source probes and random-in-span directions are stable across validation and test:

| Family | Validation Q50 tau | Test Q50 tau | Val/test Spearman | Median absolute tau change |
|---|---:|---:|---:|---:|
| ordered source probes | 16.0 | 17.0 | 0.986 | 1.0 |
| random inside `Q31` | 24.0 | 25.0 | 0.996 | 1.0 |

An artifact-only robustness analysis added lower-tail and core-loading checks without
new model forward passes, probe refits, direction resampling, or lock mutation. For
test `Q31` rotations, `Q05/Q10/Q25/Q50 tau = 10/13/18/25`, and all 512 rotations
remain above the ambient-random Q95 null. Directions in the lowest core-loading
quartile remain slow for prefixes `k = [1, 3, 5, 8]`: their test median tau stays in
`[21.0, 23.0]`, with every direction above the ambient-random Q95 null. The fat
classification is therefore not explained only by unusually high loading on the
first few source probes.

The correct pilot wording is:

> Exploratory FS1-FS4 diagnostics support a fat, structured web-document-state slow
> region with preferred axes and gradual dilution. Because test had already been
> inspected before this battery was added, claim-level-5a confirmation remains a
> prospective full-run target.

## Stage 06: Attention Alignment With Residual-PCA Controls

Robust Stage 06 used:

- Layers: `0..12`
- Components per head: `8`
- Sampled train positions: `50,000`
- Direction groups: persistent, random, low-lifetime time-lagged
- Directions per group: `31`
- Metrics:
  - max-single-PC alignment;
  - head-subspace projection norm;
  - residual-PCA controls at `k = 16, 32, 64, 128`;
  - residual-PCA containment.

Direction-group interpretation:

Stage 06 compares persistent directions against two controls. The random group controls
for generic residual-space geometry. The low-lifetime time-lagged group is drawn from
the same time-lagged probe family as the persistent directions, but from the low-\(\tau\)
end, so it controls for artifacts of the time-lagged fitting procedure. The Stage 06
question is therefore not just whether time-lagged probes align with attention, but
whether high-lifetime directions align more than both random directions and low-lifetime
time-lagged directions.

Conceptually, Stage 06 is a backward geometric test: starting from long-lived residual
directions, it asks whether those directions lie unusually close to the subspaces that
attention heads write into the residual stream. It does not by itself show that those
heads route or causally maintain the state.

Raw max-single-PC group stats:

| Direction group | Median max-over-head alignment | Q90 max-over-head alignment |
|---|---:|---:|
| random | 0.0740 | 0.0809 |
| low-lifetime time-lagged | 0.0783 | 0.0910 |
| persistent | 0.1285 | 0.1694 |

Persistent-minus-random median differences:

| Residual PCA k | Max-PC diff | Max-PC CI | Head-subspace norm diff | Head-subspace CI |
|---:|---:|---|---:|---|
| 0 | 0.0545 | `[0.0441, 0.0748]` | 0.0890 | `[0.0703, 0.1018]` |
| 16 | 0.0543 | `[0.0425, 0.0706]` | 0.0870 | `[0.0666, 0.1036]` |
| 32 | 0.0466 | `[0.0362, 0.0755]` | 0.0776 | `[0.0654, 0.1058]` |
| 64 | 0.0415 | `[0.0361, 0.0606]` | 0.0725 | `[0.0646, 0.0906]` |
| 128 | 0.0422 | `[0.0335, 0.0482]` | 0.0644 | `[0.0502, 0.0751]` |

Metric scale:

The max-PC alignment is an absolute cosine similarity between a residual direction and
one attention-head output PC, maximized over heads and PCs. In a `d_model = 2304`
space, a single random vector-vector cosine has natural scale about `1/sqrt(2304) =
0.0208`, and the max-over-head statistic gives random directions many chances. This is
why the random median max-over-head alignment is already `0.0740`. The relevant effect
size is therefore not whether `0.1285` is close to 1, but that persistent directions
reach `0.1285` versus the matched random baseline of `0.0740`, about `1.74x` higher.

The head-subspace norm is a projection length into one head's 8D PCA output subspace,
maximized over heads. Its square is the fraction of the direction's squared norm inside
that subspace, but the reported value itself is not explained variance. For one fixed
8D subspace in 2304D, a random direction has projection norm scale about
`sqrt(8/2304) = 0.0589`; Stage 06 again compares against the empirical max-over-head
random baseline. The residual-PCA-controlled `k=128` gap of `0.0644` is therefore a
robust extra projection-length shift, not a "6.44% variance explained" number.

Residual PCA containment:

| Residual PCA k | Persistent median removed | Random median removed | Low-lifetime TL median removed |
|---:|---:|---:|---:|
| 16 | 0.0069 | 0.0056 | 0.0002 |
| 32 | 0.0305 | 0.0122 | 0.0010 |
| 64 | 0.0771 | 0.0257 | 0.0059 |
| 128 | 0.2016 | 0.0538 | 0.0324 |

Layer-level head-subspace norm deltas, persistent minus random:

| Layer | k=0 | k=128 |
|---:|---:|---:|
| 7 | 0.0375 | 0.0276 |
| 8 | 0.0256 | 0.0361 |
| 9 | 0.0090 | 0.0320 |
| 10 | 0.0455 | 0.0379 |
| 11 | 0.0634 | 0.0621 |
| 12 | 0.0809 | 0.0711 |

Top persistent heads by head-subspace norm:

| Residual PCA k | Top heads |
|---:|---|
| 0 | `L12H7`, `L11H0`, `L10H6`, `L7H2`, `L12H6` |
| 128 | `L12H3`, `L11H0`, `L9H3`, `L8H0`, `L10H6` |

Head-output PCA variance kept:

Stage 06 also records per-component explained variance in `subspace/head_output_pca.parquet`.
Summing the eight fitted PCs per `(layer, head)`, the median head keeps `0.3466` of its
attention-output variance, with mean `0.3782`, interquartile range `0.2747-0.4313`,
and range `0.1280-0.9734` across all 104 heads.

The later M3 head groups have the following 8-PC variance-kept distributions:

| Head group | Mean kept | Median kept | Min | Max |
|---|---:|---:|---:|---:|
| raw_top | 0.4915 | 0.5025 | 0.3053 | 0.6844 |
| resid_top | 0.4283 | 0.4390 | 0.3241 | 0.5681 |
| high_var | 0.3756 | 0.3989 | 0.2703 | 0.4345 |
| low_align | 0.3514 | 0.3917 | 0.1878 | 0.4336 |
| random_ctrl | 0.2605 | 0.2633 | 0.1941 | 0.3135 |

The highest variance-kept heads are dominated by early-layer heads (`L0H2`, `L0H6`,
`L0H1`, `L0H3`), but among the Stage 06/M3-relevant late heads, aligned heads also
have higher compact-output variance than random controls. This matters because Stage 06
alignment is not only selecting geometrically persistent heads; it is also somewhat
biased toward heads whose output distribution is well captured by the top eight PCs.

Interpretation:

- Persistent directions align with attention-output geometry more than random directions.
- This survives removing up to the top 128 residual PCs.
- Persistent directions are more PCA-contained than random directions, but the attention alignment is not explained away by that containment.
- The signal is strongest in late layers, especially layers 10-12.

## M0: Block-Output Subspace Comparison

M0 compared layer-wise attention block, MLP block, and residual PCA subspaces.
It asks whether the Stage 06 attention alignment is specifically tied to attention as
a forward write space, rather than merely reflecting generic high-variance residual
geometry or broad computational-output geometry.

The important residual-PCA comparison is an excess-over-baseline test: attention overlap
must exceed what would be expected from residual PCA containment. In other words, the
claim is not only that persistent directions have high attention overlap, but that their
attention-minus-residual-PCA excess is larger than the matched random-direction excess.

Run settings:

- Layers: `[7, 8, 9, 10, 11, 12]`
- Components per subspace: `32`
- Direction groups: persistent, random, low-lifetime time-lagged
- Directions per group: `31`
- Bootstrap note: CIs are direction-bootstrap over probe directions, not document-bootstrap.
- Across-layer summaries reduce to max-over-layers per `probe_id` before bootstrapping.

M0 top-level status:

```text
attention_specific
```

Across-layer excess over residual PCA geometry:

| Residual PCA k | Delta E attention | Attention CI | Attention positive | Delta E MLP | MLP CI | MLP positive |
|---:|---:|---|---|---:|---|---|
| 0 | 0.0990 | `[0.0513, 0.1217]` | true | 0.0057 | `[-0.0124, 0.0345]` | false |
| 16 | 0.0834 | `[0.0560, 0.1075]` | true | -0.0020 | `[-0.0132, 0.0231]` | false |
| 32 | 0.1069 | `[0.0870, 0.1244]` | true | 0.0317 | `[0.0181, 0.0462]` | true |
| 64 | 0.0946 | `[0.0775, 0.1133]` | true | 0.0410 | `[0.0263, 0.0475]` | true |
| 128 | 0.0813 | `[0.0721, 0.0926]` | true | 0.0349 | `[0.0232, 0.0439]` | true |

Per-layer pattern:

- Attention excess is positive and robust in late layers, especially `L10`, `L11`, and `L12`.
- `L11` attention excess is strong at every control setting.
- `L12` attention excess is strongest after residual-PCA control.
- MLP excess is not positive raw, but becomes positive in late layers under stronger residual-PCA controls, especially `L10-L12`.

Interpretation:

> Persistent directions are preferentially coupled to attention block-output geometry beyond residual PCA geometry. There is also a secondary late-MLP signal after residual-PCA control, but MLP is not the primary raw geometry signal.

This justified running M3 attention transport, while keeping the possibility open that MLPs write or update part of the persistent state.

### PCA-vs-Persistent Geometry Follow-Up

A post-hoc PCA-vs-persistent analysis compared the top persistent directions against
random directions, low-lifetime time-lagged directions, top residual PCs, later residual
PCs, and random directions sampled inside the top-128 residual-PCA span.

Selected group medians:

| Group | Median tau | Max-PC alignment | Head-subspace norm | M0 attn excess | M0 MLP excess | Residual PCA128 containment |
|---|---:|---:|---:|---:|---:|---:|
| persistent top31 | 16.0 | 0.1285 | 0.1869 | 0.1164 | 0.0274 | 0.2043 |
| random31 | 1.0 | 0.0740 | 0.0979 | 0.0193 | 0.0222 | 0.0538 |
| low-lifetime TL31 | 1.0 | 0.0964 | 0.1348 | 0.1314 | 0.1117 | 0.0380 |
| top PCA31 | 1.0 | 0.1679 | 0.2080 | -0.3991 | -0.5423 | 1.0000 |
| PCA 129-159 | 1.0 | 0.1033 | 0.1442 | 0.1445 | 0.1476 | 0.0000 |
| random in top-PCA128 span | n/a | 0.1204 | 0.1694 | -0.1715 | -0.2038 | 1.0000 |

Persistent directions beat random directions on Stage 06 metrics by `+0.0545` max-PC
alignment and `+0.0890` head-subspace norm. They also beat low-lifetime time-lagged
directions on those head-level metrics by `+0.0321` and `+0.0521`, respectively, which
supports a lifetime-specific attention-head geometry signal rather than only a
time-lagged-probe artifact.

The PCA comparisons sharpen the caveat. Top residual PCs have higher raw head alignment
than persistent directions but median tau `1.0` and strongly negative M0 excess. Thus
raw attention-head alignment alone is not a persistence signature. Conversely, low-lifetime
time-lagged and later-PCA directions can show high M0 directionwise excess, so M0 should
be read as attention-specific excess beyond residual PCA/random baselines, not as a
complete lifetime classifier by itself.

## M3: Attention Transport Status

M3 was run with exclude-self transport as the primary metric.

Top-level locked status:

```text
not_positive_or_not_evaluable
```

The primary locked metric is signed Pearson correlation between:

\[
\sum_s A^{\ell,h}_{t,s}((r_s^{\mathrm{source}})^\top v)
\]

and:

\[
o_{\ell,h,t}^\top v.
\]

The run records:

- primary metric: `pearson_r_excl_self`;
- sensitivity metric: `pearson_r_incl_self`;
- source residual hook: `hook_resid_pre (pre-attention, pre-LayerNorm)`;
- per-head/per-direction CIs bootstrapped over documents;
- group comparisons computed on head-level medians to avoid inflating \(n\) from correlated directions.

Group-level signed transport:

| Head group | Head count | Median r excl self | Q90 r excl self |
|---|---:|---:|---:|
| high_var | 6 | 0.0276 | 0.1421 |
| low_align | 6 | 0.1130 | 0.2146 |
| random_ctrl | 6 | 0.2165 | 0.3522 |
| raw_top | 5 | 0.3214 | 0.4698 |
| resid_top | 5 | 0.1116 | 0.2123 |

Aligned-vs-control comparison:

| Control | Aligned median r | Control median r | Difference | Positive |
|---|---:|---:|---:|---|
| low_align | 0.2007 | 0.1130 | 0.0877 | true |
| random_ctrl | 0.2007 | 0.2165 | -0.0158 | false |
| high_var | 0.2007 | 0.0276 | 0.1731 | true |

Locked interpretation:

> M3 is not positive. Aligned heads beat low-alignment and high-variance controls, but do not beat same-layer random controls. Claim level 7b is not supported by the locked group-level transport gate.

One artifact issue remains: the summary reports `m0_status: missing` even though M0 was run and was positive. This is a reporting-linkage issue, not evidence against M0.

### Exploratory M3b Diagnostics

Per-head diagnostics show real transport-like structure, but not clean aligned-head group specificity.

Top persistent signed transport heads:

| Head group | Head | Median signed r |
|---|---|---:|
| raw_top | L12H7 | 0.5543 |
| random_ctrl | L11H2 | 0.4103 |
| raw_top | L7H2 | 0.3430 |
| raw_top | L10H6 | 0.3214 |
| raw_top | L11H0 | 0.3062 |
| random_ctrl | L8H4 | 0.2941 |
| resid_top | L8H0 | 0.2587 |
| raw_top | L12H6 | -0.5764 |

Top persistent absolute transport heads:

| Head group | Head | Median abs r |
|---|---|---:|
| raw_top | L12H6 | 0.5764 |
| raw_top | L12H7 | 0.5543 |
| random_ctrl | L11H2 | 0.4103 |
| raw_top | L7H2 | 0.3430 |
| raw_top | L10H6 | 0.3214 |
| raw_top | L11H0 | 0.3062 |
| random_ctrl | L8H4 | 0.2941 |

Largest persistent-minus-random absolute transport gaps:

| Head group | Head | Persistent minus random abs |
|---|---|---:|
| raw_top | L12H6 | 0.3610 |
| raw_top | L7H2 | 0.3004 |
| raw_top | L12H7 | 0.2621 |
| random_ctrl | L11H2 | 0.2360 |
| raw_top | L10H6 | 0.2183 |
| raw_top | L11H0 | 0.1307 |
| low_align | L9H7 | 0.1281 |
| random_ctrl | L8H4 | 0.1203 |

Exploratory interpretation:

> Several raw-top heads show strong persistent-direction transport, and `L12H6` shows a large sign-flipped or suppressive transport signal. However, transport-like correlations also appear in random-control heads. The best current interpretation is that transport is a broader late-attention property, with some head-specific persistent-direction structure, not a clean Stage-06-aligned-head group result.

These M3b diagnostics are post-hoc exploratory and do not support claim level 7b.

### Exploratory M1-lite Implication

The broad locked M1 was not run as claim-level evidence because locked M3 did not pass. Instead, exploratory M1-lite was run with head sets selected from M3b diagnostics.

| Set | Heads | Purpose |
|---|---|---|
| positive transport candidates | `L12H7`, `L10H6`, `L11H0`, `L7H2` | Test whether positive persistent transport predicts causal persistence support |
| negative transport candidate | `L12H6` | Test suppressive or anti-transport interpretation; track whether ablation increases persistence |
| transport-positive controls | `L11H2`, `L8H4`, `L12H1` | Control for generic transport strength not selected by Stage 06 alignment |
| low/negative controls | `L11H4`, `L8H6`, `L12H2` | Weak or negative transport comparison |

M1-lite status:

```text
exploratory_completed
```

M1-lite preserves the claim boundary:

- `post_hoc_head_selection=true`;
- `m0_status=attention_specific`;
- `claim_level_8_supported=false`;
- evaluation probes: fixed `J_lag-heldout`, 128 probes;
- baseline clean median \(\tau\): 4.0.

Head-level M1-lite results:

| Head group | Head | C_head | Clean median tau | Ablated median tau | Effect |
|---|---|---:|---:|---:|---|
| positive_transport_candidate | `L12H7` | 0.375 | 4.0 | 2.5 | ablation reduced persistence |
| positive_transport_candidate | `L11H0` | 0.250 | 4.0 | 3.0 | ablation reduced persistence |
| transport_positive_control | `L11H2` | 0.250 | 4.0 | 3.0 | ablation reduced persistence |
| positive_transport_candidate | `L7H2` | 0.000 | 4.0 | 4.0 | no median change |
| positive_transport_candidate | `L10H6` | 0.000 | 4.0 | 4.0 | no median change |
| transport_positive_control | `L8H4` | 0.000 | 4.0 | 4.0 | no median change |
| transport_positive_control | `L12H1` | 0.000 | 4.0 | 4.0 | no median change |
| low_negative_control | `L8H6` | 0.000 | 4.0 | 4.0 | no median change |
| low_negative_control | `L11H4` | 0.000 | 4.0 | 4.0 | no median change |
| low_negative_control | `L12H2` | 0.000 | 4.0 | 4.0 | no median change |
| negative_transport_candidate | `L12H6` | -0.250 | 4.0 | 5.0 | ablation increased persistence |

Group-level exploratory comparison:

| Comparison | Candidate median C | Control median C | Difference |
|---|---:|---:|---:|
| positive candidates vs transport-positive controls | 0.125 | 0.000 | 0.125 |
| positive candidates vs low/negative controls | 0.125 | 0.000 | 0.125 |

Exploratory interpretation:

> M1-lite suggests a head-specific causal relationship: `L12H7` and `L11H0` reduce held-out persistence when ablated, while `L12H6` appears suppressive because ablating it increases persistence. However, `L11H2`, a transport-positive control, matches `L11H0`, so the result is not cleanly specific to the positive transport candidate group. The clean median \(\tau\) is only 4.0, so exact \(C_{\mathrm{head}}\) magnitudes are quantized and should be interpreted cautiously.

These results are validation-informed and exploratory. They do not support claim level 8.

## Semantic Dossier Follow-Up

The semantic dossiers are a first-pass, validation-split qualitative analysis of the
top persistent directions. They should be treated as hypothesis generation, not as
semantic labels.

Main readout:

- The directions do not look like clean single-topic features.
- They look more like persistent document-state, register, domain, and source-template
  axes.
- Recurring high/low-side patterns include technical or instructional exposition,
  legal/privacy boilerplate, SEO/product/catalog repetition, numeric/table/reference
  structure, biomedical/scientific citation style, narrative/personal/devotional prose,
  and noisy scraped-template text.
- High-activation examples often persist over long contiguous spans within a document,
  supporting document-level or section-level state rather than isolated token triggers.

A shallow label-enrichment check gives coarse support for several direction-level
patterns, for example technical/software enrichment for rank 1, product/catalog
enrichment for rank 3, numeric/table enrichment for several directions, biomedical
enrichment for ranks 19 and 30, recipe/lifestyle enrichment for rank 29, and
religious/devotional enrichment for ranks 23 and 31. These labels are regex/density
checks on small top/bottom/span samples, so they are noisy and overlapping. The safe
interpretation is that the persistent subspace appears to encode durable document
register/template state, not that any direction has a settled semantic name.

A controlled validation-only comparison now adds sampled rotations and matched
controls:

| Direction group | Directions | Median tau | Median Q95 spans with length at least 8 | Median retained span length |
|---|---:|---:|---:|---:|
| selected persistent references | 10 | 36.5 | 530.5 | 17.16 |
| random vectors inside `Q31` | 10 | 21.5 | 473.0 | 14.21 |
| random vectors inside PCA-31 | 10 | 3.0 | 28.0 | 9.61 |
| random vectors inside PCA-128 | 10 | 1.0 | 20.5 | 9.41 |
| random vectors inside PCA-256 | 10 | 1.0 | 14.0 | 9.18 |
| ambient random residual directions | 10 | 1.0 | 11.5 | 9.17 |

All 10 sampled `Q31` rotations are slow, with validation tau from `14` to `171`.
Their tails often separate durable document regimes: formal technical or
institutional exposition, legal/commercial pages, travel and narrative text, and
scraped template artifacts. Matched PCA-span and ambient-random controls can still
produce narratable examples, but their persistence and retained-span counts are much
weaker. The qualitative interpretation should therefore be framed as web-document
state geometry, not a basis of discrete semantic concepts.

A representative document-deduplication sanity check collapsed the exported top 12
windows and spans to distinct source documents. For sampled `Q31` rotations, the top
12 windows cover a median of `7` documents, the exported top 12 retained spans cover
a median of `8.5` documents, and the largest single document accounts for a median of
one third of exported top spans. This rules out the simplest one-unusual-page failure
mode. A full artifact-only document-deduplicated recount over all retained spans is
still pending and should replace this representative check when available.

## Current Scientific Interpretation

Current findings:

1. Layer-12 residual directions in Gemma-2-2B have nontrivial temporal persistence.
2. Time-lagged residual probes reveal a heavy-tailed persistence distribution that random and individual PCA probes mostly miss.
3. The persistence depends strongly on ordered document structure.
4. The top signal is spread across roughly tens of nonduplicate directions.
5. Fixed held-out time-lagged probes collapse on both validation and test under residual-first projection, but not random-control projection.
6. The held-out persistent directions are strongly PCA-contained as a subspace.
7. Persistent directions align with attention-output geometry beyond residual-PCA controls.
8. M0 supports attention-specific block-output geometry, especially in late layers.
9. Exploratory M1-lite suggests head-specific causal effects, including support-like `L12H7`/`L11H0` and suppressive `L12H6`, but does not support locked causal head-support.
10. Post-hoc semantic dossiers suggest persistent directions track durable document-state/register/template axes rather than token-local triggers.
11. Exploratory FS1-FS4 diagnostics show that generic sampled rotations inside the exact `Q31` candidate pool remain slow on validation and test, including the lower tail and lower-ranked bands.
12. Artifact-only robustness checks show that the fat-span result is not explained only by high loading on the first few source probes.

Important refinements and caveats:

1. The current compact-subspace story is also a PCA-contained-subspace story.
2. Individual PCA axes are mostly not persistent, despite the persistent subspace being PCA-contained.
3. M0 CIs are direction-bootstrap, not document-bootstrap.
4. M0 shows a secondary late-MLP signal after residual-PCA control; attention is the primary raw and robust signal, but MLP involvement remains plausible.
5. Locked M3 is negative because aligned heads do not beat same-layer random controls.
6. Exploratory M3b shows head-specific persistent transport structure, but not clean aligned-head group specificity.
7. M1-lite is exploratory because the head sets are selected from validation-stage transport diagnostics.
8. M1-lite \(C_{\mathrm{head}}\) values are quantized by a short clean median \(\tau=4.0\).
9. Top residual PCs can have stronger raw attention-head alignment than persistent directions despite median tau `1.0`; raw alignment alone is not the persistence claim.
10. Semantic labels are qualitative validation-set hypotheses, not validated concept labels.
11. Generic random rotations inside the pilot `Q31` span are empirically slow, but
    this is exploratory evidence because test was already inspected before FS1-FS4.
    A prospective claim-level-5a confirmation requires validation-locked directions,
    metrics, and interpretation rules applied unchanged to untouched full-run test.
12. `k_80pct_lifetime_excess = 31` defines the current candidate pool, not an assumed
    intrinsic dimension. The measured slowness-concentration participation ratio
    `4.71` remains substantially below geometric rank `28.22`, and the nested curve
    shows gradual dilution. No smaller effective-core cutoff was validation-locked.
13. The semantic rotation comparison is validation-only and qualitative. A full
    artifact-only document-deduplicated recount remains pending, and cleaner-corpus or
    source-template robustness is still required before claiming abstract semantic
    state rather than web-document state.

## Best Current Summary

In a Gemma-2-2B layer-12 residual-stream pilot on C4, random and individual PCA probes are almost entirely short-timescale, while time-lagged residual probes reveal a heavy-tailed persistence distribution. The top persistent signal disappears under document permutation and is spread across roughly tens of nonduplicate directions: one lead direction is large, but the top-31 set remains high-rank and nonredundant. Robust projection collapse with fixed held-out probes confirms on both validation and test that independently fit persistent directions are removed by residual-first and PCA bases, but barely by random-control bases. Exploratory FS1-FS4 diagnostics further show that generic sampled rotations inside the exact `Q31` candidate pool remain slow on validation and test, including lower-tail directions and lower-ranked bands, while direct PCA-span controls remain mostly short-lived. This supports a fat, structured web-document-state slow region with preferred axes and gradual dilution. It does not prospectively establish claim level 5a because test had already been inspected before the FS battery was added. Persistent directions align with attention-output geometry beyond residual-PCA controls, and M0 shows attention-specific block-output geometry, especially in late layers. PCA-vs-persistent diagnostics show that raw attention-head alignment alone is not sufficient, because top residual PCs align strongly but are not persistent. Locked M3 does not show aligned-head transport above all matched controls. Exploratory per-head diagnostics and M1-lite suggest head-specific transport and causal effects, including support-like `L12H7`/`L11H0` and suppressive `L12H6`, but those mechanistic findings are post-hoc and not claim-level evidence. Semantic dossiers and controlled rotation samples suggest the slow region tracks durable web-document register/template state rather than clean single-topic concepts.

## Claims Not Yet Supported

Do not yet claim:

1. The directions are validated interpretable discourse variables or clean semantic concepts.
2. The identified heads causally write, route, or maintain the persistent state.
3. Locked group-specific attention transport has been demonstrated.
4. Head ablation has shown causal support.
5. The phenomenon explains SAE feature autocorrelation.
6. The persistent subspace is PCA-orthogonal or invisible to PCA as a subspace.
7. The full result generalizes beyond this layer/model/corpus before full-scale confirmation.
8. Claim-level-5a prospective confirmation has been established. The pilot FS battery
   is exploratory because test was already inspected before it was added.
9. Every direction inside the top-31 span is slow. The finite random-in-span sample
   supports a generic sampled-direction statement, not a universal statement.
10. A particular smaller effective slow-core dimension has been established. The
    nested curve was measured, but no cutoff rule or interval was validation-locked
    before test inspection.

## Immediate Next Steps

1. Regenerate `reports/residual_geometry_report.md` and `reports/residual_geometry_summary.json` with M0, M3, M1-lite, FS1-FS4, artifact-only robustness, and semantic-control follow-ups included.
2. Stop adding post-hoc M3 variants for this pilot. Treat M3b/M1-lite and the semantic dossiers as exploratory.
3. If a stronger mechanism claim is needed, write a prospectively locked M3-v2/M1-v2 follow-up before running new tests.
4. Finish the full artifact-only semantic document-deduplication recount and add its
   result next to the representative top-12 sanity check.
5. If a stronger semantic claim is needed, validate predeclared dossier labels on a
   cleaner corpus or source-template-controlled split.
6. Prospectively lock the full-run FS1-FS4 sweep, sampled directions, null-band
   definition, interpretation rules, and any effective-core cutoff rule before
   inspecting full-run test.
7. For the current pilot writeup, claim levels `1-5`, `6`, and `7a` are supported.
   Claim level `5a` remains exploratory rather than prospectively confirmed, and
   claim levels `7b` and `8` are not supported by locked tests.

## Artifact Paths

Local artifact root:

```text
outputs/persistent_state_residual_geometry/pilot/residual_geometry_pilot/
```

Important files:

```text
reports/residual_geometry_summary.json
subspace/dimensionality_summary.json
subspace/projection_collapse_summary.json
subspace/projection_collapse_summary.parquet
subspace/attention_alignment_summary.json
subspace/residual_direction_head_alignment.parquet
subspace/block_output_subspace_summary.json
subspace/block_output_subspace_overlap.parquet
subspace/attention_transport_summary.json
subspace/attention_transport.parquet
subspace/head_ablation_m1_lite_summary.json
subspace/head_ablation_m1_lite_autocorr.parquet
subspace/persistent_direction_analysis/summary.md
subspace/pca_vs_persistent_alignment/summary.md
subspace/semantic_dossiers/analysis_summary.md
subspace/semantic_dossiers/shallow_label_enrichment/shallow_label_enrichment_summary.md
subspace/fat_subspace/fat_subspace_summary.json
subspace/fat_subspace/nested_dimension_sweep.parquet
subspace/fat_subspace/source_probe_rank_curve.parquet
subspace/fat_subspace/pca_embedding_geometry.parquet
subspace/fat_subspace/source_probe_split_stability.parquet
subspace/fat_subspace/artifact_robustness/summary.json
subspace/fat_subspace/artifact_robustness/family_lower_tail_summary.csv
subspace/fat_subspace/artifact_robustness/core_loading_sensitivity.csv
subspace/fat_subspace/artifact_robustness/split_stability_summary.csv
subspace/fat_subspace/plots/random_in_span_tau_histogram.png
subspace/fat_subspace/plots/nested_dimension_tau_curve.png
subspace/fat_subspace/plots/source_probe_rank_tau_curve.png
subspace/fat_subspace/plots/mean_autocorrelation_profiles.png
subspace/fat_subspace/plots/pca_containment_curve.png
subspace/semantic_axis_comparison_saved/exploratory_semantic_comparison.md
subspace/semantic_axis_comparison_saved/group_summary.csv
subspace/semantic_axis_comparison_saved/representative_document_dedup_summary.csv
subspace/semantic_axis_comparison_saved/representative_document_dedup_group_summary.csv
```
