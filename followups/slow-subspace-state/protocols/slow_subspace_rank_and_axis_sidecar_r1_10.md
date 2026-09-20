# R1.10 Sidecar — Rank-Extent and Interpretable-Coordinate Audit

**Status:** Proposed / freeze before execution  
**Purpose:** Cheap, mostly artifact-only characterization of (A) how far the reproducible slow region extends beyond the canonical rank-31 slice and (B) whether the already-stable leading TICA axes / content contrasts admit simple descriptive interpretation.

This is a **sidecar**, not a new headline experiment.

It must not:
- redefine the canonical slow object post hoc;
- change any R1.7–R1.9 conclusions;
- launch new semantic labeling campaigns;
- introduce new forward passes unless an already-required projection artifact is genuinely missing.

The canonical object for all established claims remains the frozen signed-TICA \(Q_{31}\).

---

# 0. Scientific questions

## A. Rank extent

The original \(k=31\) cutoff was operational:

\[
k_{80\%\,\mathrm{lifetime\ excess}}=31.
\]

It was never claimed to be the intrinsic dimension of persistent geometry.

R1.7 showed split-fit stability increasing through the registered nested ranks:

\[
S_8=.7698,\quad
S_{13}=.7479,\quad
S_{21}=.8446,\quad
S_{31}=.8641.
\]

The sidecar asks:

> Does the reproducible, generically slow region extend substantially beyond rank 31?

The target is a **rank curve**, not a new optimized rank.

---

## B. Interpretable stable coordinates

R1.8 established that the first five matched TICA axes are highly reproducible across split fits:

\[
|\cos| =
[.9810,.9813,.8999,.9717,.9757].
\]

The sidecar asks:

> Do these stable dynamical axes correspond to simple persistent variables, and are the already-identified topic/content contrasts themselves individually reproducible and interpretable?

This is descriptive model biology.

---

# 1. Frozen inputs

Reuse existing artifacts only wherever possible.

Required:

- canonical signed-TICA candidate bank and ordered probe metadata;
- canonical \(Q_k\) construction procedure;
- split-A and split-B signed-TICA candidate banks from R1.7;
- R1.7 selected-axis / generalized-eigenvalue metadata;
- existing validation/test residual projection artifacts or exact evaluator;
- R1.8 semantic tensor and document split;
- R1.8/R1.9 topic labels;
- R1.9 fingerprint labels/features;
- R1.9 conditioned topic classifiers / coefficient matrices if saved;
- document text or context IDs for qualitative examples.

No estimator refitting is authorized unless the full ordered A/B candidate banks were not saved and cannot otherwise form the registered nested spans.

No new Gemma capture unless a required existing projection cannot be reconstructed exactly from saved artifacts.

---

# 2. Part A — Rank-extent curve

## 2.1 Registered ranks

Use exactly:

\[
k\in\{8,13,21,31,48,64,96,128\}.
\]

Do not add ranks after inspecting the curve.

Ranks \(8,13,21,31\) connect directly to prior results.  
Ranks \(48,64,96,128\) test whether the stable/generically slow region extends materially beyond the original operational cutoff.

---

## 2.2 Nested span construction

For canonical, split-A, and split-B fits, construct the nested top-\(k\) span from the exact frozen signed-TICA ranking / candidate-selection procedure.

For each \(k\), save orthonormal bases:

\[
Q_{\mathrm{canon}}^{(k)},
\quad
Q_A^{(k)},
\quad
Q_B^{(k)}.
\]

Do not rerank using this sidecar's outputs.

---

## 2.3 Geometric stability

For each \(k\), compute:

\[
S_k
=
\frac{1}{k}
\left\|
Q_A^{(k)\top}Q_B^{(k)}
\right\|_F^2.
\]

Also report the ambient random expectation:

\[
E[S_k^{\rm rand}]=\frac{k}{2304}.
\]

Primary comparison uses chance-corrected overlap:

\[
\boxed{
\widetilde S_k
=
\frac{
S_k-k/2304
}{
1-k/2304
}
}
\]

so stability is more comparable across ranks.

Also save:
- canonical correlations;
- median principal angle;
- minimum canonical correlation.

These are descriptive only.

---

## 2.4 Functional slow-span curve

For each canonical \(Q_{\mathrm{canon}}^{(k)}\):

1. sample 512 random unit directions uniformly in the span;
2. use the canonical row-wise random-in-span procedure;
3. use frozen sampling seed 31;
4. evaluate the original **signed** timescale metric on untouched test data.

For each \(k\), report:

- Q05;
- Q25;
- median;
- Q75;
- Q90;
- Q95;
- max;
- mean;
- censoring fraction.

Primary functional statistic:

\[
m_k
=
\operatorname{median}
\tau(\mathrm{random\ in}\ Q_{\mathrm{canon}}^{(k)}).
\]

If exact test projection artifacts for larger \(k\) already exist, reuse them.

If not, project only the needed frozen \(Q_k\) directions / sampled in-span directions during one minimal held-out pass. Do not store raw 2304D residuals.

---

## 2.5 Split-fit functional confirmation

For \(k\in\{31,64,128\}\) only, repeat the 512-direction random-in-span test for \(Q_A^{(k)}\) and \(Q_B^{(k)}\) on the common untouched test set.

Purpose:

> verify that any apparent broad extension is reproducible across independent fits, rather than only present in the canonical full-data ranking.

Do not run split-fit functional curves at every rank.

---

## 2.6 Lifetime-excess context

From the frozen candidate bank, report cumulative fraction of positive lifetime excess captured through each registered rank:

\[
F_k
=
\frac{
\sum_{j=1}^{k}\Delta\tau_j
}{
\sum_j\Delta\tau_j
}.
\]

This is descriptive context only.

Do not choose a new canonical rank from \(F_k\).

---

# 3. Rank-curve interpretation

Do **not** define

\[
k^\star=\arg\max_k S_k
\]

or otherwise optimize a rank from the sidecar.

The only allowed interpretations are descriptive.

### `COMPACT_CORE`

Use if:
- test random-in-span median drops sharply after the low ranks;
- e.g. larger-\(k\) medians approach ordinary random/PCA timescales;
- and/or split-fit stability degrades materially as \(k\) grows.

Interpretation:

> \(Q_{31}\) is a reasonable compact reference slice of a more rapidly diluting slow spectrum.

### `BROAD_PERSISTENT_REGION`

Use if:
- median random-in-span timescale remains clearly elevated through \(k=64\) or beyond;
- and split-fit stability remains high.

Interpretation:

> the persistent region is broader than the original operational rank-31 cutoff; \(Q_{31}\) remains the frozen canonical compact slice for historical claims.

### `GRADED_SLOW_SPECTRUM`

Use when there is no sharp boundary.

Interpretation:

> persistent geometry forms a graded spectrum rather than a naturally discrete-dimensional subspace.

No outcome changes the canonical R1.7–R1.9 object retrospectively.

---

# 4. Part B — Leading TICA-axis dossiers

## 4.1 Authorized axes

Interpret only the first five registered matched axes from R1.8.

An axis remains authorized only if its A/B matched absolute cosine is:

\[
|\cos(v_j^A,v_{\pi(j)}^B)|\ge0.80.
\]

Use the matched direction identity, not naive rank equality.

Do not interpret tail axes that fail the stability gate.

---

## 4.2 Per-axis quantitative dossier

For each authorized axis \(v_j\), compute on the existing semantic dataset:

### Variance structure
- between-document variance fraction;
- within-document variance fraction.

### Association with content
Use the existing topic labels.

Report:
- one-way explained variance / ANOVA-style effect size across the three topic classes;
- train-only linear prediction if already implemented;
- held-out effect summary.

### Association with fingerprint
Using R1.9 variables, report compact associations with:
- source type;
- register;
- formatting;
- document position.

Purpose: determine whether an axis is primarily content-like, register-like, structural, positional, or mixed.

No new labels.

---

## 4.3 Qualitative examples

For each authorized axis, inspect a **small fixed sample**:

- 8 highest document-level mean projections;
- 8 lowest document-level mean projections;
- 8 highest positive token-level deviations from document mean;
- 8 lowest token-level deviations from document mean.

For token-level examples, show a short local text window around the token.

The distinction between document mean and within-document deviation is mandatory:

\[
a_{d,t}
=
\bar a_d
+
(a_{d,t}-\bar a_d).
\]

This prevents a document-level topic/source signal from being conflated with within-document state changes.

No more than 32 examples per axis.

---

## 4.4 Descriptive axis labels

Do not force a semantic name.

Allowed outcome labels per axis:

```text
CONTENT_DOMINANT
REGISTER_DOMINANT
SOURCE_STYLE_MIXED
POSITION_STRUCTURAL
MIXED_INTERPRETABLE
NO_SIMPLE_INTERPRETATION
```

These labels are descriptive, not inferential claims.

A label may be assigned only if:
- quantitative associations and examples agree;
- the pattern appears in both split-fit matched directions.

Otherwise use `NO_SIMPLE_INTERPRETATION`.

---

# 5. Part C — Content-contrast coordinate dossiers

The R1.9 classifier has three topic classes:

```text
other_or_unclear
home_food_and_lifestyle
business_and_finance
```

Individual softmax class weights are not canonical because adding the same vector to all class weights leaves probabilities unchanged.

Therefore interpret only centered or pairwise contrast directions.

---

## 5.1 Canonical contrast construction

For each split-fit conditioned classifier:

\[
\tilde w_c
=
w_c-\frac{1}{C}\sum_{j=1}^{C}w_j.
\]

Map the slow-coordinate classifier weights back to residual space first.

Construct the three pairwise contrasts:

\[
v_{BF-HF}
=
\tilde w_{\rm business}
-
\tilde w_{\rm home},
\]

\[
v_{BF-O}
=
\tilde w_{\rm business}
-
\tilde w_{\rm other},
\]

\[
v_{HF-O}
=
\tilde w_{\rm home}
-
\tilde w_{\rm other}.
\]

Normalize each to unit norm.

---

## 5.2 Contrast reproducibility

For each fixed contrast, compute:

\[
c_{\rm contrast}
=
|\cos(v_{\rm contrast}^{A},v_{\rm contrast}^{B})|.
\]

Also evaluate each A-derived contrast on B/test and B-derived contrast on A/test using the already-frozen classifier framework.

Define a contrast as `REPRODUCIBLE_CONTRAST` if:

\[
|\cos|\ge0.80
\]

and held-out ordering / discrimination has the same sign in both fits.

No new permutation bank is required; R1.8/R1.9 already established the 2D semantic plane globally.

---

## 5.3 Contrast examples

For each reproducible contrast:

- 10 highest document means;
- 10 lowest document means;
- optionally 10 strongest within-document positive/negative deviations if those deviations are nontrivial.

Use the same fixed text-display format as the TICA-axis dossiers.

Purpose:

> determine whether the abstract topic plane contains simple continuous semantic coordinates or only becomes interpretable jointly as a 2D subspace.

---

# 6. Optional relation between slow axes and content contrasts

Compute, for each authorized TICA axis \(v_j\) and each reproducible content contrast \(c_\ell\):

\[
|\langle v_j,c_\ell\rangle|.
\]

Report the matrix.

This is descriptive only.

Possible outcomes:

- a leading dynamical axis nearly coincides with a semantic contrast;
- semantics is distributed across several slow axes;
- leading axes and semantic plane are largely distinct within the same slow region.

Do not claim causal or exclusive semantic localization.

---

# 7. Explicit non-goals

Do **not** perform:

- new semantic labeling;
- SAE analysis;
- causal intervention;
- matched-suffix experiments;
- new model/corpus/layer;
- new TICA objective;
- new rank selection rule;
- axis ablation;
- semantic steering;
- exhaustive activation-max mining;
- broad ontology construction.

This sidecar exists only to characterize already-established objects.

---

# 8. Compute budget and stopping rule

Target:

- CPU/offline analysis wherever possible;
- at most one minimal held-out forward pass if larger-rank signed projections are genuinely unavailable;
- no full residual recapture;
- no iterative branching.

Stop after:

1. the registered rank curve;
2. top-five axis dossiers;
3. fixed topic-contrast dossiers.

Do not launch follow-up computation automatically.

---

# 9. Required outputs

```text
r1_10_sidecar_config.json
provenance.json

rank_curve/
    nested_geometry.csv
    random_in_span_test.parquet
    splitfit_functional_summary.csv
    lifetime_excess_curve.csv
    rank_curve_decision.json

axes/
    axis_quantitative_summary.csv
    axis_examples.md
    axis_labels.json

content_contrasts/
    contrast_stability.csv
    contrast_examples.md
    contrast_axis_alignment.csv

r1_10_sidecar_report.md
```

---

# 10. Final interpretation target

The sidecar should answer only:

1. **How broad is the reproducible slow region beyond the frozen Q31 slice?**
2. **Do the stable leading temporal axes admit simple interpretation?**
3. **Do the identified content contrasts form reproducible interpretable coordinates inside the persistent region?**

A successful sidecar may support wording like:

> “The canonical Q31 is a compact slice of a broader graded persistent region, and several individually reproducible temporal axes / content contrasts admit stable descriptive interpretation.”

or:

> “The slow region is reproducible at the subspace level, but simple axis semantics remain limited.”

Neither outcome changes the already-frozen R1.7–R1.9 claims.
