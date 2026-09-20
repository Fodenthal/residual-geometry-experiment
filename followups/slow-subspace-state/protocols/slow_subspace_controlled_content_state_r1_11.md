# R1.11 — Controlled Persistent Content-State Test

**Status:** Proposed / freeze before execution  
**Purpose:** Test whether a frozen, independently reproducible content coordinate inside the canonical signed-TICA slow subspace retains a donor-topic-specific effect for tens to hundreds of tokens after the subsequent token sequence has been made exactly identical.

This is the first experiment in the line whose primary object is explicitly:

\[
\boxed{\text{persistent semantic state under a controlled history intervention}}
\]

rather than static document-level decoding.

## 0. Motivation

R1.7–R1.10 established:

1. a reproducible signed-TICA persistent region;
2. a canonical compact \(Q_{31}\) with a broader persistent shoulder to roughly rank 64;
3. strong enrichment for document-level state;
4. reproducible topic information inside \(Q_{31}\);
5. approximately 87% retention of the topic effect after conditioning on measured source/register/formatting fingerprint;
6. a reproducible two-dimensional content plane;
7. reproducible pairwise content contrasts with A/B cosines around \(0.89\)–\(0.90\);
8. leading TICA axes that mostly encode register/source structure rather than coinciding with semantic contrast axes.

The remaining key ambiguity is:

> Does the identified content geometry merely correlate with document-level content, or does it behave like a state variable whose value depends on earlier semantic history and persists after the local token sequence is fixed?

An earlier generic matched-suffix splice experiment showed strong remote-history sensitivity, but same-topic swaps were also large, so it could not isolate abstract topic from generic splice/document-identity effects.

R1.11 reuses the matched-suffix intervention idea with the **frozen R1.9 content coordinate as the instrument** and with matched same-topic donors as the generic-splice control.

## 1. Scientific question

Take two sequences with exactly the same recipient suffix but different prefixes.

Reference:

\[
x^{B}
=
B_{1:t_0}
\Vert
B_{t_0+1:T}.
\]

Spliced:

\[
x^{A\rightarrow B}
=
A_{1:t_0}
\Vert
B_{t_0+1:T}.
\]

For every measured horizon \(k\ge0\),

\[
x^{A\rightarrow B}_{t_0+1:t_0+k}
=
x^{B}_{t_0+1:t_0+k}.
\]

Thus the observed tokens after the splice are identical.

The primary question is:

\[
\boxed{
\text{Does changing only the earlier topic shift the frozen content coordinate
toward the donor topic at }t_0+k\text{?}
}
\]

If yes at long enough \(k\), then the current residual representation carries a semantic effect of remote history despite an identical subsequent token sequence.

## 2. Scientific object

Use only the two substantive R1.9 topic classes:

```text
business_and_finance
home_food_and_lifestyle
```

Do not use `other_or_unclear` in the primary intervention because it is intentionally heterogeneous.

Let the frozen fingerprint-conditioned canonical residual-space contrast be

\[
c_{\rm BH}
=
\frac{
w_{\rm business}-w_{\rm home}
}{
\|w_{\rm business}-w_{\rm home}\|_2
}.
\]

This contrast lies inside the canonical \(Q_{31}\) slow space by construction.

Also retain the independently fitted split-A and split-B versions:

\[
c_{\rm BH}^{A},
\qquad
c_{\rm BH}^{B}.
\]

These are frozen before R1.11 data are captured.

No semantic probe is refit on R1.11 data.

## 3. Primary signed semantic score

For a recipient \(B\), let the different-topic donor have class \(y_A\neq y_B\).

Define orientation

\[
s_{A,B}
=
\begin{cases}
+1,& y_A=\mathrm{business},\ y_B=\mathrm{home},\\
-1,& y_A=\mathrm{home},\ y_B=\mathrm{business}.
\end{cases}
\]

For condition \(c\) and horizon \(k\), define the paired residual difference

\[
\Delta h_{c,k}
=
h_{c,t_0+k}
-
h_{\rm ref,t_0+k}.
\]

The donor-directed content displacement is

\[
\boxed{
T_c(k)
=
s_{A,B}\,
c_{\rm BH}^{\top}\Delta h_{c,k}.
}
\]

Positive \(T_c(k)\) means the representation moved toward the donor's topic relative to the unspliced recipient.

The same orientation \(s_{A,B}\) is used for the matched same-topic control belonging to that recipient.

This is essential: a generic splice shock is not sufficient. The effect must point in the semantic direction specified *before* the activation is observed.

## 4. Conditions

Each recipient \(B\) receives exactly three forward sequences.

### C0 — reference / no-op

\[
B_{1:t_0}\Vert B_{t_0+1:T}.
\]

This is the natural recipient sequence and the reference activation.

### C1 — matched same-topic splice

\[
A_{\rm same,1:t_0}\Vert B_{t_0+1:T}
\]

where

\[
y_{A_{\rm same}}=y_B.
\]

Purpose:

> measure generic effects of replacing the prefix, including document identity, splice unnaturalness, source/register mismatch, and other non-topic history changes.

### C2 — matched different-topic splice

\[
A_{\rm diff,1:t_0}\Vert B_{t_0+1:T}
\]

where

\[
y_{A_{\rm diff}}\neq y_B.
\]

Purpose:

> add a controlled semantic change in earlier content while preserving the same recipient suffix.

No additional splice families are authorized.

## 5. Primary estimand

For each recipient:

\[
D_B(k)
=
T_{\rm diff}(k)-T_{\rm same}(k).
\]

Population estimand:

\[
\boxed{
M(k)
=
E_B[D_B(k)].
}
\]

Interpretation:

- \(M(k)>0\): different-topic history pushes the current content representation toward the donor topic beyond the generic same-topic splice floor;
- \(M(k)\approx0\): any content-plane perturbation is adequately explained by generic prefix replacement;
- \(M(k)<0\): the intervention does not behave as predicted by donor topic.

This difference-in-differences style contrast is the primary identification device.

## 6. Primary horizon and trajectory

Use frozen splice boundary:

\[
t_0=256.
\]

Use the existing 1024-token convention.

Measure:

\[
k\in\{0,32,64,128,256\}.
\]

The **primary confirmatory horizon** is:

\[
\boxed{k=64}.
\]

Reason:

- it is long relative to token-local effects;
- it lies well beyond the ambient/random \(\tau\sim1\) baseline;
- it is within the established slow-region timescale regime;
- it avoids making the primary claim depend on an extremely long 256-token tail.

The full trajectory is required for interpretation but does not multiply the primary test.

## 7. Fresh document pool

Use a fresh document-disjoint C4 pool not used in R1.8/R1.9 semantic fitting or evaluation.

Freeze corpus/version and context-preprocessing contract before capture.

Requirements:

- at least 1024 tokens;
- no BOS / same tokenizer convention as canonical residual-geometry work;
- exact layer-12 `hook_resid_post`;
- no document overlap with the 1,200-document semantic audit.

Use the already-frozen R1.9 topic and fingerprint labeling procedures.

Do not tune the ontology on R1.11.

## 8. Candidate-pool stopping rule

Scan and label fresh documents until there are at least:

\[
224
\]

eligible `business_and_finance` documents and

\[
224
\]

eligible `home_food_and_lifestyle` documents.

Then stop scanning.

This provides enough support for 128 recipient triplets with no document reuse plus matching slack.

No residual forward pass is run on the unused candidate pool.

## 9. Recipient and donor allocation

Use:

\[
N_{\rm recipient}=128.
\]

Balance recipient topic exactly:

```text
64 business recipients
64 home/lifestyle recipients
```

Each recipient uses two unique donors:

- one same-topic donor;
- one different-topic donor.

No document may appear more than once anywhere in the 128 triplets.

Total production documents used:

```text
128 recipients
128 same-topic donors
128 different-topic donors
384 unique documents
```

This keeps the recipient as the inferential unit and avoids donor-reuse dependence.

## 10. Fingerprint-matched donor pairing

For each recipient, choose the same-topic and different-topic donors as a **matched donor pair**.

The donor pair should be similar on the frozen R1.9 fingerprint variables while differing in topic.

Matching variables:

- source type;
- host suffix / URL-structure features where available;
- register;
- formatting/template vector;
- document length summaries;
- non-lexical structural statistics.

Do not use residual activations or R1.11 outcome scores for matching.

Do not use a high-capacity semantic embedding for matching.

### 10.1 Matching objective

Standardize continuous fingerprint variables on the candidate pool.

For candidate donor pair \((A_{\rm same},A_{\rm diff})\), define a frozen fingerprint distance

\[
d_{\rm fp}(A_{\rm same},A_{\rm diff}).
\]

Use exact agreement where feasible for coarse categorical variables and weighted Euclidean distance for standardized continuous variables.

The exact weights are frozen before production matching.

Use minimum-cost one-to-one assignment under topic and document-uniqueness constraints.

### 10.2 Boundary-structure balance

Because different-topic splices may be less locally natural, also balance **nonsemantic** boundary structure.

Using only token/format statistics from:

- the last 32 donor-prefix tokens;
- the first 32 recipient-suffix tokens;

construct frozen boundary features such as:

- punctuation class;
- newline indicators;
- capitalization;
- token length;
- numeric fraction;
- whitespace/format structure.

Require the same-topic and different-topic donor to be close on this boundary-structure score.

Do not use semantic embeddings or future model activations.

This reduces generic splice-discontinuity differences without removing the topic manipulation itself.

## 11. Reciprocal-direction balance

The two semantic transitions are analyzed separately and then pooled after sign orientation:

```text
business donor -> home recipient
home donor -> business recipient
```

There are 64 recipients of each orientation.

For every horizon, report:

\[
M_{\rm B\to H}(k)
\]

and

\[
M_{\rm H\to B}(k)
\]

after donor-directed sign normalization.

A pooled positive should not be driven entirely by one direction.

## 12. Forward-pass capture

For each recipient triplet, run:

```text
reference
same-topic splice
different-topic splice
```

through frozen Gemma-2-2B.

Capture only layer-12 residual vectors at:

\[
t_0+\{0,32,64,128,256\}.
\]

Do not store full residual trajectories.

Do not recapture unrelated layers.

Expected production sequence count:

\[
128\times3=384.
\]

## 13. Primary statistical test

Primary statistic:

\[
M(64)
=
\frac1N
\sum_B
D_B(64).
\]

Uncertainty:

- paired document bootstrap;
- recipient is the resampling unit;
- preserve the full triplet within each resampled recipient;
- 2,000 replicates.

Primary success requires:

\[
\boxed{
Q_{0.025}\left(M^{\rm bootstrap}(64)\right)>0.
}
\]

No token-level bootstrap.

No multiplicity correction is needed for the trajectory because \(k=64\) is the sole confirmatory horizon.

## 14. Split-readout replication

Evaluate the exact same captured activations with:

\[
c_{\rm BH}^{A}
\quad\text{and}\quad
c_{\rm BH}^{B}.
\]

No new forward passes.

For the strongest positive interpretation, require:

1. canonical \(M(64)\) has bootstrap lower bound \(>0\);
2. split-A \(M_A(64)>0\);
3. split-B \(M_B(64)>0\);
4. both reciprocal topic-transition subgroups have positive point estimates at \(k=64\).

Do not impose additional arbitrary effect-size ratios.

## 15. Retention curve

If \(M(0)>0\), define descriptive retention:

\[
R(k)
=
\frac{M(k)}{M(0)}.
\]

Report:

\[
R(32),R(64),R(128),R(256).
\]

Do not require monotonic decay.

Do not fit an exponential lifetime unless the observed trajectory clearly supports that model.

## 16. Generic splice diagnostic in the content plane

Let \(U_{\rm content}\in\mathbb R^{2304\times2}\) be the frozen orthonormal R1.9 content plane.

For each condition, compute:

\[
G_c(k)
=
\left\|
U_{\rm content}^{\top}\Delta h_{c,k}
\right\|_2.
\]

Report:

\[
G_{\rm same}(k),
\qquad
G_{\rm diff}(k).
\]

Interpretation:

- large \(G_{\rm same}\) and \(G_{\rm diff}\) but positive \(M(k)\): generic splice sensitivity exists, but different-topic history adds a donor-directed semantic component;
- large norms with \(M(k)\approx0\): generic content-plane perturbation without topic-specific state;
- small same-topic norm and large donor-directed different-topic effect: especially clean semantic specificity.

This diagnostic replaces a large new splice-control battery.

## 17. PCA semantic-readout control

Reuse the frozen R1.9 fingerprint-conditioned PCA-31 topic contrast.

Compute the same donor-directed statistic:

\[
M_{\rm PCA}(k).
\]

Purpose:

> determine whether the controlled semantic-history effect is especially persistent in the temporally selected slow content coordinate or is similarly visible in an equal-rank high-variance content representation.

This is secondary and does **not** gate the primary slow-state claim.

Do not add random-subspace semantic interventions.

## 18. No-op numerical audit

Before production interpretation, verify that the reference construction reproduces itself under the splice code path.

Construct a small fixed audit set where the donor prefix is the recipient's own prefix.

Require residual differences at every measured horizon to be at numerical noise relative to ordinary activation scale.

If this fails:

```text
TECHNICAL_FAILURE
```

and stop before interpreting production results.

## 19. Frozen outcomes

### A. `PERSISTENT_CONTENT_STATE`

Requires:

- no-op audit passes;
- canonical \(M(64)\) bootstrap 95% lower bound \(>0\);
- split-A and split-B \(M(64)\) are both positive;
- both reciprocal transition directions have positive \(k=64\) point estimates.

Interpretation:

> Changing earlier topic content causes a donor-directed change in a frozen, independently reproducible semantic coordinate that remains detectable after 64 identical subsequent tokens, beyond the generic same-topic splice floor.

This establishes persistent **history-to-representation semantic state**.

It does not establish downstream causal use.

### B. `SHORT_LIVED_CONTENT_HISTORY_EFFECT`

Use when donor-directed specificity is clear at \(k=0\) and/or \(k=32\), but the registered \(k=64\) gate fails.

### C. `GENERIC_SPLICE_SENSITIVITY`

Use when same-topic and different-topic content-plane norms are large, but donor-directed differences are near zero.

### D. `NO_CONTROLLED_CONTENT_EFFECT`

Use when no meaningful donor-directed content displacement appears even at early horizons.

### E. `TECHNICAL_FAILURE`

Use for no-op failure, token mismatch, provenance mismatch, invalid matching, or duplicated documents.

## 20. Required integrity checks

Before analysis, verify for every triplet:

1. same sequence length;
2. exact token equality between all three conditions from \(t_0\) onward;
3. exact absolute position equality;
4. recipient/donor document IDs are distinct where required;
5. no document reuse;
6. topic classes satisfy the condition contract;
7. donor-pair fingerprint matching metrics are stored;
8. frozen readout hashes match R1.9 artifacts;
9. layer/hook/model/tokenizer revision matches canonical provenance.

Fail closed on violations.

## 21. Explicit non-goals

Do not perform:

- new TICA fitting;
- Q64 substitution for Q31;
- new semantic ontology;
- broad natural within-document labeling;
- SAE analysis;
- activation steering;
- downstream behavioral causal intervention;
- layer/model/corpus sweep;
- alternative splice lengths;
- alternative primary horizons;
- semantic embedding matching;
- post-hoc donor filtering based on residual outcomes.

## 22. Compute strategy

### Label/matching stage
- scan fresh C4 text;
- apply frozen topic/fingerprint labelers;
- stop after candidate support target;
- match offline.

### Production stage
- 384 sequences;
- one model;
- one layer;
- five residual positions per sequence;
- no full-trajectory residual storage.

### Analysis stage
All semantic readouts, bootstrap, PCA comparison, and diagnostics are offline.

## 23. Required outputs

```text
r1_11_config.json
provenance.json

pool/
    candidate_manifest.parquet
    topic_fingerprint_labels.parquet
    selected_triplets.parquet
    matching_qc.json

integrity/
    token_identity_audit.json
    noop_audit.json

capture/
    residual_horizons.npz
    capture_manifest.json

analysis/
    canonical_trajectory.csv
    split_readout_trajectory.csv
    reciprocal_direction_trajectory.csv
    content_plane_norms.csv
    pca_trajectory.csv
    bootstrap_primary.parquet

decision.json
r1_11_report.md
```

`decision.json` must include:

- canonical \(M(k)\) at all horizons;
- 95% bootstrap interval for \(M(64)\);
- split-A/B \(M(64)\);
- reciprocal-direction \(M(64)\);
- retention curve if defined;
- same/different content-plane norms;
- PCA comparison;
- final frozen outcome.

## 24. Stopping rule

Stop after the frozen R1.11 outcome is assigned.

Do not automatically launch a behavioral intervention or a new semantic-label experiment.

If `PERSISTENT_CONTENT_STATE` passes, the next scientific fork becomes:

\[
\boxed{
\text{representation only}
\quad\text{vs}\quad
\text{causally used semantic state}
}
\]

A later experiment may then intervene directly on the frozen content coordinate and ask whether downstream model behavior moves as predicted.

If R1.11 fails, diagnose only the registered failure mode.

## 25. Interpretation target

The strongest successful statement is:

> **Changing only earlier document content shifts an independently reproducible semantic coordinate inside the canonical slow subspace toward the donor topic, and that shift remains detectable after 64 identical subsequent tokens beyond a matched same-topic splice control.**

This would be evidence for a persistent semantic state representation.

It would **not** yet show that the model causally uses that state to determine later behavior.
