# R1.9 — Content vs Document-Fingerprint Identification

**Status:** Proposed / freeze before execution  
**Purpose:** Determine whether the reproduced topic signal in the signed-TICA slow subspace reflects content information beyond document source/style/template variables, rather than merely a correlated document fingerprint.

## 0. Motivation

R1.8 established that:

- the signed-TICA slow subspace is reproducible across independent fits;
- slow-31 is strongly enriched for between-document variation;
- topic decoding reproduces across canonical Q31, split-A QA, and split-B QB;
- the ambient topic-discriminant geometry is itself reproducible;
- several leading TICA axes are individually stable.

The semantic result is therefore real, but its interpretation remains ambiguous.

Two explanations remain:

### H-content
The slow subspace directly contains document content/topic information.

### H-fingerprint
The slow subspace primarily carries persistent document fingerprint variables such as source/domain, genre/register, formatting/template, page structure, or corpus provenance; topic is decodable because these correlate with topic in C4.

The next experiment should resolve this fork rather than broaden into a large semantic atlas.

## 1. Scientific question

Let

\[
Z = Q^\top h
\]

be slow-subspace coordinates, let

\[
Y = Y_{\rm topic}
\]

be the existing coarse topic label, and let

\[
N = \{\text{source/domain},\text{genre/register},\text{format/template}\}
\]

be a compact nuisance set intended to capture persistent non-content document fingerprint.

Primary question:

\[
\boxed{I(Y;Z\mid N)>0?}
\]

Operationally:

\[
\boxed{
\Delta CE_{\rm topic\mid fingerprint}
=
CE(Y\mid N)-CE(Y\mid N,Z)
}
\]

on held-out documents.

The experiment asks whether slow-space coordinates add topic information **after conditioning on document fingerprint**.

## 2. Scope

This is an identification experiment, not a label-discovery experiment.

Do **not** add a broad ontology of semantic variables.

Do **not** add sentiment, POS, entity type, language, author identity, etc. unless one is required to construct the frozen fingerprint nuisance set.

The experiment remains focused on:

\[
\boxed{\text{content} \quad\text{vs}\quad \text{source/style/template}.}
\]

## 3. Frozen representations

Primary:

- canonical signed-TICA slow-31 Q31;
- split-A slow-31 QA;
- split-B slow-31 QB.

Controls:

- PCA-31;
- existing random-31 bank.

All bases are frozen. No TICA refitting.

## 4. Dataset and reuse policy

Prefer complete reuse of the R1.8 semantic dataset:

- 1,200 C4 documents;
- 800/200/200 semantic train/validation/sealed-test split;
- existing 12 sampled residual positions per document;
- existing residual tensor;
- existing topic labels;
- existing nuisance features.

No new Gemma forward pass unless fingerprint labels cannot be computed from saved text/metadata.

The sealed test split remains unchanged.

## 5. Fingerprint nuisance construction

Construct the **smallest** nuisance representation that captures plausible non-content document fingerprint strongly enough that survival of topic gain is meaningful.

Target three families only:

1. source/domain
2. genre/register
3. format/template

Do not optimize the nuisance set for maximal topic prediction. Its purpose is to block plausible fingerprint paths without directly absorbing arbitrary semantic content.

### 5.1 Source/domain

Preferred order:

1. explicit URL/domain metadata if already available;
2. host/domain extracted from saved metadata;
3. if unavailable, one frozen coarse source-type label.

Do not infer website identity from residual activations.

If explicit source/domain metadata are unavailable, record this nuisance family as partially observed.

### 5.2 Genre/register

Use one frozen coarse ontology, e.g.:

- informational/expository;
- commercial/promotional;
- personal/blog-like;
- forum/Q&A/discussion;
- news/reporting;
- instructional/how-to;
- other/unclear.

If existing labels capture equivalent structure, reuse them.

### 5.3 Format/template

Prefer document-derived structural variables:

- newline density;
- punctuation profile;
- list/bullet frequency;
- heading-like structure;
- sentence-length summaries;
- token-class proportions;
- numeric density;
- capitalization profile;
- existing formatting/local-window nuisance summaries.

Avoid a high-capacity text embedding that could directly encode topic.

## 6. Labeling policy

If new source/genre labels are required:

- freeze ontology and prompt before residual-space analysis;
- use one pinned deterministic labeler;
- permit abstention / `other_or_unclear`;
- do not tune labels after observing slow-space effects;
- spot-check a small fixed sample for label validity.

No broad relabeling campaign.

If support is too sparse, collapse categories prospectively before evaluating semantic gains.

## 7. Primary models

Use the exact R1.8 classifier family, regularization selection, and split contract.

For topic Y, fit:

### M0 — fingerprint only
\[
Y \sim N
\]

### M1 — fingerprint + subspace
\[
Y \sim N + Z
\]

Primary endpoint:

\[
\Delta CE(Q\mid N)
=
CE(M0)-CE(M1).
\]

Compute for:

- canonical Q31;
- QA;
- QB;
- PCA-31;
- each existing random-31 control.

All preprocessing is train-fit only.

## 8. Primary decision rule

Let

\[
\Delta_A=\Delta CE(Q_A\mid N),\qquad
\Delta_B=\Delta CE(Q_B\mid N).
\]

Let

\[
q_{.95}^{\rm rand}
\]

be the sealed-test Q95 of the fingerprint-conditioned gain across random-31 controls.

Define:

### `CONTENT_SURVIVES_FINGERPRINT`

if both

\[
\Delta_A>q_{.95}^{\rm rand}
\]

and

\[
\Delta_B>q_{.95}^{\rm rand}.
\]

Otherwise:

### `CONTENT_NOT_IDENTIFIED`

PCA-31 is reported but does not determine this gate.

## 9. Effect-retention diagnostic

For each slow fit define:

\[
R_{\rm topic}
=
\frac{
\Delta CE_{\rm slow\mid fingerprint}
}{
\Delta CE_{\rm slow\mid old\ nuisance}
}.
\]

Report descriptively.

Interpretation:

- near 1: fingerprint controls explain little of prior slow topic effect;
- intermediate: content and fingerprint share substantial variance;
- near 0: prior topic result is largely fingerprint-mediated.

Do not use an arbitrary retention cutoff for the primary decision.

## 10. Conditional semantic geometry

Only if `CONTENT_SURVIVES_FINGERPRINT`, test whether the content-specific readout itself reproduces.

Let B_A and B_B be the slow-coordinate topic classifier coefficients after including fingerprint nuisance. Map to residual space:

\[
W_A = Q_A B_A,\qquad
W_B = Q_B B_B.
\]

Remove the softmax-common direction and compare effective discriminant subspaces:

\[
S_{\rm content}
=
\frac1r
\left\|
\hat W_A^\top \hat W_B
\right\|_F^2.
\]

Use the same 100-replicate label-shuffle null style as R1.8.

Define:

### `CONTENT_GEOMETRY_REPRODUCED`

if

\[
S_{\rm content}>Q95(S_{\rm shuffle}).
\]

Use the plus-one empirical p-value.

## 11. Fingerprint-prediction diagnostics

For interpretation only, report how well the subspaces predict each nuisance family.

For each fingerprint family N_j, compute a simple held-out prediction metric from:

- slow-31;
- PCA-31;
- random-31.

Purpose:

- quantify whether the slow space actually carries source/style/template information;
- distinguish “topic survives despite strong fingerprint representation” from “the fingerprint control barely touched the slow space.”

These are diagnostics only. Do not turn them into a broad semantic atlas.

## 12. Frozen outcomes

### A. `CONTENT_STATE_IDENTIFIED`

Requires:

- `CONTENT_SURVIVES_FINGERPRINT`
- `CONTENT_GEOMETRY_REPRODUCED`

Interpretation:

> The reproducible slow subspace contains topic/content information that cannot be explained by the measured source/style/template fingerprint variables, and independently fitted slow spaces recover the same content-specific residual geometry.

Natural next step: one within-document changing semantic-state target.

### B. `TOPIC_PARTLY_FINGERPRINT_MEDIATED`

Use when conditional topic gain remains above typical random controls but one registered gate fails or the effect is substantially reduced.

Interpretation:

> The slow-space topic signal is partly shared with document fingerprint, while some content information may remain.

Do not automatically broaden labels.

### C. `DOCUMENT_FINGERPRINT_DOMINANT`

Use when:

- conditional slow gain collapses to the random-control range for both split fits;
- fingerprint nuisance itself predicts topic strongly.

Interpretation:

> The original topic signal was largely explained by persistent source/style/template structure.

This does not invalidate the slow-space result; it changes the ontology toward persistent document fingerprint/register state.

### D. `FINGERPRINT_CONTROL_INADEQUATE`

Use when:

- source/domain cannot be measured adequately;
- nuisance support is too sparse;
- nuisance labels fail reliability checks;
- or the nuisance baseline is too weak to block the intended confound.

Do not interpret surviving topic gain as content-specific under this outcome.

## 13. Minimal controls

Only:

1. PCA-31
2. existing random-31 bank
3. label-shuffle null for conditional semantic geometry
4. fingerprint-only topic baseline

No new broad control bank.

## 14. Explicit non-goals

Do not perform:

- broad multi-label semantic atlas;
- sentiment/POS probes;
- SAE analysis;
- new TICA fitting;
- causal intervention;
- new corpus;
- new model/layer;
- dynamic-state labeling;
- axis-level semantics;
- source/style intervention;
- new geometry stability tests.

These are downstream.

## 15. Compute strategy

Preferred path:

1. reuse R1.8 residual tensor;
2. reuse existing topic labels;
3. derive structural fingerprint variables from saved text/metadata;
4. label only the minimum missing source/genre variables;
5. run all probe fitting offline;
6. reuse existing PCA/random bases.

Expected GPU requirement: **none**.

## 16. Required outputs

```text
r1_9_config.json
provenance.json

fingerprint/
    fingerprint_schema.json
    fingerprint_labels.parquet
    fingerprint_qc.json

topic/
    conditional_gain_summary.csv
    retention_summary.csv
    fingerprint_topic_baseline.csv

geometry/
    conditional_semantic_overlap.json
    conditional_semantic_shuffle_null.parquet

diagnostics/
    fingerprint_decoding_summary.csv

decision.json
r1_9_report.md
```

`decision.json` must include:

- fingerprint nuisance families actually available;
- fingerprint baseline topic CE;
- conditional semantic gains for canonical / A / B / PCA / random;
- random Q95;
- effect-retention ratios;
- conditional semantic geometry overlap;
- shuffle Q95 and plus-one p-value;
- final outcome;
- next authorized branch.

## 17. Stopping rule

Stop once one frozen outcome is assigned.

Do not automatically start a within-document semantic-state experiment.

If `CONTENT_STATE_IDENTIFIED`, the only authorized positive continuation is to design one **within-document changing semantic variable** with a naturally long timescale, such as:

- current discourse subject;
- currently active entity/participant;
- current event/state of affairs;
- narrative/location state.

That next experiment should test:

\[
Y_{d,t}\leftrightarrow Q^\top h_{d,t}
\]

with Y changing within documents, and ask whether slow-space changes track semantic-state changes.

If content does not survive, do not force the dynamic-state branch. First reinterpret the slow region as document fingerprint/register/source state and decide whether that object is scientifically valuable.

## 18. Interpretation target

The strongest successful statement is:

> **The reproducible slow residual subspace contains topic/content information beyond measured source, genre/register, and formatting/template variables, and independent fits recover the same content-specific residual geometry.**

That is the smallest sufficient experiment before moving from document-static semantics to a genuinely changing semantic-state target.
