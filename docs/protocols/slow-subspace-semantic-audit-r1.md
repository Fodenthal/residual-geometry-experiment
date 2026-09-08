# Slow-Subspace Semantic Audit
## Minimal Prospective Specification

**Protocol revision:** `slow_subspace_semantic_audit_r1`  
**Status:** prospective side branch of Residual Sequence-Memory Geometry  
**Primary model:** `google/gemma-2-2b`  
**Primary layer / hook:** layer 12, `blocks.12.hook_resid_post`  
**Primary corpus:** fresh document-disjoint C4/en  
**Inherited slow object:** original layer-12 C4 slow subspace, rank \(k_S=31\), if exact provenance-compatible artifact is available  
**Inferential unit:** document  
**Goal:** determine what information the original slow subspace preferentially represents, without assuming that it is the model's privileged causal state carrier.

## Prospective minimum-sufficient-compute amendment (R1.1)

**Freeze time:** 2026-09-04, before any Slurm submission, residual capture, label result, or semantic result from this audit.

This amendment narrows the first production read without changing the scientific question. Where it conflicts with later default compute quantities in R1, this amendment controls.

The initial S2 campaign is:

```text
documents: 1,200
fit / validation / sealed test: 800 / 200 / 200
primary slow representation: S31 only
deterministic matched representation: PCA31
empirical controls: 20 frozen random rank-31 subspaces in fit-only PCA256
fitted target families: topic/domain and register/document-state
formatting and source/template: deterministic descriptive controls; source is also a topic nuisance when viable
document bootstrap: 1,000 replicates over cached held-out per-document losses, with no refitting
S3 label-shuffle refits: 50 initially
```

One frozen labeler call returns both topic and register for each document. Residual capture remains a single forward pass per batch and saves only the 12 frozen layer-12 positions plus local embedding summaries.

The following are extensions, not automatic parts of the initial campaign:

- rank 8, only after a positive or boundary S31 result makes concentration decision-relevant;
- more than 20 random controls, only if the S31 result is near the empirical-control tail;
- a second 1,200-document tranche, only if validation demonstrates inadequate precision;
- more than 50 S3 shuffles, only if observed stability is near the null boundary.

Any extension is frozen from validation diagnostics before further sealed-test access. S4 remains governed by the original causal gate and requires a separate micro-spec.

**Labeler checkpoint implementation note (frozen before document selection):** the initially proposed 2B instruction checkpoint was neither cached nor authenticated on Beehive. The unchanged frozen prompt, ontologies, and abstention rule therefore use the already-cached provenance-addressed `google/gemma-2-9b-it` snapshot at revision `11c9b309abf73637e4b6f9a3fa1e92e615547819`. The failed profile attempt produced no document pool, labels, or activations.

---

# 0. Motivation and scope

The original residual-geometry pilot established a real low-dimensional persistence phenomenon. The top-31 slow candidate span is geometrically high-rank but internally structured: generic rotations inside the span remain slow, while persistence is concentrated into only a few effective modes. Qualitative inspection suggested that the slow region often tracks durable web-document state such as register, domain, formatting, source templates, and scraping artifacts.

Later live-state work changed the interpretation. A task-local slow subspace could be enriched for a real running-state variable while still failing to contain the canonical stable state direction and being causally weaker than matched PCA or supervised state geometry. Therefore this experiment does **not** test:

> the slow subspace is the privileged functional state carrier.

It tests the narrower and still-open question:

\[
\boxed{
\text{Does timescale-selected residual geometry have a distinctive information profile,}
\atop
\text{especially for durable document/semantic state?}
}
\]

The intended possible result is closer to:

\[
\boxed{
\text{slow geometry is selectively enriched for information worth maintaining across many tokens}
}
\]

than to:

\[
\boxed{
\text{all important semantics or state is localized in the slow subspace.}
}
\]

This is deliberately a bounded side experiment. Expensive causal work is conditional on a clear descriptive positive.

---

# 1. Inherited objects and facts

The experiment inherits, but does not re-estimate from semantic labels:

```text
model: google/gemma-2-2b
layer: 12
residual width: 2304
original slow rank: 31
original slow ordering: by frozen persistence / lifetime-excess ordering
original broad PCA parent: residual PCA geometry from the same model/layer
```

The original slow span is denoted

\[
S=\operatorname{span}(Q_S),
\qquad
Q_S\in\mathbb R^{2304\times31},
\qquad
Q_S^\top Q_S=I.
\]

Two frozen nested slow spaces are primary:

\[
S_8=\operatorname{span}(Q_{S,1:8}),
\]

\[
S_{31}=S.
\]

Rank 8 is included because the original nested lifetime curve showed that the leading slow modes carry much of the persistence concentration. Rank 31 is the full original candidate span.

Rank 16 is **not** run by default. Add it only if the difference between rank 8 and rank 31 is scientifically decision-relevant after the primary read.

---

# 2. Scientific questions

The audit has four ordered questions.

## Q1 — SAE bridge

Do independently learned sparse features preferentially align with the slow subspace, and if so, what kinds of features align?

For SAE decoder direction \(d_f\), define unit-normalized direction

\[
\hat d_f=\frac{d_f}{\|d_f\|_2}
\]

and slow containment

\[
\boxed{
c_f(S_k)
=
\|Q_{S,k}^\top\hat d_f\|_2^2.
}
\]

This lies in \([0,1]\).

If feature activation statistics are already available, also report activation-weighted slow contribution

\[
\boxed{
e_f(S_k)
=
\mathbb E[a_f^2]\,
\|Q_{S,k}^\top d_f\|_2^2.
}
\]

The directional containment \(c_f\) is primary because it is invariant to the SAE feature-rescaling gauge. The activation-weighted quantity is a secondary measure of how much natural feature contribution lands in the slow span.

The question is not merely whether some SAE directions overlap \(S\). The question is whether high-overlap SAE features have a **distinctive semantic profile** relative to matched SAE controls.

---

## Q2 — supervised information profile

Does the slow subspace contain more held-out information about durable semantic/document variables than equal-rank alternative residual subspaces?

For target \(Y\), nuisance variables \(N\), and representation \(X\), define conditional held-out information gain

\[
\boxed{
\Delta H_Y(X)
=
\mathcal L_N(Y)
-
\mathcal L_{N,X}(Y)
}
\]

in nats per evaluated position.

The operational target is

\[
I(Y;X\mid N),
\]

with \(\Delta H\) as the estimator under the frozen predictive family.

The key comparison is not raw \(\Delta H_Y(S_k)\), but slow-specific excess over same-capacity residual controls:

\[
\boxed{
E_Y(S_k)
=
\Delta H_Y(S_k)
-
\operatorname{median}_b
\Delta H_Y(R_{k,b}).
}
\]

---

## Q3 — canonical semantic geometry inside the slow span

If a target is genuinely enriched in \(S\), does semantic supervision identify a stable lower-dimensional subspace inside \(S\)?

For standardized slow coordinates \(z_S\), fit

\[
\ell(y\mid N,z_S)
=
\ell_N(y\mid N)+Bz_S.
\]

After class-centering \(B\), take

\[
B_c=U\Sigma V^\top.
\]

The columns of \(V\) define candidate supervised semantic directions inside the already-frozen slow span.

This stage is run only for targets that pass the supervised enrichment gate.

---

## Q4 — causal relevance

Only after a stable semantic object is found, ask whether manipulating that semantic geometry changes model behavior more than matched controls.

Generic task degradation after deleting the entire slow span is not the primary causal endpoint because it mixes:

- semantic content;
- perturbation magnitude;
- high-variance residual geometry;
- generic functional importance;
- off-distribution deletion effects.

A causal semantic branch is activated only if the descriptive object is strong enough to justify a targeted instrument.

---

# 3. Governing execution order

```text
S0  provenance + artifact audit
 |
 v
S1  SAE decoder-overlap screen                 mostly artifact-only
 |
 +---- optional SAE context/label inspection if overlap is healthy
 |
 v
S2  one fresh C4 residual capture
    + frozen semantic/document labels
    + equal-rank slow/PCA/random readouts
 |
 v
STOP if no distinctive slow profile
 |
 v
S3  stable semantic geometry inside slow span  offline
 |
 v
OPTIONAL S4
targeted causal / functional micro-test
```

The experiment stops as soon as the scientific question is resolved.

---

# 4. S0 — provenance and slow-basis audit

## 4.1 Preferred path: exact original basis

Use the exact original layer-12 rank-31 slow basis if it is available with compatible provenance.

Require:

```text
model identity / revision compatible
layer = 12
d_model = 2304
basis rank = 31
orthonormality check passes
source ordering is recoverable
basis hash recorded
```

Record:

```text
slow_basis_status = ORIGINAL_FROZEN
```

No semantic result may alter or reorder the inherited basis.

---

## 4.2 Fallback: prospective refit

If the original artifact is unavailable, do **not** silently recreate it and call it the original basis.

Instead fit a new slow basis on a document pool disjoint from the semantic evaluation pool using the frozen legacy time-lagged recipe as closely as practical.

Record:

```text
slow_basis_status = PROSPECTIVE_REFIT
```

The fallback does not need to reproduce every historical pilot number. It must pass only the minimum health checks needed to establish that the object is genuinely slow:

1. rank fixed prospectively at 31;
2. random-in-span median timescale clearly exceeds ambient-random;
3. the leading nested directions retain the expected qualitative slow-to-fast gradient;
4. split-half subspace overlap is healthy enough that the basis is not numerical noise.

If these fail:

```text
SLOW_BASIS_NOT_RECOVERED
```

and stop the experiment.

Do not spend additional compute trying alternative slow estimators in this branch.

---

# 5. Data contract

## 5.1 Semantic evaluation pool

Use fresh C4/en documents from a corpus split not previously used for the semantic result.

Default:

```text
documents: 2,400
context length: 1,024 tokens
fit: 1,600 documents
validation: 400 documents
sealed test: 400 documents
```

If a prospective slow-basis refit is required, its documents are disjoint from all 2,400 semantic-evaluation documents.

The document is always the inferential/bootstrap unit.

---

## 5.2 Evaluated positions

The target is durable state, so dense token capture is unnecessary.

Use a frozen grid of interior positions, for example:

```text
[256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960]
```

subject to exact tokenizer/context validity.

This gives 12 measurements per document while avoiding early-context transients.

Store only the required layer-12 residual vectors at these positions.

If SAE activations are needed for S1 context inspection, collect them in the **same forward pass** rather than launching a separate capture.

---

# 6. S1 — SAE bridge

## 6.1 SAE provenance

Use one pretrained SAE whose input is provenance-compatible with the layer-12 residual object.

Before analysis record:

```text
SAE release / checkpoint
hook convention
input normalization convention
dictionary width
decoder shape
decoder-norm convention
feature IDs
```

If no compatible SAE is available without substantial new training:

```text
SAE_BRIDGE_NOT_EVALUATED
```

and continue directly to S2.

Training a new SAE is outside the budget of this side experiment.

---

## 6.2 Directional alignment

For every eligible feature \(f\), compute

\[
c_f(S_8),
\qquad
c_f(S_{31}).
\]

Report:

```text
median
q90/q95/q99
top 32 / 64 / 128 features
decoder norms
feature frequency where available
activation variance where available
```

Do not interpret top labels yet.

---

## 6.3 Slow-span coverage by SAE decoders

For the top \(m\) SAE features ranked by \(c_f(S_{31})\), normalize and orthonormalize their decoder directions to obtain \(Q_{\rm SAE,m}\).

Report

\[
\boxed{
C_{\rm SAE}(m)
=
\frac{1}{31}
\|Q_S^\top Q_{\rm SAE,m}\|_F^2
}
\]

for

\[
m\in\{32,64,128\}.
\]

Controls:

1. random eligible SAE features;
2. SAE features matched on activation frequency and decoder norm;
3. if historical SAE timescales exist, fast-feature controls matched on frequency.

Use 100 offline control draws unless 50 already gives a clearly separated distribution.

This is descriptive geometry. It does not by itself establish semantic meaning.

---

## 6.4 Semantic inspection set

Inspect only a small frozen set:

```text
64 highest-containment SAE features
64 matched-control SAE features
```

For each feature, use either existing trusted feature descriptions or top activating contexts from the shared semantic capture.

If new automatic labels are required, use one frozen labeling prompt/model and blind it to whether the feature came from the slow or control set.

Each feature receives one coarse category from the following frozen ontology:

```text
TOPIC_OR_DOMAIN
REGISTER_OR_DISCOURSE
ENTITY_OR_LONG_RANGE_REFERENCE
LEXICAL_LOCAL
SYNTAX_OR_GRAMMAR
FORMAT_OR_MARKUP
SOURCE_TEMPLATE_OR_BOILERPLATE
POSITION_OR_LENGTH
OTHER_OR_UNCLEAR
```

The primary S1 statistic is category enrichment of slow-aligned features relative to matched controls.

The feature labels are navigation evidence, not a standalone semantic claim.

---

## 6.5 S1 interpretation

Possible outcomes:

```text
SAE_S0_NO_BRIDGE
    little usable decoder alignment and no semantic-category enrichment

SAE_S1_DURABLE_STRUCTURE_ENRICHED
    high-overlap features are enriched mainly for formatting/source/register state

SAE_S2_DURABLE_SEMANTICS_ENRICHED
    high-overlap features are enriched for topic/domain/entity/reference state

SAE_S3_MIXED_DURABLE_STATE
    both semantic and structural durable categories are enriched
```

S1 never gates S2. A negative SAE result can mean the SAE dictionary simply does not align cleanly with the residual slow geometry.

---

# 7. S2 — supervised slow-space information profile

## 7.1 Why this stage is primary

SAE labels are not enough. S2 is the main semantic evidence because it asks whether the same frozen slow geometry predicts independently defined targets on held-out documents better than equal-capacity residual controls.

---

# 8. Frozen target bank

Use a deliberately small target bank that separates semantic content from durable but nonsemantic document state.

## 8.1 Primary semantic target: document topic/domain

Assign one coarse topic/domain label to each document using a frozen external labeling instrument **before slow-space results are viewed**.

The exact classifier/model, prompt, category set, and abstention rule are implementation constants written to the manifest before residual projections are analyzed.

Use a modest category set, target 6–10 classes, with an `OTHER/UNCLEAR` class or abstention.

The labeler sees raw document text, not model activations.

This is the primary target:

\[
Y_{\rm topic}.
\]

---

## 8.2 Register/discourse target

A second frozen document-level target captures style/register, for example:

```text
encyclopedic / expository
news / reporting
technical / reference
instructional
legal / administrative
conversational / forum
narrative / creative
other
```

Denote:

\[
Y_{\rm register}.
\]

This is intentionally distinct from topic.

---

## 8.3 Nonsemantic durable controls

Use two cheap controls when available.

### Formatting state

Deterministic features from raw text such as:

```text
markup / HTML density
list/table structure
code-like formatting
quotation/dialogue markers
newline density
boilerplate-like repetition
```

These may be modeled as multi-label targets or a frozen coarse class.

### Source/template identity

When C4 metadata permits, use URL host/domain or another frozen source-template label on a common-support subset with adequate documents per class.

These targets tell us whether the slow space is mostly a web-source/template carrier.

---

## 8.4 Optional existing R7 targets

If lexical-supersense / entity-type / POS / recurrence labels can be generated by already-existing code on the same capture with negligible cost, add them as **secondary** targets.

Do not implement a new ontology pipeline solely to add these targets.

Their role is to compare durable document-scale state with local/token-scale semantics.

---

# 9. Local-context nuisance model

A slow subspace should not receive semantic credit merely because the current few words identify the document topic.

For each evaluated position \(t\), construct a frozen local nuisance representation \(N_t\) from information available in a short local window.

Default:

```text
absolute position
current token class
mean input embedding over tokens [t-16, t+16]
32D PCA compression of that local embedding summary
newline / formatting indicators local to the window
```

Fit the local-embedding PCA on fit documents only.

The same \(N_t\) is used for every residual representation.

The headline statistic is incremental over this local context.

Also report raw-without-local-nuisance gain as a diagnostic.

---

# 10. Residual representations and controls

Let centered residual be

\[
\bar h_t=h_t-\mu,
\]

where \(\mu\) is estimated on permitted fit data only or inherited from the compatible frozen residual-geometry artifact.

## 10.1 Slow spaces

\[
z_{S,k}=Q_{S,k}^\top\bar h_t,
\qquad
k\in\{8,31\}.
\]

Standardize each coordinate using fit-document statistics.

Do **not** normalize each slow state to unit norm. Slow-coordinate amplitude may itself carry held state.

---

## 10.2 Top PCA controls

Fit residual PCA on fit documents only, unless an exact compatible frozen PCA basis already exists.

Use:

\[
P_8,
\qquad
P_{31}.
\]

These are the first matched controls because the original slow region lives partly in broad high-variance PCA geometry.

---

## 10.3 Random empirical residual controls

The strongest generic control is a same-rank random subspace sampled from a frozen high-variance empirical parent, default PCA-256.

For each rank \(k\),

\[
R_{k,b}\subseteq\operatorname{span}(P_{256}),
\qquad
\dim R_{k,b}=k.
\]

Use 50 frozen draws initially. Extend to 100 only for boundary results.

Where practical, match draws approximately on mean projected activation energy to the slow space. If energy matching materially complicates implementation, report projected energy explicitly and keep PCA-\(k\) as the primary deterministic control.

---

## 10.4 Broad descriptive ceiling

Use PCA-256 or a regularized full-residual decoder as a descriptive ceiling.

It is not an equal-capacity comparator and cannot establish slow-space privilege.

---

# 11. Predictive model

For each categorical target \(Y\), fit the same nuisance-offset multinomial model:

\[
\ell(y\mid N,z_X)
=
\ell_N(y\mid N)+B_X z_X.
\]

Use:

```text
isotropic L2 regularization
document-grouped cross-validation
float64 fitting
sum-to-zero class coefficient convention
no token-random CV
```

All model selection occurs on fit/validation documents.

Test is read once after the representation family, target definitions, and regularization procedure are frozen.

Losses are averaged with equal document weight.

---

# 12. Primary statistics

For every target \(Y\) and representation \(X\):

\[
\Delta H_Y(X)
=
\mathcal L_N(Y)
-
\mathcal L_{N,X}(Y).
\]

For slow rank \(k\), define:

\[
E_Y(S_k)
=
\Delta H_Y(S_k)
-
\operatorname{median}_b\Delta H_Y(R_{k,b}).
\]

Also report deterministic PCA margin:

\[
M_Y^{\rm PCA}(S_k)
=
\Delta H_Y(S_k)-\Delta H_Y(P_k).
\]

And fraction of broad residual information retained:

\[
\rho_Y(S_k)
=
\frac{\Delta H_Y(S_k)}
{\Delta H_Y(P_{256})+\epsilon}.
\]

The \(\rho\) profile is useful across target families because raw target entropies differ.

---

# 13. Statistical uncertainty

Use a paired document bootstrap.

Default:

```text
bootstrap replicates: 1,000
```

Extend to 2,000 only when a decision boundary is within Monte Carlo error.

For random-control comparisons, hold each frozen random basis fixed within the document bootstrap and summarize the control distribution separately.

Do not treat token positions as independent replicates.

---

# 14. S2 decision logic

The main question is a **profile**, not a winner-take-all leaderboard.

## `PROFILE_GENERIC`

Use when the slow space is not clearly better than matched residual controls on any durable target.

Interpretation:

> The slow space is temporally distinctive but does not show a distinctive semantic/document-state profile under this assay.

Stop the project here.

---

## `PROFILE_STRUCTURAL_DURABLE`

Use when slow-specific excess is robust for formatting/source/register targets but not for the primary topic/domain target.

Interpretation:

> The original slow space is preferentially a durable structural / source-state carrier rather than a clean semantic substrate.

This is a useful result and normally stops the branch.

---

## `PROFILE_DURABLE_SEMANTIC`

Require on sealed test for \(Y_{\rm topic}\):

1. positive slow-space conditional gain;
2. positive paired margin over the random empirical control distribution;
3. positive or at least competitive margin versus top-PCA at the same rank;
4. effect is not entirely explained by source/template labels;
5. the point semantic gain is nontrivial, with \(0.01\) nats/example retained as a practical effect-health reference from the Arm G semantic program rather than as a universal law.

Interpretation:

> Timescale-selected residual geometry is preferentially enriched for durable semantic document state.

---

## `PROFILE_MIXED_DURABLE_STATE`

Use when topic/domain and structural/source targets are both clearly enriched.

Interpretation:

> The slow space carries a mixture of semantic and nonsemantic document state rather than a clean semantic-only code.

This may be the most plausible positive world.

---

# 15. S3 — canonical semantic geometry inside the slow span

Run only for a target that is positive under S2.

Use the full \(S_{31}\) coordinates.

Fit the full nuisance-offset semantic coefficient matrix \(B\), then compute the SVD-defined semantic basis \(V_Y\).

Evaluate nested ranks:

\[
r\in\{1,2,4,8\}.
\]

Choose the smallest rank retaining at least 90% of full-\(S_{31}\) semantic gain on validation:

\[
\frac{\Delta H_Y(S_{31}V_{Y,r})}
{\Delta H_Y(S_{31})}
\ge0.90.
\]

Do not search additional ranks unless the 8-vs-31 distinction is unresolved.

---

## 15.1 Split stability

Fit the semantic geometry independently on two document-disjoint fit halves.

Compare:

```text
principal angles
mean squared canonical correlation
direction cosine only if r = 1
held-out score-space correlation
```

Compare against full-refit label-shuffle semantic nulls.

A stable subspace may be claimed even if individual axes rotate.

---

## 15.2 S3 labels

```text
SEM0_NO_STABLE_INTERNAL_GEOMETRY
    slow space contains information but no independently stable semantic subspace is recovered

SEM1_STABLE_SEMANTIC_SUBSPACE
    stable r > 1 semantic subspace

SEM2_COMPACT_STABLE_SEMANTIC_SUBSPACE
    stable r <= 8 retaining >= 90% of full-slow semantic gain

SEM3_STABLE_SEMANTIC_DIRECTION
    stable r = 1
```

This does not imply semantic exclusivity or causal use.

---

# 16. Optional S4A — broad functional deletion diagnostic

This stage is **not** semantic causality.

Run only if S2 gives a clear positive and if the result would change whether a targeted causal follow-up is worth building.

On a small fresh set, default 256–512 documents, intervene after layer 12 by projecting out:

```text
S31
PCA31
one or more energy-matched random-31 controls
```

Measure:

```text
next-token NLL change
next-token KL
effect by position
removed residual energy
```

The intervention is

\[
h_t'
=
h_t-Q Q^\top(h_t-\mu).
\]

Interpretation is limited to:

> this subspace is broadly functionally consequential under deletion.

Do **not** interpret a larger slow-space NLL effect as proof that the semantic information found in S2 is what caused the effect.

If projected energy differs strongly between bases and no reasonable energy match is available, omit this stage rather than overinterpret it.

---

# 17. Optional S4B — targeted semantic causal branch

A true semantic causal claim requires a behavior that the semantic object could plausibly mediate.

Activate only if:

1. S2 yields `PROFILE_DURABLE_SEMANTIC` or a strong `PROFILE_MIXED_DURABLE_STATE`;
2. S3 recovers a stable semantic subspace;
3. a controlled or natural matched behavior can be defined prospectively;
4. a full slow-space or broader positive-control intervention demonstrably moves that behavior.

Then compare:

- the frozen semantic subspace inside \(S\);
- random same-rank directions inside \(S\);
- a separately fit semantic representation outside \(S\) matched for held-out semantic information;
- same-label sham swaps.

Use norm-preserving donor/target semantic-coordinate replacement rather than simple unbounded steering when possible.

The causal privilege question is:

\[
\boxed{
\text{Does semantic geometry inside the slow region cause more target-specific behavior}
\atop
\text{than generic slow directions or equally decodable semantics elsewhere?}
}
\]

If the pathway positive control fails, stop. Do not iterate indefinitely on prompt construction.

A separate micro-spec should freeze the exact behavior/instrument before this stage is launched.

---

# 18. What is explicitly not in this experiment

Do not add:

- a broad benchmark/task battery;
- MMLU or unrelated capability evaluation;
- a layer sweep;
- a model sweep;
- a corpus sweep;
- new SAE training;
- a broad ontology search after seeing feature labels;
- nonlinear semantic decoders as a rescue after linear failure;
- repeated causal prompt engineering after a failed pathway instrument;
- unsupervised clustering as a new primary semantics estimator;
- claims that top-token or top-context labels alone establish semantics.

These can be future experiments only if this small audit produces a result that makes them decision-relevant.

---

# 19. Stop rules

## Stop after S1 only if implementation is blocked

A negative SAE bridge does not stop S2, because SAE alignment and residual semantics are logically separable.

## Stop after S2 if profile is generic

If the slow space does not beat matched controls on durable targets:

```text
STOP = PROFILE_GENERIC
```

No S3 or S4.

## Stop after structural-only positive

If the result is convincingly source/template/formatting dominated and the semantic target is not enriched:

```text
STOP = PROFILE_STRUCTURAL_DURABLE
```

Do not spend compute trying to force a cleaner semantic story.

## Proceed to S3 only for semantic enrichment

S3 is offline and cheap once S2 is positive.

## Proceed to S4 only if the answer could change interpretation

A causal stage is not required to publish/report a clean descriptive profile.

---

# 20. Required artifacts

```text
slow_semantic_audit/
  manifest.json
  slow_basis_manifest.json
  data_split_manifest.json

  sae/
    sae_manifest.json
    decoder_slow_containment.parquet
    span_coverage.json
    matched_feature_sets.parquet
    feature_semantic_labels.parquet
    sae_bridge_decision.json

  capture/
    token_metadata.parquet
    residual_positions.zarr
    optional_sae_activations.zarr
    local_nuisance_features.parquet

  labels/
    labeler_manifest.json
    document_labels.parquet
    formatting_targets.parquet
    source_targets.parquet

  readout/
    representation_manifest.json
    random_basis_manifest.json
    semantic_gain_by_target_rank.parquet
    document_bootstrap.parquet
    information_profile.json
    s2_decision.json

  semantic_geometry/
    fitted_semantic_bases.npz
    rank_retention.parquet
    split_stability.parquet
    semantic_nulls.parquet
    s3_decision.json

  optional_function/
    deletion_metrics.parquet
    intervention_manifest.json

  report.md
```

Every basis artifact records:

```text
shape
orthonormality error
source
rank
hash
fit-document IDs
model/layer identity
```

---

# 21. Implementation invariants

1. The slow subspace is frozen independently of semantic labels.
2. Documents, not token positions, are the inferential unit.
3. Fit/validation/test documents are disjoint.
4. No target or label ontology is changed after slow-space results are viewed.
5. All representation families use the same nuisance features and predictive model.
6. Equal-rank controls are required for semantic-enrichment claims.
7. Broad PCA/full residual is a ceiling, not a fair same-capacity comparator.
8. A feature description is not treated as semantic evidence without matched controls.
9. A generic deletion effect is not treated as semantic causality.
10. The expensive causal branch checks the S2/S3 gate artifact before launching.
11. If the original slow basis is unavailable, the replacement is always labeled as a prospective refit.
12. No result from this branch rewrites the completed live-state causal result.

---

# 22. Minimal compute plan

## Path A — original slow basis available

```text
S0 artifact audit:       CPU only
S1 decoder geometry:     CPU only
S2 semantic capture:     one Gemma-2-2B forward capture over ~2.4M tokens
S2/S3 analysis:          CPU / small GPU
S4:                      not run unless positive
```

## Path B — slow basis must be refit

Add one bounded basis-recovery capture on a disjoint C4 pool. Reuse the legacy estimator; do not expand the estimator search.

The experiment should still remain much smaller than a new Arm D/G-scale campaign.

---

# 23. Final interpretation table

| Result | Interpretation |
|---|---|
| SAE bridge negative, supervised profile negative | Slow geometry is real temporally but not semantically distinctive under these instruments |
| SAE durable labels positive, supervised structural only | SAEs expose a real durable web-state axis, mostly formatting/source/register |
| Topic/domain slow excess positive, no stable internal semantic geometry | Slow space carries durable semantics, but not in a uniquely recoverable low-rank semantic frame |
| Topic/domain positive + stable low-rank \(V_Y\) | Slow geometry contains a canonical durable-semantic subspace |
| Slow semantics positive but PCA/equal-rank controls match it | Semantics is present but not privileged by timescale selection |
| Slow semantic readout positive, targeted causal effect weak | Slow semantics is a descriptive/possibly redundant trace |
| Slow semantic readout + semantic-specific causal privilege | Strongest outcome: slow geometry contains a durable semantic representation that is not only readable but functionally privileged |

---

# 24. Intended final claim boundary

The strongest descriptive claim licensed by S2/S3 is:

> **In Gemma-2-2B layer 12 on C4-like text, directions selected for long sequence-timescale are preferentially enriched for durable document state, with [semantic / structural / mixed] information concentrated in a reproducible low-dimensional region relative to equal-rank residual controls.**

The strongest causal claim requires a separately qualified S4B instrument.

Do not claim:

> the slow subspace is the model's unique memory store;

> all semantics is localized there;

> the slow subspace is more important than PCA in general;

> the live-state causal negative has been overturned.

The purpose of this audit is to determine **what kind of information timescale selection preferentially finds**. If the answer is mostly source/template state, that is a clean and useful resolution. If the answer is durable semantics, that is a stronger and more surprising continuation.
