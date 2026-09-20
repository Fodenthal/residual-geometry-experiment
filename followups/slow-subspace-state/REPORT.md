# Slow Subspace — Post-LessWrong Results and Current Picture

**Model:** Gemma-2-2B  
**Layer:** 12, `blocks.12.hook_resid_post`  
**Corpus:** C4 English  
**Canonical estimator:** signed time-lagged generalized eigenproblem (TICA-like)  
**Canonical compact object:** \(Q_{31}\)  
**Status:** updated through R1.11, 2026-09-09

---

# 0. Executive summary

The original LessWrong result was that most residual-stream directions forget within a token, while a compact set of time-lagged directions remains autocorrelated for much longer. Since then, the picture has become substantially stronger and more specific.

The current best description is:

> **Gemma-2-2B layer 12 contains a reproducible, graded persistent residual-state region. A compact canonical slice \(Q_{31}\) captures about 80% of positive lifetime excess, while a lower-density generically slow shoulder extends to roughly rank 64. The region is strongly enriched for persistent document-level state. Its leading temporal axes mostly organize register/source-style variables, while independently reproducible content coordinates are distributed across those temporal modes. Most importantly, controlled prefix interventions show that earlier semantic content induces a donor-directed semantic displacement in the slow space that remains detectable after 64–256 identical continuation tokens, while the analogous PCA semantic effect rapidly disappears.**

The project has therefore progressed from:

\[
\text{temporal covariance}
\rightarrow
\text{reproducible geometry}
\rightarrow
\text{persistent document state}
\rightarrow
\text{identified content}
\rightarrow
\boxed{\text{controlled persistent semantic state}}.
\]

The major remaining question is no longer whether the slow space contains semantics. It is:

\[
\boxed{\text{Does the model causally use this persistent semantic state?}}
\]

---

# 1. What the LessWrong writeup had already established

The original experiment studied signed scalar projections

\[
a_{d,t}=v^\top h_{d,t}
\]

of Gemma-2-2B layer-12 residual activations and defined a timescale from the within-document autocorrelation curve.

The main original observations were:

- random directions were almost entirely \(\tau\approx1\);
- individual PCA axes were also mostly short-lived;
- time-lagged directions had a heavy upper tail;
- within-document token-order shuffling collapsed the slow tail;
- the top persistent signal was spread across many nonduplicate directions;
- generic random directions inside the selected slow span were themselves slow;
- deleting slow/PCA spans strongly reduced persistence measured by independent held-out time-lagged probes;
- the slow span was not simply the top PCA span;
- coarse semantic/topic information appeared enriched in the slow region.

The historical \(Q_{31}\) cutoff came from:

\[
k_{80\%\text{ lifetime excess}}=31.
\]

It was an operational compactness rule, not a claim that the true persistent-state dimension was exactly 31.

---

# 2. Canonical-object audit: what \(Q_{31}\) actually is

A later audit resolved an important provenance ambiguity.

The canonical basis is:

- rank 31;
- 30 signed time-lagged generalized-eigenproblem directions;
- 1 PCA direction;
- fit in residual space using the original signed TICA-like estimator;
- scored using signed residual projections.

The canonical object is **not** the later unsigned/squared-activity nonlinear estimator that appeared in R1.5/R1.6 exploratory work.

This matters because some apparent optimizer instability in that later estimator was initially at risk of being interpreted as evidence against the original slow subspace. It was not relevant to the canonical object.

A direct reproduction recovered the canonical basis essentially exactly and re-established the original random-in-span result:

\[
\operatorname{median}\tau_{\rm val}=24,
\qquad
\operatorname{median}\tau_{\rm test}=25
\]

for 512 random directions inside frozen \(Q_{31}\).

So the historical phenomenon survived the provenance audit intact.

---

# 3. R1.7 — independent-fit identifiability

The next major question was:

> If the signed-TICA estimator is fit on different documents, does it recover the same slow region?

The original 4,000 training documents were split into two disjoint 2,000-document halves, producing independent fits \(Q_A\) and \(Q_B\).

## Full-data reproduction

The combined-data refit reproduced the canonical object:

\[
S_{31}(Q_{\rm refit},Q_{\rm canonical})=0.999125
\]

and reproduced the validation random-in-span median:

\[
\tau_{\rm median}=24.
\]

## Independent split fits

The split-fit overlap was:

\[
\boxed{S_{31}(Q_A,Q_B)=0.864059}
\]

against the ambient random expectation:

\[
31/2304\approx0.01345.
\]

The functional slow phenotype also reproduced:

\[
\operatorname{median}\tau(Q_A)=24.5,
\qquad
\operatorname{median}\tau(Q_B)=25.0.
\]

Cross-fit retention was effectively perfect:

\[
24\rightarrow24
\]

in both directions.

Nested overlaps were:

\[
S_8=.7698,\quad
S_{13}=.7479,\quad
S_{21}=.8446,\quad
S_{31}=.8641.
\]

## Interpretation

This resolved a major identification concern:

> The canonical signed-TICA slow region is not a one-fit artifact. Independent samples recover substantially the same geometric region and the same generic slow phenotype.

The identifiable object is primarily the **span**, though later work showed that several leading individual axes are identifiable too.

---

# 4. Independent replication on a separate implementation/data draw

An independent replication by Dan Braun reproduced the broad phenomenon with the same TICA idea on separately constructed data.

Headline replication results:

- time-lagged Q90 timescale: \(\sim19\);
- PCA/random Q90: \(\sim3\);
- token-position shuffle reduced the top-decile timescale from \(19\rightarrow2\);
- random directions inside the recovered slow span had median \(\tau\sim19\);
- matched PCA-span and ambient-random directions had median \(\tau\sim3\);
- about 85 directions, or \(3.7\%\) of the 2304D residual space, carried 80% of positive lifetime excess.

The exact dimensionality was broader than the original 31, which was an early indication that:

\[
\boxed{\text{the phenomenon is robust, while the exact cutoff is procedure/data dependent.}}
\]

---

# 5. R1.8 — document-level state and semantic reproducibility

R1.8 asked two questions:

1. Is slow-space variance especially document-persistent?
2. Does the topic result reproduce in independently recovered slow spaces?

## 5.1 Between-document vs within-document variation

The fraction of variance attributable to between-document differences was:

| Representation | Between-doc | Within-doc |
|---|---:|---:|
| Canonical slow-31 | 43.66% | 56.34% |
| Split A slow-31 | 42.59% | 57.41% |
| Split B slow-31 | 44.07% | 55.93% |
| PCA-31 | 19.23% | 80.77% |
| Random-31 median | ~17.05% | ~82.95% |

Slow-minus-random difference:

\[
0.2661,\qquad 95\%\,{\rm CI}=[0.2613,0.2708].
\]

So slow-31 allocates roughly \(2.6\times\) as much variance to persistent between-document differences as random 31D subspaces.

This does **not** mean the slow representation is static: roughly 56% of its variance is still within-document.

It also cannot explain away the original autocorrelation result, because the original primary timescale estimator demeaned each document before measuring temporal persistence.

The natural decomposition is therefore:

\[
z_{d,t}
=
\underbrace{\mu_d}_{\text{persistent document-level component}}
+
\underbrace{\delta z_{d,t}}_{\text{within-document dynamics}},
\]

with both pieces nontrivial.

## 5.2 Topic semantics reproduce across independent slow fits

Sealed-test topic CE gains:

| Representation | Gain |
|---|---:|
| Canonical slow-31 | 0.03337 nats |
| Split A slow-31 | 0.03366 |
| Split B slow-31 | 0.03407 |
| PCA-31 | 0.01733 |
| Random-31 Q95 | 0.01575 |

The independent slow fits essentially reproduced the entire canonical semantic effect.

This ruled out the explanation:

> “One particular \(Q_{31}\) happened to contain a topic-decodable direction.”

## 5.3 The semantic geometry itself reproduces

The topic classifiers from \(Q_A\) and \(Q_B\) were mapped back into ambient residual space.

After removing the softmax-common direction:

- effective semantic rank: 2;
- canonical correlations: \(0.9020,\ 0.8966\);
- aggregate overlap:

\[
\boxed{S_{\rm sem}=0.8087}
\]

- label-shuffle Q95: \(0.1205\);
- plus-one permutation \(p=0.0099\).

So the two independent slow spaces do not merely support similarly accurate probes. They recover nearly the **same ambient two-dimensional semantic geometry**.

---

# 6. R1.8 axis stability — several individual dynamical axes are real objects

The first five registered TICA axes had matched split-fit absolute cosines:

\[
[0.9810,\ 0.9813,\ 0.8999,\ 0.9717,\ 0.9757].
\]

Thus the leading temporal structure is stronger than “stable subspace, arbitrary rotation.”

Several individual dynamical directions are themselves reproducible.

However, later axes degrade strongly, so this does not justify treating the entire ordered 31D TICA basis as individually canonical.

A useful distinction is:

\[
\boxed{\text{leading axis identity is stable; tail-axis identity is not.}}
\]

---

# 7. R1.9 — content survives document-fingerprint conditioning

The central alternative explanation after R1.8 was:

\[
\text{slow space}
\rightarrow
\text{source/register/template}
\rightarrow
\text{topic}.
\]

R1.9 explicitly conditioned the topic probe on measured document-fingerprint variables:

- coarse source type;
- host suffix / URL structure;
- register;
- formatting/template;
- nonlexical structural variables.

## 7.1 Fingerprint controls were meaningful

Adding fingerprint variables to the old nuisance baseline improved sealed-test topic CE from:

\[
0.41648
\]

to

\[
0.36768,
\]

a gain of:

\[
0.04879\text{ nats}.
\]

So this was not a vacuous nuisance control.

## 7.2 Most of the slow-space topic effect survived

Conditional topic gains:

| Representation | Conditional gain |
|---|---:|
| Canonical slow-31 | 0.02898 nats |
| Split A slow-31 | 0.02917 |
| Split B slow-31 | 0.02977 |
| PCA-31 | 0.01443 |
| Random-31 Q95 | 0.00881 |

Retention relative to R1.8:

| Representation | Effect retained |
|---|---:|
| Canonical | 86.8% |
| Split A | 86.7% |
| Split B | 87.4% |
| PCA | 83.3% |

So measured fingerprint structure accounts for only about 13% of the slow-space topic gain.

The correct interpretation is not that 87% is a literal causal decomposition; rather:

> Most of the observed slow-space topic effect survives a nuisance set that meaningfully predicts topic.

## 7.3 Conditional semantic geometry barely changed

After conditioning:

- ambient overlap: \(0.8042\);
- canonical correlations: \(0.9047,\ 0.8888\);
- shuffle Q95: \(0.1722\);
- \(p=0.0099\).

For comparison, the unconditional overlap was:

\[
0.8087.
\]

So conditioning on fingerprint barely changed the recovered semantic plane.

This substantially weakened the hypothesis that the shared topic geometry was merely a proxy for measured source/style/template correlations.

## 7.4 The slow space still carries some fingerprint information

Balanced diagnostic probes:

| Fingerprint family | Slow balanced acc. | Random median | Random Q95 |
|---|---:|---:|---:|
| Source type | 0.471 | 0.369 | 0.471 |
| Register | 0.632 | 0.465 | 0.597 |
| Formatting | 0.259 | 0.234 | 0.308 |

The clearest non-content enrichment is **register**.

Thus the slow region is not “content rather than fingerprint.” It carries both.

A better ontology is:

\[
\boxed{
\text{persistent document/context state}
=
\text{content}
+
\text{register/source state}
+
\text{other long-lived factors}.
}
\]

### Remaining source caveat

Exact website identity could not be robustly conditioned out:

- 1,200 documents;
- 1,182 distinct hosts;
- 98.7% singleton hosts.

Therefore the valid claim is:

> content beyond the **measured** fingerprint variables,

not:

> content independent of every possible source-specific fingerprint.

---

# 8. R1.10 — the persistent region is broader than \(Q_{31}\)

R1.10 tested nested ranks:

\[
k\in\{8,13,21,31,48,64,96,128\}.
\]

## 8.1 Generic random-in-span persistence

Canonical median signed timescales:

| Rank | Median \(\tau\) | Q25–Q75 | Q95 |
|---:|---:|---:|---:|
| 8 | 85 | 49–167 | 350 |
| 13 | 30 | 21–53 | 168 |
| 21 | 26 | 19–44 | 100 |
| 31 | 25 | 17–40 | 80 |
| 48 | 20 | 15–30 | 63 |
| 64 | 17 | 12–24 | 45 |
| 96 | 5 | 4–8 | 20 |
| 128 | 4 | 3–6 | 17 |

This reveals a three-part structure:

\[
\boxed{
\text{very slow inner core }(k\lesssim8)
\subset
\text{broad persistent shoulder }(k\lesssim64)
\subset
\text{mostly ordinary outer tail }(k\gtrsim96).
}
\]

Independent split fits reproduced the shape:

| Rank | Split A median | Split B median |
|---:|---:|---:|
| 31 | 24.5 | 25 |
| 64 | 16 | 17 |
| 128 | 4 | 4 |

So the broad shoulder and later collapse are not artifacts of the full-data ranking.

## 8.2 Geometry stays reproducible beyond where generic slowness collapses

Chance-corrected split-fit overlap:

| Rank | Corrected overlap |
|---:|---:|
| 8 | 0.769 |
| 13 | 0.746 |
| 21 | 0.843 |
| 31 | 0.862 |
| 48 | 0.872 |
| 64 | 0.887 |
| 96 | 0.867 |
| 128 | 0.868 |

Thus:

\[
\boxed{
\text{geometric reproducibility}
\neq
\text{generic temporal persistence}.
}
\]

The split fits recover a broad common TICA-related geometric region through at least rank 128, but only roughly the first 64 dimensions form a fat generically slow region.

## 8.3 Why \(Q_{31}\) remains useful

Cumulative positive lifetime excess:

| Rank | Fraction |
|---:|---:|
| 8 | 57.4% |
| 13 | 64.2% |
| 21 | 72.5% |
| 31 | 80.2% |
| 48 | 87.5% |
| 64 | 91.4% |
| 96 | 97.2% |
| 128 | 99.6% |

So \(Q_{31}\):

- captures about 80% of positive lifetime excess;
- retains median random-in-span \(\tau\approx25\);
- is heavily validated across all later semantic experiments.

Ranks 32–64 contribute another 11.2% of lifetime excess and remain generically slow, but at lower density.

The current terminology should therefore be:

\[
\boxed{Q_{31}=\text{canonical compact slow slice}}
\]

inside

\[
\boxed{\text{a broader persistent shoulder extending to roughly rank 64}.}
\]

The exact intrinsic dimension is not established and probably should not be treated as a hard integer boundary.

---

# 9. R1.10 — what the leading temporal axes encode

The first five stable TICA axes were analyzed against topic, source type, register, formatting, and position.

Results:

| Axis | A/B cosine | Between-doc variance | Main interpretation |
|---:|---:|---:|---|
| 1 | 0.981 | 58% | register + source/topic mixture |
| 2 | 0.981 | 58% | formal/conversational register + content |
| 3 | 0.900 | 53% | no simple standalone interpretation |
| 4 | 0.972 | 76% | register/source style |
| 5 | 0.976 | 76% | strong register/source structure |

For all five:

\[
\eta^2_{\rm position}<0.005,
\qquad
\eta^2_{\rm formatting}<0.007.
\]

So the leading temporal axes are not primarily trivial token-position or formatting coordinates.

They mostly describe high-level persistent document register/source-style variables, with correlated semantic content.

This supports the view that TICA is discovering dynamical coordinates of a mixed persistent state, not one human concept per axis.

---

# 10. R1.10 — semantic coordinates are reproducible but distributed across temporal axes

The three registered topic contrasts all reproduced across independent slow fits:

| Contrast | A/B cosine |
|---|---:|
| Business vs home/food | 0.895 |
| Business vs other | 0.903 |
| Home/food vs other | 0.891 |

Each had the same signed held-out ordering in both fits.

Qualitative extremes were semantically sensible.

However, no semantic contrast nearly coincided with one leading TICA axis.

Largest alignments included:

- axis 3 ↔ business-vs-other: \(0.498\);
- axis 3 ↔ home-vs-other: \(0.405\);
- axis 4 ↔ home-vs-other: \(0.392\);
- axis 5 ↔ business-vs-home: \(0.348\).

Thus:

\[
\boxed{
\text{stable dynamical axes}
\neq
\text{individual semantic contrast axes}.
}
\]

The best interpretation is that the persistent region supports **multiple meaningful bases**:

- a temporal eigenbasis selected for dynamics;
- semantic contrast directions selected for content.

Semantic variables are distributed across several temporal modes rather than cleanly identical to individual TICA axes.

---

# 11. Earlier matched-suffix work: generic history sensitivity but unresolved semantics

Before the R1.9/R1.11 identification improvements, a matched-suffix intervention changed remote prefixes while holding later tokens fixed.

That experiment showed that remote history strongly affected later slow-space representations.

However, same-topic swaps were also large.

Therefore it established:

\[
\text{remote history}
\rightarrow
\text{later representation}
\]

but did **not** isolate:

\[
\text{topic/content}
\rightarrow
\text{later semantic state}.
\]

The remaining alternatives included document identity, source, style, entities, subtopic, and splice mismatch.

R1.11 was designed specifically to resolve that ambiguity with a frozen content readout and matched same-topic donor controls.

---

# 12. R1.11 — controlled persistent content state

R1.11 is the strongest post-LessWrong result.

The intervention changed only the first 256 tokens, then held the continuation exactly identical.

Each recipient had:

1. natural/reference prefix;
2. matched same-topic donor prefix;
3. matched different-topic donor prefix.

The measurement used the already-frozen R1.9 business-vs-home content direction.

The primary difference-in-differences statistic was:

\[
D(k)=T_{\rm different}(k)-T_{\rm same}(k),
\]

where positive values mean the current representation is shifted toward the different-topic donor in the preregistered semantic direction.

## 12.1 Primary trajectory

| Horizon | \(D(k)\) | 95% CI | Retained from \(k=0\) |
|---:|---:|---:|---:|
| 0 | 5.5202 | [4.5699, 6.4602] | 100.0% |
| 32 | 1.2721 | [0.9209, 1.6268] | 23.0% |
| **64** | **0.9256** | **[0.6764, 1.1642]** | **16.8%** |
| 128 | 0.5745 | [0.3879, 0.7876] | 10.4% |
| 256 | 0.3313 | [0.1742, 0.4774] | 6.0% |

The preregistered endpoint was \(k=64\). It passed by a wide margin.

The effect remained positive through 256 identical continuation tokens.

The trajectory is best described as:

\[
\boxed{
\text{large immediate semantic transient}
+
\text{smaller slowly decaying persistent tail}.
}
\]

A descriptive exponential approximation gives a tail half-life of roughly 115 tokens, but this was not preregistered and should not be treated as a mechanistic fit.

## 12.2 The effect is specifically donor-directed

Canonical same-topic vs different-topic scores:

| Horizon | Same-topic | Different-topic | Difference |
|---:|---:|---:|---:|
| 0 | 1.6819 | 7.2021 | 5.5202 |
| 32 | -0.0714 | 1.2007 | 1.2721 |
| 64 | -0.0527 | 0.8729 | 0.9256 |
| 128 | -0.0408 | 0.5337 | 0.5745 |
| 256 | -0.1375 | 0.1938 | 0.3313 |

After the immediate boundary transient, the matched same-topic effect is approximately zero.

The persistent tail is therefore not simply:

> every foreign prefix causes a long-lasting representation shift.

It is specifically:

> a different-topic prefix leaves a later residual displacement in the donor-topic direction.

## 12.3 Generic splice sensitivity does not explain the signed result

Content-plane norms at \(k=64\):

\[
G_{\rm same}=2.3817,
\qquad
G_{\rm different}=2.5166.
\]

The norm difference is only:

\[
0.135.
\]

Yet the donor-directed signed difference is:

\[
0.9256.
\]

So both splice conditions perturb the content plane, but only the different-topic splice produces the predicted semantic orientation.

This makes “different-topic splices just disrupt the model more” a poor explanation of the primary effect.

## 12.4 Independent semantic readouts reproduce the effect

At \(k=64\):

\[
D_{\rm canonical}=0.9256,
\]

\[
D_A=0.8330,
\]

\[
D_B=0.9712.
\]

The whole trajectory is similar under both independently learned readouts.

Thus the controlled state effect is not dependent on one particular fitted semantic direction.

## 12.5 Reciprocal topic directions reproduce

At \(k=64\):

\[
D_{\rm business\ prefix\rightarrow home\ recipient}=0.9052,
\]

\[
D_{\rm home\ prefix\rightarrow business\ recipient}=0.9461.
\]

So the pooled effect is not driven by a one-sided “business/formality” perturbation.

The semantic orientation reverses correctly with the donor topic.

---

# 13. R1.11 PCA control — semantic representation vs persistent semantic representation

The frozen PCA-31 semantic readout showed:

| Horizon | Slow \(D(k)\) | PCA \(D(k)\) |
|---:|---:|---:|
| 0 | 5.5202 | 6.3966 |
| 32 | 1.2721 | 0.4185 |
| **64** | **0.9256** | **0.0229** |
| 128 | 0.5745 | -0.2616 |
| 256 | 0.3313 | -0.0005 |

This is especially informative.

At the splice boundary:

\[
D_{\rm PCA}(0)>D_{\rm slow}(0).
\]

So the slow readout is not simply a more responsive semantic detector.

But as the identical continuation accumulates:

\[
D_{\rm PCA}(k)\rightarrow0
\]

while:

\[
D_{\rm slow}(k)>0.
\]

This cleanly separates:

\[
\boxed{\text{semantic representation}}
\]

from:

\[
\boxed{\text{persistent semantic representation}}.
\]

The time-lagged slow geometry is specifically enriched for the part of semantic history that remains present later.

---

# 14. Current ontology of the slow region

The accumulated evidence no longer fits a picture like:

> “there are 31 topic neurons/directions.”

A better picture is:

\[
\boxed{\text{a multiplexed, graded persistent state region}}
\]

with several properties.

## 14.1 Graded temporal structure

\[
\text{very slow core}
\subset
\text{persistent shoulder}
\subset
\text{stable outer TICA geometry}.
\]

There is no established sharp intrinsic rank.

## 14.2 Multiple long-lived variables

The region carries:

- content/topic;
- register;
- source/style;
- probably other persistent document/context variables.

## 14.3 Temporal and semantic bases differ

The leading TICA basis gives natural dynamical modes.

The semantic classifier gives natural content coordinates.

These are different rotations of overlapping persistent geometry.

## 14.4 State is history-dependent, not just static document identity

R1.11 shows that manipulating earlier semantic history changes later semantic state under identical subsequent tokens.

So the slow region is not merely correlated with static labels assigned to a document.

At least one content coordinate behaves as a genuine history-dependent state variable.

---

# 15. What is now strongly supported

## 15.1 Existence

A heavy-tailed distribution of residual-stream persistence exists in Gemma-2-2B layer 12.

## 15.2 Sequence dependence

The slow tail depends strongly on ordered document structure; token-position shuffling collapses it.

## 15.3 Fat slow geometry

Generic random directions inside the compact slow span remain slow.

## 15.4 Reproducibility

Independent data splits recover substantially the same signed-TICA slow region.

## 15.5 Graded extent

The generically slow region extends beyond \(Q_{31}\) to roughly rank 64, while generic persistence falls sharply by ranks 96–128.

## 15.6 Document-state enrichment

The slow region is strongly enriched for between-document variation relative to PCA/random controls.

## 15.7 Semantic content

Coarse topic information is strongly enriched in the slow region.

## 15.8 Content beyond measured fingerprint

Most of the topic effect survives controls for measured source, register, URL/host structure, formatting, and other nonlexical document features.

## 15.9 Reproducible semantic geometry

Independent TICA fits recover nearly the same low-dimensional content plane and the same pairwise content contrasts.

## 15.10 Interpretable leading temporal structure

Several leading TICA axes are individually reproducible and primarily reflect register/source-style state.

## 15.11 Controlled persistent semantic state

Changing earlier topic content causes a directionally appropriate change in later slow-space semantic state after 64–256 identical continuation tokens.

## 15.12 Temporal privilege over PCA

An equal-rank PCA semantic direction has a strong immediate semantic response but almost no persistent donor-specific effect by 64 tokens.

---

# 16. What is *not* established

## 16.1 Causal use of the slow semantic state

R1.11 establishes:

\[
do(\text{earlier content})
\rightarrow
\text{later semantic representation}.
\]

It does **not** yet establish:

\[
do(\text{slow semantic coordinate})
\rightarrow
\text{later computation or behavior}.
\]

So the slow content state is causally induced by history, but causal necessity/sufficiency of the representation itself remains open.

## 16.2 Exclusive localization

Content is not exclusive to the slow region.

PCA-31 also contains topic information, and representations in transformers are distributed/redundant.

The supported claim is preferential persistence/enrichment, not exclusive storage.

## 16.3 Exact intrinsic dimension

31 is a useful canonical compact cutoff.

The broader slow shoulder extends to about 64.

There is no evidence for one exact natural integer dimension.

## 16.4 Universal axis-level interpretability

Several leading axes are stable and interpretable, but some stable axes remain mixed or have no simple one-label explanation.

## 16.5 Complete source independence

Exact host identity could not be conditioned out because nearly all hosts were singletons.

The content claim is conditional on measured fingerprint variables.

## 16.6 Generality beyond the current setting

The full semantic-state result has not yet been established across:

- other layers;
- other models;
- other corpora;
- other semantic contrasts.

## 16.7 Behavioral meaning

The demonstrated state is contextual semantic content history.

It should not yet be called:

- a belief;
- a goal;
- a world model;
- a factual memory;
- a planning state.

---

# 17. Separate result that cautions against assuming generic causal privilege

In a separate Gemma-2-9B-IT running-state task, slow-subspace coordinates were descriptively enriched for state information, but causal state-swap effects were much stronger in PCA geometry than in the task-local slow subspace.

That experiment is a different model/task/estimator setting and should not be directly merged with the Gemma-2-2B content-state result.

But it is a useful warning:

\[
\boxed{
\text{temporal specialization}
\neq
\text{causal privilege by default}.
}
\]

Therefore the next causal experiment should test the frozen semantic coordinate directly rather than assuming that slowness itself implies downstream causal use.

---

# 18. Current best concise claim

A conservative version:

> **Gemma-2-2B layer 12 contains a reproducible low-dimensional residual region selected for temporal persistence. A compact canonical 31D slice captures about 80% of lifetime excess, with a broader generically slow shoulder extending to roughly 64 dimensions. The region is strongly enriched for persistent document-level state, including both register/source structure and content information. Independent fits recover nearly the same semantic content geometry, and most topic signal survives measured source/style controls. In matched-prefix interventions, changing earlier topic induces a donor-directed semantic displacement in the slow region that remains detectable after 64–256 identical continuation tokens, while an equal-rank PCA semantic effect rapidly disappears.**

A shorter conceptual version:

> **The residual stream contains a reproducible persistent state region in which earlier semantic context leaves a slowly decaying, directionally specific content trace.**

---

# 19. How the story changed since the LessWrong post

The LessWrong post's central message was roughly:

\[
\boxed{\text{a compact part of the residual stream forgets slowly}.}
\]

The post-LessWrong work progressively changed that into:

### Step 1 — It is reproducible

Independent signed-TICA fits recover the same slow region.

### Step 2 — It carries persistent document state

Slow-space variation is much more document-level than PCA/random variation.

### Step 3 — It contains reproducible semantics

Independent slow fits recover nearly the same topic plane.

### Step 4 — The semantics are not mostly source/style proxy

About 87% of the topic effect survives measured document-fingerprint controls.

### Step 5 — The region is graded, not exactly 31D

\(Q_{31}\) is a compact high-density slice; the generically slow shoulder extends to about rank 64.

### Step 6 — Temporal axes and semantic axes are different

Leading TICA modes mostly capture register/source state; semantic contrasts are distributed across several temporal modes.

### Step 7 — The content behaves like state

Changing earlier topic changes the later semantic coordinate after the intervening tokens are held exactly fixed.

This is the main conceptual upgrade:

\[
\boxed{
\text{slow geometry}
\rightarrow
\text{persistent semantic state}.
}
\]

---

# 20. The next decisive experiment

At this point, further static characterization has diminishing returns.

The main unresolved question is:

\[
\boxed{\text{Does the model use the identified persistent semantic coordinate?}}
\]

The natural next experiment is to intervene directly on the frozen content coordinate / semantic plane while holding the surrounding residual state as controlled as possible, then ask whether:

1. the induced perturbation propagates to later residual states in the predicted direction;
2. downstream logits or behavior change in the corresponding semantic direction;
3. matched PCA/random semantic directions with similar decodability do not produce the same persistent causal effect.

A positive result would complete the strongest desired chain:

\[
\boxed{
\text{earlier content}
\rightarrow
\text{persistent semantic state}
\rightarrow
\text{downstream computation/behavior}.
}
\]

That would move the project from identifying a persistent semantic representation to identifying a **causally used semantic state variable**.
