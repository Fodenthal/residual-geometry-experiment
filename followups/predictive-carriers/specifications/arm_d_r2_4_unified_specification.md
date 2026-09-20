# Residual Sequence-Memory Geometry — Predictive Carrier Resolution (unified)
## Arm D-R2.4, source specification folded with the R2.4 amendment ledger

```text
protocol_revision:
arm_d_r2_4_predictive_carrier_resolution_unified

status:
EXECUTING

source_specification:
docs/specifications/arm_d/arm_d_r2_4_source_specification.md
sha256 a566209ee26feaa74776801523a101368d249280973df896710e8ef69c9f26b2
```

## 0. Precedence

The source specification governs every section it states. This document is normative only
where it amends or operationalizes the source. Where the two disagree, this document
governs and the disagreement is recorded in the ledger of section 0.1. The source file is
immutable lineage and is never edited.

The R2.4 lineage is R1 (`exp_01kzmd2x1hfaas5are07m3bqpe`), R2.2
(`exp_01kzs0q57aet88bm4svwasaxex`), R2.3 (`exp_01kzsrwrndepmbc5jd18t1npf7`) and R2.3c
(`exp_01kzt4r3zfexna89meakdkxr3r`). No frozen verdict of any of those runs is altered here.

### 0.1 Amendment ledger

Every amendment below was decided on 2026-08-12, before any R2.4 development number was
computed, and is therefore prospective with respect to every result this run reports. The
`when` column records that fact explicitly.

| id | when | what it changes | source section amended |
|----|------|-----------------|------------------------|
| A1 | 2026-08-12, before any DEV number was read | the multi-horizon operators are built from the persistence-residualized cross-covariance, matching the score | 8.1, 8.2 |
| A2 | 2026-08-12, before any DEV number was read | each lag's contribution is trace-normalized before summing | 8.3 |
| A3 | 2026-08-12, before any DEV number was read | the energy-matched deletion family F becomes REQUIRED and primary for the L axis | 5.3, 5.4, 12.1 |
| A4 | 2026-08-12, before any DEV number was read | a confirmatory replay of the frozen deletion family is added as step C6b; `L1_GEOMETRY_SPECIFIC_DELETION` may not be emitted without it | 5.4, 19 |
| A5 | 2026-08-12, before any DEV number was read | ridge stability becomes a reported axis: the carrier is refit across a frozen ridge grid and compared | 8, 17 |
| A6 | 2026-08-12, before any DEV number was read | an ANISOTROPIC no-prediction world is added to the required worlds | 18 |
| A7 | 2026-08-12, before any DEV number was read | the rank-efficiency ratio gets a multi-horizon shared-source ceiling; the per-lag free-source ceiling is kept as a looser upper bound | 7.2, 7.3 |
| A8 | 2026-08-12, before any DEV number was read | sizing is set by split-half MSC at the largest rank, and the reproduction requirement gets a numeric threshold frozen at S6 | 4.3, 15 |
| A9 | 2026-08-12, before any DEV number was read | the M-stage data rule is fixed now: fresh pool, or composition statistics from the untouched test split only | 14, 25 |
| A10 | 2026-08-12, before any DEV number was read | the `P0` identifier collision is resolved before any code is written | 5, 13.1, 19 |
| A11 | 2026-08-12, from a synthetic implementation smoke world, before any DEV-Q or CONFIRM-Q number was computed | the primary score is the baseline-augmented gain; source section 3's literal form is co-reported | 3, 7.3, 9 |
| A12 | 2026-08-12, from the DEV-Q variance audit, before the freeze and before any CONFIRM-Q number existed | the source's energy-matched deletion family is arithmetically unattainable in this aperture, so the deletion contrast is dose-matched top-versus-lower instead | 12.4, 13.4 |
| A13 | 2026-08-12, from the DEV-Q ceilings stage, before the freeze and before any CONFIRM-Q number existed | the decoder ridge grid is extended upward and every selection records whether it sits at a grid edge | 3, 7.3 |
| A14 | 2026-08-12, from the DEV-Q baseline bank under the extended grid, before the freeze and before any CONFIRM-Q number existed | a richer baseline must beat the simpler incumbent by 1e-3 held-out R-squared to displace it | 3, 7.2 |
| D1 | 2026-08-12, after the five labels were read, at the researcher's direction | rank-matched random comparators are added at every frozen rank; the frozen family drew them only at the primary rank | 11.2, 13.1 |
| D2 | 2026-08-12, after the five labels were read, at the researcher's direction | the complement carrier is compared with random subspaces drawn inside the same complement; the frozen 0.50 relative-gain threshold is weaker than that level and no longer carries the label | 12.2, 13.5 |
| D3 | 2026-08-12, after the five labels were read, at the researcher's direction | amendment A7 is recorded as defective as written: the construction it names a shared-source ceiling maximizes an in-sample trace form, not the held-out score, so it is a same-family comparator and the only true upper bound is the full aperture | 9.4, 11.3 |
| D4 | 2026-08-12, after the five labels were read, at the researcher's direction | section 10.3's material-disagreement clause is applied: the two estimators disagree on the P axis, so METRIC_DEPENDENT_PREDICTIVE_CARRIERS is recorded beside the frozen concordance label | 10.3, 13.3 |
| D5 | 2026-08-12, after the five labels were read, at the researcher's direction | split-half stability is reported as a position between the association-destroying null and the resampling ceiling, beside its p-value | 13.2, 19.4 |
| D6 | 2026-08-12, after the five labels were read, at the researcher's direction | the frozen `P0` label is scoped in writing to the primary estimator's own saturation profile: it states where that estimator's rank curve reaches its 0.90 criterion and does not state that compact prediction is unattainable, since the co-reported estimator attains it at rank 16 | 8.2, 13.1 |
| D7 | 2026-08-12, after the five labels were read, at the researcher's direction | the researcher's challenge to the G-axis threshold is recorded as INCORRECT and resolved rather than left open: `metric_concordance_fraction` multiplies the within-half self-overlap at the comparison rank, so the bar is 0.3175 and the observed 0.4725 clears it, exceeds the 0.3969 ceiling outright and stands at 7.626 times the measured chance mean; `G1_METRIC_CONCORDANT_CARRIER` stands on its own statistic | 10.2, 10.3 |

### 0.1.1 The D-series is post-freeze and separately marked

Rows A1 to A14 were fixed before any CONFIRM-Q number existed. Rows D1 to D5 were not: they were
directed by the researcher after the five labels were read, and every artifact they produce is
stamped `POST_FREEZE_DIAGNOSTIC`. They may change what the report claims and what a later stage
should believe. They do not rewrite a frozen label, and none of them touched the sealed test
split, which remains at one read.

Two of the seven record defects in the amendments or the reporting rather than additions to
either. D7 records a challenge that did not survive contact with the frozen rule, and it is kept
in the ledger for the same reason the defects are: a review objection that turns out to be wrong
should be resolvable from the record, not silently dropped.

Of the substantive findings, two record defects rather than additions. D3 finds amendment A7 wrong as written:
its construction attains the maximum of the in-sample trace form it is defined to maximize at
every rank, and is still outscored on held-out gain by the run's own carriers, so it optimizes a
different functional than the score the report grades carriers on and cannot bound that score.
D4 finds a clause of the source specification that was never implemented: section 10.3 requires a
metric-dependent label when the two estimators disagree materially, they do disagree about
compactness, and the run emitted only the concordance label.

### 0.2 A13 — the scoring regularizer must not be chosen by the grid

On DEV-Q every decoder in the scoring path selected the inherited grid's top value, 1e4, at
every lag and every rank. A regularizer pinned to the edge of its grid is chosen by the grid
rather than by the data, and here it is chosen inside the quantity that decides the run: the
held-out gain. Two consequences follow. The grid is extended to 1e10, so the selection can be
interior. And every scored carrier records, per lag and per score variant, whether its
selection still sits at an edge, so a later reader can see the condition rather than
reconstruct it. The development stages were rerun under the extended grid before the freeze;
nothing confirmatory had been computed, and no carrier had been promoted.

### 0.3 A14 — the baseline bank must not be decided by noise

Under A13's extended grid the increment baseline took lag 32 with a pooled held-out R-squared
of +1.933e-5 against the zero baseline's exact zero, at a ridge of 1e6 where its prediction is
numerically nil. The prediction was unaffected; the recorded label was not, and a label that
flips on the eighth decimal is not reproducible. The bank is therefore ordered from the least
presumptuous prediction upward, zero then persistence then the fitted increment, and a richer
candidate must beat the incumbent by 1e-3 held-out R-squared to displace it. The DEV-Q trees
predate this rule, and their lag-32 baseline label reads B2 for that reason; the rule resolves
that case to B0, which is what the confirmatory run uses.

---

## 1. A10 — one label per name

The source uses `P0` for three different objects. This run uses:

```text
S0 .. S6      development steps (source section 19's R2.4-P0 .. R2.4-P6)
C0 .. C9      confirmatory steps (source section 19, unchanged)
D-AUDIT       the variance-deletion audit (source section 5, "Preflight P0")
P0_NO_COMPACT_PREDICTIVE_COMPRESSION   the P-axis label (source section 13.1, unchanged)
```

Only the P-axis label keeps the `P0` string, because it is a result label rather than a
step identifier. Step identifiers are `S`/`C`; the deletion audit is named for what it is.

Development steps:

```text
S0   lock this specification and the existing-run provenance
S1   D-AUDIT: the variance-dose deletion audit on DEV-Q
S2   direction-versus-magnitude diagnostic on DEV-Q
S3   ceilings: implement and unit-test the full and reduced-rank ceilings
S4   estimators: implement and unit-test Q-E and Q-W on the known-answer worlds
S5   DEV rank curves, stability learning curves, ridge-stability sweep, sizing
S6   FREEZE: lags, weights, ridges, rank rule, nulls, thresholds, sample size
```

Confirmatory steps are the source's C0 .. C9 with one insertion:

```text
C6b  replay the frozen D-AUDIT deletion family on CONFIRM-Q (amendment A4)
```

---

## 2. The analysis object

Inherited unchanged from R2.2/R2.3/R2.3c, so every comparator remains comparable:

```text
model            google/gemma-2-2b, revision c5ebcd40d208330abc697524c919956e692655cf
layer            12, residual stream
corpus           allenai/c4, en, revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2
aperture         R1's frozen mean and basis, width 256 (never refit)
row              one document; condition "main"; suffix length 64
x_t              the paired remote-history difference at the source position, source hook
y_{t,k}          x_{t+k}, the same aperture, the same hook
K_Q              (1, 2, 4, 8, 16, 32, 64); lag 128 descriptive only
```

The suffix length 64 is R2.3's and R2.3c's analysis object; R2.4 inherits it so the
descriptive comparators of source section 11 are scored on the object they were fitted for.

---

## 3. The score, and A1

For a carrier `Q` and lag `k`, the primary score is source section 3's held-out gain over
the frozen carrier-independent baseline `b_k`:

\[
\Gamma_Q(k) = R^2(y_k, D_{Q,k}Q^\top x_t) - R^2(y_k, b_k(x_t)).
\]

`b_k` is R2.3c's global baseline bank — `B0` zero, `B1` full-aperture persistence `x_t`,
`B2` a document-grouped ridge map from the aperture increment `u_t = x_t - x_{t-1}` — with
the per-lag winner chosen by document-grouped cross-validation on fit documents only, and
then used for every carrier and every rank. `R^2` is the equal-document score: per-document
mean squared row norm, averaged over documents.

**A1.** The operators of source sections 8.1 and 8.2 are built from the
persistence-residualized cross-covariance

\[
C_k = \mathbb{E}\big[x_t\,(y_{t,k} - b_k(x_t))^\top\big],
\]

and the destination covariance used by Q-W is the covariance of that same residualized
object, \(\tilde\Sigma_k = \operatorname{Cov}(y_{t,k} - b_k(x_t))\). The raw-\(C_k\) form is
computed and reported as a diagnostic, never as the primary carrier.

Rationale: the score is a gain over persistence, while raw \(C_k\) rewards predicting the
persistent component, which at `k=1,2` is nearly the whole target. Building the objective on
the raw target readmits the energy confound through the objective after the whitened metric
was chosen to exclude it.

Two scores are reported for every carrier, rank and lag, from the same fits:

```text
gamma_raw    source section 3 exactly: predictor D_{Q,k} z_t, gain over b_k   [PRIMARY]
gamma_aug    predictor b_k(x_t) + Dtilde_{Q,k} z_t, gain over b_k             [co-reported]
gamma_white  the same contrast scored in the destination-whitened metric      [source section 9]
```

**A11.** `gamma_aug` is the primary score and decides every label; `gamma_raw` is
co-reported. The two answer different questions: how much a carrier adds to what persistence
already gives, and how much of the future it reconstructs on its own.

Rationale, decided from a synthetic implementation world before any DEV-Q or CONFIRM-Q number
existed. Under the literal form, a rank-`r` carrier must also reconstruct the persistent
component of the future out of `r` dimensions, and persistence is nearly the whole target at
the shortest lags. Its gain over persistence is therefore large and negative at every rank
below the full aperture: on the smoke world a rank-3 carrier that recovers the planted span
at 0.9624 subspace correlation still scores -0.1732 under the literal form and +0.03780 under
the augmented form. Both efficiency ratios of source section 7.3 would then divide a negative
number by a positive one, and the section 9 compression summary would be taken over a curve
with no meaningful zero. The augmented form is also exactly what amendment A1's residualized
objective optimizes, so the estimator and the score finally agree, which was A1's purpose.

The literal form remains informative and is reported at every rank and lag: it is the honest
answer to "could this carrier stand in for the aperture", and its distance from zero measures
how much of the target is simply the present.

---

## 4. A2 — lag normalization

Both operators sum trace-normalized per-lag terms:

\[
G_E=\sum_{k\in K_Q} \frac{w_k}{\tau^E_k}\,W_0C_kC_k^\top W_0,
\qquad
\tau^E_k=\operatorname{tr}\!\left(W_0C_kC_k^\top W_0\right),
\]

\[
G_W=\sum_{k\in K_Q} \frac{w_k}{\tau^W_k}\,\left(W_0C_k\widetilde W_k\right)\left(W_0C_k\widetilde W_k\right)^\top,
\qquad
\tau^W_k=\operatorname{tr}\!\left(\cdot\right),
\]

with `w_k = 1/|K_Q|` and \(\widetilde W_k=(\tilde\Sigma_k+\epsilon_k I)^{-1/2}\). The
unnormalized form is a required sensitivity. The long-horizon weighting
`K = [8,16,32,64]` of source section 8.3 remains a required sensitivity.

Rationale: equal weights apply to objective terms rather than to scores, and
cross-covariance magnitude decays with lag, so without normalization the multi-horizon
carrier is a short-lag carrier wearing a multi-horizon label.

---

## 5. Ridge, and A5

\(W_0=(\Sigma_0+\epsilon_0 I)^{-1/2}\) and \(\widetilde W_k=(\tilde\Sigma_k+\epsilon_k I)^{-1/2}\),
with `eps = rho * trace(Sigma)/p` and one shared relative scale `rho` for source and
destination. `rho` is resolved on fit documents only, by document-grouped cross-validated
multi-horizon gain of `Q_W` at the reference rank 16, over the frozen grid

```text
RHO_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
```

**A5.** `Q_W` and `Q_E` are refit at every `rho` in the grid and the mean subspace
correlation between the selected carrier and every other grid variant is reported as its own
axis:

```text
R1_RIDGE_STABLE_CARRIER      min MSC over the grid neighbourhood >= RIDGE_STABILITY_MIN
R0_RIDGE_DEPENDENT_CARRIER   otherwise
```

`RIDGE_STABILITY_MIN` is frozen at S6 from the DEV sweep. A carrier that moves materially
with `rho` is a regularization choice and is not promoted to the reference object, whatever
its split-half stability says.

Rationale: a ridge artifact is common to both document halves, so split-half agreement
structurally cannot detect it. The whitened estimator inverts the destination covariance and
is designed to look where variance is small; in this aperture the leading 16 directions hold
0.6722 of the variance, so the tail it looks at is estimated from little data.

If the selected `rho` exceeds the top of the grid, or the source or destination condition
number after regularization exceeds `1e8`, the metric is marked `NUMERICALLY_UNSTABLE` and
its carrier is not turned into a reference state (source section 17).

---

## 6. A7 — the ceilings

Three ceilings are computed, all on fit documents, all scored through the identical
held-out path:

```text
Gamma_full(k)         full 256-dimensional aperture, document-grouped ridge
Gamma_rrr(r,k)        per-lag reduced-rank regression: source and destination free
Gamma_shared(r)       ONE source subspace U_r shared across every lag in K_Q
```

`Gamma_shared` maximizes the score-weighted sum

\[
U_r=\arg\max_{U^\top U=I_r}\;\sum_{k\in K_Q}\frac{1}{\mathbb E\|\tilde y_k\|^2}\,
\operatorname{tr}\!\left(U^\top \Sigma_0^{-1/2}C_kC_k^\top\Sigma_0^{-1/2}U\right),
\]

that is, the best single rank-`r` source bottleneck for the reported score rather than for
either estimator's own objective. It differs from `Q-E` in weighting (target energy rather
than term trace) and from `Q-W` in metric. Two efficiency ratios are reported:

\[
\eta_{\rm global}(Q,k)=\frac{\Gamma_Q(k)}{\Gamma_{\rm full}(k)},
\qquad
\eta_{\rm rank}(Q,k)=\frac{\Gamma_Q(k)}{\Gamma_{\rm shared}(r)(k)},
\]

with the per-lag `Gamma_rrr` ratio kept and labelled as a looser upper bound. Because
`Gamma_shared` is fitted on fit documents and scored held out, `eta_rank` may exceed one;
that is reported, not clipped.

Rationale: the per-lag ceiling frees the source subspace per lag while every candidate
carrier is one subspace shared across lags, so the per-lag ratio sits below one for a
structural reason rather than for an inefficiency, and the penalty grows with the spread of
`K_Q`.

---

## 7. A3 and A4 — deletion, dose response, and load-bearingness

The D-AUDIT deletion families, all with a FULL refit of the predictive model from the
surviving source coordinates to the unchanged 256-dimensional target:

```text
A   PCA1:16
B   PCA17:32
C   rank-16 Haar deletions in the whole aperture              (20 draws)
D   mixed: m top PCs + (16-m) complement-Haar, m in 0,2,4,8,12,16
E   small-angle rotations of PCA1:16 toward lower PCs         (frozen angle grid)
F   energy-matched, rank-varying: lower PCs taken in order until V(D) >= V_PCA16   [REQUIRED]
G   the rank-varying top-PC series PCA1:r for r in the rank grid (dose context)
```

**A3.** Family F is required and is the primary L-axis comparator; the dose-response curve
across every family is the continuous context. The source's impossibility argument forbids
matching rank and captured variance simultaneously near the variance-maximizing subspace; it
does not forbid matching captured variance at higher rank, which is what F does (expected
rank 40 to 60 for `V_PCA16 = 0.6722`).

The L axis is decided by:

```text
L1_GEOMETRY_SPECIFIC_DELETION   the target's surviving gain is BELOW family F's by more than
                                DELETION_MARGIN_MIN, with a document-bootstrap lower bound on
                                the difference above zero, AND the confirmatory replay agrees
L0_ENERGY_EXPLAINS_DELETION     the target's surviving gain is inside the dose-response
                                envelope at its deleted variance
L_UNDERRESOLVED                 otherwise, or if Gamma_full does not clear the positive-gain
                                floor at the lags concerned
```

**A4.** `L1_GEOMETRY_SPECIFIC_DELETION` may not be emitted from DEV-Q. It requires step C6b,
the replay of the frozen deletion family on CONFIRM-Q. Until C6b runs, the strongest
admissible DEV statement is `LOAD_BEARINGNESS_GEOMETRY_SUGGESTIVE`, exactly as source
section 5.4 allows.

---

## 8. Stability, nulls, and the metric-geometry rule

Per source section 10: independent fits on fit-half-A and fit-half-B, mean subspace
correlation, full principal-angle spectrum, containment both ways.

```text
analytic Haar reference          E[MSC] = r/256   (sanity check only)
inferential null                 full-refit association destruction: the future block is
                                 deranged across documents inside the fit half, every
                                 covariance and both operators are recomputed from scratch,
                                 400 frozen replicates per estimator and rank
attainable ceiling               within-half document bootstrap refits, 200 replicates
```

```text
I1 requires   split-half MSC above every null replicate (empirical p <= 1/401) at the frozen
              rank, for the estimator concerned
G1 requires   MSC(Q_E, Q_W) >= METRIC_CONCORDANCE_FRACTION x the within-half bootstrap mean
              self-overlap of Q_W at the same rank, with both estimators stable
G0            both estimators stable, MSC below that fraction
G_UNDERRESOLVED  either estimator not stable beyond its null
```

`METRIC_CONCORDANCE_FRACTION` is frozen at S6. Comparing the cross-metric overlap against
the attainable sampling ceiling rather than against a fixed constant keeps the rule honest
when the estimator itself is noisy.

---

## 9. A8 — sizing and the reproduction threshold

The confirmatory document count is chosen from `{2000, 4000, 8000}` on DEV-Q by the
**slowest-converging endpoint**, named in advance as split-half MSC at the largest rank in
the grid (64), evaluated at the confirmatory fit-half fraction 0.35. Document-resampling
learning curves are measured at DEV subsample sizes, the curve `MSC(n) = a - b/n` is fitted,
and the rule is:

```text
choose the smallest N whose predicted split-half MSC at 0.35N is within SIZING_MSC_TOLERANCE
of the fitted asymptote a, AND whose predicted document-bootstrap standard error of the
multi-horizon mean gain is at most SIZING_SE_MAX; if no candidate qualifies, choose 8000
```

`SIZING_MSC_TOLERANCE` and `SIZING_SE_MAX` are frozen at S6 before the curves are read for
the decision.

The source section 15 point 5 requirement that the test read reproduce the result "in sign
and material magnitude" is given a number at S6:

```text
REPRODUCTION_MIN_FRACTION   the test multi-horizon mean gain must be at least this fraction
                            of the validation value, with the same sign, and the test
                            split-half MSC at the frozen rank at least this fraction of the
                            validation value
```

---

## 10. A6 — the required worlds

Source section 18's worlds, plus one:

```text
Q-S0   isotropic no-prediction null
Q-S1   low-rank predictive source under isotropic covariance
Q-S2   energy-only nuisance: high variance, no extra predictive relation
Q-S3   low-variance strongly predictive carrier
Q-S4   two redundant predictive carriers
Q-S5   metric-disagreement world
Q-S6   ANISOTROPIC no-prediction null: a steeply decaying spectrum matching the real
       aperture, with no predictive structure anywhere                       [amendment A6]
```

Q-S6 is the world the existing suite misses. Q-S0 is isotropic, and Q-S3 only shows that
Q-W *can* find a real low-variance carrier; nothing in the source suite shows it does not
manufacture one out of a poorly estimated low-variance tail. Q-W must not report a stable
carrier on Q-S6: its recovered subspace must fail the split-half stability rule of section 8
at the same replicate count used on real data.

These worlds run the real estimators end to end through the real scoring path. They validate
implementation and interpretation; they authorize no candidate-superiority test.

---

## 11. A9 — the data rule the next stage inherits

The autonomy stage that receives this run's carrier object must either draw a fresh document
pool, or take every composition statistic only from R2.4's untouched test split. Composition
retention is a ratio whose denominator is the carrier's own predictive gain, so computing it
where the carrier was fitted and selected reintroduces the denominator problem that
invalidated R2.3b, one stage later. This rule is recorded in the handoff manifest at C9 and
is binding on the next specification.

---

## 12. Data boundary

```text
DEV-Q      the spent 2000-document R2.2 pool (its sealed 300 were opened by R2.2).
           Method development only.  Nothing computed on it promotes a carrier.
CONFIRM-Q  a fresh C4/en capture, document-disjoint by content hash from R1's 2400-document
           pool and from R2.2's 2000-document pool, split 35/35/15/15 by document into
           fit-half-A, fit-half-B, validation, test.
```

Disjointness is proved by hash intersection, not by shard arithmetic; the fresh pool is
drawn from C4 train shards disjoint from both prior pools and the intersection is still
computed and recorded. The test split is read exactly once, after C7 has frozen every label,
rank, carrier identity and threshold; a test-access counter starts at zero at S6 and the
code refuses a second read.

---

## 13. Numerical contract

Source section 17 in full, plus: float64 throughout; covariance symmetry asserted;
orthonormality of every reported carrier asserted; predictions invariant under within-carrier
orthogonal rotation; the summed per-lag operator equal to its direct block form; the
reduced-rank ceiling checked against a dense reference in a small synthetic aperture; both
eigensystems replayed; no validation or test row in any mean, covariance, ridge resolution or
carrier fit; document-weighted scoring.

Every substantive label cites the section that mandates it. A missing input yields an
explicit unavailable record naming the quantity, and an unavailable input poisons every
decision that consumes it. Any similarity diagnostic asserts first that the compared objects
differ.

---

## 14. Non-claims

R2.4 establishes a predictive representation and nothing more. It does not establish that
the recovered carrier is autonomous, composes, is causal, is mechanistically privileged, is
the unique state, or generalizes across layers, corpora, tasks or models. No causal,
autonomy or generalization language appears anywhere in its output.
