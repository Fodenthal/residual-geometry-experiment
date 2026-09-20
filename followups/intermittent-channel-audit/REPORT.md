# Intermittent-Channel Audit Pilot Results

## Status and scope

Completed artifact-only pilot of the frozen C4 residual-geometry projections.
This is an estimator audit: it does not fit new directions, run the model again,
or make causal or semantic claims about the selected directions.

The corresponding locked protocol is [`PROTOCOL.md`](PROTOCOL.md). The large
frozen projection tensors and corpus data are intentionally omitted from this
public repository.

The pilot used all frozen projection chunks: 32 train, 4 validation, and 4 test.
It used 20 validation token-permutation replicates. The train profile array had
two memory-heavy shards retried with 96 GB; both completed, and the final
reducer, selector, and report completed successfully.

## Locked primary selection rule

For each probe, the primary statistic is norm-controlled Q95 coactivation lift
minus one, averaged over the inclusive long-range lag band 33--128. Candidates
were selected on validation only when all of the following held:

1. broad signed long-lag persistence was at most 0.05;
2. observed long-range norm-controlled activity was at least 0.02;
3. additive excess over the median token-permutation null was at least 0.02;
4. the empirical token-permutation percentile was at least 0.95.

The additive excess is

\[
\Delta_j=S^{\mathrm{obs}}_j-\operatorname{median}_b S^{\mathrm{perm}}_{j,b}.
\]

This supersedes the earlier fractional-collapse diagnostic, which was unstable
when its observed denominator was near zero. The local band 1--32 is reported
as a diagnostic only. Candidate directions were deduplicated at absolute cosine
similarity 0.95 and frozen before test characterization.

## Headline result

The pilot selected 43 of 1,024 probes. The frozen validation-selected
population reproduced on test.

| Primary statistic: long-range norm-controlled coactivation excess | Median |
|---|---:|
| All validation probes | -0.0050 |
| Validation candidates | 0.0270 |
| All test probes | -0.0012 |
| Frozen validation candidates on test | 0.0264 |

Forty-two of 43 frozen candidates had a positive test statistic, and 31 of 43
remained at or above the 0.02 observed-effect floor. Validation-to-test rank
correlation across candidates was 0.625. The candidate median therefore exceeds
the all-test median under a candidate set frozen on validation.

On validation, candidate probes had median token-permutation null -0.0221 and
median additive excess 0.0479. All 43 candidates had empirical percentile 1.0:
each exceeded all 20 of its validation token-permutation replicates.

## Temporal controls

For the selected probes, long-range norm-controlled null summaries were:

| Control | Median | Q95 |
|---|---:|---:|
| Token permutation | -0.0218 | -0.0149 |
| Matched random gate | -0.0190 | -0.0134 |
| Block permutation, 8 tokens | -0.0272 | -0.0190 |
| Block permutation, 32 tokens | -0.0303 | -0.0213 |

Every validation candidate exceeded its own 95th-percentile null under all four
controls. Token permutation defines selection; the matched-gate and block
controls are confirming diagnostics and remain validation-resident controls.

Local coactivation was much stronger than the primary long-range effect:

| Norm-controlled Q95 coactivation excess | Validation candidates | Frozen test candidates |
|---|---:|---:|
| Local band, lags 1--32 | 0.1596 | 0.1553 |
| Long-range band, lags 33--128 | 0.0270 | 0.0264 |

The 32-token block permutation retained local median 0.1088, while the
8-token block permutation reduced it to 0.0523. The strongest supported
interpretation is therefore temporally clustered activity with a smaller,
order-dependent component across 32-token blocks, not equally strong
persistence across all long lags.

## Activity, payload, and support

The primary effect appears only after the frozen global-amplitude proxy control:
raw long-range coactivation had candidate medians near zero (-0.0002 on
validation and -0.0020 on test), whereas the norm-controlled statistic was
positive. This makes amplitude normalization essential to the claim.

The effect is not supported as persistent signed payload:

- broad signed long-lag median: 0.0080 validation, 0.0084 test;
- norm-gated active-pair payload-correlation median: 0.0166 validation,
  0.0341 test (all-test median 0.0296);
- long-range envelope persistence was near zero and had positive bootstrap
  lower bounds for only 11/43 validation and 14/43 test candidates.

It is, however, well supported in active-pair count: median long-range
active-active pairs were 130,910 on validation and 132,441 on test. Natural
Q95 duty was about 5%; candidate bursts averaged 4.6 tokens with inter-burst
gaps around 84 tokens.

## Stability, families, and geometry

Document-resampling intervals for the primary statistic had positive lower
95% bounds for 37/43 candidates on validation and 37/43 on test. These
intervals quantify document sampling variation, not post-selection uncertainty;
the frozen test replication remains the primary safeguard.

Candidate composition was:

| Family | Selected / available | Selection rate |
|---|---:|---:|
| Random | 31 / 512 | 6.05% |
| Time-lagged | 8 / 256 | 3.13% |
| PCA | 4 / 256 | 1.56% |

The effect is not enriched among the existing time-lagged or PCA probe banks;
the selected set is mostly random directions. Its span is also largely distinct
from the earlier slow subspace: mean-squared containment 0.0245, effective
shared dimensions 0.76, and median principal angle 82.6 degrees.

## Claim boundary and handoff

Supported claim:

> A held-out-replicating subset of frozen residual directions has weak broad
> signed persistence but positive, normalized, order-sensitive activity
> coactivation at long range.

Not supported:

- stable signed payload recurrence;
- a semantic or causal memory channel;
- a dedicated intermittent subspace;
- enrichment in the pre-existing slow-direction families.

If pursued, the next experiment should be separately specified: learn or test
a gated/fourth-order estimator using this activity statistic, then evaluate it
with the same validation-freeze and held-out characterization discipline.
