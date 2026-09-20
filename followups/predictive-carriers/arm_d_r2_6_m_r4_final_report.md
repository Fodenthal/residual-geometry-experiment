# Arm D-R2.6-M-r4 final report

## Result

**GRADED_PREDICTIVE_DECOMPOSITION**

R2.6-M-r4 qualified prospectively and completed end to end. The fixed sequence accepted one
new above-floor private predictive component, Q3, after the inherited Q1/Q2 seed family. Q3 did
not have a reproducible spectral boundary above the realistic graded-spectrum envelope, so it is
not classified as a discrete predictive pocket. Q4 failed the matched-random L1 test and the
protocol stopped without testing Q5 or Q6.

The result is therefore three accepted predictive components at this estimator and resolution:
Q1 (rank 16), corrected Q2 (rank 32), and Q3 (rank 32), for cumulative rank 80. This is evidence
for additional structured predictive state beyond Q1/Q2, but against a discrete modular-pocket
interpretation at the frozen R2.6 resolution.

## Qualification repair

The repair did not tune a decision threshold to the fresh result. It corrected mismatches between
the synthetic qualification worlds and the estimator being qualified:

1. Conditionalization is symmetric and cross-fitted: the accepted union is regressed from both
   complement source scores and future targets before conditional-operator estimation.
2. W5 uses a simultaneous gap envelope: the q95 threshold is taken over the within-replicate
   maximum across all searched ranks and sequential steps.
3. The W3 graded plant places the effect in the rank-16 candidate actually tested, while the
   denominator retains the full rank-64 tail energy.
4. L1 ridge robustness tests cumulative-union recovery. Spectral-gap persistence remains a
   separate L2 requirement rather than being imposed on L1.

At 4,000 documents the repaired instrument still missed required graded-W3 power, so that sample
size was rejected. At 8,000 documents all 1,050 W0--W6 tasks passed. W0 and W6 family-wise error,
W4 false L1, and W5 false L2/L3 were all 0. W1 L1/L2 power and W2 weak power were all 1.0. Every
W3 cell passed at sharing fractions 0.5, 0.8, and 0.95, at both m* and 2m*: discrete L1/L2 power
was 1.0, graded L1 power was 1.0, and graded false-L2 rate was 0.

The selected resolution floor was m* = 0.0055246791. The simultaneous W5 spectral-gap q95 was
0.2213224824 and the temporal-profile shape q95 was 0.0015966624.

## Prospective controls and fresh capture

The protocol was frozen with `fresh_data_unread=true` before capture. It fixed N=8,000, candidate
ranks {4, 8, 16, 32}, four possible new components, a maximum cumulative rank of 128, 200 matched
random controls per step, 2,000 sealed bootstrap replicates, ridge values {0.03, 0.10, 0.30}, and
the Q1/Q2 hashes.

The fresh capture was healthy: 8,000 documents, 16,000 sequences, 16,000 invariant checks, zero
invariant failures, exact no-op difference 0.0, and aperture width 256. Content-hash audits found
zero overlap with R1, R2.2, R2.4, or R2.5 pools. The deterministic split contained 2,773 FitA,
2,858 FitB, 1,179 validation, and 1,190 sealed-test documents.

The capture-health record retains the inherited `arm_d_r2_2` capture-harness label; the governing
prospective protocol and analysis artifacts are independently frozen as `arm_d_r2_6_m_r4`.

## Frozen discovery sequence

The complete sequence was frozen with zero test rows used:

| Candidate | Rank | Cumulative rank | Selected gap | W5 q95 | Split-stable | Ridge-stable | Boundary reproduces |
|---|---:|---:|---:|---:|---|---|---|
| Q3 | 32 | 80 | 0.01927130 | 0.22132248 | yes | yes | no |
| Q4 | 32 | 112 | 0.00528345 | 0.22132248 | yes | yes | no |
| Q5 | 8 | 120 | 0.00543846 | 0.22132248 | yes | yes | no |
| Q6 | 8 | 128 | 0.00822760 | 0.22132248 | yes | yes | no |

All four cumulative paths beat their random-subspace MSC q95 and reached 85.6--87.7% of the
bootstrap-attainable median stability. No frozen candidate had an L2-qualifying spectral boundary.

## Sealed fixed-sequence decisions

The sealed test was opened exactly once.

| Candidate | Delta MR | Random q95 | Excess T | Bootstrap LCB95 | L1 | L2 | Decision |
|---|---:|---:|---:|---:|---|---|---|
| Q3 | 0.01447505 | 0.01129512 | 0.00317993 | 0.00232144 | pass | fail | structured predictive component without discrete boundary |
| Q4 | 0.00681437 | 0.00719627 | -0.00038190 | -0.00089944 | fail | fail | private component rejected |
| Q5 | -- | -- | -- | -- | not tested | not tested | stopped after Q4 L1 failure |
| Q6 | -- | -- | -- | -- | not tested | not tested | stopped after Q4 L1 failure |

Q3 clears the practical floor, the matched-random floor, and the positive one-sided bootstrap
criterion. Q4's raw increment is slightly above m*, but it is smaller than the matched-random q95;
its excess and confidence bound are negative. The fixed stopping rule therefore prevents any claim
about Q5 or Q6 on sealed data.

## Predictive accounting

| Accepted family | Cumulative rank | Gain | Fraction of PCA64 | Fraction of full aperture |
|---|---:|---:|---:|---:|
| Q1 | 16 | 0.21393784 | 0.834920 | 0.776955 |
| Q1+Q2 | 48 | 0.24175158 | 0.943467 | 0.877966 |
| Q1+Q2+Q3 | 80 | 0.25593875 | 0.998834 | 0.929489 |

The full-aperture gain was 0.27535425 and PCA64 gain was 0.25623746. Q3 nearly closes the seed
family's gap to PCA64 and raises full-aperture coverage from 87.80% to 92.95%. The seed-to-full
budget was 0.03360266, giving only an arithmetic upper bound of six further m*-sized increments;
that bound is not evidence that six further components exist.

Exact leave-one-out unique gains were 0.016264 for Q1, 0.017342 for Q2, and 0.014187 for Q3.
Shapley allocations were 0.094561, 0.094990, and 0.066387, respectively.

## Robustness and geometry

Q3's cumulative split-half MSC was 0.7151 versus random q95 0.6652 and 87.7% of the attainable
bootstrap median. Its cumulative union had MSC 0.9700, 1.0000, and 0.9647 against the primary path
at ridge rho 0.03, 0.10, and 0.30. Leave-one-out refitting recovered Q3 with MSC 1.0.

The covariance-whitened deflation sensitivity analysis also found exactly one above-m* new
component and no W5-exceeding boundary. The mapped whitened Q3 had MSC 0.7476 to the Euclidean Q3,
which supports path-level agreement without claiming the coordinate representations are identical.

Q3 accounts for 11.72% of source variance and has only 27.47% containment in PCA64, compared with
46.10% for Q1 and 48.12% for Q2. This suggests Q3 recovers predictive directions that are less
aligned with the dominant-variance subspace.

The components remain representationally related. Cross-decoding R2 values involving Q3 range
from 0.402 to 0.542, its mean canonical correlation is 0.627 with Q1 and 0.471 with Q2, and its
prediction-space CKA is 0.896 with Q1 and 0.857 with Q2. These relationships argue against reading
orthogonal carrier coordinates as semantic independence.

Temporal profile shapes differ beyond W5 for all pairs and reproduce across fit halves. L3 is
nevertheless false by contract because L3 specialization requires at least one L2-qualified
discrete pocket, and no component passed L2.

## Interpretation

At the frozen estimator, sample size, rank search, and resolution floor, the data support one
additional stable predictive direction family beyond Q1/Q2. They do not support a clean spectral
boundary separating that family from a graded tail. The strongest warranted conclusion is thus
"structured graded predictive state," not "a third autonomous module."

The experiment is predictive and representational. It does not establish semantic content,
causal use, autonomous dynamics, a unique basis, or a resolution-independent natural module count.

## Provenance and verification

- Protocol revision: `arm_d_r2_6_m_r4`
- Corrected Q2 SHA-256: `d2aa4443c1529360af57f0528c4afd0b58f31e891f1d71f01a84facd78d72f59`
- Q3 SHA-256: `524d24086b4c7dca0c4c42265d9762ae20fc4ca08a62c133dd2c7f8784402ae9`
- Qualification SHA-256: `87b384e81ae4f268c8d0e2400f01b644d467c2a87da8cc35b47d6542a04078b6`
- Frozen candidate bundle SHA-256: `049d5aa6ed7be30cb5f345f9d0a189910776fdf2be5b9d949adfeec11a890b28`
- Artifact-tree status: `R2_6_M_ARTIFACT_TREE_VERIFIED`
- Manifested files: 3,393; every SHA-256 entry verified
- Sealed-test access count: 1
- Large Beehive capture artifacts are intentionally omitted from this public
  snapshot; the compact verification records and hashes remain in `results/`.
