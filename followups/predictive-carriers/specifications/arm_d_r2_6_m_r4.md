# Arm D-R2.6-M-r4 — Qualification repair amendment

This prospective amendment inherits Arm D-R2.6-M-r3 in full. R3 completed all 1,050
qualification worlds at N=4,000, 8,000, and 12,000 and was finalized as
`R2_6_M_INSTRUMENT_NOT_QUALIFIED`; no fresh capture was started and the sealed test was
never opened. R4 makes only the two known-answer repairs below, then repeats W5 calibration
and the complete qualification matrix from new namespaced seeds.

## A3.1 — Candidate-level graded effect plant

R3 correctly normalized shared/private coordinate variance and solved the total private
gain exactly, but attached the W3 graded effect label to the full 64-coordinate tail. The
instrument's L1 statistic instead tests the prospectively selected candidate, which was
rank 16 in every m-star graded profile and formal world. At sharing 0.95, the R3 N=12,000
median tested increment was 0.005528 against m-star 0.005525, making the required 0.80
power impossible for a statistic centered on its own hard cutoff.

R4 keeps the same smooth 64-coordinate tail but solves coefficient scale so the first
rank-16 candidate has the requested population private gain. Total target energy still
includes the entire tail, so omitted tail energy is not hidden from the denominator.
Discrete W3 continues to target its planted rank-8 candidate. This changes only synthetic
known-answer construction; the real-data practical floor remains exactly m-star.

## A3.2 — Separate L1 ridge stability from L2 boundary persistence

R3 computed cumulative-union MSC across the frozen estimator-ridge neighborhood but did
not use it. Its L1 ridge gate instead required a spectral-gap ratio. That makes a smooth
graded component fail L1 for lacking a discrete boundary, although absence of such a
boundary is precisely what distinguishes L1 from L2 in the claim ladder.

R4 uses cumulative-union MSC at least 0.50 as the L1 ridge-robustness gate. The existing
gap-ratio requirement is retained unchanged and is required for L2, together with the W5
simultaneous envelope and independent boundary reproduction. Alpha, bootstraps, random
controls, practical floor, candidate ranks, sequence, W5 specificity gates, and every
real-data acceptance threshold are unchanged.

## Qualification and freshness

R4 uses new namespaced seeds and repeats W5 calibration and all 1,050 W0--W6 worlds at the
prespecified N=4,000/8,000/12,000 sequence, selecting the smallest passing N. Fresh capture,
protocol freeze, and sealed analysis remain forbidden until every qualification gate passes.
