# Arm D: predictive carriers of remote-history state

This is the completed predictive-carrier line built on Gemma-2-2B layer-12
residual activations. It asks whether the difference between two histories can
be predicted from a compact subspace, whether that subspace is unique, and
whether additional predictive structure remains after the first carrier.

## Headline results

- R2.4: an energy-metric carrier at rank 16 recovers 80.29% of the full
  aperture's predictive gain and beats the best rank-matched random subspace by
  1.527× on held-out documents.
- The advantage decays with rank: 1.527× at rank 16, 1.322× at rank 32, and
  1.223× at rank 64. The carrier is therefore compact but not a unique channel.
- R2.6-M-r4 prospectively accepts one additional stable predictive component,
  Q3, giving cumulative rank 80 and 92.95% of full-aperture gain.
- No clean spectral boundary or autonomous module count is supported. The
  strongest interpretation is structured graded predictive state.

## Layout

```text
arm_d_r2_6_m_r4_final_report.md  completed report
specifications/                  frozen R2.4/R2.6 protocols
results/                         compact verification and decision records
code/src/                        carrier, scoring, qualification, and report code
```

Large model captures, activation arrays, cluster bundles, and external artifact
URLs are omitted. The compact records preserve the frozen decisions, sealed-test
read count, carrier metrics, and verification hashes.

## Claim boundary

This line establishes predictive representation and redundancy at the tested
estimator, rank grid, model, layer, and corpus. It does not establish semantic
content, causal use, autonomous dynamics, a unique basis, or a universal module
count.
