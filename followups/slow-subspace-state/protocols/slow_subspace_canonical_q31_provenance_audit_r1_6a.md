# Slow-Subspace R1.6A — Canonical Q31 Provenance Audit

## Status

Prospectively frozen after the R1.5/R1.6 estimator mismatch was identified and before
evaluating the canonical basis with the current runtime.

## Goal

Determine whether the current model/runtime and signed timescale evaluator reproduce
the original canonical Q31 fat-subspace result. This is a provenance audit, not a new
representation-learning experiment.

## Frozen inputs

- Exact basis artifact hash:
  `966736e0346dd354559f1354efa86f71c4e6c01cbb929ff70b18f7a9420cce6c`.
- Basis rank: 31; source composition: 30 time-lagged probes plus one PCA probe.
- Original 500 validation and 500 test documents and ordering.
- Gemma-2-2B layer-12 `hook_resid_post`, context length 1,024.
- Original random-in-span sampler: 512 row-wise Gaussian coefficient vectors,
  normalized to unit length, NumPy `default_rng(seed=31)`.
- Signed scalar projection observable, not absolute or squared activity.
- Within-document autocorrelation, maximum lag 512, minimum 50 valid documents,
  minimum valid-lag fraction 0.8, smoothing width 5, threshold 1/e.

## Procedure

1. Verify the basis hash, rank, orthonormality, source IDs, and source families.
2. Generate the exact 512 residual-space directions with the original sampler.
3. During each model forward pass, project the layer-12 residual directly onto those
   residual-space directions on GPU. Do not reconstruct through the newer PCA-512
   capture: the canonical basis has only 0.9294 containment in that newer PCA span.
4. Transfer only the 512 scalar projections and accumulate the original signed
   within-document autocorrelation estimator.
5. Persist direction-level timescales, mean profiles, and small fixed trace samples
   for later numeric comparison if required.

## Frozen gate

The canonical report gives validation/test median timescales 24/25.

- `CANONICAL_Q31_REPRODUCED` if both current medians are within three tokens of their
  respective canonical values.
- `CANONICAL_Q31_EVALUATOR_MISMATCH` otherwise.

If reproduced, the R1.5/R1.6 discrepancy is explained by estimator/observable and
artifact substitution, and a faithful generalized-eigenproblem rerun may be specified
prospectively. If mismatched, compare the saved fixed trace samples against original
saved projections before fitting anything new.

No TICA refit is authorized by R1.6A itself.
