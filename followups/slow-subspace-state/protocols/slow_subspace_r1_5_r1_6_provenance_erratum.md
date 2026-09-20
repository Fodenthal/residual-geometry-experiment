# Provenance Erratum — Slow-Subspace R1.5 and R1.6

## Status

This erratum was written after R1.6 and before the R1.6A canonical-basis audit.
R1.5/R1.6 artifacts are preserved, but conclusions about the original signed TICA
slow subspace are suspended.

## Confirmed mismatch

The original residual-geometry experiment fit signed time-lagged directions by the
regularized generalized eigenproblem

\[
\Sigma_{\mathrm{lag}}v=\lambda(\Sigma_0+\epsilon I)v.
\]

Its canonical rank-31 artifact is:

```text
inputs/bases/residual_first_kstar.npz
sha256 = 966736e0346dd354559f1354efa86f71c4e6c01cbb929ff70b18f7a9420cce6c
```

It contains 30 `time_lagged` source probes and one `pca` source probe.

R1.5 and the five fresh R1.6 fits instead used a later nonconvex optimizer of
unsigned/squared-projection autocorrelation with eight random starts and 300
iterations. That estimator is a separate scientific object.

R1.6 also used the following artifact as its descriptive “historical bank”:

```text
outputs/unsigned_state_lifetime_geometry/probes/unsigned_time_lagged_residual.npz
sha256 = c3841c7de43dfb49b7962f706753e6aad1e2f6a00df0b40a67e385ce32e8418a
```

This is not the canonical original Q31 basis. Finally, R1.6 evaluated squared
projection activity through `score_unsigned_projections`; the original fat-subspace
battery evaluated signed scalar residual projections.

## Consequence

- `RESTART_LOW` in R1.5 describes identifiability of the replacement nonconvex
  unsigned optimizer, not identifiability of the original generalized eigenspace.
- `FULL_DATA_FAT_COLLAPSED` in R1.6 describes random directions inside spans selected
  by that replacement estimator under squared scoring. It does not refute the
  original signed TICA Q31 result.
- The earlier interpretation that the reported approximately-25 median probably
  referred to Q13 is withdrawn. The canonical original report records rank-31
  validation/test medians of 24/25.

No R1.5/R1.6 files are deleted or rewritten. They remain valid records of a different
estimator and observable, with their scientific scope corrected by this erratum.

## Required next action

Before new science, evaluate the exact canonical hash above using the original
512-direction sampler (`seed=31`) and signed within-document timescale estimator on
the frozen validation and test documents. Do not infer anything about a new TICA fit
until this provenance gate is resolved.
