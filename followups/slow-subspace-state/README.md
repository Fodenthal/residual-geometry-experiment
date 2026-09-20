# Slow-Subspace State Follow-ups

This directory contains the completed post-publication experiments that test
whether the original slow residual subspace is reproducible, what information
it carries, and whether earlier semantic content leaves a persistent trace in
that geometry.

The short version is:

> Gemma-2-2B layer 12 contains a reproducible, graded persistent-state region.
> A compact 31-dimensional slice captures most of the lifetime excess, while a
> lower-density slow shoulder extends to roughly rank 64. Earlier topic content
> produces a donor-directed displacement in a frozen semantic coordinate that
> remains detectable after 64–256 identical continuation tokens.

The full evidence and claim boundaries are in [REPORT.md](REPORT.md).

## Experiment index

| Stage | Question | Result |
|---|---|---|
| R1.3–R1.6 | Are later nonlinear/unsigned fits stable? | No; these runs exposed an estimator-provenance mismatch and are retained as an audit trail. |
| R1.6a | What was the historical object? | The canonical object is the original signed TICA-like `Q31`, not the later unsigned optimizer. |
| R1.7 | Do independent document halves recover it? | Yes: `S31 = 0.8641`, with held-out median timescales 24.5 and 25.0. |
| R1.8 | Is it document-persistent and semantically reproducible? | Yes: slow-space between-document variance is substantially enriched and independent fits recover similar topic geometry. |
| R1.9 | Does topic survive measured source/style controls? | Yes: 86.7–87.4% of the topic effect remains after conditioning on the measured fingerprint. |
| R1.10 | Is 31 the exact dimension? | No: `Q31` is a compact reference object inside a graded shoulder extending to roughly rank 64. |
| R1.11 | Does earlier content leave controlled persistent state? | Yes: the preregistered donor-directed effect at 64 identical continuation tokens is 0.9256, 95% CI [0.6764, 1.1642]. |

## Layout

```text
REPORT.md       consolidated scientific summary and limitations
protocols/      frozen protocols plus the estimator-provenance erratum
results/        compact reports and decision records; no large activations
code/scripts/   numbered experiment entry points
code/src/       reusable implementation used by the numbered stages
code/tests/     focused unit tests for R1.7–R1.11
```

## Running the focused tests

The code snapshot preserves the original research-workspace import layout.
From this directory:

```bash
cd code
PYTHONPATH=. pytest -q tests
```

The artifact-only stages require the frozen inputs named in their protocols.
GPU capture stages additionally require the pinned Gemma-2-2B model revision
and document manifests. Generated residual tensors, model weights, C4 text,
and cluster-local caches are intentionally not redistributed here.

## Claim boundary

R1.11 establishes that changing earlier content changes a later semantic
representation under an identical continuation. It does **not** establish that
the model causally uses that coordinate for downstream computation or behavior.
The proposed causal intervention experiment is therefore not included as a
completed result.

