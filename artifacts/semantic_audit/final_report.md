# Slow-Subspace Semantic Audit R1 — Final Report

- S0: `PASS` / `ORIGINAL_FROZEN`
- S1: `SAE_BRIDGE_NOT_EVALUATED`
- S2 profile: `PROFILE_DURABLE_SEMANTIC`
- S3: `SEM2_COMPACT_STABLE_SEMANTIC_SUBSPACE`

## Sealed-test document-bootstrap results

| target | rank | statistic | point (nats) | 95% CI |
|---|---:|---|---:|---|
| register | 31 | slow_delta_h | 0.009233 | [-0.016490, 0.035111] |
| register | 31 | slow_minus_random_median | 0.007311 | [-0.014982, 0.030875] |
| register | 31 | slow_minus_pca | 0.003210 | [-0.009883, 0.016708] |
| topic | 31 | slow_delta_h | 0.033151 | [0.013353, 0.055950] |
| topic | 31 | slow_minus_random_median | 0.023766 | [0.012704, 0.036279] |
| topic | 31 | slow_minus_pca | 0.015859 | [0.004753, 0.027428] |

## Claim boundary

The frozen coarse-topic endpoint was statistically positive. This is moderate evidence for a semantic interpretation because topic may still be confounded with other durable document properties such as source or style.

No causal claim is licensed by this run. S4 was not queued.
