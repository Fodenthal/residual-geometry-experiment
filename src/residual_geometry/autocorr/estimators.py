from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


VAR_EPS = 1e-12
E_INV = 1.0 / math.e


@dataclass(frozen=True)
class AutocorrResult:
    profiles: np.ndarray
    valid_doc_counts: np.ndarray


class AutocorrAccumulator:
    def __init__(self, n_features: int, max_lag: int) -> None:
        self.n_features = n_features
        self.max_lag = max_lag
        self.sum_corr = np.zeros((n_features, max_lag + 1), dtype=np.float64)
        self.valid_doc_counts = np.zeros((n_features, max_lag + 1), dtype=np.int64)

    def update(self, result: AutocorrResult) -> None:
        k = result.profiles.shape[1] - 1
        valid = result.valid_doc_counts[:, : k + 1] > 0
        self.sum_corr[:, : k + 1] += np.where(valid, result.profiles[:, : k + 1], 0.0) * result.valid_doc_counts[:, : k + 1]
        self.valid_doc_counts[:, : k + 1] += result.valid_doc_counts[:, : k + 1]

    def finalize(self) -> AutocorrResult:
        profiles = np.full_like(self.sum_corr, np.nan, dtype=np.float64)
        valid = self.valid_doc_counts > 0
        profiles[valid] = self.sum_corr[valid] / self.valid_doc_counts[valid]
        profiles[:, 0] = 1.0
        return AutocorrResult(profiles=profiles, valid_doc_counts=self.valid_doc_counts.copy())


@dataclass
class QuantileReservoir:
    n_features: int
    capacity: int = 20_000
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.samples = np.zeros((self.n_features, self.capacity), dtype=np.float32)
        self.sample_counts = np.zeros(self.n_features, dtype=np.int64)
        self.positive_counts = np.zeros(self.n_features, dtype=np.int64)

    def update(self, activations: np.ndarray) -> None:
        flat = activations.reshape(-1, activations.shape[-1])
        for feature_i in range(self.n_features):
            positive = flat[:, feature_i][flat[:, feature_i] > 0]
            if not len(positive):
                continue
            old_seen = int(self.positive_counts[feature_i])
            self.positive_counts[feature_i] += len(positive)
            current_count = int(min(self.sample_counts[feature_i], self.capacity))
            for offset, value in enumerate(positive):
                seen_index = old_seen + offset
                if current_count < self.capacity:
                    self.samples[feature_i, current_count] = value
                    current_count += 1
                else:
                    replace_at = int(self.rng.integers(0, seen_index + 1))
                    if replace_at < self.capacity:
                        self.samples[feature_i, replace_at] = value
            self.sample_counts[feature_i] = current_count

    def table(self, quantiles: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99)) -> pd.DataFrame:
        rows = []
        for feature_i in range(self.n_features):
            n = int(self.sample_counts[feature_i])
            positive = self.samples[feature_i, :n]
            row = {
                "selected_feature_position": feature_i,
                "positive_count": int(self.positive_counts[feature_i]),
                "reservoir_count": n,
                "quantile_approximation": bool(self.positive_counts[feature_i] > self.capacity),
            }
            for q in quantiles:
                row[f"q{int(q * 100):02d}"] = float(np.quantile(positive, q)) if n else 0.0
            row["threshold_fallback"] = bool(self.positive_counts[feature_i] < 1000)
            rows.append(row)
        return pd.DataFrame(rows)


def _lag_corr(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=1)
    y_mean = y.mean(axis=1)
    xc = x - x_mean[:, None]
    yc = y - y_mean[:, None]
    x_var = np.mean(xc * xc, axis=1)
    y_var = np.mean(yc * yc, axis=1)
    valid = (x_var >= VAR_EPS) & (y_var >= VAR_EPS)
    corr = np.full(x.shape[0], np.nan, dtype=np.float64)
    denom = np.sqrt(x_var[valid] * y_var[valid])
    corr[valid] = np.mean(xc[valid] * yc[valid], axis=1) / denom
    return corr, valid


def _autocorr_feature_batch(
    arr_b: np.ndarray,
    k_max: int,
    active_b: np.ndarray | None,
    profiles_b: np.ndarray,
    valid_counts_b: np.ndarray,
) -> None:
    """Fill profiles_b and valid_counts_b in-place for one contiguous feature batch."""
    docs, positions, _ = arr_b.shape
    for lag in range(1, k_max + 1):
        x = arr_b[:, : positions - lag, :]
        y = arr_b[:, lag:, :]
        xc = x - x.mean(axis=1, keepdims=True)
        yc = y - y.mean(axis=1, keepdims=True)
        x_var = np.mean(xc * xc, axis=1)
        y_var = np.mean(yc * yc, axis=1)
        cov   = np.mean(xc * yc, axis=1)
        valid = (x_var >= VAR_EPS) & (y_var >= VAR_EPS)
        if active_b is not None:
            valid = valid & active_b
        denom = np.where(valid, np.sqrt(x_var * y_var), 1.0)
        corr  = np.where(valid, cov / denom, np.nan)
        valid_per = valid.sum(axis=0)
        corr_sum  = np.where(valid, corr, 0.0).sum(axis=0)
        has_valid = valid_per > 0
        profiles_b[:, lag] = np.where(has_valid, corr_sum / np.where(has_valid, valid_per, 1), np.nan)
        valid_counts_b[:, lag] = valid_per


def _autocorr_fft_exact_batch(
    arr_b: np.ndarray,
    k_max: int,
    active_b: np.ndarray | None,
    profiles_b: np.ndarray,
    valid_counts_b: np.ndarray,
    n_fft: int,
) -> None:
    """FFT + prefix-sum autocorrelation. Exact numerical match with _autocorr_feature_batch.

    Strategy: FFT gives all cross-product sums Sxy[k] at once; prefix sums give window
    means and window variances for each lag in O(1) per lag. No Python loop over positions.
    Works for any estimator (raw, within, binary) since centering is handled externally.
    """
    docs, T, feats = arr_b.shape
    K = k_max
    k_int = np.arange(1, K + 1, dtype=int)              # (K,)
    pair_counts = (T - k_int).astype(np.float64)         # (K,)
    pc = pair_counts[np.newaxis, :, np.newaxis]           # (1, K, 1) for broadcasting

    # Prefix sums: cs_full[:, t, :] = sum arr_b[0:t]
    cs_full = np.empty((docs, T + 1, feats), dtype=np.float64)
    cs_full[:, 0, :] = 0.0
    np.cumsum(arr_b, axis=1, out=cs_full[:, 1:, :])
    cs2_full = np.empty((docs, T + 1, feats), dtype=np.float64)
    cs2_full[:, 0, :] = 0.0
    np.cumsum(arr_b * arr_b, axis=1, out=cs2_full[:, 1:, :])

    S_total = cs_full[:, T, :]    # (docs, feats): sum of full sequence
    S2_total = cs2_full[:, T, :]  # (docs, feats): sum of squares

    # FFT cross-products: irfft(|FFT(x)|^2)[k] = (1/n_fft) * sum_{t=0}^{T-k-1} x[t]*x[t+k]
    X = np.fft.rfft(arr_b, n=n_fft, axis=1)
    power = X.real ** 2 + X.imag ** 2
    del X
    acf_fft = np.fft.irfft(power, n=n_fft, axis=1)  # (docs, n_fft, feats)
    del power

    # Vectorized over all K lags simultaneously
    idx_end = T - k_int                                  # [T-1, T-2, ..., T-K]
    S1x = cs_full[:, idx_end, :]                         # (docs, K, feats)
    S1y = S_total[:, np.newaxis, :] - cs_full[:, k_int, :]  # (docs, K, feats)
    S2x = cs2_full[:, idx_end, :]                        # (docs, K, feats)
    S2y = S2_total[:, np.newaxis, :] - cs2_full[:, k_int, :]  # (docs, K, feats)
    # irfft(|rfft(x,n)|^2,n)[k] = Σ_{t=0}^{T-k-1} x[t]*x[t+k]  (the raw sum, not 1/n_fft)
    Sxy = acf_fft[:, k_int, :]                           # (docs, K, feats)
    del acf_fft, cs_full, cs2_full

    mu_x = S1x / pc                                      # (docs, K, feats)
    mu_y = S1y / pc
    del S1x, S1y
    var_x = S2x / pc - mu_x ** 2
    var_y = S2y / pc - mu_y ** 2
    del S2x, S2y
    cov_xy = Sxy / pc - mu_x * mu_y
    del Sxy, mu_x, mu_y

    valid = (var_x >= VAR_EPS) & (var_y >= VAR_EPS)     # (docs, K, feats)
    if active_b is not None:
        valid = valid & active_b[:, np.newaxis, :]       # broadcast (docs, 1, feats) → (docs, K, feats)
    denom = np.where(valid, np.sqrt(np.maximum(var_x * var_y, 0.0)), 1.0)
    corr = np.where(valid, cov_xy / denom, np.nan)       # (docs, K, feats)
    del var_x, var_y, cov_xy, denom

    # Average over valid docs; valid_per is (K, feats), same for all lags given fixed valid_mask
    valid_per = valid.sum(axis=0)                         # (K, feats)
    corr_sum = np.where(valid, corr, 0.0).sum(axis=0)    # (K, feats)
    has_valid = valid_per > 0                             # (K, feats)
    safe_vp = np.where(has_valid, valid_per, 1).astype(np.float64)

    # profiles_b: (feats, K+1); corr_sum.T: (feats, K)
    profiles_b[:, 1:] = np.where(has_valid.T, corr_sum.T / safe_vp.T, np.nan)
    valid_counts_b[:, 1:] = valid_per.T


def compute_document_autocorr(
    activations: np.ndarray,
    max_lag: int,
    estimator: str,
    thresholds: np.ndarray | None = None,
    active_min_firings: int = 5,
) -> AutocorrResult:
    """Compute feature autocorrelation with equal document weighting.

    activations has shape (documents, positions, features).
    """
    if activations.ndim != 3:
        raise ValueError("activations must have shape (documents, positions, features)")
    docs, positions, features = activations.shape
    k_max = min(max_lag, positions - 1)
    arr = activations.astype(np.float64, copy=False)

    active_doc_mask = None
    if estimator == "active_doc":
        active_doc_mask = (activations > 0).sum(axis=1) >= active_min_firings  # (docs, features)
        arr = arr - arr.mean(axis=1, keepdims=True)
    elif estimator == "within":
        arr = arr - arr.mean(axis=1, keepdims=True)
    elif estimator == "binary":
        if thresholds is None:
            raise ValueError("binary autocorr requires thresholds")
        arr = (arr > thresholds.reshape(1, 1, features)).astype(np.float64)
    elif estimator != "raw":
        raise ValueError(f"Unknown estimator: {estimator}")

    profiles = np.full((features, k_max + 1), np.nan, dtype=np.float64)
    valid_counts = np.zeros((features, k_max + 1), dtype=np.int64)
    profiles[:, 0] = 1.0
    valid_counts[:, 0] = (
        active_doc_mask.sum(axis=0).astype(np.int64) if active_doc_mask is not None else docs
    )

    # Use FFT+prefix-sum kernel when k_max is large enough to benefit.
    # Falls back to lag loop for tiny sequences where overhead isn't worth it.
    use_fft = k_max >= 4 and positions >= 16
    n_fft = 0
    if use_fft:
        n_fft = 1
        while n_fft < 2 * positions:
            n_fft <<= 1

    # Process features in cache-friendly batches (~20 MB working set per batch).
    bytes_per_feature = docs * positions * 8
    feat_batch = max(1, (20 * 1024 * 1024) // bytes_per_feature)
    for fs in range(0, features, feat_batch):
        fe = min(fs + feat_batch, features)
        arr_b = np.ascontiguousarray(arr[:, :, fs:fe])
        active_b = active_doc_mask[:, fs:fe] if active_doc_mask is not None else None
        if use_fft:
            _autocorr_fft_exact_batch(arr_b, k_max, active_b, profiles[fs:fe], valid_counts[fs:fe], n_fft)
        else:
            _autocorr_feature_batch(arr_b, k_max, active_b, profiles[fs:fe], valid_counts[fs:fe])

    return AutocorrResult(profiles=profiles, valid_doc_counts=valid_counts)


def compute_document_autocorr_matrix(
    activations: np.ndarray,
    max_lag: int,
    estimator: str,
    thresholds: np.ndarray | None = None,
    active_min_firings: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-document correlations with shape (docs, features, lags)."""
    docs, positions, features = activations.shape
    k_max = min(max_lag, positions - 1)
    arr = activations.astype(np.float64, copy=False)

    active_doc_mask = None
    if estimator == "active_doc":
        active_doc_mask = (activations > 0).sum(axis=1) >= active_min_firings
        arr = arr - arr.mean(axis=1, keepdims=True)
    elif estimator == "within":
        arr = arr - arr.mean(axis=1, keepdims=True)
    elif estimator == "binary":
        if thresholds is None:
            raise ValueError("binary autocorr requires thresholds")
        arr = (arr > thresholds.reshape(1, 1, features)).astype(np.float64)
    elif estimator != "raw":
        raise ValueError(f"Unknown estimator: {estimator}")

    corr_matrix = np.full((docs, features, k_max + 1), np.nan, dtype=np.float32)
    valid_matrix = np.zeros((docs, features, k_max + 1), dtype=bool)
    corr_matrix[:, :, 0] = 1.0
    valid_matrix[:, :, 0] = True

    use_fft = k_max >= 4 and positions >= 16
    n_fft = 0
    if use_fft:
        n_fft = 1
        while n_fft < 2 * positions:
            n_fft <<= 1

    # For the matrix version docs can be large (all val docs concatenated for bootstrap).
    # Use a K-loop over lags but precompute FFT cross-products and prefix sums once per
    # feature batch, so each lag step is O(docs × feats_b) arithmetic — no position loop.
    bytes_per_feature = docs * positions * 8
    feat_batch = max(1, (20 * 1024 * 1024) // bytes_per_feature)
    for fs in range(0, features, feat_batch):
        fe = min(fs + feat_batch, features)
        arr_b = np.ascontiguousarray(arr[:, :, fs:fe])
        active_b = active_doc_mask[:, fs:fe] if active_doc_mask is not None else None
        if use_fft:
            T = positions
            # Prefix sums for window means and variances
            cs = np.empty((docs, T + 1, fe - fs), dtype=np.float64)
            cs[:, 0, :] = 0.0
            np.cumsum(arr_b, axis=1, out=cs[:, 1:, :])
            cs2 = np.empty((docs, T + 1, fe - fs), dtype=np.float64)
            cs2[:, 0, :] = 0.0
            np.cumsum(arr_b * arr_b, axis=1, out=cs2[:, 1:, :])
            S_tot = cs[:, T, :]    # (docs, feats_b)
            S2_tot = cs2[:, T, :]  # (docs, feats_b)
            # FFT cross-products: acf_fft[:, k, :] = (1/n_fft)*sum_t x[t]*x[t+k]
            X = np.fft.rfft(arr_b, n=n_fft, axis=1)
            power = X.real ** 2 + X.imag ** 2
            del X
            acf_fft = np.fft.irfft(power, n=n_fft, axis=1)  # (docs, n_fft, feats_b)
            del power
            for k in range(1, k_max + 1):
                T_k = T - k
                Sxy_k = acf_fft[:, k, :]                     # (docs, feats_b) — raw sum
                mu_x = (cs[:, T_k, :]) / T_k
                mu_y = (S_tot - cs[:, k, :]) / T_k
                var_x = cs2[:, T_k, :] / T_k - mu_x ** 2
                var_y = (S2_tot - cs2[:, k, :]) / T_k - mu_y ** 2
                cov = Sxy_k / T_k - mu_x * mu_y
                valid = (var_x >= VAR_EPS) & (var_y >= VAR_EPS)
                if active_b is not None:
                    valid = valid & active_b
                denom = np.where(valid, np.sqrt(np.maximum(var_x * var_y, 0.0)), 1.0)
                corr_matrix[:, fs:fe, k] = np.where(valid, cov / denom, np.nan).astype(np.float32)
                valid_matrix[:, fs:fe, k] = valid
            del acf_fft, cs, cs2
        else:
            for lag in range(1, k_max + 1):
                x = arr_b[:, : positions - lag, :]
                y = arr_b[:, lag:, :]
                xc = x - x.mean(axis=1, keepdims=True)
                yc = y - y.mean(axis=1, keepdims=True)
                x_var = np.mean(xc * xc, axis=1)
                y_var = np.mean(yc * yc, axis=1)
                cov   = np.mean(xc * yc, axis=1)
                valid = (x_var >= VAR_EPS) & (y_var >= VAR_EPS)
                if active_b is not None:
                    valid = valid & active_b
                denom = np.where(valid, np.sqrt(x_var * y_var), 1.0)
                corr  = np.where(valid, cov / denom, np.nan).astype(np.float32)
                corr_matrix[:, fs:fe, lag] = corr
                valid_matrix[:, fs:fe, lag] = valid

    return corr_matrix, valid_matrix


def smooth_profile(profile: np.ndarray, width: int = 5) -> np.ndarray:
    out = profile.astype(np.float64, copy=True)
    if len(out) <= 2 or width <= 1:
        return np.clip(out, -1.0, 1.0)
    radius = width // 2
    for idx in range(1, len(profile)):
        lo = max(1, idx - radius)
        hi = min(len(profile), idx + radius + 1)
        window = profile[lo:hi]
        out[idx] = np.nan if np.all(np.isnan(window)) else np.nanmean(window)
    out[0] = 1.0
    return np.clip(out, -1.0, 1.0)


def extract_tau(
    profile: np.ndarray,
    max_lag: int,
    valid_counts: np.ndarray,
    min_valid_docs: int,
    min_valid_lag_fraction: float,
    smoothing_width: int = 5,
) -> dict:
    available = min(max_lag, len(profile) - 1)
    valid_lags = valid_counts[1 : available + 1] >= min_valid_docs
    tau_valid = bool(valid_lags.mean() >= min_valid_lag_fraction) if available else False
    smoothed = smooth_profile(profile[: available + 1], width=smoothing_width)
    crossing = np.where((np.arange(len(smoothed)) >= 1) & (smoothed < E_INV))[0]
    if len(crossing):
        tau = int(crossing[0])
        right_censored = False
    else:
        tau = int(available + 1)
        right_censored = True
    return {
        "tau": tau,
        "right_censored": right_censored,
        "tau_valid": tau_valid,
        "valid_lag_fraction": float(valid_lags.mean()) if available else 0.0,
    }


def build_timescale_table(
    feature_indices: np.ndarray,
    estimator_results: dict[str, AutocorrResult],
    max_lag: int,
    min_valid_docs: int,
    min_valid_lag_fraction: float,
    smoothing_width: int = 5,
) -> pd.DataFrame:
    rows = []
    stat_columns = [
        f"{key}_{estimator}"
        for estimator in estimator_results
        for key in ["tau", "right_censored", "tau_valid", "valid_lag_fraction"]
    ]
    for selected_i, feature_index in enumerate(feature_indices):
        row: dict[str, object] = {"feature_index": int(feature_index)}
        for estimator, result in estimator_results.items():
            stats = extract_tau(
                result.profiles[selected_i],
                max_lag=max_lag,
                valid_counts=result.valid_doc_counts[selected_i],
                min_valid_docs=min_valid_docs,
                min_valid_lag_fraction=min_valid_lag_fraction,
                smoothing_width=smoothing_width,
            )
            for key, value in stats.items():
                row[f"{key}_{estimator}"] = value
        rows.append(row)
    table = pd.DataFrame(rows, columns=["feature_index", *stat_columns])
    for estimator in ["within", "binary", "raw"]:
        tau_col = f"tau_{estimator}"
        valid_col = f"tau_valid_{estimator}"
        rank_col = f"rankpct_{estimator}"
        if tau_col in table:
            valid = table[valid_col] if valid_col in table else True
            table[rank_col] = table.loc[valid, tau_col].rank(pct=True)
    slow_score = np.zeros(len(table), dtype=np.float64)
    if "rankpct_within" in table:
        slow_score += table["rankpct_within"].fillna(0).to_numpy(dtype=np.float64)
    if "rankpct_binary" in table:
        slow_score += 0.5 * table["rankpct_binary"].fillna(0).to_numpy(dtype=np.float64)
    if "rankpct_raw" in table:
        slow_score += 0.25 * table["rankpct_raw"].fillna(0).to_numpy(dtype=np.float64)
    table["slow_score"] = slow_score
    return table


def positive_activation_quantiles(
    activations: np.ndarray,
    quantiles: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99),
) -> pd.DataFrame:
    features = activations.shape[-1]
    flat = activations.reshape(-1, features)
    rows = []
    for feature_i in range(features):
        positive = flat[:, feature_i][flat[:, feature_i] > 0]
        row = {"selected_feature_position": feature_i, "positive_count": int(len(positive))}
        for q in quantiles:
            row[f"q{int(q * 100):02d}"] = float(np.quantile(positive, q)) if len(positive) else 0.0
        row["threshold_fallback"] = bool(len(positive) < 1000)
        rows.append(row)
    return pd.DataFrame(rows)


def event_triggered_persistence(
    activations: np.ndarray,
    feature_indices: np.ndarray,
    quantile_table: pd.DataFrame,
    offsets: list[int],
) -> pd.DataFrame:
    docs, positions, features = activations.shape
    rows = []
    for feature_i in range(features):
        qrow = quantile_table.iloc[feature_i]
        threshold_name = "q99" if int(qrow["positive_count"]) >= 500 else "q95"
        threshold = float(qrow[threshold_name])
        q50 = float(qrow["q50"])
        scale = threshold - q50 + 1e-6
        feature_values = activations[:, :, feature_i].astype(np.float64, copy=False)
        event_doc, event_pos = np.where(feature_values > threshold)
        for offset in offsets:
            valid = event_pos + offset < positions
            if valid.any():
                values = feature_values[event_doc[valid], event_pos[valid] + offset] / scale
                mean_value = float(np.mean(values))
                n_events = int(valid.sum())
            else:
                mean_value = float("nan")
                n_events = 0
            rows.append(
                {
                    "feature_index": int(feature_indices[feature_i]),
                    "offset": int(offset),
                    "event_threshold": threshold_name,
                    "event_persistence": mean_value,
                    "n_events": n_events,
                }
            )
    return pd.DataFrame(rows)


class EventTriggeredAccumulator:
    def __init__(self, feature_indices: np.ndarray, quantile_table: pd.DataFrame, offsets: list[int]) -> None:
        self.feature_indices = feature_indices
        self.quantile_table = quantile_table.reset_index(drop=True)
        self.offsets = list(offsets)
        self.sums = np.zeros((len(feature_indices), len(offsets)), dtype=np.float64)
        self.counts = np.zeros((len(feature_indices), len(offsets)), dtype=np.int64)

    def update(self, activations: np.ndarray) -> None:
        _, positions, features = activations.shape
        for feature_i in range(features):
            qrow = self.quantile_table.iloc[feature_i]
            threshold_name = "q99" if int(qrow["positive_count"]) >= 500 else "q95"
            threshold = float(qrow[threshold_name])
            q50 = float(qrow["q50"])
            scale = threshold - q50 + 1e-6
            values = activations[:, :, feature_i].astype(np.float64, copy=False)
            event_doc, event_pos = np.where(values > threshold)
            for offset_i, offset in enumerate(self.offsets):
                valid = event_pos + offset < positions
                if valid.any():
                    normalized = values[event_doc[valid], event_pos[valid] + offset] / scale
                    self.sums[feature_i, offset_i] += float(np.sum(normalized))
                    self.counts[feature_i, offset_i] += int(valid.sum())

    def table(self) -> pd.DataFrame:
        rows = []
        for feature_i, feature_index in enumerate(self.feature_indices):
            qrow = self.quantile_table.iloc[feature_i]
            threshold_name = "q99" if int(qrow["positive_count"]) >= 500 else "q95"
            for offset_i, offset in enumerate(self.offsets):
                count = int(self.counts[feature_i, offset_i])
                rows.append(
                    {
                        "feature_index": int(feature_index),
                        "offset": int(offset),
                        "event_threshold": threshold_name,
                        "event_persistence": float(self.sums[feature_i, offset_i] / count) if count else float("nan"),
                        "n_events": count,
                    }
                )
        return pd.DataFrame(rows)
