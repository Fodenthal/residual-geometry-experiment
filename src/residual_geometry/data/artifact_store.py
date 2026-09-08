from __future__ import annotations

import os

from residual_geometry.utils.io import ensure_dir


class PersistentStateArtifactStore:
    def __init__(self, run_dir: str) -> None:
        self._run_dir = os.path.abspath(run_dir)
        for directory in [
            self.contexts_dir,
            self.features_dir,
            self.activations_dir,
            self.residual_probes_dir,
            self.autocorr_dir,
            self.nulls_dir,
            self.burn_in_dir,
            self.truncation_dir,
            self.causal_persistence_dir,
            self.fanout_dir,
            self.subspace_dir,
            self.robustness_dir,
            self.qualitative_dir,
            self.reports_dir,
            self.configs_dir,
            self.logs_dir,
            self.qc_dir,
        ]:
            ensure_dir(directory)

    @property
    def run_dir(self) -> str:
        return self._run_dir

    @property
    def contexts_dir(self) -> str:
        return os.path.join(self._run_dir, "contexts")

    @property
    def features_dir(self) -> str:
        return os.path.join(self._run_dir, "features")

    @property
    def activations_dir(self) -> str:
        return os.path.join(self._run_dir, "activations")

    @property
    def residual_probes_dir(self) -> str:
        return os.path.join(self._run_dir, "residual_probes")

    @property
    def residual_probe_directions_dir(self) -> str:
        return os.path.join(self.residual_probes_dir, "directions")

    @property
    def residual_projection_dir(self) -> str:
        return os.path.join(self.residual_probes_dir, "projections")

    @property
    def residual_autocorr_dir(self) -> str:
        return os.path.join(self.residual_probes_dir, "autocorr")

    @property
    def autocorr_dir(self) -> str:
        return os.path.join(self._run_dir, "autocorr")

    @property
    def nulls_dir(self) -> str:
        return os.path.join(self._run_dir, "nulls")

    @property
    def burn_in_dir(self) -> str:
        return os.path.join(self._run_dir, "burn_in")

    @property
    def truncation_dir(self) -> str:
        return os.path.join(self._run_dir, "truncation")

    @property
    def causal_persistence_dir(self) -> str:
        return os.path.join(self._run_dir, "causal_persistence")

    @property
    def fanout_dir(self) -> str:
        return os.path.join(self._run_dir, "fanout")

    @property
    def subspace_dir(self) -> str:
        return os.path.join(self._run_dir, "subspace")

    @property
    def fat_subspace_dir(self) -> str:
        path = os.path.join(self.subspace_dir, "fat_subspace")
        ensure_dir(path)
        return path

    @property
    def robustness_dir(self) -> str:
        return os.path.join(self._run_dir, "robustness")

    @property
    def qualitative_dir(self) -> str:
        return os.path.join(self._run_dir, "qualitative")

    @property
    def reports_dir(self) -> str:
        return os.path.join(self._run_dir, "reports")

    @property
    def configs_dir(self) -> str:
        return os.path.join(self._run_dir, "configs")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self._run_dir, "logs")

    @property
    def qc_dir(self) -> str:
        return os.path.join(self._run_dir, "qc")

    @property
    def context_pool_path(self) -> str:
        return os.path.join(self.contexts_dir, "context_pool.parquet")

    @property
    def context_split_path(self) -> str:
        return os.path.join(self.contexts_dir, "context_split.parquet")

    @property
    def feature_stats_path(self) -> str:
        return os.path.join(self.features_dir, "feature_stats.parquet")

    @property
    def selected_features_path(self) -> str:
        return os.path.join(self.features_dir, "selected_features.parquet")

    @property
    def selection_config_path(self) -> str:
        return os.path.join(self.features_dir, "selection_config.json")

    @property
    def timescales_path(self) -> str:
        return os.path.join(self.autocorr_dir, "timescales.parquet")

    @property
    def timescale_summary_path(self) -> str:
        return os.path.join(self.autocorr_dir, "timescale_summary.json")

    @property
    def event_triggered_path(self) -> str:
        return os.path.join(self.autocorr_dir, "event_triggered_persistence.parquet")

    @property
    def bootstrap_timescales_path(self) -> str:
        return os.path.join(self.autocorr_dir, "bootstrap_timescales.parquet")

    @property
    def bootstrap_event_triggered_path(self) -> str:
        return os.path.join(self.autocorr_dir, "bootstrap_event_triggered.parquet")

    @property
    def high_confidence_slow_features_path(self) -> str:
        return os.path.join(self.autocorr_dir, "high_confidence_slow_features.parquet")

    @property
    def burn_in_metrics_path(self) -> str:
        return os.path.join(self.burn_in_dir, "burn_in_metrics.parquet")

    @property
    def burn_in_summary_path(self) -> str:
        return os.path.join(self.burn_in_dir, "burn_in_summary.json")

    @property
    def burn_in_profiles_path(self) -> str:
        return os.path.join(self.burn_in_dir, "bucket_profiles.npz")

    @property
    def stage_b_gate_path(self) -> str:
        return os.path.join(self.run_dir, "gates", "stage_b_autocorr_gate.json")

    @property
    def protocol_log_path(self) -> str:
        return os.path.join(self.run_dir, "protocol_log.json")

    def activation_split_dir(self, split: str) -> str:
        path = os.path.join(self.activations_dir, split)
        ensure_dir(path)
        return path

    def activation_chunk_path(self, split: str, chunk_index: int) -> str:
        return os.path.join(self.activation_split_dir(split), f"chunk_{chunk_index:05d}.npz")

    def autocorr_profiles_path(self, split: str, estimator: str) -> str:
        return os.path.join(self.autocorr_dir, f"profiles_{split}_{estimator}.npz")

    def qc_summary_path(self, stage_name: str) -> str:
        return os.path.join(self.qc_dir, f"{stage_name}_summary.json")

    @property
    def resolved_model_architecture_path(self) -> str:
        return os.path.join(self.configs_dir, "resolved_model_architecture.json")

    @property
    def random_residual_directions_path(self) -> str:
        return os.path.join(self.residual_probe_directions_dir, "random_residual.npz")

    @property
    def residual_pca_directions_path(self) -> str:
        return os.path.join(self.residual_probe_directions_dir, "residual_pca.npz")

    @property
    def time_lagged_residual_directions_path(self) -> str:
        return os.path.join(self.residual_probe_directions_dir, "time_lagged_residual.npz")

    @property
    def residual_pca_fit_summary_path(self) -> str:
        return os.path.join(self.residual_probes_dir, "pca_fit_summary.json")

    @property
    def residual_covariance_spectrum_path(self) -> str:
        return os.path.join(self.residual_probes_dir, "residual_covariance_spectrum.json")

    @property
    def residual_covariance_eigenvalues_path(self) -> str:
        return os.path.join(self.residual_probes_dir, "residual_covariance_eigenvalues.npy")

    @property
    def time_lagged_fit_summary_path(self) -> str:
        return os.path.join(self.residual_probes_dir, "time_lagged_fit_summary.json")

    @property
    def time_lagged_covariances_path(self) -> str:
        return os.path.join(self.residual_probes_dir, "time_lagged_covariances.npz")

    @property
    def time_lagged_ridge_sensitivity_path(self) -> str:
        return os.path.join(self.residual_probes_dir, "time_lagged_ridge_sensitivity.parquet")

    @property
    def residual_probe_timescales_path(self) -> str:
        return os.path.join(self.residual_autocorr_dir, "probe_timescales.parquet")

    @property
    def residual_probe_summary_path(self) -> str:
        return os.path.join(self.residual_autocorr_dir, "residual_probe_summary.json")

    @property
    def residual_bootstrap_timescales_path(self) -> str:
        return os.path.join(self.residual_autocorr_dir, "bootstrap_probe_timescales.parquet")

    @property
    def residual_geometry_report_path(self) -> str:
        return os.path.join(self.reports_dir, "residual_geometry_report.md")

    @property
    def residual_geometry_summary_path(self) -> str:
        return os.path.join(self.reports_dir, "residual_geometry_summary.json")

    @property
    def fat_subspace_summary_path(self) -> str:
        return os.path.join(self.fat_subspace_dir, "fat_subspace_summary.json")

    def residual_projection_split_dir(self, split: str) -> str:
        path = os.path.join(self.residual_projection_dir, split)
        ensure_dir(path)
        return path

    def residual_projection_chunk_path(self, split: str, chunk_index: int) -> str:
        return os.path.join(self.residual_projection_split_dir(split), f"chunk_{chunk_index:05d}.npz")

    def residual_autocorr_profiles_path(self, split: str, probe_family: str, estimator: str) -> str:
        return os.path.join(self.residual_autocorr_dir, f"profiles_{split}_{probe_family}_{estimator}.npz")
