from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    name: str = "google/gemma-2-2b"
    revision: str = "main"
    hook_name: str = "blocks.12.hook_resid_post"
    layer_index: int = 12
    d_model: int = 2304
    vocab_size: int = 256000


class SAEConfig(BaseModel):
    release: str = "gemma-scope-2b-pt-res"
    sae_id: str = "layer_12/width_16k/average_l0_82"
    sae_id_fallbacks: list[str] = Field(
        default_factory=lambda: [
            "layer_12/width_16k/average_l0_176",
            "layer_12/width_16k/average_l0_41",
            "layer_12/width_16k/average_l0_22",
            "layer_12/width_16k/average_l0_445",
        ]
    )
    n_features: int = 16384


class DatasetConfig(BaseModel):
    dataset_name: str = "allenai/c4"
    dataset_config_name: str | None = "en"
    split: str = "validation"
    streaming: bool = True
    document_count: int
    max_scan_rows: int


class ContextProcessingConfig(BaseModel):
    min_tokens: int = 512
    max_tokens: int = 512
    add_special_tokens: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> "ContextProcessingConfig":
        if self.min_tokens <= 0:
            raise ValueError("min_tokens must be positive")
        if self.max_tokens < self.min_tokens:
            raise ValueError("max_tokens must be >= min_tokens")
        return self


class ContextSplitConfig(BaseModel):
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 2

    @model_validator(mode="after")
    def validate_fractions(self) -> "ContextSplitConfig":
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("context split fractions must sum to 1.0")
        return self


class FeatureSelectionConfig(BaseModel):
    selected_feature_count: int
    min_density: float = 1e-4
    max_density: float = 0.20
    min_active_documents: int = 50
    density_bins: int = 8
    selection_seed: int = 0
    stats_batch_size: int = 16

    @model_validator(mode="after")
    def validate_selection(self) -> "FeatureSelectionConfig":
        if self.selected_feature_count <= 0:
            raise ValueError("selected_feature_count must be positive")
        if not 0.0 <= self.min_density < self.max_density <= 1.0:
            raise ValueError("density bounds must satisfy 0 <= min < max <= 1")
        if self.density_bins <= 0:
            raise ValueError("density_bins must be positive")
        return self


class ActivationConfig(BaseModel):
    batch_size: int = 16
    chunk_documents: int = 64
    feature_chunk_size: int = 2048
    save_chunks: bool = True

    @model_validator(mode="after")
    def validate_activation(self) -> "ActivationConfig":
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.chunk_documents <= 0:
            raise ValueError("chunk_documents must be positive")
        if self.feature_chunk_size <= 0:
            raise ValueError("feature_chunk_size must be positive")
        return self


class AutocorrConfig(BaseModel):
    max_lag: int
    bootstrap_replicates: int
    valid_doc_threshold: int = 100
    min_valid_lag_fraction: float = 0.8
    smoothing_width: int = 5
    binary_primary_quantile: float = 0.99
    binary_fallback_quantile: float = 0.95
    binary_min_positive_activations: int = 1000
    event_offsets: list[int]

    @model_validator(mode="after")
    def validate_autocorr(self) -> "AutocorrConfig":
        if self.max_lag <= 0:
            raise ValueError("max_lag must be positive")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0.0 < self.min_valid_lag_fraction <= 1.0:
            raise ValueError("min_valid_lag_fraction must be in (0, 1]")
        return self


class NullConfig(BaseModel):
    matched_sparsity_replicates: int
    document_permutation_replicates: int
    random_residual_directions: int = 512
    residual_pca_components: int = 256
    residual_pca_max_positions: int = 1_000_000
    seed: int = 4


class ResidualRandomProbeConfig(BaseModel):
    directions: int
    seed: int = 4


class ResidualPCAConfig(BaseModel):
    max_positions: int
    components: int
    seed: int = 4
    batch_size: int = 4096


class ResidualTimeLaggedConfig(BaseModel):
    whitening_pcs: int
    output_directions: int
    lag_set: list[int]
    ridge_scale: float = 1e-4
    max_ridge_scale: float = 1e-1
    condition_number_threshold: float = 1e4
    seed: int = 4

    @model_validator(mode="after")
    def validate_lags(self) -> "ResidualTimeLaggedConfig":
        if self.whitening_pcs <= 0:
            raise ValueError("whitening_pcs must be positive")
        if self.output_directions <= 0:
            raise ValueError("output_directions must be positive")
        if not self.lag_set or any(lag <= 0 for lag in self.lag_set):
            raise ValueError("lag_set must contain positive lags")
        if self.ridge_scale <= 0 or self.max_ridge_scale < self.ridge_scale:
            raise ValueError("ridge scales must satisfy 0 < ridge_scale <= max_ridge_scale")
        return self


class ResidualProjectionConfig(BaseModel):
    batch_size: int = 32
    chunk_documents: int = 128
    storage_dtype: Literal["float16", "float32"] = "float16"

    @model_validator(mode="after")
    def validate_projection(self) -> "ResidualProjectionConfig":
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.chunk_documents <= 0:
            raise ValueError("chunk_documents must be positive")
        if self.chunk_documents % self.batch_size != 0:
            raise ValueError("chunk_documents must be a multiple of batch_size")
        return self


class ResidualSubspaceConfig(BaseModel):
    k_values: list[int]
    dedup_abs_cosine_threshold: float = 0.95


class ResidualGeometryConfig(BaseModel):
    max_tokens: int = 1024
    max_lag: int
    sensitivity_max_lag: int | None = None
    random: ResidualRandomProbeConfig
    pca: ResidualPCAConfig
    time_lagged: ResidualTimeLaggedConfig
    projections: ResidualProjectionConfig = Field(default_factory=ResidualProjectionConfig)
    subspace: ResidualSubspaceConfig | None = None
    permutation_replicates: int
    bootstrap_replicates: int
    bootstrap_seed: int = 13
    valid_doc_threshold: int
    min_valid_lag_fraction: float = 0.8
    smoothing_width: int = 5
    ci_lags: list[int]

    @model_validator(mode="after")
    def validate_residual_geometry(self) -> "ResidualGeometryConfig":
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_lag <= 0:
            raise ValueError("max_lag must be positive")
        if self.sensitivity_max_lag is not None and self.sensitivity_max_lag < self.max_lag:
            raise ValueError("sensitivity_max_lag must be >= max_lag")
        max_allowed_lag = self.sensitivity_max_lag or self.max_lag
        if any(lag <= 0 or lag > max_allowed_lag for lag in self.ci_lags):
            raise ValueError("ci_lags must be positive and <= max configured lag")
        if self.permutation_replicates <= 0:
            raise ValueError("permutation_replicates must be positive")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if self.valid_doc_threshold <= 0:
            raise ValueError("valid_doc_threshold must be positive")
        if not 0.0 < self.min_valid_lag_fraction <= 1.0:
            raise ValueError("min_valid_lag_fraction must be in (0, 1]")
        return self


class TruncationConfig(BaseModel):
    windows: list[int]
    target_position_start: int = 128
    target_position_stop: int = 512
    max_positions_per_document: int = 128
    position_seed: int = 7
    match_seed: int = 8


class CausalPersistenceConfig(BaseModel):
    future_offsets: list[int]
    features_per_bin: int = 32
    positions_per_stratum: int = 32
    match_seed: int = 10


class RuntimeConfig(BaseModel):
    synthetic_activations: bool = False
    validate_sae_load: bool = True
    disk_light_stage_b: bool = False


class PersistentStateConfig(BaseModel):
    mode: Literal["smoke", "pilot", "full"]
    run_name: str
    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    sae: SAEConfig = Field(default_factory=SAEConfig)
    dataset: DatasetConfig
    context_processing: ContextProcessingConfig = Field(default_factory=ContextProcessingConfig)
    context_split: ContextSplitConfig = Field(default_factory=ContextSplitConfig)
    feature_selection: FeatureSelectionConfig
    activations: ActivationConfig = Field(default_factory=ActivationConfig)
    autocorr: AutocorrConfig
    nulls: NullConfig
    truncation: TruncationConfig
    causal_persistence: CausalPersistenceConfig
    residual_geometry: ResidualGeometryConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output_base: str = "outputs/persistent_state"


def mode_defaults(mode: Literal["smoke", "pilot", "full"]) -> dict:
    if mode == "smoke":
        return {
            "dataset": {"document_count": 128, "max_scan_rows": 10_000},
            "feature_selection": {
                "selected_feature_count": 256,
                "min_active_documents": 1,
                "stats_batch_size": 2,
            },
            "autocorr": {
                "max_lag": 64,
                "bootstrap_replicates": 20,
                "valid_doc_threshold": 1,
                "event_offsets": [1, 2, 4, 8, 16, 32, 64],
            },
            "nulls": {
                "matched_sparsity_replicates": 5,
                "document_permutation_replicates": 5,
                "random_residual_directions": 64,
            },
            "truncation": {"windows": [32, 128, 512]},
            "causal_persistence": {"future_offsets": [0, 1, 2, 4, 8, 16, 32, 64]},
            "residual_geometry": {
                "max_tokens": 1024,
                "max_lag": 128,
                "random": {"directions": 64},
                "pca": {"max_positions": 100_000, "components": 64},
                "time_lagged": {"whitening_pcs": 128, "output_directions": 32, "lag_set": [8, 16, 32]},
                "subspace": {"k_values": [8, 16, 32]},
                "permutation_replicates": 5,
                "bootstrap_replicates": 20,
                "valid_doc_threshold": 8,
                "ci_lags": [1, 2, 4, 8, 16, 32, 64, 128],
            },
        }
    if mode == "pilot":
        return {
            "dataset": {"document_count": 5_000, "max_scan_rows": 500_000},
            "feature_selection": {"selected_feature_count": 4_096},
            "autocorr": {
                "max_lag": 256,
                "bootstrap_replicates": 200,
                "event_offsets": [1, 2, 4, 8, 16, 32, 64, 128, 256],
            },
            "nulls": {
                "matched_sparsity_replicates": 20,
                "document_permutation_replicates": 20,
            },
            "truncation": {"windows": [16, 32, 64, 128, 256, 512]},
            "causal_persistence": {"future_offsets": [0, 1, 2, 4, 8, 16, 32, 64]},
            "residual_geometry": {
                "max_tokens": 1024,
                "max_lag": 512,
                "random": {"directions": 512},
                "pca": {"max_positions": 1_000_000, "components": 256},
                "time_lagged": {"whitening_pcs": 512, "output_directions": 256, "lag_set": [8, 16, 32, 64, 128]},
                "subspace": {"k_values": [8, 16, 32, 64, 128, 256]},
                "permutation_replicates": 20,
                "bootstrap_replicates": 200,
                "valid_doc_threshold": 100,
                "ci_lags": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            },
        }
    return {
        "dataset": {"document_count": 50_000, "max_scan_rows": 2_000_000},
        "feature_selection": {"selected_feature_count": 16_384},
        "autocorr": {
            "max_lag": 256,
            "bootstrap_replicates": 200,
            "event_offsets": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        },
        "nulls": {
            "matched_sparsity_replicates": 20,
            "document_permutation_replicates": 20,
            "residual_pca_max_positions": 5_000_000,
        },
        "truncation": {"windows": [16, 32, 64, 128, 256, 512]},
        "causal_persistence": {
            "future_offsets": [0, 1, 2, 4, 8, 16, 32, 64, 96, 128],
            "positions_per_stratum": 128,
        },
        "residual_geometry": {
            "max_tokens": 1024,
            "max_lag": 512,
            "sensitivity_max_lag": 768,
            "random": {"directions": 2048},
            "pca": {"max_positions": 5_000_000, "components": 512},
            "time_lagged": {"whitening_pcs": 1024, "output_directions": 512, "lag_set": [8, 16, 32, 64, 128]},
            "subspace": {"k_values": [8, 16, 32, 64, 128, 256, 512, 1024]},
            "permutation_replicates": 20,
            "bootstrap_replicates": 200,
            "valid_doc_threshold": 100,
            "ci_lags": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768],
        },
    }
