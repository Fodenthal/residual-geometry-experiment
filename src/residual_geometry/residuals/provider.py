from __future__ import annotations

from typing import Any

import torch
from transformer_lens import HookedTransformer

from residual_geometry.config.schema import PersistentStateConfig
from residual_geometry.utils.device import get_device
from residual_geometry.utils.logging import get_logger


class ResidualStreamProvider:
    def __init__(self, config: PersistentStateConfig):
        self.config = config
        self.device = get_device()
        self.logger = get_logger(__name__)
        self._model: HookedTransformer | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        self.logger.info("Loading residual-only model %s", self.config.model.name)
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._model = HookedTransformer.from_pretrained(
            self.config.model.name,
            revision=self.config.model.revision,
            device=str(self.device),
            dtype=model_dtype,
        ).to(self.device)
        self._model.eval()

    @property
    def model(self) -> HookedTransformer:
        if self._model is None:
            self.load()
        assert self._model is not None
        return self._model

    def get_tokenizer(self) -> Any:
        return self.model.tokenizer

    def compute_sequence_residuals(self, token_batch: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            _, cache = self.model.run_with_cache(
                token_batch.to(self.device),
                names_filter=self.config.model.hook_name,
                return_type=None,
            )
        return cache[self.config.model.hook_name]

    def architecture_metadata(self) -> dict[str, object]:
        cfg = self.model.cfg
        raw_cfg = getattr(self.model, "cfg", None)
        metadata: dict[str, object] = {
            "model_name": self.config.model.name,
            "hook_name": self.config.model.hook_name,
            "target_layer_index": int(self.config.model.layer_index),
            "n_layers": getattr(cfg, "n_layers", None),
            "d_model": getattr(cfg, "d_model", None),
            "n_heads": getattr(cfg, "n_heads", None),
            "n_key_value_heads": getattr(cfg, "n_key_value_heads", None),
            "n_query_groups": getattr(cfg, "n_key_value_heads", None),
            "n_ctx": getattr(cfg, "n_ctx", None),
            "sliding_window": getattr(cfg, "sliding_window", None),
            "window_size": getattr(cfg, "window_size", None),
            "use_local_attn": getattr(cfg, "use_local_attn", None),
            "attn_scores_soft_cap": getattr(cfg, "attn_scores_soft_cap", None),
            "final_logit_softcap": getattr(cfg, "final_logit_softcap", None),
            "output_logits_soft_cap": getattr(cfg, "output_logits_soft_cap", None),
        }
        attn_types = getattr(cfg, "attn_types", None)
        if attn_types is not None:
            metadata["per_layer_attention_type"] = list(attn_types)
        hf_config = getattr(getattr(self.model, "hf_model", None), "config", None)
        if hf_config is not None:
            for key in [
                "num_hidden_layers",
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "max_position_embeddings",
                "sliding_window",
                "attn_logit_softcapping",
                "final_logit_softcapping",
            ]:
                metadata[f"hf_{key}"] = getattr(hf_config, key, None)
            layer_types = getattr(hf_config, "layer_types", None)
            if layer_types is not None:
                metadata["per_layer_attention_type"] = list(layer_types)
        layer_types = metadata.get("per_layer_attention_type")
        layer_index = self.config.model.layer_index
        if isinstance(layer_types, list) and 0 <= layer_index < len(layer_types):
            target_attention_type = layer_types[layer_index]
            attention_type_source = "transformerlens_cfg_attn_types"
        elif metadata.get("sliding_window") or metadata.get("hf_sliding_window") or metadata.get("window_size"):
            target_attention_type = "sliding_window_exposed_no_per_layer_type"
            attention_type_source = "window_size_without_per_layer_type"
        else:
            target_attention_type = "unknown"
            attention_type_source = "not_exposed"
        metadata["target_layer_attention_type"] = target_attention_type
        metadata["target_layer_attention_type_source"] = attention_type_source
        metadata["target_layer_full_1024_context_if_window_ge_1024"] = bool(
            (metadata.get("sliding_window") or metadata.get("hf_sliding_window") or metadata.get("window_size") or 0) >= 1024
        )
        metadata["raw_transformerlens_cfg_class"] = type(raw_cfg).__name__ if raw_cfg is not None else None
        return metadata
