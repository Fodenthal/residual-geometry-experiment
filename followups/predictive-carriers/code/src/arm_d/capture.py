"""Model loading, the frozen hook pair, residual capture, and patching (R1 §26).

Raw Hugging Face hooks are used rather than TransformerLens.  Two reasons, both
recorded in the run manifest: the job image ships ``transformers`` 5.x, which
the repository's pinned ``transformer-lens==2.6.0`` does not support; and
TransformerLens' default weight processing (LayerNorm folding, writing-weight
centring) alters the residual stream it reports, whereas a raw pre-hook on the
decoder layer returns the exact tensor the model computes.  Arm D measures
differences in that tensor, so exactness matters more than the repository's
usual tooling.

Hook pair (R1 §26.1):

* source      pre-hook input of ``layers[12]``  -- the residual immediately
              before layer 12's ``input_layernorm`` and its K/V formation,
              i.e. ``resid_pre`` at layer 12;
* destination output of ``layers[12]``          -- ``resid_post`` at layer 12.

Attention at token ``t + k`` reads keys and values built from the source hook's
tensor at token ``t``, so a genuine sequence-mixing path exists.
:func:`verify_interface` proves it on the loaded graph instead of assuming it,
and also proves the *absence* of a backward path (R1 addition A6).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
import torch

MODEL_NAME = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
PRIMARY_LAYER = 12
D_MODEL = 2304


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    layers: Sequence[torch.nn.Module]
    final_norm: torch.nn.Module
    lm_head: torch.nn.Module
    device: torch.device
    metadata: dict[str, object]


def resolve_decoder_layers(model: torch.nn.Module, expected: int) -> Sequence[torch.nn.Module]:
    """Find the decoder-layer container semantically, never by a hardcoded path."""

    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == expected:
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError(f"no ModuleList of length {expected} found in {type(model).__name__}")
    # Prefer the shallowest match; deeper ones would be sub-containers.
    candidates.sort(key=lambda item: item[0].count("."))
    return candidates[0][1]


def resolve_named_module(model: torch.nn.Module, suffixes: tuple[str, ...]) -> torch.nn.Module:
    for suffix in suffixes:
        for name, module in model.named_modules():
            if name.endswith(suffix):
                return module
    raise RuntimeError(f"none of {suffixes} resolved in {type(model).__name__}")


def load_model(
    *,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    revision: str = MODEL_REVISION,
    name: str = MODEL_NAME,
) -> LoadedModel:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if resolved_device.type != "cuda":
        dtype = torch.float32
    config = AutoConfig.from_pretrained(name, revision=revision)
    if config.model_type != "gemma2":
        raise RuntimeError(f"unexpected model_type {config.model_type!r}")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(resolved_device)
    model.eval()

    layers = resolve_decoder_layers(model, config.num_hidden_layers)
    if len(layers) != config.num_hidden_layers:
        raise RuntimeError("resolved decoder-layer count disagrees with the config")
    if config.hidden_size != D_MODEL:
        raise RuntimeError(f"unexpected hidden size {config.hidden_size}")
    final_norm = resolve_named_module(model, ("model.norm", ".norm"))
    lm_head = resolve_named_module(model, ("lm_head",))
    if next(model.parameters()).dtype != dtype:
        raise RuntimeError("model dtype does not match the requested dtype")

    metadata = {
        "model_name": name,
        "model_revision": revision,
        "model_class": type(model).__name__,
        "model_type": config.model_type,
        "n_layers": int(config.num_hidden_layers),
        "d_model": int(config.hidden_size),
        "n_heads": int(config.num_attention_heads),
        "n_key_value_heads": int(getattr(config, "num_key_value_heads", -1)),
        "sliding_window": int(getattr(config, "sliding_window", -1)),
        "attn_logit_softcapping": float(getattr(config, "attn_logit_softcapping", 0.0) or 0.0),
        "final_logit_softcapping": float(getattr(config, "final_logit_softcapping", 0.0) or 0.0),
        "attn_implementation": "eager",
        "dtype": str(dtype),
        "device": str(resolved_device),
        "layer_types": list(getattr(config, "layer_types", []) or []),
        "primary_layer": PRIMARY_LAYER,
        "source_hook": f"decoder_layers[{PRIMARY_LAYER}].forward_pre_hook.hidden_states (resid_pre)",
        "destination_hook": f"decoder_layers[{PRIMARY_LAYER}].forward_hook.output[0] (resid_post)",
        "hook_backend": "raw_huggingface_module_hooks",
        "sliding_window_covers_full_context": bool(int(getattr(config, "sliding_window", 0) or 0) >= 1024),
    }
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
        device=resolved_device,
        metadata=metadata,
    )


@contextmanager
def residual_taps(
    loaded: LoadedModel,
    *,
    layer_index: int = PRIMARY_LAYER,
    capture_final_norm: bool = False,
    patch: Callable[[torch.Tensor], torch.Tensor] | None = None,
    extra_layers: Sequence[int] = (),
    capture_attention: bool = False,
) -> Iterator[dict[str, torch.Tensor]]:
    """Capture ``resid_pre`` and ``resid_post`` at ``layer_index``.

    ``patch`` receives the source tensor and returns the tensor the layer should
    actually consume, which is how every causal intervention is applied: the
    intervention and the measurement therefore use the same interface, as R1
    §26.2 requires.

    ``extra_layers`` taps additional layers in the SAME forward pass (R1.1
    amendment 3).  The layer sweep is paired by construction because every layer
    sees the identical spliced sequence; a separate pass per layer would not be.
    The patch is applied only at ``layer_index``, so the secondary layers are
    observational.

    ``capture_attention`` stores the attention probabilities at ``layer_index``
    (R1.1 amendment 4).  Attention scores are a cross-position bilinear form, so
    a patch that acts by routing rather than by content transport should move
    them; the destination-content shift alone cannot distinguish the two.
    """

    store: dict[str, torch.Tensor] = {}
    layer = loaded.layers[layer_index]

    def pre_hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]
        store["source"] = hidden.detach()
        if patch is None:
            return None
        patched = patch(hidden)
        store["patched_source"] = patched.detach()
        if "hidden_states" in kwargs:
            kwargs = {**kwargs, "hidden_states": patched}
            return args, kwargs
        return (patched,) + tuple(args[1:]), kwargs

    def post_hook(module, args, kwargs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store["destination"] = tensor.detach()
        return None

    handles = [
        layer.register_forward_pre_hook(pre_hook, with_kwargs=True),
        layer.register_forward_hook(post_hook, with_kwargs=True),
    ]

    def _make_extra_hooks(index: int) -> tuple[Callable, Callable]:
        def extra_pre(module, args, kwargs):
            hidden = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]
            store[f"source_layer_{index}"] = hidden.detach()
            return None

        def extra_post(module, args, kwargs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            store[f"destination_layer_{index}"] = tensor.detach()
            return None

        return extra_pre, extra_post

    for index in extra_layers:
        if index == layer_index:
            continue
        extra_pre, extra_post = _make_extra_hooks(index)
        extra_layer = loaded.layers[index]
        handles.append(extra_layer.register_forward_pre_hook(extra_pre, with_kwargs=True))
        handles.append(extra_layer.register_forward_hook(extra_post, with_kwargs=True))

    if capture_attention:

        def attention_hook(module, args, kwargs, output):
            # The eager attention path returns (attn_output, attn_weights).  Some
            # transformers versions return None for the weights unless asked, so
            # record availability rather than assuming it and let the caller
            # report the diagnostic as unavailable instead of fabricating it.
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            store["attention"] = weights.detach() if weights is not None else None
            return None

        handles.append(
            loaded.layers[layer_index].self_attn.register_forward_hook(
                attention_hook, with_kwargs=True
            )
        )
    if capture_final_norm:

        def norm_hook(module, args, output):
            store["final_norm"] = (output[0] if isinstance(output, tuple) else output).detach()
            return None

        handles.append(loaded.final_norm.register_forward_hook(norm_hook))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


def forward_tokens(
    loaded: LoadedModel, tokens: torch.Tensor, *, output_attentions: bool = False
) -> None:
    with torch.inference_mode():
        loaded.model(
            input_ids=tokens.to(loaded.device),
            use_cache=False,
            output_attentions=output_attentions,
        )


def verify_interface(
    loaded: LoadedModel,
    tokens: np.ndarray,
    source_position: int,
    *,
    layer_index: int = PRIMARY_LAYER,
    magnitude: float = 1.0,
    lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
) -> dict[str, object]:
    """Prove the hook pair is wired as claimed, on the loaded graph (R1 §26.1).

    Forward path: perturbing the source hook at ``t`` must change the
    destination hook at ``t + k`` for every evaluated ``k >= 1``.
    Backward path: it must leave the destination hook at every token ``< t``
    bitwise unchanged.  Causal masking makes the second a real wiring test.
    """

    batch = torch.as_tensor(np.asarray(tokens)[None, :], dtype=torch.long)
    with residual_taps(loaded, layer_index=layer_index) as clean:
        forward_tokens(loaded, batch)
        clean_destination = clean["destination"].clone().float()
        clean_source = clean["source"].clone().float()

    generator = torch.Generator(device="cpu").manual_seed(0)
    direction = torch.randn(clean_source.shape[-1], generator=generator)
    direction = direction / direction.norm()
    scale = float(clean_source[0, source_position].norm()) * magnitude
    delta = (direction * scale).to(loaded.device).to(clean["source"].dtype if "source" in clean else torch.float32)

    def patch(hidden: torch.Tensor) -> torch.Tensor:
        patched = hidden.clone()
        patched[:, source_position, :] = patched[:, source_position, :] + delta.to(patched.dtype)
        return patched

    with residual_taps(loaded, layer_index=layer_index, patch=patch) as perturbed:
        forward_tokens(loaded, batch)
        perturbed_destination = perturbed["destination"].clone().float()

    difference = (perturbed_destination - clean_destination)[0]
    norms = difference.norm(dim=-1)
    reference = clean_destination[0].norm(dim=-1)
    relative = (norms / reference.clamp_min(1e-12)).cpu().numpy()

    before = relative[:source_position]
    forward_by_lag = {
        int(lag): float(relative[source_position + lag])
        for lag in lags
        if source_position + lag < relative.shape[0]
    }
    return {
        "source_position": int(source_position),
        "layer_index": int(layer_index),
        "perturbation_relative_norm": float(magnitude),
        "max_relative_change_before_source": float(before.max()) if before.size else 0.0,
        "relative_change_at_source": float(relative[source_position]),
        "relative_change_by_lag": forward_by_lag,
        "forward_path_present": bool(all(value > 1e-4 for value in forward_by_lag.values())),
        "no_backward_path": bool(before.size == 0 or float(before.max()) == 0.0),
        "evaluated_lags": [int(lag) for lag in forward_by_lag],
    }


# --------------------------------------------------------------------------
# Aperture fitting and coordinate capture
# --------------------------------------------------------------------------


class StreamingMoments:
    """Float64 mean and second moment over pooled source/destination tokens.

    Accumulated on the compute device: the Gram update is ``d_model``-squared
    work per token and moving 16k x 2304 activations to the host each batch
    would dominate the capture.  Float64 throughout, because a float32 running
    second moment over a million tokens loses digits where it matters (the
    aperture's small eigenvalues).
    """

    def __init__(self, width: int, device: torch.device | None = None) -> None:
        self.width = int(width)
        self.count = 0
        self.device = device or torch.device("cpu")
        self.total = torch.zeros(self.width, dtype=torch.float64, device=self.device)
        self.gram = torch.zeros((self.width, self.width), dtype=torch.float64, device=self.device)

    def update(self, values: torch.Tensor) -> None:
        array = values.reshape(-1, self.width).to(device=self.device, dtype=torch.float64)
        self.count += int(array.shape[0])
        self.total += array.sum(dim=0)
        self.gram += array.T @ array

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("not enough tokens to estimate an aperture")
        mean = (self.total / self.count).cpu().numpy()
        gram = (self.gram / self.count).cpu().numpy()
        covariance = gram - np.outer(mean, mean)
        return mean, 0.5 * (covariance + covariance.T)


def aperture_from_covariance(mean: np.ndarray, covariance: np.ndarray, width: int) -> dict[str, np.ndarray]:
    """Eigenvalue-ordered PCA basis; smaller apertures are column prefixes."""

    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1][: int(width)]
    return {
        "mean": mean.astype(np.float64),
        "basis": vectors[:, order].astype(np.float64),
        "eigenvalues": values[order].astype(np.float64),
        "total_variance": np.array(float(np.sum(np.clip(values, 0, None)))),
    }


def project(values: torch.Tensor, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    array = values.to(torch.float32).cpu().numpy().astype(np.float64)
    return (array - mean) @ basis


@dataclass(frozen=True)
class CaptureOffsets:
    source: tuple[int, ...]
    destination: tuple[int, ...]


def capture_batch(
    loaded: LoadedModel,
    token_batch: np.ndarray,
    source_positions: np.ndarray,
    offsets: CaptureOffsets,
    mean: np.ndarray,
    basis: np.ndarray,
    *,
    layer_index: int = PRIMARY_LAYER,
    patch: Callable[[torch.Tensor], torch.Tensor] | None = None,
    nll_offsets: tuple[int, ...] | None = None,
    extra_layers: Sequence[int] = (),
    extra_bases: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
    capture_attention: bool = False,
) -> dict[str, np.ndarray]:
    """One forward pass; return aperture coordinates at the requested offsets.

    ``nll_offsets`` adds next-token negative log-likelihood at those offsets
    from the source position, computed from the same forward pass by tapping the
    final norm and applying the language-model head at only those positions.
    Running it as a second forward pass would double the capture cost.
    """

    tokens = torch.as_tensor(np.asarray(token_batch), dtype=torch.long)
    with residual_taps(
        loaded,
        layer_index=layer_index,
        patch=patch,
        capture_final_norm=nll_offsets is not None,
        extra_layers=extra_layers,
        capture_attention=capture_attention,
    ) as store:
        forward_tokens(loaded, tokens, output_attentions=capture_attention)
        source = store["source"]
        destination = store["destination"]
        rows = np.arange(tokens.shape[0])
        source_index = np.asarray(source_positions)[:, None] + np.asarray(offsets.source)[None, :]
        destination_index = np.asarray(source_positions)[:, None] + np.asarray(offsets.destination)[None, :]
        source_slice = source[torch.as_tensor(rows)[:, None], torch.as_tensor(source_index)]
        destination_slice = destination[torch.as_tensor(rows)[:, None], torch.as_tensor(destination_index)]
        source_coordinates = project(source_slice, mean, basis)
        destination_coordinates = project(destination_slice, mean, basis)
        output = {
            "source": source_coordinates.astype(np.float32),
            "destination": destination_coordinates.astype(np.float32),
        }
        # Secondary layers use their OWN aperture basis: residual bases are not
        # comparable across layers, so projecting layer 10 through the layer-12
        # basis would compare a representation to a coordinate system it never
        # lived in.
        for index in extra_layers:
            if index == layer_index or extra_bases is None or index not in extra_bases:
                continue
            layer_mean, layer_basis = extra_bases[index]
            layer_source = store[f"source_layer_{index}"][
                torch.as_tensor(rows)[:, None], torch.as_tensor(source_index)
            ]
            layer_destination = store[f"destination_layer_{index}"][
                torch.as_tensor(rows)[:, None], torch.as_tensor(destination_index)
            ]
            output[f"source_layer_{index}"] = project(
                layer_source, layer_mean, layer_basis
            ).astype(np.float32)
            output[f"destination_layer_{index}"] = project(
                layer_destination, layer_mean, layer_basis
            ).astype(np.float32)
            # Also read the secondary layer through the PRIMARY layer's frozen
            # interface (R1.1 amendment 7.6).  The residual stream is the same
            # vector space at every layer, so Q_0 fitted at layer 12 can be
            # applied directly to layer 10 and 14 residuals; that is a sharper
            # test than comparing independently fitted per-layer objects.
            output[f"source_layer_{index}_primary_basis"] = project(
                layer_source, mean, basis
            ).astype(np.float32)
            output[f"destination_layer_{index}_primary_basis"] = project(
                layer_destination, mean, basis
            ).astype(np.float32)
        if capture_attention:
            attention = store.get("attention")
            if attention is None:
                output["attention_available"] = np.zeros((), dtype=bool)
            else:
                # Keep only the attention ROWS at the evaluated destination
                # positions.  The full pattern is (batch, heads, query, key) and
                # storing it for every sequence would dwarf the coordinates.
                selected = attention[
                    torch.as_tensor(rows)[:, None], :, torch.as_tensor(destination_index)
                ]
                output["attention_rows"] = selected.float().cpu().numpy().astype(np.float32)
                output["attention_available"] = np.ones((), dtype=bool)
        if nll_offsets is not None:
            output["nll"] = _nll_from_hidden(
                loaded, store["final_norm"], tokens, np.asarray(source_positions), np.asarray(nll_offsets)
            )
    return output


def _nll_from_hidden(
    loaded: LoadedModel,
    hidden: torch.Tensor,
    tokens: torch.Tensor,
    source_positions: np.ndarray,
    offsets: np.ndarray,
    chunk: int = 8,
) -> np.ndarray:
    """Next-token NLL at ``source_positions + offsets`` from a tapped forward."""

    batch, sequence_length = tokens.shape[0], tokens.shape[1]
    losses = np.zeros((batch, offsets.shape[0]), dtype=np.float64)
    softcap = float(loaded.metadata.get("final_logit_softcapping", 0.0) or 0.0)
    device_tokens = tokens.to(loaded.device)
    index = source_positions[:, None] + offsets[None, :]

    # The NLL at position i reads the token at i + 1, so the last position with a
    # next-token target is sequence_length - 2.  The largest source position plus
    # the largest lag reaches the final position of the sequence, whose target
    # lies one past the end.  Those entries are recorded as NaN rather than
    # clamped: clamping would silently score a different position and quietly
    # bias the offsets that hit the boundary most often.
    evaluable = index + 1 < sequence_length
    safe_index = np.clip(index, 0, sequence_length - 2)
    rows = torch.as_tensor(np.arange(batch))[:, None]
    with torch.inference_mode():
        for start in range(0, offsets.shape[0], chunk):
            window = torch.as_tensor(safe_index[:, start : start + chunk])
            selected = hidden[rows, window]
            logits = loaded.lm_head(selected).float()
            if softcap > 0:
                logits = torch.tanh(logits / softcap) * softcap
            targets = device_tokens[rows, window + 1]
            gathered = torch.log_softmax(logits, dim=-1).gather(-1, targets[..., None]).squeeze(-1)
            stop = start + window.shape[1]
            losses[:, start:stop] = np.where(
                evaluable[:, start:stop], (-gathered).cpu().numpy(), np.nan
            )
    return losses


def sequence_nll(
    loaded: LoadedModel,
    token_batch: np.ndarray,
    positions: np.ndarray,
    *,
    layer_index: int = PRIMARY_LAYER,
    patch: Callable[[torch.Tensor], torch.Tensor] | None = None,
    chunk: int = 16,
) -> np.ndarray:
    """Mean next-token NLL at the requested positions (R1 §16.3, §27.3).

    The final-norm output is tapped and the language-model head applied only at
    the requested positions, so the 256k-wide logit tensor is never
    materialized for the full context.
    """

    tokens = torch.as_tensor(np.asarray(token_batch), dtype=torch.long)
    with residual_taps(loaded, layer_index=layer_index, capture_final_norm=True, patch=patch) as store:
        forward_tokens(loaded, tokens)
        hidden = store["final_norm"]
        batch = tokens.shape[0]
        losses = np.zeros((batch, len(positions)), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(positions), chunk):
                window = np.asarray(positions[start : start + chunk])
                selected = hidden[:, window, :]
                logits = loaded.lm_head(selected).float()
                softcap = float(loaded.metadata.get("final_logit_softcapping", 0.0) or 0.0)
                if softcap > 0:
                    logits = torch.tanh(logits / softcap) * softcap
                targets = tokens.to(loaded.device)[:, window + 1]
                log_probabilities = torch.log_softmax(logits, dim=-1)
                gathered = log_probabilities.gather(-1, targets[..., None]).squeeze(-1)
                losses[:, start : start + len(window)] = (-gathered).cpu().numpy()
    return losses


def capture_recovery_profile(
    loaded: LoadedModel,
    spliced_tokens: np.ndarray,
    reference_tokens: np.ndarray,
    boundary_positions: np.ndarray,
    offsets: np.ndarray,
    basis: np.ndarray,
    *,
    layer_index: int = PRIMARY_LAYER,
) -> dict[str, np.ndarray]:
    """Excess loss and state divergence at offsets AFTER the splice boundary.

    R1.1 amendment 10 item 4.  The confirmatory capture stores nothing between
    the boundary and the source position: ``HISTORY_OFFSETS`` starts at -4 and
    ``NLL_OFFSETS`` at -1, both relative to the source position, so the window
    in which the model adapts to the splice was never observed.  This reads it.

    The discriminating comparison is between two timescales measured on the same
    forward passes: how fast the model's next-token loss returns to its
    unspliced level, against how fast the state difference decays.  A boundary
    shock should relax on the timescale over which local surprise recovers.  A
    remote-history component should persist well past it.  If the state
    difference decays no more slowly than the loss does, the boundary-shock
    reading is not excluded.

    Returns per-sequence, per-offset arrays: the excess next-token loss of the
    spliced run over its own unspliced reference, the full-residual difference
    norm, and the norm of that difference projected onto the aperture.

    No aperture mean is taken: this projects a DIFFERENCE of two residuals, and
    the mean cancels in the subtraction.  Passing one would imply a centring
    step that does not happen.
    """

    spliced = torch.as_tensor(np.asarray(spliced_tokens), dtype=torch.long)
    reference = torch.as_tensor(np.asarray(reference_tokens), dtype=torch.long)
    if spliced.shape != reference.shape:
        raise ValueError("spliced and reference token batches must have the same shape")
    boundaries = np.asarray(boundary_positions, dtype=np.int64)
    window = np.asarray(offsets, dtype=np.int64)
    positions = boundaries[:, None] + window[None, :]
    if positions.min() < 0 or positions.max() >= spliced.shape[1]:
        raise ValueError(
            f"recovery offsets leave the sequence: positions span "
            f"[{int(positions.min())}, {int(positions.max())}] for length {int(spliced.shape[1])}"
        )

    captured: dict[str, np.ndarray] = {}
    residuals: dict[str, torch.Tensor] = {}
    for name, batch in (("spliced", spliced), ("reference", reference)):
        with residual_taps(loaded, layer_index=layer_index, capture_final_norm=True) as store:
            with torch.no_grad():
                loaded.model(batch.to(loaded.device), use_cache=False)
            residuals[name] = store["source"].detach().float().cpu()
            captured[f"nll_{name}"] = _nll_from_hidden(
                loaded, store["final_norm"], batch, boundaries, window
            )

    difference = residuals["spliced"] - residuals["reference"]
    rows = torch.arange(difference.shape[0])[:, None]
    selected = difference[rows, torch.as_tensor(positions)]
    projector = torch.as_tensor(np.asarray(basis), dtype=torch.float32)
    projected = selected @ projector

    captured["excess_nll"] = captured["nll_spliced"] - captured["nll_reference"]
    captured["full_difference_norm"] = selected.norm(dim=-1).numpy().astype(np.float32)
    captured["projected_difference_norm"] = projected.norm(dim=-1).numpy().astype(np.float32)
    captured["offsets"] = window
    return captured
