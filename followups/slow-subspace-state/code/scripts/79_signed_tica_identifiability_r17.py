#!/usr/bin/env python
"""R1.7 faithful signed-TICA split-fit identifiability pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.persistent_state.autocorr.estimators import AutocorrAccumulator, build_timescale_table, compute_document_autocorr
from src.persistent_state.residuals.pca import fit_residual_pca
from src.persistent_state.residuals.probes import ResidualProbeSet, probe_set_from_directions
from src.persistent_state.residuals.time_lagged import fit_time_lagged_from_covariances
from src.persistent_state.slow_semantic.r17 import (
    EXPECTED_CANONICAL_SHA256, EXPECTED_CONFIG_SHA256, LAGS, RANDOM_IN_SPAN_COUNT,
    RANDOM_IN_SPAN_SEED, RANKS, SPLIT_SEED, candidate_probe_set,
    combine_raw_statistics, frozen_half_indices, frozen_outcome,
    functional_classification, geometry_classification, random_in_span,
    raw_statistics_to_centered, sampled_original_positions, save_json,
    select_nested_bases, sha256_file, subspace_overlap, summarize_tau,
)


MODEL = "google/gemma-2-2b"
REVISION = "main"
HOOK = "blocks.12.hook_resid_post"
D_MODEL = 2304
POSITIONS = 1024
MAX_LAG = 512
PCA_POSITIONS = 1_000_000
PCA_COMPONENTS = 256
TICA_COMPONENTS = 256


def load_documents(pool_path: Path, split_path: Path) -> dict[str, pd.DataFrame]:
    pool = pd.read_parquet(pool_path)
    split = pd.read_parquet(split_path)
    docs = pool.merge(split, on="context_id", how="inner")
    out = {name: docs.loc[docs.split == name].reset_index(drop=True) for name in ("train", "val", "test")}
    if tuple(len(out[x]) for x in ("train", "val", "test")) != (4000, 500, 500):
        raise ValueError("frozen train/validation/test counts changed")
    return out


def token_matrix(frame: pd.DataFrame) -> np.ndarray:
    tokens = np.asarray([list(x) for x in frame.token_ids], dtype=np.int64)
    if tokens.shape != (len(frame), POSITIONS):
        raise ValueError(f"unexpected token matrix {tokens.shape}")
    return tokens


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    source = os.environ.get("SLOW_SEMANTIC_MODEL_PATH", MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=None if source != MODEL else REVISION, local_files_only=source != MODEL
    )
    hf_model = None
    if source != MODEL:
        hf_model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, torch_dtype=dtype)
    model = HookedTransformer.from_pretrained(
        MODEL, revision=REVISION if hf_model is None else None, tokenizer=tokenizer,
        hf_model=hf_model, local_files_only=hf_model is not None, device=device, dtype=dtype,
    ).eval()
    del hf_model
    return model, device


def residual_batch(model, device: str, batch_tokens: np.ndarray) -> torch.Tensor:
    with torch.inference_mode():
        _, cache = model.run_with_cache(
            torch.as_tensor(batch_tokens, dtype=torch.long, device=device),
            names_filter=lambda name: name == HOOK, return_type=None,
        )
    result = cache[HOOK]
    if result.shape[-1] != D_MODEL or not torch.isfinite(result).all():
        raise FloatingPointError(f"invalid residual batch {tuple(result.shape)}")
    return result


def init_gpu_stats(device: str) -> dict[str, torch.Tensor]:
    result = {
        "token_count": torch.zeros((), dtype=torch.float64, device=device),
        "sum": torch.zeros(D_MODEL, dtype=torch.float64, device=device),
        "second_moment_sum": torch.zeros((D_MODEL, D_MODEL), dtype=torch.float64, device=device),
    }
    for lag in LAGS:
        result[f"lag_{lag}_count"] = torch.zeros((), dtype=torch.float64, device=device)
        result[f"lag_{lag}_left_sum"] = torch.zeros(D_MODEL, dtype=torch.float64, device=device)
        result[f"lag_{lag}_right_sum"] = torch.zeros(D_MODEL, dtype=torch.float64, device=device)
        result[f"lag_{lag}_cross_sum"] = torch.zeros((D_MODEL, D_MODEL), dtype=torch.float64, device=device)
    return result


def update_gpu_stats(acc: dict[str, torch.Tensor], sequences: torch.Tensor) -> None:
    x = sequences.float()
    flat = x.reshape(-1, D_MODEL)
    acc["token_count"] += flat.shape[0]
    acc["sum"] += flat.double().sum(0)
    acc["second_moment_sum"] += (flat.T @ flat).double()
    for lag in LAGS:
        left = x[:, :-lag].reshape(-1, D_MODEL)
        right = x[:, lag:].reshape(-1, D_MODEL)
        acc[f"lag_{lag}_count"] += left.shape[0]
        acc[f"lag_{lag}_left_sum"] += left.double().sum(0)
        acc[f"lag_{lag}_right_sum"] += right.double().sum(0)
        acc[f"lag_{lag}_cross_sum"] += (left.T @ right).double()


def cpu_stats(acc: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in acc.items()}


def stage_initialize(args) -> int:
    run = args.run_dir
    for rel in ("provenance", "capture", "full_refit", "split_A", "split_B", "geometry", "timescales"):
        (run / rel).mkdir(parents=True, exist_ok=True)
    if sha256_file(args.canonical_basis) != EXPECTED_CANONICAL_SHA256:
        raise ValueError("canonical Q31 hash mismatch")
    with np.load(args.canonical_basis, allow_pickle=False) as z:
        if str(z["config_hash"].item()) != EXPECTED_CONFIG_SHA256:
            raise ValueError("canonical configuration hash mismatch")
        if z["basis"].shape != (D_MODEL, 31):
            raise ValueError("canonical basis shape changed")
    docs = load_documents(args.context_pool, args.context_split)
    a, b = frozen_half_indices()
    split = pd.DataFrame({
        "train_index": np.arange(4000), "context_id": docs["train"].context_id.astype(str),
        "half": np.where(np.isin(np.arange(4000), a), "A", "B"),
    })
    split.to_parquet(run / "split_indices.parquet", index=False)
    save_json(run / "split_indices.json", {
        "seed": SPLIT_SEED, "partition_source": "new prospective deterministic split; no verified historical 2000/2000 artifact existed",
        "A": a.tolist(), "B": b.tolist(),
    })
    shutil.copy2(args.spec, run / "provenance" / "slow_subspace_signed_tica_identifiability_r1_7.md")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    config = {
        "model": MODEL, "revision": REVISION, "hook": HOOK, "d_model": D_MODEL,
        "documents": {"train": 4000, "validation": 500, "test": 500}, "positions": POSITIONS,
        "lags": list(LAGS), "whitening_pcs": 512, "tica_candidates": 256,
        "ridge_scale": 1e-4, "max_ridge_scale": 1e-1, "condition_threshold": 1e4,
        "pca_positions": PCA_POSITIONS, "pca_components": PCA_COMPONENTS, "pca_seed": 4,
        "ambient_random_candidates": 512, "ambient_random_seed": 4,
        "dedup_abs_cosine_threshold": 0.95, "ranks": list(RANKS),
        "random_in_span_count": RANDOM_IN_SPAN_COUNT, "random_in_span_seed": RANDOM_IN_SPAN_SEED,
        "max_lag": MAX_LAG, "valid_document_threshold": 100, "min_valid_lag_fraction": 0.8,
        "smoothing_width": 5, "split_seed": SPLIT_SEED,
    }
    save_json(run / "r1_7_config.json", config)
    save_json(run / "provenance.json", {
        "status": "PROVENANCE_LOCKED", "canonical_basis_sha256": EXPECTED_CANONICAL_SHA256,
        "canonical_config_sha256": EXPECTED_CONFIG_SHA256,
        "canonical_probe_commit": "3b76a6c04247ca0da7457f379dc2e11e32fb2962",
        "canonical_selection_commit": "06ba01a716cb9b174935b52f0f028e18953260ed",
        "implementation_commit": commit,
        "authoritative_paths": ["src/persistent_state/residuals/time_lagged.py", "src/persistent_state/residuals/pca.py", "src/persistent_state/subspace/residual_geometry.py", "scripts/persistent_state/residual_geometry/05b_fat_subspace_diagnostics.py", "scripts/persistent_state/residual_geometry/78_audit_canonical_q31_signed.py"],
        "signed_evaluator_audit_decision": "CANONICAL_Q31_REPRODUCED",
    })
    return 0


def stage_capture(args) -> int:
    docs = load_documents(args.context_pool, args.context_split)
    tokens = token_matrix(docs["train"])
    split = pd.read_parquet(args.run_dir / "split_indices.parquet")
    half = split.half.astype(str).to_numpy()
    idx_a = np.flatnonzero(half == "A")
    idx_b = np.flatnonzero(half == "B")
    sample_positions = {
        "full": sampled_original_positions(np.arange(4000)),
        "A": sampled_original_positions(idx_a),
        "B": sampled_original_positions(idx_b),
    }
    profile = args.profile_batches is not None
    sample_maps = {}
    if not profile:
        for label in ("full", "A", "B"):
            path = args.run_dir / "capture" / f"pca_sample_{label}.npy"
            if path.exists() and not args.overwrite:
                raise FileExistsError(path)
            sample_maps[label] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(PCA_POSITIONS, D_MODEL)
            )
    model, device = load_model()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    stats = {"A": init_gpu_stats(device), "B": init_gpu_stats(device)}
    timings = []
    limit = len(tokens) if not profile else min(len(tokens), args.profile_batches * args.batch_size)
    for batch_index, start in enumerate(range(0, limit, args.batch_size)):
        stop = min(limit, start + args.batch_size)
        t0 = time.perf_counter()
        residual = residual_batch(model, device, tokens[start:stop])
        for label in ("A", "B"):
            mask = torch.as_tensor(half[start:stop] == label, device=device)
            if bool(mask.any()):
                update_gpu_stats(stats[label], residual[mask])
        written = {}
        if not profile:
            global_start, global_stop = start * POSITIONS, stop * POSITIONS
            flat = residual.reshape(-1, D_MODEL)
            for label, selected in sample_positions.items():
                lo, hi = np.searchsorted(selected, [global_start, global_stop])
                if hi > lo:
                    local = torch.as_tensor(selected[lo:hi] - global_start, device=device)
                    sample_maps[label][lo:hi] = flat[local].float().cpu().numpy()
                written[label] = int(hi - lo)
        torch.cuda.synchronize() if device == "cuda" else None
        row = {"batch": batch_index, "documents": stop-start, "seconds": time.perf_counter()-t0, "sample_rows": written}
        timings.append(row); print(json.dumps(row), flush=True)
    if profile:
        mean = float(np.mean([x["seconds"] for x in timings]))
        save_json(args.run_dir / "capture" / "profile.json", {
            "status": "PROFILE_ONLY", "batches": len(timings), "batch_size": args.batch_size,
            "mean_seconds_per_batch": mean, "projected_capture_seconds": mean * math.ceil(4000/args.batch_size),
            "pca_sample_storage_bytes": 3 * PCA_POSITIONS * D_MODEL * 4, "timings": timings,
        })
        return 4
    for label in ("A", "B"):
        np.savez_compressed(args.run_dir / "capture" / f"raw_statistics_{label}.npz", **cpu_stats(stats[label]))
    for array in sample_maps.values():
        array.flush()
    save_json(args.run_dir / "capture" / "manifest.json", {
        "status": "PASS", "full_forward_passes": math.ceil(4000/args.batch_size),
        "full_forward_passes_per_batch": 1, "raw_statistics": "uncentered additive A/B moments",
        "matrix_multiply_precision": "CUDA FP32 with TF32 disabled; accumulated FP64",
        "pca_samples": {k: {"rows": PCA_POSITIONS, "original_flat_indices_sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k,v in sample_positions.items()},
        "timings": timings,
    })
    return 0


def load_raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def save_fit(run: Path, label: str, time_fit, pca_fit) -> None:
    folder = run / ("full_refit" if label == "full" else f"split_{label}")
    probes = candidate_probe_set(time_fit.probes, pca_fit.probes)
    np.savez_compressed(
        folder / "candidates.npz",
        directions=probes.directions,
        probe_ids=probes.probe_ids.astype(str),
        probe_family=probes.probe_family.astype(str),
        time_lagged_generalized_eigenvalues=time_fit.generalized_eigenvalues,
        pca_explained_variance=pca_fit.eigenvalues,
        time_lagged_mean=time_fit.mean,
        pca_mean=pca_fit.mean,
    )
    save_json(folder / "fit_metadata.json", {
        "label": label, "time_lagged": time_fit.summary, "pca": pca_fit.diagnostics,
        "candidate_family_counts": pd.Series(probes.probe_family.astype(str)).value_counts().to_dict(),
    })


def stage_fit(args) -> int:
    if args.profile_fit:
        sample = np.load(args.run_dir / "capture" / "pca_sample_full.npy", mmap_mode="r")
        profile_rows = min(100_000, len(sample))
        profile_components = 64
        t0 = time.perf_counter()
        fit_residual_pca(sample[:profile_rows], n_components=profile_components, seed=4)
        elapsed = time.perf_counter() - t0
        save_json(args.run_dir / "capture" / "fit_profile.json", {
            "status": "PROFILE_ONLY", "sample_rows": profile_rows,
            "components": profile_components, "seconds": elapsed,
            "projected_seconds_per_full_pca_fit": elapsed * (PCA_POSITIONS/profile_rows) * (PCA_COMPONENTS/profile_components),
            "projected_seconds_for_three_fits": 3 * elapsed * (PCA_POSITIONS/profile_rows) * (PCA_COMPONENTS/profile_components),
            "projection_model": "linear in sampled rows and requested randomized-PCA components",
        })
        return 4
    labels = [args.fit_label] if args.fit_label in ("A", "B") else ["full"]
    raw_a = load_raw(args.run_dir / "capture" / "raw_statistics_A.npz")
    raw_b = load_raw(args.run_dir / "capture" / "raw_statistics_B.npz")
    raw_by_label = {"A": raw_a, "B": raw_b, "full": combine_raw_statistics(raw_a, raw_b)}
    for label in labels:
        mean, sigma0, sigma_lag, pairs, _ = raw_statistics_to_centered(raw_by_label[label])
        time_fit = fit_time_lagged_from_covariances(
            mean=mean, sigma0=sigma0, sigma_lag=sigma_lag, pair_count=pairs,
            lag_set=list(LAGS), whitening_pcs=512, output_directions=TICA_COMPONENTS,
            ridge_scale=1e-4, max_ridge_scale=1e-1, condition_threshold=1e4,
        )
        sample = np.load(args.run_dir / "capture" / f"pca_sample_{label}.npy", mmap_mode="r")
        pca_fit = fit_residual_pca(sample, n_components=PCA_COMPONENTS, seed=4)
        save_fit(args.run_dir, label, time_fit, pca_fit)
    return 0


def load_candidates(run: Path, label: str) -> ResidualProbeSet:
    folder = run / ("full_refit" if label == "full" else f"split_{label}")
    with np.load(folder / "candidates.npz", allow_pickle=False) as z:
        return ResidualProbeSet(z["directions"], z["probe_ids"], z["probe_family"])


def score_blocks(model, device: str, frame: pd.DataFrame, blocks: dict[str, np.ndarray], batch_size: int):
    tokens = token_matrix(frame)
    gpu = {key: torch.as_tensor(value.T, dtype=torch.float32, device=device) for key, value in blocks.items()}
    accum = {key: AutocorrAccumulator(n_features=value.shape[0], max_lag=MAX_LAG) for key,value in blocks.items()}
    timings = []
    for bi,start in enumerate(range(0, len(tokens), batch_size)):
        stop = min(len(tokens), start+batch_size); t0=time.perf_counter()
        residual = residual_batch(model, device, tokens[start:stop]).float()
        for key, directions in gpu.items():
            projection = torch.einsum("btd,dp->btp", residual, directions).cpu().numpy().astype(np.float32)
            accum[key].update(compute_document_autocorr(projection, max_lag=MAX_LAG, estimator="within"))
        timings.append({"batch":bi,"documents":stop-start,"seconds":time.perf_counter()-t0})
    tables={}
    for key,acc in accum.items():
        result=acc.finalize()
        table=build_timescale_table(np.arange(blocks[key].shape[0]), {"within":result}, MAX_LAG, 100, 0.8, 5)
        tables[key]=(table,result.profiles)
    return tables,timings


def candidate_table(probes: ResidualProbeSet, table: pd.DataFrame, profiles: np.ndarray) -> pd.DataFrame:
    out=table.copy(); out["probe_id"]=probes.probe_ids.astype(str); out["probe_family"]=probes.probe_family.astype(str)
    eligible=np.ones(len(out),dtype=bool); tl=out.probe_family.astype(str).to_numpy()=="time_lagged"
    eligible[tl]=np.nanmedian(profiles[tl][:,[8,16,32]],axis=1)>0
    out["gk_positive_validation_persistence"]=eligible
    return out


def save_selected(run: Path, label: str, bases: dict[int,np.ndarray], ranked: pd.DataFrame) -> None:
    folder=run/("full_refit" if label=="full" else f"split_{label}")
    np.save(folder/"q31.npy",bases[31]); ranked.head(31).to_json(folder/"selected_probes.json",orient="records",indent=2)
    ranked.to_parquet(folder/"ranked_deduplicated_candidates.parquet",index=False)
    np.savez_compressed(folder/"nested_bases.npz",**{f"q{k}":v for k,v in bases.items()})


def load_bases(run: Path,label: str)->dict[int,np.ndarray]:
    folder=run/("full_refit" if label=="full" else f"split_{label}")
    with np.load(folder/"nested_bases.npz",allow_pickle=False) as z:return {k:z[f"q{k}"] for k in RANKS}


def stage_rank(args)->int:
    docs=load_documents(args.context_pool,args.context_split); model,device=load_model()
    labels=["full"] if args.rank_scope=="full" else ["A","B"]
    probes={label:load_candidates(args.run_dir,label) for label in labels}
    scored,timings=score_blocks(model,device,docs["val"],{label:p.directions for label,p in probes.items()},args.batch_size)
    bases={}
    for label in labels:
        table=candidate_table(probes[label],*scored[label]); table.to_parquet(args.run_dir/("full_refit" if label=="full" else f"split_{label}")/"validation_candidate_timescales.parquet",index=False)
        bases[label],ranked=select_nested_bases(probes[label],table); save_selected(args.run_dir,label,bases[label],ranked)
    eval_blocks={}
    for label in labels:
        for k in RANKS:
            directions,coeff=random_in_span(bases[label][k]); eval_blocks[f"{label}_k{k}_random"]=directions
            np.savez_compressed(args.run_dir/("full_refit" if label=="full" else f"split_{label}")/f"random_in_span_k{k}.npz",directions=directions,coefficients=coeff)
        eval_blocks[f"{label}_selected_axes"]=bases[label][31].T
    eval_scored,eval_timing=score_blocks(model,device,docs["val"],eval_blocks,args.batch_size)
    rows=[]
    for key,(table,_) in eval_scored.items():
        parts=key.split("_"); table["fit"]=parts[0]; table["split"]="val"; table["direction_kind"]="selected_axis" if key.endswith("selected_axes") else "random_in_span"; table["rank"]=31 if key.endswith("selected_axes") else int(parts[1][1:]); rows.append(table)
    pd.concat(rows,ignore_index=True).to_parquet(args.run_dir/"timescales"/f"{args.rank_scope}_validation.parquet",index=False)
    if args.rank_scope=="full":
        canonical=np.load(args.canonical_basis,allow_pickle=False)["basis"]
        overlap,singular=subspace_overlap(bases["full"][31],canonical)
        summary=summarize_tau(eval_scored["full_k31_random"][0]); ranked=json.loads((args.run_dir/"full_refit"/"selected_probes.json").read_text()); composition=pd.Series([x["probe_family"] for x in ranked[:31]]).value_counts().to_dict()
        passed=overlap>=0.99 and 20<=summary["median"]<=28 and composition=={"time_lagged":30,"pca":1}
        gate={"status":"PASS" if passed else "PROVENANCE_REFIT_FAILED","S31_vs_canonical":overlap,"canonical_correlations":singular.tolist(),"validation_random_in_span":summary,"selected_family_composition":composition,"timings":timings+eval_timing}
        save_json(args.run_dir/"full_refit"/"provenance_gate.json",gate)
        if not passed:
            save_json(args.run_dir/"decision.json",{"provenance_gate":"PROVENANCE_REFIT_FAILED","final_outcome":"PROVENANCE_REFIT_FAILED","reason":gate})
            (args.run_dir/"r1_7_report.md").write_text("# R1.7 Signed-TICA Identifiability\n\nOutcome: `PROVENANCE_REFIT_FAILED`. Split fits were not interpreted.\n")
            return 5
    return 0


def stage_evaluate(args)->int:
    docs=load_documents(args.context_pool,args.context_split); model,device=load_model(); bases={label:load_bases(args.run_dir,label) for label in ("full","A","B")}
    blocks={}
    for label in ("full","A","B"):
        for k in RANKS: blocks[f"{label}_k{k}_random"]=random_in_span(bases[label][k])[0]
        blocks[f"{label}_selected_axes"]=bases[label][31].T
    scored,timings=score_blocks(model,device,docs["test"],blocks,args.batch_size)
    rows=[]
    for key,(table,_) in scored.items():
        parts=key.split("_"); table["fit"]=parts[0];table["split"]="test";table["direction_kind"]="selected_axis" if key.endswith("selected_axes") else "random_in_span";table["rank"]=31 if key.endswith("selected_axes") else int(parts[1][1:]);rows.append(table)
    test=pd.concat(rows,ignore_index=True)
    test.loc[test.direction_kind=="random_in_span"].to_parquet(args.run_dir/"timescales"/"random_in_span_test.parquet",index=False)
    test.loc[test.direction_kind=="selected_axis"].to_parquet(args.run_dir/"timescales"/"selected_axis_test.parquet",index=False)
    val=pd.concat([pd.read_parquet(args.run_dir/"timescales"/"full_validation.parquet"),pd.read_parquet(args.run_dir/"timescales"/"split_validation.parquet")],ignore_index=True)
    val.loc[val.direction_kind=="random_in_span"].to_parquet(args.run_dir/"timescales"/"random_in_span_validation.parquet",index=False)
    val.loc[val.direction_kind=="selected_axis"].to_parquet(args.run_dir/"timescales"/"selected_axis_validation.parquet",index=False)
    split_manifest=pd.read_parquet(args.run_dir/"split_indices.parquet"); train=docs["train"].copy();train["half"]=split_manifest.half.to_numpy(); cross_blocks={label:random_in_span(bases[label][31])[0] for label in ("A","B")}
    cross=[]
    for eval_half in ("A","B"):
        frame=train.loc[train.half==eval_half].reset_index(drop=True); result,t=score_blocks(model,device,frame,cross_blocks,args.batch_size);timings+=t
        for source,(table,_) in result.items():table["source_fit"]=source;table["evaluation_half"]=eval_half;cross.append(table)
    cross_table=pd.concat(cross,ignore_index=True);cross_table.to_parquet(args.run_dir/"timescales"/"crossfit_timescales.parquet",index=False)
    med={(s,e):float(cross_table.loc[(cross_table.source_fit==s)&(cross_table.evaluation_half==e)&cross_table.tau_valid_within.astype(bool),"tau_within"].median()) for s in ("A","B") for e in ("A","B")}
    pd.DataFrame([{"source_fit":s,"median_A":med[(s,"A")],"median_B":med[(s,"B")],"transfer_ratio":med[(s,"B")]/med[(s,"A")] if s=="A" else med[(s,"A")]/med[(s,"B")]} for s in ("A","B")]).to_csv(args.run_dir/"timescales"/"crossfit_summary.csv",index=False)
    selected=pd.concat([val[val.direction_kind=="selected_axis"],test[test.direction_kind=="selected_axis"]]); summary=[]
    for (fit,split_name),group in selected.groupby(["fit","split"]):
        x=group.loc[group.tau_valid_within.astype(bool),"tau_within"].to_numpy(float);q=np.quantile(x,[.25,.5,.75,.9]);summary.append({"fit":fit,"split":split_name,"q25":q[0],"median":q[1],"q75":q[2],"q90":q[3],"max":x.max()})
    pd.DataFrame(summary).to_csv(args.run_dir/"timescales"/"selected_axis_summary.csv",index=False)
    save_json(args.run_dir/"timescales"/"evaluation_timing.json",{"timings":timings})
    return 0


def stage_finalize(args)->int:
    gate=json.loads((args.run_dir/"full_refit"/"provenance_gate.json").read_text())
    if gate["status"]!="PASS":return 0
    bases={label:load_bases(args.run_dir,label) for label in ("A","B")};geometry=[];corr={}
    for k in RANKS:
        score,singular=subspace_overlap(bases["A"][k],bases["B"][k]);geometry.append({"rank":k,"overlap":score,"ambient_expectation":k/D_MODEL});corr[f"k{k}"]=singular
    pd.DataFrame(geometry).to_csv(args.run_dir/"geometry"/"nested_overlap.csv",index=False);np.savez_compressed(args.run_dir/"geometry"/"canonical_correlations.npz",**corr)
    test=pd.read_parquet(args.run_dir/"timescales"/"random_in_span_test.parquet"); summaries={}
    for (fit,k),group in test.groupby(["fit","rank"]):summaries[f"{fit}_k{int(k)}"]=summarize_tau(group)
    m_a=summaries["A_k31"]["median"];m_b=summaries["B_k31"]["median"];s31=next(x["overlap"] for x in geometry if x["rank"]==31);gc=geometry_classification(s31);fc=functional_classification(m_a,m_b);outcome=frozen_outcome(gc,fc)
    decision={"provenance_gate":"PASS","nested_overlaps":{str(x["rank"]):x["overlap"] for x in geometry},"test_random_in_span":summaries,"m_A_test":m_a,"m_B_test":m_b,"geometry_classification":gc,"functional_classification":fc,"final_outcome":outcome}
    save_json(args.run_dir/"decision.json",decision)
    nested_lines = []
    for row in geometry:
        rank = row["rank"]
        nested_lines.append(
            f"- k={rank}: S={row['overlap']:.4f}; "
            f"A/B test medians={summaries[f'A_k{rank}']['median']:.1f}/"
            f"{summaries[f'B_k{rank}']['median']:.1f}"
        )
    nested = "\n".join(nested_lines)
    (args.run_dir/"r1_7_report.md").write_text(f"# R1.7 Faithful Signed-TICA Split-Fit Identifiability\n\nOutcome: `{outcome}`\n\nFull-data provenance gate: PASS (S31={gate['S31_vs_canonical']:.6f}, validation median tau={gate['validation_random_in_span']['median']:.1f}).\n\n## Nested geometry and function\n\n{nested}\n\nGeometry: `{gc}`. Function: `{fc}`.\n")
    removed = []
    for label in ("full", "A", "B"):
        path = args.run_dir / "capture" / f"pca_sample_{label}.npy"
        if path.exists():
            removed.append({"path": str(path), "bytes": path.stat().st_size})
            path.unlink()
    save_json(args.run_dir/"capture"/"temporary_sample_cleanup.json", {
        "status": "REMOVED_AFTER_ALL_FITS_AND_EVALUATIONS_COMPLETED",
        "removed": removed,
        "retained": ["raw_statistics_A.npz", "raw_statistics_B.npz"],
    })
    return 0


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("initialize","capture","fit","rank","evaluate","finalize"));p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--context-pool",type=Path);p.add_argument("--context-split",type=Path);p.add_argument("--canonical-basis",type=Path);p.add_argument("--spec",type=Path);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--profile-batches",type=int);p.add_argument("--profile-fit",action="store_true");p.add_argument("--fit-label",choices=("full","A","B"),default="full");p.add_argument("--rank-scope",choices=("full","split"),default="full");p.add_argument("--overwrite",action="store_true");args=p.parse_args()
    return {"initialize":stage_initialize,"capture":stage_capture,"fit":stage_fit,"rank":stage_rank,"evaluate":stage_evaluate,"finalize":stage_finalize}[args.stage](args)


if __name__=="__main__":raise SystemExit(main())
