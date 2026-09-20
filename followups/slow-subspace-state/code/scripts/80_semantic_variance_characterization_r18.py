#!/usr/bin/env python
"""R1.8 basis-invariant variance, semantic, and axis characterization."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.persistent_state.slow_semantic.r18 import (
    axis_decision, bootstrap_variance, local_eigengap, match_axes,
    plus_one_p, readout_overlap, variance_components,
)
from src.persistent_state.slow_semantic.readout import (
    cross_entropy_rows, decision_logits, fit_offset_multinomial,
)
from src.persistent_state.slow_semantic.r17 import save_json, sha256_file


POSITIONS = (256,320,384,448,512,576,640,704,768,832,896,960)
BOOTSTRAP_SEED = 730253
SHUFFLE_SEED = 731803


def token_classes(tokens: np.ndarray) -> np.ndarray:
    return np.column_stack([tokens%2,tokens%3,tokens%5,tokens%7]).astype(np.float64)


def load_basis(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy": result=np.load(path)
    else:
        with np.load(path,allow_pickle=False) as z: result=z[key]
    result=np.asarray(result,dtype=np.float64)
    if result.shape!=(2304,31):raise ValueError(f"invalid basis {path}: {result.shape}")
    if np.linalg.norm(result.T@result-np.eye(31),ord=2)>1e-4:raise ValueError(f"non-orthonormal basis {path}")
    return result


def initialize(args)->int:
    run=args.run_dir
    for rel in ("variance","semantics","axes","provenance","profiles"): (run/rel).mkdir(parents=True,exist_ok=True)
    required={
        "semantic_residuals":args.semantic_run/"capture/residuals.npy",
        "semantic_local_context":args.semantic_run/"capture/local_embedding_means.npy",
        "semantic_labels":args.semantic_run/"labels/document_labels.parquet",
        "semantic_tokens":args.semantic_run/"labels/token_ids.npy",
        "semantic_procedure":args.semantic_run/"supervised/procedure_freeze.json",
        "semantic_pca":args.semantic_run/"supervised/residual_pca256.npz",
        "semantic_random_bases":args.semantic_run/"supervised/random_bases.npz",
        "semantic_prior_gains":args.semantic_run/"supervised/semantic_gain_by_target_rank.parquet",
        "canonical":args.canonical_basis,
        "r17_A":args.r17_run/"split_A/q31.npy", "r17_B":args.r17_run/"split_B/q31.npy",
        "r17_A_candidates":args.r17_run/"split_A/candidates.npz", "r17_B_candidates":args.r17_run/"split_B/candidates.npz",
        "r17_A_selected":args.r17_run/"split_A/selected_probes.json", "r17_B_selected":args.r17_run/"split_B/selected_probes.json",
        "r17_decision":args.r17_run/"decision.json",
    }
    missing=[str(p) for p in required.values() if not p.exists()]
    if missing:raise FileNotFoundError(f"missing frozen inputs: {missing}")
    decision=json.loads((args.r17_run/"decision.json").read_text())
    if decision.get("final_outcome")!="SIGNED_TICA_IDENTIFIABLE":raise ValueError("R1.7 licensing outcome changed")
    residual=np.load(required["semantic_residuals"],mmap_mode="r")
    if residual.shape!=(1200,12,2304):raise ValueError(f"semantic residual shape changed: {residual.shape}")
    frame=pd.read_parquet(required["semantic_labels"])
    if frame.split.value_counts().to_dict()!={"fit":800,"validation":200,"test":200}:raise ValueError("semantic split changed")
    shutil.copy2(args.spec,run/"provenance"/args.spec.name)
    hashes={key:sha256_file(path) for key,path in required.items() if path.is_file() and path.stat().st_size<20_000_000}
    save_json(run/"r1_8_config.json",{
        "variance_bootstrap_replicates":200,"variance_bootstrap_seed":BOOTSTRAP_SEED,
        "semantic_target":"topic","baseline_c":0.001,"slow_representation_l2":0.1,
        "semantic_shuffle_replicates":100,"semantic_shuffle_seed":SHUFFLE_SEED,
        "axis_count":10,"axis_cosine_threshold":0.80,"axis_gate":"at least 3 of first 5 matched axes",
        "document_level_operationalization":"95% document-bootstrap lower bound of median(slow R_between)-median(random R_between) exceeds zero",
        "dynamic_operationalization":"registered semantic gates pass but the registered between-vs-random contrast is not clearly positive",
    })
    commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip() or "unknown"
    save_json(run/"provenance.json",{"status":"PASS","implementation_commit":commit,"input_hashes":hashes,"new_labels":False,"new_capture":False,"r17_outcome":decision["final_outcome"]})
    return 0


def semantic_design(semantic_run: Path):
    frame=pd.read_parquet(semantic_run/"labels/document_labels.parquet")
    residual=np.asarray(np.load(semantic_run/"capture/residuals.npy",mmap_mode="r"),dtype=np.float64)
    local=np.asarray(np.load(semantic_run/"capture/local_embedding_means.npy",mmap_mode="r"),dtype=np.float64)
    tokens=np.load(semantic_run/"labels/token_ids.npy",mmap_mode="r")
    n_docs,n_pos,d=residual.shape;split=frame.split.astype(str).to_numpy();fit_docs=np.flatnonzero(split=="fit");test_docs=np.flatnonzero(split=="test")
    fit_rows=(fit_docs[:,None]*n_pos+np.arange(n_pos)[None]).ravel();flat=residual.reshape(-1,d);mean=flat[fit_rows].mean(0);centered=flat-mean
    local_flat=local.reshape(-1,d);lpca=PCA(n_components=32,svd_solver="randomized",random_state=730257).fit(local_flat[fit_rows]);local_scores=lpca.transform(local_flat)
    pos=np.tile(np.eye(n_pos),(n_docs,1));current=np.asarray(tokens[:,POSITIONS]).reshape(-1);raw=np.column_stack([pos,token_classes(current),local_scores]);nuisance=StandardScaler().fit(raw[fit_rows]).transform(raw)
    labels=frame.topic.astype(str).to_numpy();counts=pd.Series(labels[fit_docs]).value_counts();classes=sorted(counts[counts>=30].index.tolist());mapping={v:i for i,v in enumerate(classes)};y_doc=np.asarray([mapping.get(v,-1) for v in labels]);y=np.repeat(y_doc,n_pos)
    allowed_fit=np.intersect1d(fit_docs,np.flatnonzero(y_doc>=0));allowed_test=np.intersect1d(test_docs,np.flatnonzero(y_doc>=0));tr=(allowed_fit[:,None]*n_pos+np.arange(n_pos)[None]).ravel();te=(allowed_test[:,None]*n_pos+np.arange(n_pos)[None]).ravel()
    return frame,residual,centered,nuisance,y,allowed_fit,allowed_test,tr,te,classes


def fit_semantic(nuisance,x,y,tr,te,baseline_c=.001,l2=.1,train_y=None):
    yy=y[tr] if train_y is None else train_y
    baseline=LogisticRegression(C=baseline_c,max_iter=600,solver="lbfgs",tol=1e-7).fit(nuisance[tr],yy);classes=len(np.unique(yy));base_tr=decision_logits(baseline,nuisance[tr],classes);base_te=decision_logits(baseline,nuisance[te],classes)
    scaler=StandardScaler().fit(x[tr]);xs=scaler.transform(x);fit=fit_offset_multinomial(base_tr,xs[tr],yy,l2)
    if not fit.success:raise RuntimeError("frozen topic offset fit did not converge")
    gain=float(np.mean(cross_entropy_rows(base_te,y[te])-cross_entropy_rows(fit.logits(base_te,xs[te]),y[te])))
    return gain,fit.coefficients/scaler.scale_[:,None]


def selected_tica(run: Path,label: str):
    folder=run/f"split_{label}";selected=json.loads((folder/"selected_probes.json").read_text());selected=[x for x in selected if x["probe_family"]=="time_lagged"][:10]
    with np.load(folder/"candidates.npz",allow_pickle=False) as z:
        ids=z["probe_ids"].astype(str);directions=z["directions"].astype(np.float64);eigen=z["time_lagged_generalized_eigenvalues"].astype(np.float64)
    lookup={v:i for i,v in enumerate(ids)};candidate_indices=np.asarray([int(x["probe_id"].rsplit("_",1)[1]) for x in selected]);return directions[[lookup[x["probe_id"]] for x in selected]],candidate_indices,eigen,selected


def analyze(args)->int:
    t0=time.perf_counter();reps_boot=20 if args.profile else 200;reps_shuffle=5 if args.profile else 100
    frame,residual,centered,nuisance,y,fit_docs,test_docs,tr,te,classes=semantic_design(args.semantic_run)
    canonical=load_basis(args.canonical_basis,"basis");qa=load_basis(args.r17_run/"split_A/q31.npy","");qb=load_basis(args.r17_run/"split_B/q31.npy","")
    with np.load(args.semantic_run/"supervised/residual_pca256.npz",allow_pickle=False) as z:pca=z["directions"].astype(np.float64)[:,:31]
    random={}
    with np.load(args.semantic_run/"supervised/random_bases.npz",allow_pickle=False) as z:
        for i in range(20):random[f"random_{i:03d}"]=z[f"rank31_draw{i:03d}"].astype(np.float64)
    bases={"canonical":canonical,"A":qa,"B":qb,"pca":pca,**random}
    coordinates={label:np.einsum("dtm,mk->dtk",residual,basis,optimize=True).astype(np.float32) for label,basis in bases.items()}
    point_rows=[]
    for label,value in coordinates.items():point_rows.append({"basis":label,**variance_components(value)})
    point=pd.DataFrame(point_rows);boot,boot_summary=bootstrap_variance(coordinates,list(random),reps_boot,BOOTSTRAP_SEED)
    prior=pd.read_parquet(args.semantic_run/"supervised/semantic_gain_by_target_rank.parquet");prior=prior[(prior.target=="topic")&(prior["rank"]==31)&(prior.split=="test")];prior_map=dict(zip(prior.family,prior.delta_h));random_gains=np.asarray([prior_map[f"random_31_{i:03d}"] for i in range(20)],float);random_q95=float(np.quantile(random_gains,.95))
    semantic={};ambient={}
    representations = {label: centered @ basis for label,basis in (("canonical",canonical),("A",qa),("B",qb),("pca",pca))}
    for label,basis in (("canonical",canonical),("A",qa),("B",qb),("pca",pca)):
        gain,coef=fit_semantic(nuisance,representations[label],y,tr,te,l2=.1);semantic[label]=gain;ambient[label]=basis@coef
    if abs(semantic["canonical"]-prior_map["slow_31"])>1e-6 or abs(semantic["pca"]-prior_map["pca_31"])>1e-6:raise ValueError("frozen semantic implementation failed to reproduce prior canonical/PCA gains")
    observed,singular,rank=readout_overlap(ambient["A"],ambient["B"]);rng=np.random.default_rng(SHUFFLE_SEED);null=[];fit_doc_labels=y.reshape(len(frame),12)[:,0]
    for _ in range(reps_shuffle):
        ya=np.repeat(rng.permutation(fit_doc_labels[fit_docs]),12);yb=np.repeat(rng.permutation(fit_doc_labels[fit_docs]),12)
        _,ca=fit_semantic(nuisance,representations["A"],y,tr,te,l2=.1,train_y=ya);_,cb=fit_semantic(nuisance,representations["B"],y,tr,te,l2=.1,train_y=yb);null.append(readout_overlap(qa@ca,qb@cb)[0])
    null=np.asarray(null);shuffle_q95=float(np.quantile(null,.95));p_value=plus_one_p(observed,null)
    da,ia,ea,sa=selected_tica(args.r17_run,"A");db,ib,eb,sb=selected_tica(args.r17_run,"B");matches=match_axes(da,db);matches["a_candidate_index"]=ia[matches.a_selected_rank.to_numpy()-1];matches["b_candidate_index"]=ib[matches.b_selected_rank.to_numpy()-1];matches["a_probe_id"]=[sa[i-1]["probe_id"] for i in matches.a_selected_rank];matches["b_probe_id"]=[sb[i-1]["probe_id"] for i in matches.b_selected_rank];matches["a_eigenvalue"]=[ea[i] for i in matches.a_candidate_index];matches["b_eigenvalue"]=[eb[i] for i in matches.b_candidate_index];matches["a_local_eigengap"]=[local_eigengap(ea,i) for i in matches.a_candidate_index];matches["b_local_eigengap"]=[local_eigengap(eb,i) for i in matches.b_candidate_index];axis_gate=axis_decision(matches)
    if args.profile:
        save_json(args.run_dir/"profiles/profile.json",{"status":"PROFILE_ONLY","bootstrap_replicates":reps_boot,"shuffle_replicates":reps_shuffle,"seconds":time.perf_counter()-t0,"projected_seconds":(time.perf_counter()-t0)*max(200/reps_boot,100/reps_shuffle)});return 4
    point.to_csv(args.run_dir/"variance/variance_decomposition.csv",index=False);boot.to_parquet(args.run_dir/"variance/variance_bootstrap.parquet",index=False);boot_summary.to_csv(args.run_dir/"variance/variance_bootstrap_summary.csv",index=False)
    gains=pd.DataFrame([{"basis":k,"delta_ce":v} for k,v in semantic.items()]+[{"basis":f"random_{i:03d}","delta_ce":v} for i,v in enumerate(random_gains)]);gains.to_csv(args.run_dir/"semantics/semantic_gain_summary.csv",index=False)
    pd.DataFrame({"replicate":np.arange(100),"semantic_readout_overlap":null}).to_parquet(args.run_dir/"semantics/semantic_shuffle_null.parquet",index=False)
    semantic_reproduced=bool(semantic["A"]>random_q95 and semantic["B"]>random_q95);geometry_reproduced=bool(observed>shuffle_q95)
    save_json(args.run_dir/"semantics/semantic_readout_overlap.json",{"observed":observed,"canonical_correlations":singular.tolist(),"effective_discriminant_rank":rank,"shuffle_q95":shuffle_q95,"plus_one_p":p_value,"semantics_reproduced":semantic_reproduced,"semantic_geometry_reproduced":geometry_reproduced})
    matches.to_csv(args.run_dir/"axes/axis_matching.csv",index=False);matches[["a_probe_id","b_probe_id","a_eigenvalue","b_eigenvalue","a_local_eigengap","b_local_eigengap"]].to_csv(args.run_dir/"axes/eigengap_summary.csv",index=False);save_json(args.run_dir/"axes/axis_decision.json",{"decision":axis_gate,"stable_top5":int((matches.head(5).matched_abs_cosine>=.8).sum()),"top5_cosines":matches.head(5).matched_abs_cosine.tolist()})
    contrast=boot_summary[(boot_summary.basis=="slow")&(boot_summary.statistic=="slow_minus_random_median")].iloc[0];elevated=bool(contrast.ci_low>0)
    if not (semantic_reproduced and geometry_reproduced):outcome="SEMANTICS_NOT_STABLE";branch="stop semantic extrapolation; diagnose current topic task and labels"
    elif elevated:outcome="DOCUMENT_LEVEL_SEMANTIC_STATE";branch="one compact content-vs-source/style disentangling experiment"
    else:outcome="DYNAMIC_SEMANTIC_STATE_PLAUSIBLE";branch="one within-document dynamic semantic-state labeling experiment"
    decision={"between_document_variance":{row.basis:float(row.r_between) for row in point.itertuples()},"slow_minus_random_median":contrast.to_dict(),"semantic_gains":{**semantic,"random":random_gains.tolist()},"random_q95":random_q95,"semantic_readout_overlap":observed,"semantic_shuffle_q95":shuffle_q95,"semantic_plus_one_p":p_value,"semantics_gate":"SEMANTICS_REPRODUCED" if semantic_reproduced else "SEMANTICS_NOT_REPRODUCED","semantic_geometry_gate":"SEMANTIC_GEOMETRY_REPRODUCED" if geometry_reproduced else "SEMANTIC_GEOMETRY_NOT_REPRODUCED","top5_axis_matched_cosines":matches.head(5).matched_abs_cosine.tolist(),"axis_gate":axis_gate,"overall_scientific_outcome":outcome,"authorized_next_branch":branch}
    save_json(args.run_dir/"decision.json",decision)
    (args.run_dir/"r1_8_report.md").write_text(f"# R1.8 Slow-Subspace Semantic and Variance Characterization\n\nOutcome: `{outcome}`. Axis status: `{axis_gate}`.\n\nA/B topic gains: {semantic['A']:.6f}/{semantic['B']:.6f} nats; random Q95: {random_q95:.6f}. Semantic readout overlap: {observed:.4f} (shuffle Q95 {shuffle_q95:.4f}, p={p_value:.4f}).\n\nMedian slow R_between minus random median: {contrast.point:.4f} (95% CI [{contrast.ci_low:.4f}, {contrast.ci_high:.4f}]).\n")
    return 0


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("initialize","analyze"));p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--semantic-run",type=Path,required=True);p.add_argument("--r17-run",type=Path,required=True);p.add_argument("--canonical-basis",type=Path,required=True);p.add_argument("--spec",type=Path,required=True);p.add_argument("--profile",action="store_true");args=p.parse_args();return initialize(args) if args.stage=="initialize" else analyze(args)
if __name__=="__main__":raise SystemExit(main())
