#!/usr/bin/env python
"""R1.9 content-vs-document-fingerprint identification on frozen R1.8 data."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.persistent_state.slow_semantic.r18 import plus_one_p, readout_overlap
from src.persistent_state.slow_semantic.r19 import final_outcome, fingerprint_frame, one_hot_frozen
from src.persistent_state.slow_semantic.readout import cross_entropy_rows, decision_logits, fit_offset_multinomial


POSITIONS=(256,320,384,448,512,576,640,704,768,832,896,960)
SHUFFLE_SEED=740111


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_classes(tokens: np.ndarray) -> np.ndarray:
    return np.column_stack([tokens%2,tokens%3,tokens%5,tokens%7]).astype(np.float64)


def load_basis(path: Path, key: str="basis") -> np.ndarray:
    if path.suffix==".npy": result=np.load(path)
    else:
        with np.load(path,allow_pickle=False) as z: result=z[key]
    result=np.asarray(result,dtype=np.float64)
    if result.shape!=(2304,31): raise ValueError(f"invalid basis {path}: {result.shape}")
    if np.linalg.norm(result.T@result-np.eye(31),ord=2)>1e-4: raise ValueError(f"non-orthonormal basis {path}")
    return result


def initialize(args) -> int:
    run=args.run_dir
    for rel in ("fingerprint","topic","geometry","diagnostics","provenance","profiles"):
        (run/rel).mkdir(parents=True,exist_ok=True)
    required={
        "semantic_residuals":args.semantic_run/"capture/residuals.npy",
        "semantic_local_context":args.semantic_run/"capture/local_embedding_means.npy",
        "semantic_labels":args.semantic_run/"labels/document_labels.parquet",
        "semantic_tokens":args.semantic_run/"labels/token_ids.npy",
        "semantic_pca":args.semantic_run/"supervised/residual_pca256.npz",
        "semantic_random_bases":args.semantic_run/"supervised/random_bases.npz",
        "canonical":args.canonical_basis,
        "r17_A":args.r17_run/"split_A/q31.npy", "r17_B":args.r17_run/"split_B/q31.npy",
        "r17_decision":args.r17_run/"decision.json",
        "r18_decision":args.r18_run/"decision.json",
        "r18_gains":args.r18_run/"semantics/semantic_gain_summary.csv",
    }
    missing=[str(p) for p in required.values() if not p.exists()]
    if missing: raise FileNotFoundError(f"missing frozen inputs: {missing}")
    r17=json.loads(required["r17_decision"].read_text());r18=json.loads(required["r18_decision"].read_text())
    if r17.get("final_outcome")!="SIGNED_TICA_IDENTIFIABLE": raise ValueError("R1.7 licensing outcome changed")
    if r18.get("overall_scientific_outcome")!="DOCUMENT_LEVEL_SEMANTIC_STATE": raise ValueError("R1.8 licensing outcome changed")
    if "content-vs-source/style" not in r18.get("authorized_next_branch",""): raise ValueError("R1.8 did not authorize R1.9")
    residual=np.load(required["semantic_residuals"],mmap_mode="r")
    if residual.shape!=(1200,12,2304): raise ValueError(f"semantic residual shape changed: {residual.shape}")
    frame=pd.read_parquet(required["semantic_labels"])
    if frame.split.value_counts().to_dict()!={"fit":800,"validation":200,"test":200}: raise ValueError("semantic split changed")
    shutil.copy2(args.spec,run/"provenance"/args.spec.name)
    hashes={key:sha256_file(path) for key,path in required.items() if path.is_file() and path.stat().st_size<20_000_000}
    save_json(run/"r1_9_config.json",{
        "topic_target":"topic","topic_classes":"fit support >=30, identical to R1.8",
        "old_nuisance":"R1.8 position, token-class, and train-fit local-context PCA features",
        "conditional_nuisance":"old nuisance plus frozen document fingerprint",
        "fingerprint_families":["source/domain","genre/register","format/template"],
        "categorical_minimum_fit_support":20,"baseline_c":0.001,"slow_representation_l2":0.1,
        "fingerprint_diagnostic_c":0.01,"fingerprint_diagnostic_class_weight":"balanced",
        "semantic_shuffle_replicates":100,"semantic_shuffle_seed":SHUFFLE_SEED,
        "fingerprint_adequacy":"URL and host coverage >=95%, all three families available, and >=2 supported coarse source types",
        "fingerprint_topic_strong":"fingerprint-conditioned baseline improves sealed-test CE over the old nuisance baseline (>0 nats)",
        "primary_gate":"A and B conditional gains both exceed random-31 Q95",
    })
    commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip() or "unknown"
    save_json(run/"provenance.json",{"status":"PASS","implementation_commit":commit,"input_hashes":hashes,
        "new_labels":False,"new_capture":False,"r17_outcome":r17["final_outcome"],"r18_outcome":r18["overall_scientific_outcome"]})
    return 0


def semantic_design(semantic_run: Path):
    frame=pd.read_parquet(semantic_run/"labels/document_labels.parquet")
    residual=np.asarray(np.load(semantic_run/"capture/residuals.npy",mmap_mode="r"),dtype=np.float64)
    local=np.asarray(np.load(semantic_run/"capture/local_embedding_means.npy",mmap_mode="r"),dtype=np.float64)
    tokens=np.load(semantic_run/"labels/token_ids.npy",mmap_mode="r")
    n_docs,n_pos,d=residual.shape;split=frame.split.astype(str).to_numpy();fit_docs=np.flatnonzero(split=="fit");test_docs=np.flatnonzero(split=="test")
    fit_rows=(fit_docs[:,None]*n_pos+np.arange(n_pos)[None]).ravel();flat=residual.reshape(-1,d);mean=flat[fit_rows].mean(0);centered=flat-mean
    local_flat=local.reshape(-1,d);lpca=PCA(n_components=32,svd_solver="randomized",random_state=730257).fit(local_flat[fit_rows]);local_scores=lpca.transform(local_flat)
    pos=np.tile(np.eye(n_pos),(n_docs,1));current=np.asarray(tokens[:,POSITIONS]).reshape(-1);raw=np.column_stack([pos,token_classes(current),local_scores]);old_nuisance=StandardScaler().fit(raw[fit_rows]).transform(raw)
    labels=frame.topic.astype(str).to_numpy();counts=pd.Series(labels[fit_docs]).value_counts();classes=sorted(counts[counts>=30].index.tolist());mapping={v:i for i,v in enumerate(classes)};y_doc=np.asarray([mapping.get(v,-1) for v in labels]);y=np.repeat(y_doc,n_pos)
    allowed_fit=np.intersect1d(fit_docs,np.flatnonzero(y_doc>=0));allowed_test=np.intersect1d(test_docs,np.flatnonzero(y_doc>=0));tr=(allowed_fit[:,None]*n_pos+np.arange(n_pos)[None]).ravel();te=(allowed_test[:,None]*n_pos+np.arange(n_pos)[None]).ravel()
    return frame,residual,centered,old_nuisance,y,y_doc,fit_docs,test_docs,tr,te,classes


def build_fingerprint(frame: pd.DataFrame, fit_docs: np.ndarray, old_nuisance: np.ndarray):
    fp,schema=fingerprint_frame(frame,fit_docs);numeric=fp[schema["numeric"]].to_numpy(float)
    scaler=StandardScaler().fit(numeric[fit_docs]);numeric_scaled=scaler.transform(numeric)
    categorical=[]
    for column in ("source_type","host_suffix_class","register","formatting"):
        categorical.append(one_hot_frozen(fp[column].to_numpy(),schema["categorical"][column]))
    document_design=np.column_stack([numeric_scaled,*categorical])
    repeated=np.repeat(document_design,12,axis=0)
    return fp,schema,document_design,np.column_stack([old_nuisance,repeated])


def baseline_logits(nuisance,y,tr,te,train_y=None):
    yy=y[tr] if train_y is None else train_y
    model=LogisticRegression(C=.001,max_iter=600,solver="lbfgs",tol=1e-7).fit(nuisance[tr],yy)
    classes=len(np.unique(yy));return decision_logits(model,nuisance[tr],classes),decision_logits(model,nuisance[te],classes)


def conditional_fit(nuisance,x,y,tr,te,train_y=None):
    yy=y[tr] if train_y is None else train_y;base_tr,base_te=baseline_logits(nuisance,y,tr,te,yy)
    scaler=StandardScaler().fit(x[tr]);xs=scaler.transform(x);fit=fit_offset_multinomial(base_tr,xs[tr],yy,.1)
    if not fit.success: raise RuntimeError("frozen topic offset fit did not converge")
    gain=float(np.mean(cross_entropy_rows(base_te,y[te])-cross_entropy_rows(fit.logits(base_te,xs[te]),y[te])))
    return gain,fit.coefficients/scaler.scale_[:,None],float(np.mean(cross_entropy_rows(base_te,y[te])))


def nuisance_decoding(bases,centered,fp,fit_docs,test_docs):
    rows=[];doc_features={label:(centered@basis).reshape(len(fp),12,31).mean(1) for label,basis in bases.items()}
    for family in ("source_type","register","formatting"):
        y=fp[family].astype(str).to_numpy();supported=sorted(pd.Series(y[fit_docs]).value_counts().loc[lambda s:s>=20].index.tolist())
        use_fit=np.asarray([i for i in fit_docs if y[i] in supported]);use_test=np.asarray([i for i in test_docs if y[i] in supported])
        if len(supported)<2 or not len(use_test):
            rows.append({"family":family,"basis":"NOT_EVALUABLE","classes":len(supported),"documents":len(use_test),"accuracy":np.nan,"balanced_accuracy":np.nan,"macro_f1":np.nan});continue
        mapping={v:i for i,v in enumerate(supported)}
        for label,x in doc_features.items():
            scaler=StandardScaler().fit(x[use_fit]);model=LogisticRegression(C=.01,class_weight="balanced",max_iter=600,solver="lbfgs",tol=1e-7).fit(scaler.transform(x[use_fit]),[mapping[y[i]] for i in use_fit]);pred=model.predict(scaler.transform(x[use_test]));truth=np.asarray([mapping[y[i]] for i in use_test])
            rows.append({"family":family,"basis":label,"classes":len(supported),"documents":len(use_test),"accuracy":accuracy_score(truth,pred),"balanced_accuracy":balanced_accuracy_score(truth,pred),"macro_f1":f1_score(truth,pred,average="macro",zero_division=0)})
    return pd.DataFrame(rows)


def analyze(args) -> int:
    started=time.perf_counter();shuffle_reps=5 if args.profile else 100
    frame,residual,centered,old_nuisance,y,y_doc,fit_docs,test_docs,tr,te,classes=semantic_design(args.semantic_run)
    fp,schema,fp_document,nuisance=build_fingerprint(frame,fit_docs,old_nuisance)
    canonical=load_basis(args.canonical_basis);qa=load_basis(args.r17_run/"split_A/q31.npy");qb=load_basis(args.r17_run/"split_B/q31.npy")
    with np.load(args.semantic_run/"supervised/residual_pca256.npz",allow_pickle=False) as z:pca=z["directions"].astype(np.float64)[:,:31]
    random={}
    with np.load(args.semantic_run/"supervised/random_bases.npz",allow_pickle=False) as z:
        for i in range(20):random[f"random_{i:03d}"]=z[f"rank31_draw{i:03d}"].astype(np.float64)
    bases={"canonical":canonical,"A":qa,"B":qb,"pca":pca,**random};representations={label:centered@basis for label,basis in bases.items()}
    old_tr,old_te=baseline_logits(old_nuisance,y,tr,te);fp_tr,fp_te=baseline_logits(nuisance,y,tr,te)
    old_ce=float(np.mean(cross_entropy_rows(old_te,y[te])));fingerprint_ce=float(np.mean(cross_entropy_rows(fp_te,y[te])));fingerprint_improvement=old_ce-fingerprint_ce
    gains={};ambient={}
    for label in bases:
        gain,coef,_=conditional_fit(nuisance,representations[label],y,tr,te);gains[label]=gain
        if label in {"A","B"}: ambient[label]=bases[label]@coef
    random_values=np.asarray([gains[f"random_{i:03d}"] for i in range(20)]);random_q95=float(np.quantile(random_values,.95));survives=bool(gains["A"]>random_q95 and gains["B"]>random_q95)
    old_gains=pd.read_csv(args.r18_run/"semantics/semantic_gain_summary.csv").set_index("basis").delta_ce.to_dict();retention={label:gains[label]/old_gains[label] for label in gains}
    observed=np.nan;singular=np.asarray([]);rank=0;null=np.asarray([]);shuffle_q95=np.nan;p_value=np.nan;geometry=False
    if survives:
        observed,singular,rank=readout_overlap(ambient["A"],ambient["B"]);rng=np.random.default_rng(SHUFFLE_SEED);fit_doc_labels=y[tr].reshape(-1,12)[:,0];values=[]
        for _ in range(shuffle_reps):
            ya=np.repeat(rng.permutation(fit_doc_labels),12);yb=np.repeat(rng.permutation(fit_doc_labels),12)
            _,ca,_=conditional_fit(nuisance,representations["A"],y,tr,te,ya);_,cb,_=conditional_fit(nuisance,representations["B"],y,tr,te,yb);values.append(readout_overlap(qa@ca,qb@cb)[0])
        null=np.asarray(values);shuffle_q95=float(np.quantile(null,.95));p_value=plus_one_p(observed,null);geometry=bool(observed>shuffle_q95)
    elapsed=time.perf_counter()-started
    if args.profile:
        save_json(args.run_dir/"profiles/profile.json",{"status":"PROFILE_ONLY","shuffle_replicates":shuffle_reps,"seconds":elapsed,"projected_seconds":elapsed*max(1,100/shuffle_reps),"content_gate_in_profile":survives});return 4
    fp.to_parquet(args.run_dir/"fingerprint/fingerprint_labels.parquet",index=False)
    host_counts=frame.host.astype(str).value_counts();adequate=bool(frame.url.notna().mean()>=.95 and frame.host.notna().mean()>=.95 and len(schema["categorical"]["source_type"])>=2)
    qc={"status":"PASS" if adequate else "FAIL","documents":len(frame),"url_coverage":float(frame.url.notna().mean()),"host_coverage":float(frame.host.notna().mean()),"unique_hosts":int(frame.host.nunique()),"singleton_host_fraction":float((host_counts==1).mean()),"source_identity_generalization_limit":"exact-domain effects are weakly estimable because nearly all hosts are singletons","category_counts":{c:fp[c].value_counts().to_dict() for c in ("source_type","host_suffix_class","register","formatting")},"new_labeling":False}
    save_json(args.run_dir/"fingerprint/fingerprint_schema.json",schema);save_json(args.run_dir/"fingerprint/fingerprint_qc.json",qc)
    pd.DataFrame([{"basis":k,"conditional_delta_ce":v,"random_q95":random_q95,"passes_random_q95":v>random_q95} for k,v in gains.items()]).to_csv(args.run_dir/"topic/conditional_gain_summary.csv",index=False)
    pd.DataFrame([{"basis":k,"old_delta_ce":old_gains[k],"conditional_delta_ce":gains[k],"retention_ratio":retention[k]} for k in gains]).to_csv(args.run_dir/"topic/retention_summary.csv",index=False)
    pd.DataFrame([{"model":"old_nuisance","sealed_test_ce":old_ce},{"model":"old_nuisance_plus_fingerprint","sealed_test_ce":fingerprint_ce},{"model":"fingerprint_improvement","sealed_test_ce":fingerprint_improvement}]).to_csv(args.run_dir/"topic/fingerprint_topic_baseline.csv",index=False)
    pd.DataFrame({"replicate":np.arange(len(null)),"conditional_semantic_overlap":null}).to_parquet(args.run_dir/"geometry/conditional_semantic_shuffle_null.parquet",index=False)
    save_json(args.run_dir/"geometry/conditional_semantic_overlap.json",{"status":"RUN" if survives else "NOT_RUN_GATE_CLOSED","observed":None if np.isnan(observed) else observed,"canonical_correlations":singular.tolist(),"effective_discriminant_rank":rank,"shuffle_q95":None if np.isnan(shuffle_q95) else shuffle_q95,"plus_one_p":None if np.isnan(p_value) else p_value,"content_geometry_reproduced":geometry})
    diagnostic_bases={"canonical":canonical,"pca":pca,**random};diagnostics=nuisance_decoding(diagnostic_bases,centered,fp,fit_docs,test_docs);diagnostics.to_csv(args.run_dir/"diagnostics/fingerprint_decoding_summary.csv",index=False)
    both_in_random=bool(gains["A"]<=random_q95 and gains["B"]<=random_q95);outcome,branch=final_outcome(adequate,survives,geometry,both_in_random,fingerprint_improvement)
    decision={"fingerprint_nuisance_families":schema["families"],"fingerprint_adequate":adequate,"fingerprint_baseline_topic_ce":fingerprint_ce,"old_nuisance_topic_ce":old_ce,"fingerprint_topic_ce_improvement":fingerprint_improvement,"conditional_semantic_gains":gains,"random_q95":random_q95,"effect_retention_ratios":retention,"content_survives_fingerprint":survives,"conditional_semantic_geometry_overlap":None if np.isnan(observed) else observed,"semantic_shuffle_q95":None if np.isnan(shuffle_q95) else shuffle_q95,"semantic_plus_one_p":None if np.isnan(p_value) else p_value,"content_geometry_reproduced":geometry,"final_outcome":outcome,"next_authorized_branch":branch}
    save_json(args.run_dir/"decision.json",decision)
    (args.run_dir/"r1_9_report.md").write_text(f"# R1.9 Content vs Document-Fingerprint Identification\n\nOutcome: `{outcome}`.\n\nFingerprint baseline changed sealed-test topic CE by {fingerprint_improvement:+.6f} nats. Conditional A/B gains: {gains['A']:.6f}/{gains['B']:.6f}; random Q95: {random_q95:.6f}. Retention A/B: {retention['A']:.3f}/{retention['B']:.3f}.\n\nConditional geometry overlap: {observed if not np.isnan(observed) else 'not run'}; shuffle Q95: {shuffle_q95 if not np.isnan(shuffle_q95) else 'not run'}; p={p_value if not np.isnan(p_value) else 'not run'}.\n\nSource limitation: exact host identity has {qc['singleton_host_fraction']:.1%} singleton prevalence, so the claim is conditional on measured coarse source, register, and structural fingerprint rather than exhaustive website identity.\n")
    return 0


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("initialize","analyze"));p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--semantic-run",type=Path,required=True);p.add_argument("--r17-run",type=Path,required=True);p.add_argument("--r18-run",type=Path,required=True);p.add_argument("--canonical-basis",type=Path,required=True);p.add_argument("--spec",type=Path,required=True);p.add_argument("--profile",action="store_true");args=p.parse_args();return initialize(args) if args.stage=="initialize" else analyze(args)


if __name__=="__main__": raise SystemExit(main())
