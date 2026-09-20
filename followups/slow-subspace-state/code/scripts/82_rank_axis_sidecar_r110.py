#!/usr/bin/env python
"""R1.10 artifact-first rank-extent and interpretable-coordinate sidecar."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
from src.persistent_state.slow_semantic.r110 import (
    RANKS, descriptive_axis_label, eta_squared, extreme_signature, geometry_row,
    orthonormal_nested, random_in_span, rank_curve_decision, summarize_timescales,
    variance_fraction,
)


MODEL="google/gemma-2-2b";REVISION="main";HOOK="blocks.12.hook_resid_post";D_MODEL=2304;SEQ=1024;MAX_LAG=512
POSITIONS=(256,320,384,448,512,576,640,704,768,832,896,960)


def save_json(path: Path,value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def load_r19_module():
    path=Path(__file__).with_name("81_content_vs_fingerprint_r19.py");spec=importlib.util.spec_from_file_location("frozen_r19",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def load_ranked_basis(r17: Path,label: str,rank: int) -> np.ndarray:
    folder=r17/("full_refit" if label=="full" else f"split_{label}")
    ranked=pd.read_parquet(folder/"ranked_deduplicated_candidates.parquet")
    with np.load(folder/"candidates.npz",allow_pickle=False) as z:
        ids=z["probe_ids"].astype(str);directions=z["directions"].astype(np.float64)
    lookup={v:i for i,v in enumerate(ids)};selected=np.stack([directions[lookup[str(x)]] for x in ranked.probe_id.head(rank)])
    return orthonormal_nested(selected,rank)


def load_direction(r17: Path,label: str,probe_id: str) -> np.ndarray:
    folder=r17/("full_refit" if label=="full" else f"split_{label}")
    with np.load(folder/"candidates.npz",allow_pickle=False) as z:
        ids=z["probe_ids"].astype(str);directions=z["directions"].astype(np.float64)
    matches=np.flatnonzero(ids==probe_id)
    if len(matches)!=1:raise ValueError(f"{label}: missing/nonunique {probe_id}")
    return directions[matches[0]]/np.linalg.norm(directions[matches[0]])


def initialize(args) -> int:
    for rel in ("rank_curve","axes","content_contrasts","provenance","profiles"):(args.run_dir/rel).mkdir(parents=True,exist_ok=True)
    required={
        "r17_decision":args.r17_run/"decision.json","r17_full_candidates":args.r17_run/"full_refit/candidates.npz",
        "r17_full_ranked":args.r17_run/"full_refit/ranked_deduplicated_candidates.parquet",
        "r17_A_candidates":args.r17_run/"split_A/candidates.npz","r17_A_ranked":args.r17_run/"split_A/ranked_deduplicated_candidates.parquet",
        "r17_B_candidates":args.r17_run/"split_B/candidates.npz","r17_B_ranked":args.r17_run/"split_B/ranked_deduplicated_candidates.parquet",
        "r18_decision":args.r18_run/"decision.json","r18_axes":args.r18_run/"axes/axis_matching.csv",
        "r19_decision":args.r19_run/"decision.json","r19_fingerprint":args.r19_run/"fingerprint/fingerprint_labels.parquet",
        "semantic_residuals":args.semantic_run/"capture/residuals.npy","semantic_labels":args.semantic_run/"labels/document_labels.parquet",
        "semantic_tokens":args.semantic_run/"labels/token_ids.npy","context_pool":args.context_pool,"context_split":args.context_split,
    }
    missing=[str(p) for p in required.values() if not p.exists()]
    if missing:raise FileNotFoundError(f"missing frozen inputs: {missing}")
    if json.loads(required["r17_decision"].read_text()).get("final_outcome")!="SIGNED_TICA_IDENTIFIABLE":raise ValueError("R1.7 license changed")
    if json.loads(required["r19_decision"].read_text()).get("final_outcome")!="CONTENT_STATE_IDENTIFIED":raise ValueError("R1.9 license changed")
    for label in ("full_refit","split_A","split_B"):
        if len(pd.read_parquet(args.r17_run/label/"ranked_deduplicated_candidates.parquet"))<128:raise ValueError(f"{label} lacks 128 ranked candidates")
    shutil.copy2(args.spec,args.run_dir/"provenance"/args.spec.name)
    save_json(args.run_dir/"r1_10_sidecar_config.json",{
        "status":"FROZEN","ranks":list(RANKS),"canonical_functional_ranks":list(RANKS),"splitfit_functional_ranks":[31,64,128],
        "random_in_span_count":512,"random_in_span_seed":31,"signed_timescale":True,"test_documents":500,
        "axis_count":5,"axis_cosine_gate":.80,"axis_label_min_eta2":.02,"axis_extreme_documents":8,"axis_extreme_tokens":8,
        "contrast_cosine_gate":.80,"contrast_extreme_documents":10,
        "rank_outcome_rules":"BROAD if inherited function>=15 and corrected geometry>=.70 through k64 or k128; COMPACT if full k64<=5 or corrected S64<.40; else GRADED",
        "new_labels":False,"semantic_recapture":False,"heldout_passes":1,
    })
    hashes={k:sha256_file(v) for k,v in required.items() if v.stat().st_size<20_000_000}
    commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip() or "unknown"
    save_json(args.run_dir/"provenance.json",{"status":"PASS","implementation_commit":commit,"input_hashes":hashes,"canonical_object_unchanged":"signed-TICA Q31","larger_rank_status":"descriptive sidecar only"})
    return 0


def categorical_eta(values: np.ndarray,labels: np.ndarray) -> float:return eta_squared(values,np.asarray(labels,str))


def markdown_excerpt(text: str,n: int=240) -> str:return " ".join(str(text).split())[:n].replace("|","\\|")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    columns=list(frame.columns);lines=["| "+" | ".join(map(str,columns))+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    for row in frame.itertuples(index=False,name=None):
        values=[]
        for value in row:
            if isinstance(value,(float,np.floating)):values.append(f"{float(value):.6g}")
            else:values.append(str(value))
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def token_window(tokenizer,tokens: np.ndarray,doc: int,pos_index: int,width: int=14) -> str:
    center=POSITIONS[pos_index];lo=max(0,center-width);hi=min(tokens.shape[1],center+width+1)
    return markdown_excerpt(tokenizer.decode(tokens[doc,lo:hi].tolist(),skip_special_tokens=True),300)


def example_block(title: str,values: np.ndarray,frame: pd.DataFrame,fp: pd.DataFrame,tokens: np.ndarray,tokenizer) -> str:
    doc_mean=values.mean(1);deviation=values-doc_mean[:,None];lines=[f"### {title}",""]
    for heading,indices in (("Highest document means",np.argsort(doc_mean)[-8:][::-1]),("Lowest document means",np.argsort(doc_mean)[:8])):
        lines.extend([f"#### {heading}","","| score | topic | register | source | document | excerpt |","|---:|---|---|---|---|---|"])
        for i in indices:lines.append(f"| {doc_mean[i]:.4f} | {frame.topic.iloc[i]} | {fp.register.iloc[i]} | {fp.source_type.iloc[i]} | `{frame.document_id.iloc[i][:12]}` | {markdown_excerpt(frame.text.iloc[i])} |")
        lines.append("")
    flat=deviation.ravel()
    for heading,indices in (("Highest within-document deviations",np.argsort(flat)[-8:][::-1]),("Lowest within-document deviations",np.argsort(flat)[:8])):
        lines.extend([f"#### {heading}","","| deviation | document | sampled position | local window |","|---:|---|---:|---|"])
        for index in indices:
            doc,pos=divmod(int(index),12);lines.append(f"| {flat[index]:.4f} | `{frame.document_id.iloc[doc][:12]}` | {POSITIONS[pos]} | {token_window(tokenizer,tokens,doc,pos)} |")
        lines.append("")
    return "\n".join(lines)


def prepare(args) -> int:
    bases={}
    for label in ("full","A","B"):
        for rank in RANKS:bases[(label,rank)]=load_ranked_basis(args.r17_run,label,rank)
    np.savez_compressed(args.run_dir/"rank_curve/nested_bases.npz",**{f"{label}_k{rank}":basis for (label,rank),basis in bases.items()})
    geometry=[]
    for rank in RANKS:
        row,singular=geometry_row(bases[("A",rank)],bases[("B",rank)],rank);row["canonical_correlations_json"]=json.dumps(singular.tolist());geometry.append(row)
    pd.DataFrame(geometry).to_csv(args.run_dir/"rank_curve/nested_geometry.csv",index=False)
    ranked=pd.read_parquet(args.r17_run/"full_refit/ranked_deduplicated_candidates.parquet");random_tau=ranked.loc[ranked.probe_family=="random","tau_within"].median();excess=np.maximum(ranked.tau_within.to_numpy(float)-random_tau,0);total=excess.sum();cum=np.cumsum(excess)
    lifetime=[]
    for rank in RANKS:lifetime.append({"rank":rank,"random_tau_baseline":random_tau,"positive_lifetime_excess":float(excess[:rank].sum()),"cumulative_fraction_positive_lifetime_excess":float(cum[rank-1]/total) if total else 0.0})
    pd.DataFrame(lifetime).to_csv(args.run_dir/"rank_curve/lifetime_excess_curve.csv",index=False)

    r19=load_r19_module();frame,residual,centered,old_nuisance,y,y_doc,fit_docs,test_docs,tr,te,classes=r19.semantic_design(args.semantic_run);fp_saved=pd.read_parquet(args.r19_run/"fingerprint/fingerprint_labels.parquet");fp,schema,fp_document,nuisance=r19.build_fingerprint(frame,fit_docs,old_nuisance)
    if not fp[["document_id","source_type","host_suffix_class","register","formatting"]].equals(fp_saved[["document_id","source_type","host_suffix_class","register","formatting"]]):raise ValueError("R1.9 fingerprint reconstruction mismatch")
    tokens=np.load(args.semantic_run/"labels/token_ids.npy",mmap_mode="r");tokenizer=AutoTokenizer.from_pretrained(os.environ["SLOW_SEMANTIC_MODEL_PATH"],local_files_only=True)
    matches=pd.read_csv(args.r18_run/"axes/axis_matching.csv").head(5);axis_rows=[];axis_values={};axis_examples=["# R1.10 Leading TICA-Axis Examples",""]
    axis_metadata=[]
    for j,row in matches.iterrows():
        if row.matched_abs_cosine<.80:continue
        directions={"A":load_direction(args.r17_run,"A",row.a_probe_id),"B":load_direction(args.r17_run,"B",row.b_probe_id),"canonical":load_direction(args.r17_run,"full",row.a_probe_id)}
        if directions["A"]@directions["B"]<0:directions["B"]*=-1
        if directions["A"]@directions["canonical"]<0:directions["canonical"]*=-1
        fit_signatures={}
        for fit,direction in directions.items():
            values=np.einsum("dtm,m->dt",residual,direction,optimize=True);axis_values[(j+1,fit)]=values;means=values.mean(1);between,within=variance_fraction(values)
            use=y_doc>=0;assoc={"topic":categorical_eta(means[use],frame.topic.to_numpy()[use]),"register":categorical_eta(means,fp.register),"source_type":categorical_eta(means,fp.source_type),"formatting":categorical_eta(means,fp.formatting),"position":categorical_eta(values.ravel(),np.tile(np.arange(12),len(frame)))}
            axis_rows.append({"axis":j+1,"fit":fit,"a_probe_id":row.a_probe_id,"b_probe_id":row.b_probe_id,"matched_abs_cosine":row.matched_abs_cosine,"between_variance_fraction":between,"within_variance_fraction":within,"eta2_topic":assoc["topic"],"eta2_topic_fit":categorical_eta(means[fit_docs][y_doc[fit_docs]>=0],frame.topic.to_numpy()[fit_docs][y_doc[fit_docs]>=0]),"eta2_topic_test":categorical_eta(means[test_docs][y_doc[test_docs]>=0],frame.topic.to_numpy()[test_docs][y_doc[test_docs]>=0]),"eta2_register":assoc["register"],"eta2_source_type":assoc["source_type"],"eta2_formatting":assoc["formatting"],"eta2_position":assoc["position"]})
        rows=pd.DataFrame([x for x in axis_rows if x["axis"]==j+1]);families=["topic","register","source_type","formatting","position"];dominants={fit:max(families,key=lambda f:float(rows.loc[rows.fit==fit,f"eta2_{f}"].iloc[0])) for fit in ("canonical","A","B")}
        for fit in ("A","B"):
            values=axis_values[(j+1,fit)];means=values.mean(1);dom=dominants[fit]
            if dom=="position":fit_signatures[fit]=extreme_signature(values.ravel(),np.tile(np.arange(12),len(frame)))
            else:
                labels=frame.topic.to_numpy() if dom=="topic" else fp[dom].to_numpy();mask=(y_doc>=0) if dom=="topic" else np.ones(len(frame),bool);fit_signatures[fit]=extreme_signature(means[mask],labels[mask])
        label=descriptive_axis_label(rows,fit_signatures)
        axis_metadata.append({"axis":j+1,"label":label,"dominant_by_fit":dominants,"extreme_signatures":{k:list(v) for k,v in fit_signatures.items()},"matched_abs_cosine":row.matched_abs_cosine})
        axis_examples.append(example_block(f"Axis {j+1}: {label}",axis_values[(j+1,"canonical")],frame,fp,tokens,tokenizer))
    pd.DataFrame(axis_rows).to_csv(args.run_dir/"axes/axis_quantitative_summary.csv",index=False);save_json(args.run_dir/"axes/axis_labels.json",{"axes":axis_metadata,"policy":"label only when canonical/A/B dominant association agrees, A/B extremes agree, and all eta2>=0.02"});(args.run_dir/"axes/axis_examples.md").write_text("\n".join(axis_examples)+"\n")

    qa=bases[("A",31)];qb=bases[("B",31)];xa=centered@qa;xb=centered@qb;ga,ca,_=r19.conditional_fit(nuisance,xa,y,tr,te);gb,cb,_=r19.conditional_fit(nuisance,xb,y,tr,te);prior=json.loads((args.r19_run/"decision.json").read_text())
    if abs(ga-prior["conditional_semantic_gains"]["A"])>1e-8 or abs(gb-prior["conditional_semantic_gains"]["B"])>1e-8:raise ValueError("R1.9 conditioned coefficients did not reproduce")
    wa=qa@ca;wb=qb@cb;wa-=wa.mean(1,keepdims=True);wb-=wb.mean(1,keepdims=True);class_index={c:i for i,c in enumerate(classes)}
    contrast_defs=[("business_vs_home","business_and_finance","home_food_and_lifestyle"),("business_vs_other","business_and_finance","other_or_unclear"),("home_vs_other","home_food_and_lifestyle","other_or_unclear")]
    contrast_rows=[];contrast_vectors={};contrast_examples=["# R1.10 Content-Contrast Examples",""]
    for name,positive,negative in contrast_defs:
        va=wa[:,class_index[positive]]-wa[:,class_index[negative]];vb=wb[:,class_index[positive]]-wb[:,class_index[negative]];va/=np.linalg.norm(va);vb/=np.linalg.norm(vb);signed=float(va@vb);cosine=abs(signed)
        score_a=np.einsum("dtm,m->dt",residual,va,optimize=True).mean(1);score_b=np.einsum("dtm,m->dt",residual,vb,optimize=True).mean(1);topics=frame.topic.astype(str).to_numpy();mask_test=np.isin(np.arange(len(frame)),test_docs)
        def gap(score):return float(score[mask_test&(topics==positive)].mean()-score[mask_test&(topics==negative)].mean())
        gap_a,gap_b=gap(score_a),gap(score_b);reproducible=bool(cosine>=.80 and gap_a>0 and gap_b>0);consensus=va+np.sign(signed)*vb;consensus/=np.linalg.norm(consensus);contrast_vectors[name]=consensus
        contrast_rows.append({"contrast":name,"positive_class":positive,"negative_class":negative,"signed_cosine":signed,"absolute_cosine":cosine,"A_test_ordering_gap":gap_a,"B_test_ordering_gap":gap_b,"status":"REPRODUCIBLE_CONTRAST" if reproducible else "CONTRAST_NOT_REPRODUCIBLE"})
        if reproducible:
            values=np.einsum("dtm,m->dt",residual,consensus,optimize=True);contrast_examples.append(example_block(name,values,frame,fp,tokens,tokenizer))
    pd.DataFrame(contrast_rows).to_csv(args.run_dir/"content_contrasts/contrast_stability.csv",index=False);(args.run_dir/"content_contrasts/contrast_examples.md").write_text("\n".join(contrast_examples)+"\n")
    alignment=[]
    for meta in axis_metadata:
        axis=load_direction(args.r17_run,"full",matches.iloc[meta["axis"]-1].a_probe_id)
        for name,contrast in contrast_vectors.items():alignment.append({"axis":meta["axis"],"contrast":name,"absolute_alignment":abs(float(axis@contrast))})
    pd.DataFrame(alignment).to_csv(args.run_dir/"content_contrasts/contrast_axis_alignment.csv",index=False)
    return 0


def load_documents(pool: Path,split_path: Path) -> pd.DataFrame:
    pool_df=pd.read_parquet(pool);split=pd.read_parquet(split_path);docs=pool_df.merge(split,on="context_id",how="inner");test=docs.loc[docs.split=="test"].reset_index(drop=True)
    if len(test)!=500:raise ValueError(f"expected 500 test documents, got {len(test)}")
    return test


def token_matrix(frame: pd.DataFrame) -> np.ndarray:
    result=np.asarray([list(x) for x in frame.token_ids],dtype=np.int64)
    if result.shape!=(len(frame),SEQ):raise ValueError(result.shape)
    return result


def load_model():
    device="cuda" if torch.cuda.is_available() else "cpu";dtype=torch.bfloat16 if device=="cuda" else torch.float32;source=os.environ.get("SLOW_SEMANTIC_MODEL_PATH",MODEL)
    tokenizer=AutoTokenizer.from_pretrained(source,revision=None if source!=MODEL else REVISION,local_files_only=source!=MODEL);hf=None
    if source!=MODEL:hf=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,torch_dtype=dtype)
    model=HookedTransformer.from_pretrained(MODEL,revision=REVISION if hf is None else None,tokenizer=tokenizer,hf_model=hf,local_files_only=hf is not None,device=device,dtype=dtype).eval();del hf;return model,device


def residual_batch(model,device,tokens):
    with torch.inference_mode():_,cache=model.run_with_cache(torch.as_tensor(tokens,dtype=torch.long,device=device),names_filter=lambda n:n==HOOK,return_type=None)
    result=cache[HOOK]
    if result.shape[-1]!=D_MODEL or not torch.isfinite(result).all():raise FloatingPointError(tuple(result.shape))
    return result


def evaluate(args) -> int:
    profile=args.profile_batches is not None;docs=load_documents(args.context_pool,args.context_split);tokens=token_matrix(docs);limit=min(len(tokens),args.profile_batches*args.batch_size) if profile else len(tokens)
    with np.load(args.run_dir/"rank_curve/nested_bases.npz",allow_pickle=False) as z:
        blocks={f"full_k{k}":random_in_span(z[f"full_k{k}"]) for k in RANKS}
        for fit in ("A","B"):
            for k in (31,64,128):blocks[f"{fit}_k{k}"]=random_in_span(z[f"{fit}_k{k}"])
    model,device=load_model();gpu={key:torch.as_tensor(value.T,dtype=torch.float32,device=device) for key,value in blocks.items()};acc={key:AutocorrAccumulator(n_features=512,max_lag=MAX_LAG) for key in blocks};timings=[]
    for bi,start in enumerate(range(0,limit,args.batch_size)):
        stop=min(limit,start+args.batch_size);t0=time.perf_counter();residual=residual_batch(model,device,tokens[start:stop]).float();forward=time.perf_counter()-t0;transfer=0
        for key,directions in gpu.items():
            t1=time.perf_counter();projection=torch.einsum("btd,dp->btp",residual,directions).cpu().numpy().astype(np.float32);transfer+=time.perf_counter()-t1;acc[key].update(compute_document_autocorr(projection,max_lag=MAX_LAG,estimator="within"))
        if device=="cuda":torch.cuda.synchronize()
        row={"batch":bi,"documents":stop-start,"forward_seconds":forward,"projection_transfer_and_autocorr_seconds":transfer,"total_seconds":time.perf_counter()-t0};timings.append(row);print(json.dumps(row),flush=True)
    if profile:
        mean=float(np.mean([x["total_seconds"] for x in timings]));save_json(args.run_dir/"profiles/functional_profile.json",{"status":"PROFILE_ONLY","batches":len(timings),"batch_size":args.batch_size,"mean_seconds_per_batch":mean,"projected_seconds":mean*math.ceil(500/args.batch_size),"directions_per_batch":sum(x.shape[0] for x in blocks.values()),"gpu_to_cpu_bytes_per_batch":sum(args.batch_size*SEQ*x.shape[0]*4 for x in blocks.values()),"one_forward_pass_per_batch":True,"timings":timings});return 4
    rows=[]
    for key,value in acc.items():
        result=value.finalize();table=build_timescale_table(np.arange(512),{"within":result},MAX_LAG,100,.8,5);fit,rank=key.split("_k");table["fit"]=fit;table["rank"]=int(rank);rows.append(table)
    pd.concat(rows,ignore_index=True).to_parquet(args.run_dir/"rank_curve/random_in_span_test.parquet",index=False);save_json(args.run_dir/"profiles/functional_production.json",{"status":"PASS","batches":len(timings),"seconds":sum(x["total_seconds"] for x in timings),"timings":timings})
    return 0


def finalize(args) -> int:
    raw=pd.read_parquet(args.run_dir/"rank_curve/random_in_span_test.parquet");summary=[]
    for (fit,rank),group in raw.groupby(["fit","rank"]):summary.append({"fit":fit,"rank":int(rank),**summarize_timescales(group)})
    summary=pd.DataFrame(summary).sort_values(["fit","rank"]);summary.loc[summary.fit.isin(["A","B"]),:].to_csv(args.run_dir/"rank_curve/splitfit_functional_summary.csv",index=False)
    geometry=pd.read_csv(args.run_dir/"rank_curve/nested_geometry.csv");decision=rank_curve_decision(summary,geometry);save_json(args.run_dir/"rank_curve/rank_curve_decision.json",{"decision":decision,"canonical_q31_unchanged":True,"functional_summary":summary.to_dict("records")})
    axes=json.loads((args.run_dir/"axes/axis_labels.json").read_text())["axes"];contrasts=pd.read_csv(args.run_dir/"content_contrasts/contrast_stability.csv");repro=contrasts.loc[contrasts.status=="REPRODUCIBLE_CONTRAST","contrast"].tolist()
    lines=["# R1.10 Rank-Extent and Interpretable-Coordinate Sidecar","",f"Rank-curve outcome: `{decision}`. The historical canonical object remains signed-TICA Q31.","","## Functional rank curve","",markdown_table(summary),"","## Axis dossiers",""]
    lines.extend([f"- Axis {x['axis']}: `{x['label']}` (A/B |cos|={x['matched_abs_cosine']:.4f})" for x in axes]);lines.extend(["","## Content contrasts","",f"Reproducible fixed contrasts: {', '.join(repro) if repro else 'none'}.","","This sidecar is descriptive and changes none of R1.7-R1.9's frozen conclusions."])
    (args.run_dir/"r1_10_sidecar_report.md").write_text("\n".join(lines)+"\n");return 0


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("initialize","prepare","evaluate","finalize"));p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--r17-run",type=Path,required=True);p.add_argument("--r18-run",type=Path,required=True);p.add_argument("--r19-run",type=Path,required=True);p.add_argument("--semantic-run",type=Path,required=True);p.add_argument("--context-pool",type=Path,required=True);p.add_argument("--context-split",type=Path,required=True);p.add_argument("--spec",type=Path,required=True);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--profile-batches",type=int);args=p.parse_args();return {"initialize":initialize,"prepare":prepare,"evaluate":evaluate,"finalize":finalize}[args.stage](args)


if __name__=="__main__":raise SystemExit(main())
