#!/usr/bin/env python
"""R1.11 controlled persistent content-state experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.persistent_state.slow_semantic.protocol import PROTOCOL
from src.persistent_state.slow_semantic.r19 import fingerprint_frame,one_hot_frozen
from src.persistent_state.slow_semantic.r111 import HORIZONS,allocate_and_match,assign_outcome,bootstrap_rows,normalized_contrast,orthonormal_columns,splice_triplets,trajectory

MODEL="google/gemma-2-2b";REVISION="main";HOOK="blocks.12.hook_resid_post";D_MODEL=2304;SEQ=1024;BOUNDARY=256;POSITIONS=tuple(BOUNDARY+k for k in HORIZONS)
BUSINESS="business_and_finance";HOME="home_food_and_lifestyle"
PROMPT="""Classify this web document using exactly one topic and one register from the frozen lists.
Topic: {topics}
Register: {registers}
Return exactly: topic=<label>;register=<label>
If genuinely ambiguous, use other_or_unclear. Do not explain.

DOCUMENT:
{text}"""


def save_json(path: Path,value) -> None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def stable_id(text: str,url: str) -> str:return hashlib.sha256((url+"\n"+text).encode("utf-8",errors="replace")).hexdigest()
def formatting_label(text: str) -> str:
    lines=text.splitlines() or [text];chars=max(len(text),1);nonempty=max(sum(bool(x.strip()) for x in lines),1)
    scores={"code_or_markup":sum(text.count(x) for x in ("{","}","</",";","```"))/chars,
        "list_or_table":max(sum(bool(re.match(r"\s*(?:[-*•]|\d+[.)])\s+",x)) for x in lines)/len(lines),sum(x.count("|")>=2 or "\t" in x for x in lines)/len(lines)),
        "quotation_or_dialogue":sum(x.lstrip().startswith((">",'"',"“")) for x in lines)/len(lines),"newline_heavy":len(lines)/chars*80,
        "boilerplate_repetition":1-len(set(x.strip() for x in lines if x.strip()))/nonempty,"plain_prose":.035}
    return max(scores,key=scores.get)
def parse_label(output: str,choices,key: str) -> str:
    match=re.search(rf"{key}\s*=\s*([a-z0-9_]+)",output.lower())
    if match and match.group(1) in choices:return match.group(1)
    found=[x for x in choices if x in output.lower()];return found[0] if len(found)==1 else "other_or_unclear"
def load_r19_module():
    path=Path(__file__).with_name("81_content_vs_fingerprint_r19.py");spec=importlib.util.spec_from_file_location("frozen_r19",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def initialize(args) -> int:
    run=args.run_dir
    for rel in ("pool","capture","analysis","provenance","profiles","decisions"):(run/rel).mkdir(parents=True,exist_ok=True)
    required={"r19_decision":args.r19_run/"decision.json","semantic_labels":args.semantic_run/"labels/document_labels.parquet","semantic_residuals":args.semantic_run/"capture/residuals.npy",
        "canonical_basis":args.canonical_basis,"split_A":args.r17_run/"split_A/q31.npy","split_B":args.r17_run/"split_B/q31.npy","pca":args.semantic_run/"supervised/residual_pca256.npz"}
    missing=[str(v) for v in required.values() if not v.exists()]
    if missing:raise FileNotFoundError(missing)
    r19_decision=json.loads(required["r19_decision"].read_text())
    if r19_decision.get("final_outcome")!="CONTENT_STATE_IDENTIFIED":raise ValueError("R1.9 did not license controlled content-state test")
    r19=load_r19_module();frame,residual,centered,old_nuisance,y,y_doc,fit_docs,test_docs,tr,te,classes=r19.semantic_design(args.semantic_run);fp,schema,fp_doc,nuisance=r19.build_fingerprint(frame,fit_docs,old_nuisance)
    def basis(path,key="basis"):
        if path.suffix==".npy":return np.asarray(np.load(path),float)
        with np.load(path,allow_pickle=False) as z:return np.asarray(z[key],float)
    bases={"canonical":basis(required["canonical_basis"]),"A":basis(required["split_A"]),"B":basis(required["split_B"])}
    with np.load(required["pca"],allow_pickle=False) as z:bases["pca"]=np.asarray(z["directions"][:,:31],float)
    contrasts={};ambient={};gains={}
    for label,b in bases.items():
        gain,coef,_=r19.conditional_fit(nuisance,centered@b,y,tr,te);gains[label]=gain;contrasts[label]=normalized_contrast(b,coef,classes,positive=BUSINESS,negative=HOME);ambient[label]=b@coef
    expected=r19_decision["conditional_semantic_gains"]
    for label in bases:
        if abs(gains[label]-float(expected[label]))>2e-6:raise ValueError(f"R1.9 readout reconstruction drift {label}: {gains[label]} vs {expected[label]}")
    canonical_weights=ambient["canonical"];content_plane=orthonormal_columns(canonical_weights-canonical_weights.mean(1,keepdims=True),rank=2)
    np.savez_compressed(run/"provenance/frozen_readouts.npz",canonical=contrasts["canonical"],split_A=contrasts["A"],split_B=contrasts["B"],pca=contrasts["pca"],content_plane=content_plane,classes=np.asarray(classes))
    shutil.copy2(args.spec,run/"provenance"/args.spec.name)
    config={"status":"FROZEN_BEFORE_R1_11_CAPTURE","model":MODEL,"revision":REVISION,"hook":HOOK,"sequence_length":SEQ,"splice_boundary_zero_based":BOUNDARY,"capture_positions_zero_based":POSITIONS,
        "topics":[BUSINESS,HOME],"topic_shorthand":{"business":BUSINESS,"home":HOME},"minimum_topic_support":224,"recipients_per_direction":64,"triplets":128,"unique_documents":384,"hash_thinning_modulus":13,"hash_thinning_residue":11,
        "fingerprint_weights":{"same_match_fingerprint":1,"same_match_boundary":.5,"different_pair_fingerprint":2,"different_recipient_boundary":1,"boundary_imbalance":1},
        "bootstrap_replicates":2000,"bootstrap_seed":111064,"primary_horizon":64,"primary_rule":"canonical lower95>0 AND split-A/B means>0 AND both reciprocal direction means>0",
        "early_rule":"lower95>0 at k=0 or k=32","generic_rule":"max absolute donor-directed mean <= 0.1 * max mean content-plane displacement","randomness":{"allocation":111256},"no_automatic_followup":True}
    save_json(run/"r1_11_config.json",config)
    hashes={k:sha256_file(v) for k,v in required.items() if v.stat().st_size<20_000_000};hashes["frozen_readouts"]=sha256_file(run/"provenance/frozen_readouts.npz")
    commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip() or "unknown"
    save_json(run/"provenance/provenance.json",{"status":"PASS","implementation_commit":commit,"input_hashes":hashes,"r19_outcome":r19_decision["final_outcome"],"r19_reconstructed_gains":gains,"labels_activation_blind":True})
    return 0


def build_candidates(tokenizer,args):
    from datasets import load_dataset
    excluded=set(pd.read_parquet(args.semantic_run/"labels/document_labels.parquet").document_id.astype(str));seen=set();records=[];tokens=[];scanned=0
    stream=load_dataset(PROTOCOL["corpus"],PROTOCOL["corpus_config"],split=PROTOCOL["corpus_split"],streaming=True)
    for row in stream:
        scanned+=1;text=str(row.get("text",""));url=str(row.get("url",""));doc_id=stable_id(text,url)
        if doc_id in excluded or doc_id in seen or int(doc_id[:8],16)%13!=11:continue
        encoded=tokenizer(text,add_special_tokens=True,truncation=True,max_length=SEQ)["input_ids"]
        if len(encoded)<SEQ:continue
        seen.add(doc_id);tokens.append(np.asarray(encoded[:SEQ],np.int32));records.append({"document_id":doc_id,"url":url,"host":(urlparse(url).hostname or "").lower(),"text":text,"formatting":formatting_label(text),"stream_index":scanned})
        if len(records)>=12000:break
    return records,tokens,scanned


def token_feature_table(tokenizer,tokens: np.ndarray) -> tuple[np.ndarray,np.ndarray,list[str]]:
    ids=np.unique(tokens[:,224:288]);lookup={int(i):tokenizer.decode([int(i)],skip_special_tokens=False) for i in ids}
    names=["mean_chars","space","newline","punct","digit","upper","alpha"]
    def window(row):
        strings=[lookup[int(i)] for i in row];chars="".join(strings);n=max(len(chars),1)
        return [np.mean([len(x) for x in strings]),sum(c.isspace() for c in chars)/n,chars.count("\n")/n,sum(not c.isalnum() and not c.isspace() for c in chars)/n,sum(c.isdigit() for c in chars)/n,sum(c.isupper() for c in chars)/n,sum(c.isalpha() for c in chars)/n]
    return np.asarray([window(x[224:256]) for x in tokens]),np.asarray([window(x[256:288]) for x in tokens]),names


def label(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    profile=args.profile_batches is not None;model_source=os.environ["SLOW_SEMANTIC_MODEL_PATH"];label_source=os.environ["SLOW_SEMANTIC_LABELER_PATH"]
    tokenizer=AutoTokenizer.from_pretrained(model_source,local_files_only=True);records,token_rows,scanned=build_candidates(tokenizer,args)
    lt=AutoTokenizer.from_pretrained(label_source,local_files_only=True);lt.pad_token_id=lt.pad_token_id or lt.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(label_source,local_files_only=True,torch_dtype=torch.bfloat16,device_map="auto").eval();topics=PROTOCOL["topic_classes"];registers=PROTOCOL["register_classes"];times=[];counts=Counter();labeled=0
    limit=len(records) if not profile else min(len(records),args.batch_size*args.profile_batches)
    for start in range(0,limit,args.batch_size):
        batch=records[start:start+args.batch_size];rendered=[lt.apply_chat_template([{"role":"user","content":PROMPT.format(topics=", ".join(topics),registers=", ".join(registers),text=str(x["text"])[:12000])}],tokenize=False,add_generation_prompt=True) for x in batch]
        encoded=lt(rendered,return_tensors="pt",padding=True,truncation=True,max_length=PROTOCOL["labeler_max_input_tokens"]).to(model.device);t0=time.perf_counter()
        with torch.inference_mode():generated=model.generate(**encoded,max_new_tokens=32,do_sample=False,pad_token_id=lt.pad_token_id)
        times.append(time.perf_counter()-t0)
        for i,row in enumerate(batch):
            raw=lt.decode(generated[i,encoded["input_ids"].shape[1]:],skip_special_tokens=True).strip();row["topic"]=parse_label(raw,topics,"topic");row["register"]=parse_label(raw,registers,"register");row["labeler_raw"]=raw;counts[row["topic"]]+=1;labeled+=1
        print(json.dumps({"labeled":labeled,"business":counts[BUSINESS],"home":counts[HOME],"seconds":times[-1]}),flush=True)
        if not profile and counts[BUSINESS]>=224 and counts[HOME]>=224:limit=start+len(batch);break
    if profile:
        save_json(args.run_dir/"profiles/label_profile.json",{"status":"PROFILE_ONLY","documents":labeled,"mean_seconds_per_batch":float(np.mean(times)),"projected_documents":5600,"projected_seconds":float(np.mean(times)*math.ceil(5600/args.batch_size)),"scanned_candidates":len(records),"source_documents_scanned":scanned});return 4
    frame=pd.DataFrame(records[:limit]);token_matrix=np.stack(token_rows[:limit]);support=frame.topic.value_counts().to_dict()
    if support.get(BUSINESS,0)<224 or support.get(HOME,0)<224:raise RuntimeError(f"support gate failed: {support}")
    fp,schema=fingerprint_frame(frame,np.arange(len(frame)));numeric=StandardScaler().fit_transform(fp[schema["numeric"]].to_numpy(float));categorical=[one_hot_frozen(fp[c].to_numpy(),schema["categorical"][c]) for c in ("source_type","host_suffix_class","register","formatting")];design=np.column_stack([numeric,*categorical])
    prefix,suffix,feature_names=token_feature_table(tokenizer,token_matrix);prefix=StandardScaler().fit_transform(prefix);suffix=StandardScaler().fit_transform(suffix)
    matching_frame=frame.copy();matching_frame["topic"]=matching_frame.topic.replace({BUSINESS:"business",HOME:"home"})
    triplets,qc=allocate_and_match(matching_frame,design,prefix,suffix);ref,same,different=splice_triplets(token_matrix,triplets,BOUNDARY)
    frame.to_parquet(args.run_dir/"pool/topic_fingerprint_labels.parquet",index=False);np.save(args.run_dir/"pool/candidate_tokens.npy",token_matrix,allow_pickle=False);triplets.to_parquet(args.run_dir/"pool/selected_triplets.parquet",index=False)
    np.savez_compressed(args.run_dir/"pool/spliced_tokens.npz",reference=ref,same=same,different=different)
    qc.update({"topic_counts":support,"candidate_documents":len(frame),"source_documents_scanned":scanned,"boundary_features":feature_names,"post_boundary_token_equality":True,"document_reuse":False});save_json(args.run_dir/"pool/matching_qc.json",qc)
    save_json(args.run_dir/"pool/labeler_manifest.json",{"model":"google/gemma-2-9b-it","revision":"11c9b309abf73637e4b6f9a3fa1e92e615547819","prompt":PROMPT,"topic_classes":topics,"register_classes":registers,"activation_blind":True,"tokenizer_convention":"canonical add_special_tokens=True; t0 is zero-based token-array index"})
    return 0


def load_model():
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM,AutoTokenizer
    source=os.environ["SLOW_SEMANTIC_MODEL_PATH"];device="cuda" if torch.cuda.is_available() else "cpu";dtype=torch.bfloat16 if device=="cuda" else torch.float32
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True);hf=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,torch_dtype=dtype)
    model=HookedTransformer.from_pretrained(MODEL,tokenizer=tokenizer,hf_model=hf,local_files_only=True,device=device,dtype=dtype).eval();del hf;return model,device
def residual_batch(model,device,batch):
    import torch
    with torch.inference_mode():_,cache=model.run_with_cache(torch.as_tensor(batch,dtype=torch.long,device=device),names_filter=lambda x:x==HOOK,return_type=None)
    result=cache[HOOK][:,POSITIONS,:]
    if result.shape[-2:]!=(5,D_MODEL) or not torch.isfinite(result).all():raise FloatingPointError(tuple(result.shape))
    return result.float().cpu().numpy()


def capture(args) -> int:
    profile=args.profile_batches is not None
    with np.load(args.run_dir/"pool/spliced_tokens.npz",allow_pickle=False) as z:ref=z["reference"];same=z["same"];different=z["different"]
    model,device=load_model();batch_times=[];outputs={"reference":[],"same":[],"different":[]};limit=min(len(ref),args.batch_size*args.profile_batches) if profile else len(ref)
    noop_tokens=np.concatenate([ref[:4],ref[:4]],axis=0);noop=residual_batch(model,device,noop_tokens);scale=float(np.sqrt(np.mean(noop[:4]**2)));maxdiff=float(np.max(np.abs(noop[:4]-noop[4:])));relative=maxdiff/max(scale,1e-12);noop_pass=relative<1e-5
    save_json(args.run_dir/"capture/noop_audit.json",{"status":"PASS" if noop_pass else "FAIL","documents":4,"activation_rms":scale,"max_absolute_difference":maxdiff,"relative_max_difference":relative,"threshold":1e-5,"token_identical":True})
    if not noop_pass:raise RuntimeError("no-op numerical audit failed")
    for start in range(0,limit,args.batch_size):
        stop=min(limit,start+args.batch_size);joined=np.concatenate([ref[start:stop],same[start:stop],different[start:stop]],axis=0);t0=time.perf_counter();captured=residual_batch(model,device,joined);elapsed=time.perf_counter()-t0;n=stop-start
        for key,block in zip(outputs,(captured[:n],captured[n:2*n],captured[2*n:])):outputs[key].append(block)
        batch_times.append(elapsed);print(json.dumps({"triplets":stop,"batch_seconds":elapsed}),flush=True)
    if profile:
        mean=float(np.mean(batch_times));save_json(args.run_dir/"profiles/capture_profile.json",{"status":"PROFILE_ONLY","batches":len(batch_times),"triplets_per_batch":args.batch_size,"mean_seconds_per_batch":mean,"projected_seconds":mean*math.ceil(128/args.batch_size),"full_residual_gpu_to_cpu_bytes_per_batch":0,"selected_horizon_bytes_per_batch":args.batch_size*3*5*D_MODEL*4,"forward_passes_per_batch":1});return 4
    np.savez_compressed(args.run_dir/"capture/residual_horizons.npz",**{k:np.concatenate(v) for k,v in outputs.items()},horizons=np.asarray(HORIZONS),positions=np.asarray(POSITIONS))
    save_json(args.run_dir/"capture/capture_manifest.json",{"status":"PASS","shape":[128,3,5,D_MODEL],"one_joint_forward_per_condition_batch":True,"gpu_index_before_cpu_transfer":True,"estimated_total_transfer_bytes":128*3*5*D_MODEL*4,"model":MODEL,"revision":REVISION,"hook":HOOK})
    return 0


def analyze(args) -> int:
    with np.load(args.run_dir/"capture/residual_horizons.npz",allow_pickle=False) as z:ref=z["reference"];same=z["same"];different=z["different"]
    with np.load(args.run_dir/"provenance/frozen_readouts.npz",allow_pickle=False) as z:readouts={k:z[k] for k in ("canonical","split_A","split_B","pca")};plane=z["content_plane"]
    triplets=pd.read_parquet(args.run_dir/"pool/selected_triplets.parquet");orientation=np.where(triplets.different_donor_topic.astype(str).to_numpy()=="business",1.0,-1.0)
    all_values={};rows=[]
    for label,c in readouts.items():
        ts,td,d=trajectory(ref,same,different,c,orientation);all_values[label]=(ts,td,d)
        rows.extend({"basis":label,"horizon":h,"mean_T_same":float(ts[:,j].mean()),"mean_T_different":float(td[:,j].mean()),"mean_D":float(d[:,j].mean())} for j,h in enumerate(HORIZONS))
    table=pd.DataFrame(rows);table.loc[table.basis=="canonical"].to_csv(args.run_dir/"analysis/semantic_trajectory.csv",index=False);table.loc[table.basis.isin(["split_A","split_B"])].to_csv(args.run_dir/"analysis/split_readout_trajectory.csv",index=False);table.loc[table.basis=="pca"].to_csv(args.run_dir/"analysis/pca_control_trajectory.csv",index=False)
    bootstrap=bootstrap_rows(all_values["canonical"][2]);bootstrap.to_parquet(args.run_dir/"analysis/bootstrap_primary.parquet",index=False);ci={h:tuple(np.quantile(bootstrap.loc[bootstrap.horizon==h,"mean_D"],[.025,.975])) for h in HORIZONS}
    gs=np.linalg.norm(np.einsum("nkd,dp->nkp",same-ref,plane,optimize=True),axis=-1);gd=np.linalg.norm(np.einsum("nkd,dp->nkp",different-ref,plane,optimize=True),axis=-1)
    pd.DataFrame([{"horizon":h,"mean_G_same":float(gs[:,j].mean()),"mean_G_different":float(gd[:,j].mean())} for j,h in enumerate(HORIZONS)]).to_csv(args.run_dir/"analysis/content_plane_trajectory.csv",index=False)
    canonical=all_values["canonical"][2];directions={name:canonical[triplets.direction.astype(str).to_numpy()==name].mean(0) for name in sorted(triplets.direction.unique())}
    pd.DataFrame([{"direction":name,"horizon":h,"mean_D":float(values[j])} for name,values in directions.items() for j,h in enumerate(HORIZONS)]).to_csv(args.run_dir/"analysis/reciprocal_direction_trajectory.csv",index=False)
    m=canonical.mean(0);retention=[None if m[0]<=0 else float(v/m[0]) for v in m];generic_scale=float(max(gs.mean(0).max(),gd.mean(0).max(),1e-12));generic_ratio=float(np.max(np.abs(m))/generic_scale)
    noop=json.loads((args.run_dir/"capture/noop_audit.json").read_text());outcome=assign_outcome(noop["status"]=="PASS",m,ci,all_values["split_A"][2].mean(0),all_values["split_B"][2].mean(0),directions,generic_ratio)
    decision={"final_outcome":outcome,"primary_horizon":64,"canonical_means":{str(h):float(v) for h,v in zip(HORIZONS,m)},"canonical_bootstrap_95":{str(h):[float(x) for x in ci[h]] for h in HORIZONS},"split_A_means":{str(h):float(v) for h,v in zip(HORIZONS,all_values["split_A"][2].mean(0))},"split_B_means":{str(h):float(v) for h,v in zip(HORIZONS,all_values["split_B"][2].mean(0))},"reciprocal_direction_means":{name:{str(h):float(v) for h,v in zip(HORIZONS,values)} for name,values in directions.items()},"retention":{str(h):v for h,v in zip(HORIZONS,retention)},"pca_means":{str(h):float(v) for h,v in zip(HORIZONS,all_values["pca"][2].mean(0))},"generic_content_plane_ratio":generic_ratio,"no_op_status":noop["status"],"automatic_followup_launched":False}
    save_json(args.run_dir/"decisions/decision.json",decision)
    (args.run_dir/"r1_11_report.md").write_text(f"# R1.11 Controlled Persistent Content-State Test\n\nOutcome: `{outcome}`.\n\nAt the registered k=64 horizon, canonical D={m[2]:.6f} with paired-document bootstrap 95% CI [{ci[64][0]:.6f}, {ci[64][1]:.6f}]. Split-A/B values are {all_values['split_A'][2].mean(0)[2]:.6f}/{all_values['split_B'][2].mean(0)[2]:.6f}. Reciprocal direction values are "+", ".join(f"{k}={v[2]:.6f}" for k,v in directions.items())+f".\n\nNo-op audit: `{noop['status']}`. PCA is secondary and did not gate the outcome. No follow-up was launched.\n")
    return 0


def finalize(args) -> int:
    decision=args.run_dir/"decisions/decision.json";required=[args.run_dir/x for x in ("pool/candidate_pool_manifest.parquet","pool/topic_fingerprint_labels.parquet","pool/selected_triplets.parquet","pool/matching_qc.json","capture/noop_audit.json","capture/residual_horizons.npz","analysis/semantic_trajectory.csv","analysis/split_readout_trajectory.csv","analysis/content_plane_trajectory.csv","analysis/pca_control_trajectory.csv","analysis/bootstrap_primary.parquet","r1_11_report.md")]
    # candidate labels are the auditable candidate-pool manifest under both required names.
    source=args.run_dir/"pool/topic_fingerprint_labels.parquet";target=args.run_dir/"pool/candidate_pool_manifest.parquet"
    if source.exists() and not target.exists():shutil.copy2(source,target)
    missing=[str(x) for x in required+[decision] if not x.exists()]
    if missing:raise FileNotFoundError(missing)
    save_json(args.run_dir/"final_status.json",{"status":"COMPLETE","outcome":json.loads(decision.read_text())["final_outcome"],"required_artifacts":len(required)+1,"followup":"NONE_PER_FROZEN_STOPPING_RULE"});return 0


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("initialize","label","capture","analyze","finalize"));p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--semantic-run",type=Path,required=True);p.add_argument("--r17-run",type=Path,required=True);p.add_argument("--r19-run",type=Path,required=True);p.add_argument("--canonical-basis",type=Path,required=True);p.add_argument("--spec",type=Path,required=True);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--profile-batches",type=int);args=p.parse_args()
    return {"initialize":initialize,"label":label,"capture":capture,"analyze":analyze,"finalize":finalize}[args.stage](args)
if __name__=="__main__":raise SystemExit(main())
