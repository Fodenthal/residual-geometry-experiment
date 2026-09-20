"""Pure helpers for R1.11 controlled content-state splicing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

HORIZONS=(0,32,64,128,256)


def orthonormal_columns(matrix: np.ndarray, rank: int | None=None) -> np.ndarray:
    u,s,_=np.linalg.svd(np.asarray(matrix,float),full_matrices=False)
    keep=int(np.sum(s>max(s[0]*1e-10,1e-12))) if rank is None else min(rank,int(np.sum(s>max(s[0]*1e-10,1e-12))))
    if keep<1: raise ValueError("empty contrast span")
    return u[:,:keep]


def normalized_contrast(basis: np.ndarray, coefficients: np.ndarray, classes: list[str], positive: str="business", negative: str="home") -> np.ndarray:
    lookup={str(v):i for i,v in enumerate(classes)}
    if positive not in lookup or negative not in lookup: raise ValueError(f"missing contrast classes: {classes}")
    vector=np.asarray(basis,float)@(np.asarray(coefficients,float)[:,lookup[positive]]-np.asarray(coefficients,float)[:,lookup[negative]])
    norm=float(np.linalg.norm(vector))
    if norm<1e-10: raise ValueError("degenerate business-home contrast")
    return vector/norm


def pairwise_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a=np.asarray(left,float);b=np.asarray(right,float)
    return np.sqrt(np.maximum(((a[:,None,:]-b[None,:,:])**2).sum(-1),0.0))


def allocate_and_match(frame: pd.DataFrame, fingerprint: np.ndarray, boundary_prefix: np.ndarray, boundary_suffix: np.ndarray | None=None, seed: int=111256) -> tuple[pd.DataFrame,dict]:
    """Allocate disjoint roles then solve frozen minimum-cost donor matchings."""
    rng=np.random.default_rng(seed);roles={};boundary_suffix=np.asarray(boundary_prefix if boundary_suffix is None else boundary_suffix,float);boundary_prefix=np.asarray(boundary_prefix,float)
    for topic in ("business","home"):
        idx=np.flatnonzero(frame.topic.astype(str).to_numpy()==topic)
        idx=idx[np.argsort(frame.document_id.astype(str).to_numpy()[idx])]
        if len(idx)<224: raise ValueError(f"{topic} support {len(idx)} < 224")
        idx=rng.permutation(idx[:224]);roles[topic]={"recipient":idx[:64],"same":idx[64:128],"cross":idx[128:192],"slack":idx[192:224]}
    rows=[];cost_rows=[]
    for recipient_topic,donor_topic in (("business","home"),("home","business")):
        rec=roles[recipient_topic]["recipient"];same_pool=roles[recipient_topic]["same"]
        rs=pairwise_distance(fingerprint[rec],fingerprint[same_pool])+0.5*pairwise_distance(boundary_suffix[rec],boundary_prefix[same_pool])
        _,same_assignment=linear_sum_assignment(rs);same=same_pool[same_assignment]
        cross_pool=roles[donor_topic]["cross"]
        fp_sd=pairwise_distance(fingerprint[same],fingerprint[cross_pool])
        bd_rd=pairwise_distance(boundary_suffix[rec],boundary_prefix[cross_pool]);bd_rs=np.linalg.norm(boundary_suffix[rec]-boundary_prefix[same],axis=1)
        cost=2.0*fp_sd+bd_rd+np.abs(bd_rd-bd_rs[:,None])
        _,cross_assignment=linear_sum_assignment(cost);different=cross_pool[cross_assignment]
        for j,(r,s,d) in enumerate(zip(rec,same,different)):
            rows.append({"triplet_id":len(rows),"direction":f"{donor_topic}_to_{recipient_topic}","recipient_topic":recipient_topic,"different_donor_topic":donor_topic,
                "recipient_row":int(r),"same_donor_row":int(s),"different_donor_row":int(d),"recipient_id":frame.document_id.iloc[r],"same_donor_id":frame.document_id.iloc[s],"different_donor_id":frame.document_id.iloc[d],
                "same_fingerprint_distance":float(np.linalg.norm(fingerprint[r]-fingerprint[s])),"different_pair_fingerprint_distance":float(np.linalg.norm(fingerprint[s]-fingerprint[d])),
                "same_boundary_distance":float(np.linalg.norm(boundary_suffix[r]-boundary_prefix[s])),"different_boundary_distance":float(np.linalg.norm(boundary_suffix[r]-boundary_prefix[d])),"matching_cost":float(cost[j,cross_assignment[j]])})
    result=pd.DataFrame(rows)
    used=result[["recipient_id","same_donor_id","different_donor_id"]].to_numpy().ravel()
    qc={"status":"PASS" if len(set(used))==384 else "FAIL","triplets":len(result),"unique_documents":len(set(used)),"directions":result.direction.value_counts().to_dict(),
        "fingerprint_distance":{"same_mean":float(result.same_fingerprint_distance.mean()),"different_pair_mean":float(result.different_pair_fingerprint_distance.mean())},
        "boundary_distance":{"same_mean":float(result.same_boundary_distance.mean()),"different_mean":float(result.different_boundary_distance.mean())}}
    if qc["status"]!="PASS" or len(result)!=128: raise ValueError(f"invalid matching: {qc}")
    return result,qc


def splice_triplets(tokens: np.ndarray, triplets: pd.DataFrame, boundary: int=256) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    values=np.asarray(tokens);ref=[];same=[];different=[]
    for row in triplets.itertuples(index=False):
        recipient=values[row.recipient_row]
        ref.append(recipient.copy());same.append(np.concatenate([values[row.same_donor_row,:boundary],recipient[boundary:]]));different.append(np.concatenate([values[row.different_donor_row,:boundary],recipient[boundary:]]))
    ref=np.stack(ref);same=np.stack(same);different=np.stack(different)
    if not (np.array_equal(ref[:,boundary:],same[:,boundary:]) and np.array_equal(ref[:,boundary:],different[:,boundary:])): raise ValueError("post-boundary token mismatch")
    return ref,same,different


def trajectory(residual_ref: np.ndarray,residual_same: np.ndarray,residual_different: np.ndarray,contrast: np.ndarray,orientation: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    c=np.asarray(contrast,float);o=np.asarray(orientation,float)[:,None]
    same=o*np.einsum("nkd,d->nk",np.asarray(residual_same)-np.asarray(residual_ref),c,optimize=True)
    different=o*np.einsum("nkd,d->nk",np.asarray(residual_different)-np.asarray(residual_ref),c,optimize=True)
    return same,different,different-same


def bootstrap_rows(values: np.ndarray, replicates: int=2000, seed: int=111064) -> pd.DataFrame:
    x=np.asarray(values,float);rng=np.random.default_rng(seed);out=[]
    for replicate in range(replicates):
        indices=rng.integers(0,len(x),size=len(x));means=x[indices].mean(0)
        out.extend({"replicate":replicate,"horizon":h,"mean_D":float(v)} for h,v in zip(HORIZONS,means))
    return pd.DataFrame(out)


def assign_outcome(noop_pass: bool,canonical: np.ndarray,ci: dict[int,tuple[float,float]],split_a: np.ndarray,split_b: np.ndarray,directions: dict[str,np.ndarray],generic_ratio: float) -> str:
    if not noop_pass:return "TECHNICAL_FAILURE"
    primary=ci[64][0]>0 and split_a[2]>0 and split_b[2]>0 and all(v[2]>0 for v in directions.values())
    if primary:return "PERSISTENT_CONTENT_STATE"
    if ci[0][0]>0 or ci[32][0]>0:return "SHORT_LIVED_CONTENT_HISTORY_EFFECT"
    if generic_ratio<=0.1:return "GENERIC_SPLICE_SENSITIVITY"
    return "NO_CONTROLLED_CONTENT_EFFECT"
