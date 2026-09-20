"""Pure helpers for the R1.10 rank-and-axis characterization sidecar."""
from __future__ import annotations

import numpy as np
import pandas as pd


RANKS=(8,13,21,31,48,64,96,128)


def orthonormal_nested(directions: np.ndarray, rank: int) -> np.ndarray:
    selected=np.asarray(directions,dtype=np.float64)[:rank].T
    if selected.shape[1]<rank:
        raise ValueError(f"only {selected.shape[1]} directions for rank {rank}")
    q,r=np.linalg.qr(selected)
    if int(np.sum(np.abs(np.diag(r))>1e-8))<rank:
        raise ValueError(f"rank-deficient nested span at {rank}")
    return q[:,:rank].astype(np.float32)


def geometry_row(left: np.ndarray, right: np.ndarray, rank: int, ambient: int=2304) -> tuple[dict,np.ndarray]:
    singular=np.clip(np.linalg.svd(np.asarray(left,float).T@np.asarray(right,float),compute_uv=False),0,1)
    overlap=float(np.mean(singular**2));chance=rank/ambient
    row={"rank":rank,"overlap":overlap,"ambient_expectation":chance,
         "chance_corrected_overlap":(overlap-chance)/(1-chance),
         "median_principal_angle_degrees":float(np.degrees(np.arccos(np.median(singular)))),
         "minimum_canonical_correlation":float(singular.min())}
    return row,singular


def random_in_span(basis: np.ndarray, count: int=512, seed: int=31) -> np.ndarray:
    rng=np.random.default_rng(seed);coef=rng.normal(size=(count,basis.shape[1]));coef/=np.linalg.norm(coef,axis=1,keepdims=True)
    return (coef@np.asarray(basis,float).T).astype(np.float32)


def summarize_timescales(table: pd.DataFrame) -> dict:
    valid=table.loc[table.tau_valid_within.astype(bool),"tau_within"].to_numpy(float)
    if not len(valid): return {"count":0,"q05":None,"q25":None,"median":None,"q75":None,"q90":None,"q95":None,"max":None,"mean":None,"censoring_fraction":float(table.right_censored_within.astype(bool).mean())}
    q=np.quantile(valid,[.05,.25,.5,.75,.9,.95])
    return {"count":int(len(valid)),"q05":float(q[0]),"q25":float(q[1]),"median":float(q[2]),"q75":float(q[3]),"q90":float(q[4]),"q95":float(q[5]),"max":float(valid.max()),"mean":float(valid.mean()),"censoring_fraction":float(table.right_censored_within.astype(bool).mean())}


def rank_curve_decision(summary: pd.DataFrame, geometry: pd.DataFrame) -> str:
    def med(fit,rank): return float(summary.loc[(summary.fit==fit)&(summary["rank"]==rank),"median"].iloc[0])
    s64=float(geometry.loc[geometry["rank"]==64,"chance_corrected_overlap"].iloc[0])
    s128=float(geometry.loc[geometry["rank"]==128,"chance_corrected_overlap"].iloc[0])
    broad64=med("full",64)>=15 and med("A",64)>=15 and med("B",64)>=15 and s64>=.70
    broad128=med("full",128)>=15 and med("A",128)>=15 and med("B",128)>=15 and s128>=.70
    if broad64 or broad128: return "BROAD_PERSISTENT_REGION"
    if med("full",64)<=5 or s64<.40: return "COMPACT_CORE"
    return "GRADED_SLOW_SPECTRUM"


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    x=np.asarray(values,float);y=np.asarray(labels);ok=np.isfinite(x)&pd.notna(y)
    x=x[ok];y=y[ok]
    if len(x)<2:return float("nan")
    total=float(np.sum((x-x.mean())**2))
    if total<=0:return 0.0
    between=sum(len(x[y==label])*(float(x[y==label].mean())-float(x.mean()))**2 for label in np.unique(y))
    return float(between/total)


def variance_fraction(values: np.ndarray) -> tuple[float,float]:
    z=np.asarray(values,float);doc=z.mean(1);grand=float(z.mean())
    total=float(np.mean((z-grand)**2));between=float(np.mean((doc-doc.mean())**2));within=float(np.mean((z-doc[:,None])**2))
    return between/total,within/total


def extreme_signature(values: np.ndarray, labels: np.ndarray, count: int=40) -> tuple[str,str]:
    x=np.asarray(values,float);y=np.asarray(labels,str);order=np.argsort(x);lo=y[order[:min(count,len(order))]];hi=y[order[-min(count,len(order)):]]
    def enriched(group):
        candidates=np.unique(y);base={c:max(np.mean(y==c),1e-9) for c in candidates}
        return max(candidates,key=lambda c:np.mean(group==c)/base[c])
    return str(enriched(lo)),str(enriched(hi))


def descriptive_axis_label(rows: pd.DataFrame, signatures: dict[str,tuple[str,str]], threshold: float=.02) -> str:
    families=["topic","register","source_type","formatting","position"]
    dominants=[]
    for fit in ("canonical","A","B"):
        row=rows.loc[rows.fit==fit].iloc[0];dominants.append(max(families,key=lambda f:float(row[f"eta2_{f}"])))
    if len(set(dominants))!=1:return "NO_SIMPLE_INTERPRETATION"
    dominant=dominants[0]
    if min(float(rows.loc[rows.fit==fit,f"eta2_{dominant}"].iloc[0]) for fit in ("canonical","A","B"))<threshold:return "NO_SIMPLE_INTERPRETATION"
    if signatures.get("A")!=signatures.get("B"):return "NO_SIMPLE_INTERPRETATION"
    return {"topic":"CONTENT_DOMINANT","register":"REGISTER_DOMINANT","source_type":"SOURCE_STYLE_MIXED","formatting":"SOURCE_STYLE_MIXED","position":"POSITION_STRUCTURAL"}[dominant]
