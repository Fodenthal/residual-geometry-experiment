from __future__ import annotations

import numpy as np
import pandas as pd

from src.persistent_state.slow_semantic.r110 import (
    descriptive_axis_label, eta_squared, geometry_row, orthonormal_nested,
    random_in_span, rank_curve_decision, variance_fraction,
)


def test_nested_geometry_and_random_sampling() -> None:
    rng=np.random.default_rng(2);directions=rng.normal(size=(12,30));q=orthonormal_nested(directions,8)
    np.testing.assert_allclose(q.T@q,np.eye(8),atol=1e-6)
    row,singular=geometry_row(q,q,8,30);assert row["overlap"]>.999999 and np.allclose(singular,1)
    draws=random_in_span(q,32,31);np.testing.assert_allclose(np.linalg.norm(draws,axis=1),1,atol=1e-6)


def test_effects_variance_and_axis_label() -> None:
    values=np.asarray([0,0,1,1],float);labels=np.asarray(["a","a","b","b"]);assert eta_squared(values,labels)==1
    between,within=variance_fraction(np.asarray([[0,0],[2,2]],float));assert between==1 and within==0
    rows=pd.DataFrame([{"fit":f,"eta2_topic":.2,"eta2_register":.01,"eta2_source_type":0,"eta2_formatting":0,"eta2_position":0} for f in ("canonical","A","B")])
    assert descriptive_axis_label(rows,{"A":("a","b"),"B":("a","b")})=="CONTENT_DOMINANT"


def test_registered_rank_decision() -> None:
    summary=pd.DataFrame([{"fit":f,"rank":k,"median":20} for f in ("full","A","B") for k in (64,128)])
    geometry=pd.DataFrame({"rank":[64,128],"chance_corrected_overlap":[.8,.75]})
    assert rank_curve_decision(summary,geometry)=="BROAD_PERSISTENT_REGION"
