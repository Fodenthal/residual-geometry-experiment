from __future__ import annotations

import numpy as np
import pandas as pd

from src.persistent_state.slow_semantic.r18 import (
    axis_decision, bootstrap_variance, match_axes, plus_one_p,
    readout_overlap, variance_components,
)


def test_variance_decomposition_closes_and_detects_between_state() -> None:
    rng=np.random.default_rng(1);doc=rng.normal(size=(40,1,3));within=.2*rng.normal(size=(40,12,3));z=doc+within
    result=variance_components(z)
    assert result["relative_closure_error"]<1e-12
    assert result["r_between"]>.9
    np.testing.assert_allclose(result["r_between"]+result["r_within"],1,atol=1e-12)


def test_document_bootstrap_is_deterministic() -> None:
    rng=np.random.default_rng(2);coords={"canonical":rng.normal(size=(20,4,2)),"A":rng.normal(size=(20,4,2)),"B":rng.normal(size=(20,4,2)),"pca":rng.normal(size=(20,4,2)),"random_000":rng.normal(size=(20,4,2))}
    a,sa=bootstrap_variance(coords,["random_000"],10,3);b,sb=bootstrap_variance(coords,["random_000"],10,3)
    pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(sa,sb)


def test_readout_overlap_is_basis_invariant() -> None:
    rng=np.random.default_rng(4);w=rng.normal(size=(12,3));q,_=np.linalg.qr(rng.normal(size=(12,12)));score,singular,rank=readout_overlap(w,q@(q.T@w))
    assert rank==2
    np.testing.assert_allclose(score,1,atol=1e-12);np.testing.assert_allclose(singular,1,atol=1e-12)


def test_matching_axis_gate_and_plus_one_p() -> None:
    a=np.eye(6)[:5];b=a[[1,0,2,4,3]];matched=match_axes(a,b)
    assert np.allclose(matched.matched_abs_cosine,1)
    assert axis_decision(matched)=="AXES_INTERPRETABLE"
    assert plus_one_p(.5,np.array([.1,.6,.2]))==.5
