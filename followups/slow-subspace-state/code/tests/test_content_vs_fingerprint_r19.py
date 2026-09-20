from __future__ import annotations

import numpy as np
import pandas as pd

from src.persistent_state.slow_semantic.r19 import (
    collapse_from_fit, final_outcome, fingerprint_frame, source_type, structural_features,
)


def test_source_and_structural_features_are_deterministic() -> None:
    assert source_type("forums.example.com") == "forum_or_community"
    assert source_type("agency.gov") == "government"
    a=structural_features("TITLE:\n- item 1\n- item 2?", "https://example.org/a/b?q=1")
    b=structural_features("TITLE:\n- item 1\n- item 2?", "https://example.org/a/b?q=1")
    assert a==b and a["list_line_fraction"]>0 and a["url_path_depth"]==2


def test_collapse_uses_fit_support_only() -> None:
    values=np.asarray(["a","a","b","b","b","c"])
    collapsed,categories=collapse_from_fit(values,np.asarray([0,1,2,5]),minimum=2)
    assert categories==["a","__LOW_SUPPORT__"]
    assert collapsed.tolist()==["a","a","__LOW_SUPPORT__","__LOW_SUPPORT__","__LOW_SUPPORT__","__LOW_SUPPORT__"]


def test_fingerprint_schema_and_outcomes() -> None:
    frame=pd.DataFrame({"document_id":["a","b","c"],"url":["https://x.gov/a"]*3,"host":["x.gov"]*3,
        "text":["A.\n- 1"]*3,"register":["news"]*3,"formatting":["list"]*3})
    fp,schema=fingerprint_frame(frame,np.arange(3))
    assert len(fp)==3 and schema["families"]==["source/domain","genre/register","format/template"]
    assert final_outcome(True,True,True,False,.1)[0]=="CONTENT_STATE_IDENTIFIED"
    assert final_outcome(True,False,False,True,.1)[0]=="DOCUMENT_FINGERPRINT_DOMINANT"
