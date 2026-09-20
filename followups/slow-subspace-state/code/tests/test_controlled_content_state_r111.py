import numpy as np
import pandas as pd

from src.persistent_state.slow_semantic.r111 import allocate_and_match,bootstrap_rows,normalized_contrast,splice_triplets,trajectory


def test_contrast_and_trajectory_sign():
    basis=np.eye(3)[:,:2];coef=np.asarray([[1,0],[0,1]],float);c=normalized_contrast(basis,coef,["business","home"])
    ref=np.zeros((2,5,3));same=ref.copy();different=ref.copy();different[:,:,0]=np.asarray([[1],[ -1]])
    _,_,d=trajectory(ref,same,different,c,np.asarray([1,-1]));assert np.all(d>0)
    assert len(bootstrap_rows(d,11))==55


def test_allocation_is_disjoint_and_splice_exact():
    topics=np.asarray(["business"]*224+["home"]*224);frame=pd.DataFrame({"topic":topics,"document_id":[f"d{i:03d}" for i in range(448)]})
    rng=np.random.default_rng(1);triplets,qc=allocate_and_match(frame,rng.normal(size=(448,5)),rng.normal(size=(448,3)))
    assert qc["unique_documents"]==384
    tokens=np.arange(448*20).reshape(448,20);ref,same,diff=splice_triplets(tokens,triplets,boundary=7)
    assert np.array_equal(ref[:,7:],same[:,7:]) and np.array_equal(ref[:,7:],diff[:,7:])
