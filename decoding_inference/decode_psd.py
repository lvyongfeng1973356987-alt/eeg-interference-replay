"""Participant-held-out SEED decoding with narrow and band-pooled PSD inputs."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

import joint_decode as j
from feature_audit import PSD_CASES


GRID = (("linear",.01),("rbf",.3),("rbf",3.),("rbf",30.))


def evaluate(meta, raw, family):
    subject, session, trial, label = meta.T
    data=j.relative_to_calibration(raw,subject,session)
    for held in np.unique(subject):
        train=np.flatnonzero((subject!=held)&(session>1))
        test=np.flatnonzero((subject==held)&(session>1))
        assert (len(train),len(test))==(420,30)
        scores=[[] for _ in GRID]
        for i_train,i_valid in GroupKFold(n_splits=4).split(train,label[train],subject[train]):
            fit, valid=train[i_train],train[i_valid]
            kernels={kind:j.kernel_matrices(data[fit,0],data[valid,0],kind)
                     for kind in ("linear","rbf")}
            for i,(kind,c) in enumerate(GRID):
                k0,k1=kernels[kind]
                prediction=j.fit_kernel(k0,label[fit],c).predict(k1)
                for person in np.unique(subject[valid]):
                    rows=subject[valid]==person
                    scores[i].append(balanced_accuracy_score(label[valid][rows],prediction[rows]))
        best=int(np.argmax([np.mean(x) for x in scores]))
        kind,c=GRID[best]
        x0,x1=j.kernel_matrices(data[train,0],data[test].reshape(-1,data.shape[-1]),kind)
        pred=j.fit_kernel(x0,label[train],c).predict(x1).reshape(len(test),-1)
        for i,guesses in zip(test,pred):
            for case,y in zip(PSD_CASES,guesses):
                yield family,*map(int,meta[i]),*case,int(y),kind,c
        print(f"PSD {family}, held-out subject {held:02d}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    output=[]
    with np.load(a.cache,allow_pickle=False) as data:
        for i,family in enumerate(("narrow","pooled")):
            output.extend(evaluate(data["meta"],data[f"x{i}"],family))
    j.write_csv(a.out,["family","subject","session","trial","label","draw","snr_db",
                       "synthesis","prediction","kernel","c"],output)
