"""Refit the matched SEED PSD decoders with fold-local PCA and wider regularization."""
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"decoding_inference"))
import align_psd as aligned
import feature_audit as features

j=features.j
REPRESENTATIONS=("narrow","pooled","beta_1hz","fold_1hz")
DIMENSIONS=(0,16,32,64)
GRID={"linear":(1e-4,1e-3,1e-2,1e-1,1.),"rbf":(.03,.3,3.,30.)}
CANDIDATES=[(d,k,c) for d in DIMENSIONS for k,g in GRID.items() for c in g]


def descriptors(signal):
    return (*aligned.aligned_features(signal),*aligned.aligned_features(signal,focused=True))


def extract(path, epochs, control):
    metadata, cache=[],[]
    for identifiers, clean in j.load_raw_trials(path):
        subject, session, trial, _ = identifiers
        spectrum=features.spectrum(clean)
        clean_features=descriptors(spectrum)
        variants={(0,0,"clean"):clean_features}
        if session==1:
            variants.update({key:clean_features for key in features.PSD_CASES[1:]})
        else:
            for draw in j.REALIZATIONS:
                parts=features.seed_parts(control,epochs,draw,subject,session,trial)
                for arm, components in zip(features.MODES,parts,strict=True):
                    noise,_=features.ledger(components)
                    wave=features.spectrum(noise)
                    for snr in features.PSD_LEVELS:
                        variants[draw,snr,arm]=descriptors(spectrum+features.f.amount(clean,noise,snr)*wave)
        metadata.append(identifiers)
        cache.append([variants[key] for key in features.PSD_CASES])
        if session==3 and trial==15:
            print(f"SEED PSD features: {subject}/15",flush=True)
    meta=np.asarray(metadata)
    assert len(meta)==675
    order=np.lexsort((meta[:,1],meta[:,2],meta[:,0]))
    arrays=[np.asarray([[case[i] for case in cases] for cases in cache])[order]
            for i in range(len(REPRESENTATIONS))]
    return meta[order],arrays


def gram_bank(train,test,dimensions=DIMENSIONS):
    scaler=StandardScaler().fit(train)
    x=scaler.transform(train).astype("float64")
    y=scaler.transform(test).astype("float64")
    if any(dimensions):
        pca=PCA(n_components=max(dimensions),svd_solver="randomized",
                random_state=20261122).fit(x)
        u,v=pca.transform(x),pca.transform(y)
    for dimension in dimensions:
        a,b=(u[:,:dimension],v[:,:dimension]) if dimension else (x,y)
        intercept=a@a.T
        held=b@a.T
        diag=np.diag(intercept)
        held_diag=np.einsum("ij,ij->i",b,b)
        gamma=1/(a.shape[1]*a.var()) if a.var()>0 else 1.
        yield dimension,{"linear":(intercept,held),
                         "rbf":(np.exp(-gamma*np.maximum(diag[:,None]+diag-2*intercept,0)),
                                np.exp(-gamma*np.maximum(held_diag[:,None]+diag-2*held,0)))}


def predict(meta, raw, family):
    subject,session,trial,label=meta.T
    data=j.relative_to_calibration(raw,subject,session)
    for person in np.unique(subject):
        train=np.flatnonzero((subject!=person)&(session>1))
        test=np.flatnonzero((subject==person)&(session>1))
        assert (len(train),len(test))==(420,30)
        scores={candidate:[] for candidate in CANDIDATES}
        splitter=GroupKFold(n_splits=4)
        for fit,valid in splitter.split(train,label[train],subject[train]):
            a,b=train[fit],train[valid]
            groups={s:subject[b]==s for s in np.unique(subject[b])}
            for dim,bank in gram_bank(data[a,0],data[b,0]):
                for kernel,grid in GRID.items():
                    x,y=bank[kernel]
                    for c in grid:
                        out=j.fit_kernel(x,label[a],c).predict(y)
                        scores[dim,kernel,c].extend(
                            balanced_accuracy_score(label[b][use],out[use]) for use in groups.values())
        dim,kernel,c=max(CANDIDATES,key=lambda candidate:np.mean(scores[candidate]))
        # Recompute the same 64-component fit when the winning slice uses PCA.
        dimensions=(dim,64) if dim and dim!=64 else (dim,)
        bank=next(grams for degree,grams in gram_bank(
            data[train,0],data[test].reshape(-1,data.shape[-1]),dimensions) if degree==dim)
        x,y=bank[kernel]
        guesses=j.fit_kernel(x,label[train],c).predict(y).reshape(len(test),-1)
        for index,row in zip(test,guesses,strict=True):
            for case,value in zip(features.PSD_CASES,row,strict=True):
                yield family,*map(int,meta[index]),*case,int(value),kernel,c,dim
        print(f"SEED {family}, participant {person:02d}: {dim}, {kernel}, C={c:g}",flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--epochs",type=Path,required=True)
    a=p.parse_args()
    aligned.validate()
    control=np.load(aligned.ROOT/"S1_Data_control_sequence.npz",allow_pickle=False)["control_sequence"]
    meta,arrays=extract(a.data,features.release_epochs(a.epochs),control)
    rows=[]
    for family,x in zip(REPRESENTATIONS,arrays,strict=True):
        rows.extend(predict(meta,x,family))
    j.write_csv(HERE/"seed_wide_grid.csv",["family","subject","session","trial","label","draw","snr_db",
               "synthesis","prediction","kernel","c","pca_components"],rows)
