"""SEED joint-artifact paired decoding, with tuning on clean training subjects only."""
import argparse
from pathlib import Path
import json
import re
from collections import Counter
import hashlib

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import loadmat
from scipy.signal import resample_poly, sosfilt
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from followup import BANDS, amount, components, drift_and_pulse, filter_design, load_artifacts, power, unit_noise, sensitivity, write_csv

BASE_SEED = 20261122
LEVELS = (5, 0, -5)
REALIZATIONS = (1, 2)
CASES = [(seed, db, scheme) for seed in REALIZATIONS for db in LEVELS
         for scheme in ("direct", "antialias")]
T = 7680
REF = 640
WINDOW = 512
STEP = 117
CHANNELS = (
    "Fp1 Fpz Fp2 AF3 AF4 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 "
    "FT8 T7 C5 C3 C1 Cz C2 C4 C6 T8 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 P7 P5 P3 P1 "
    "Pz P2 P4 P6 P8 PO7 PO5 PO3 POz PO4 PO6 PO8 CB1 O1 Oz O2 CB2"
).split()
EOG = dict(Fp1=1., Fpz=1., Fp2=1., AF3=.88, AF4=.88, F7=.78, F5=.68, F3=.58,
           F1=.58, Fz=.58, F2=.58, F4=.58, F6=.68, F8=.78, FC5=.43, FC1=.28, FC2=.28, FC6=.43)
EMG = dict(FT7=.85, FT8=.85, T7=1., T8=1., TP7=.85, TP8=.85, FC5=.55, FC6=.55,
           C3=.4, C4=.4, CP5=.72, CP6=.72, P7=.45, P8=.45)


def artifact(epochs, profile, rng):
    fade = 10
    ramp = np.linspace(0, 1, fade, dtype="float32")
    stream = np.empty(T + epochs.shape[1], dtype="float32")
    filled = 0
    while filled < T:
        piece = epochs[rng.integers(len(epochs))] * (-1 if rng.random() < .5 else 1)
        if filled:
            stream[filled-fade:filled] *= 1-ramp
            stream[filled-fade:filled] += ramp*piece[:fade]
            piece = piece[fade:]
        stream[filled:filled+len(piece)] = piece
        filled += len(piece)
    result = np.zeros((len(CHANNELS), T), dtype="float32")
    for name, weight in profile.items():
        shift = rng.integers(-2, 3)
        gain = rng.uniform(.92, 1.08)
        result[CHANNELS.index(name)] = np.roll(stream[:T], shift) * weight * gain
    return unit_noise(result)


def paired_noise(control, artifacts, seed, subject, session_index, trial):
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, seed, subject, session_index, trial]))
    start = rng.integers(0, len(control)-T+1)
    env = control[start:start+T]
    phase = rng.uniform(0, 2*np.pi, 3)
    drift, pulse = drift_and_pulse(env, rng)
    white = .1*rng.standard_normal((len(CHANNELS), T), dtype="float32")
    gain = (np.exp(-.2*np.abs(np.arange(len(CHANNELS))-len(CHANNELS)//2))*
            rng.uniform(.7, 1, len(CHANNELS))).astype("float32")[:, None]
    physical_background = drift + pulse + white
    physiology = (np.sqrt(.4)*artifact(artifacts[0], EOG, rng)
                  + np.sqrt(.35)*artifact(artifacts[1], EMG, rng))
    for lowpass in (False, True):
        line = components(env, phase, lowpass).sum(axis=0)
        simulated = unit_noise(gain*(line+physical_background))
        yield np.asarray(.5*simulated+physiology, dtype="float32")


def spans(x):
    return sliding_window_view(x[..., REF:], WINDOW, axis=-1)[..., ::STEP, :]


def centered(x):
    return x-x.mean(axis=-1, keepdims=True)


def half_logvar(var):
    return .5*np.log2(np.maximum(var,1e-10))


def descriptors(logvar, ref, band_data):
    """Three temporal bins, scalp power contrasts, and wideband spatial correlations."""
    average = np.stack([g.mean(axis=-1) for g in np.array_split(logvar,3,axis=-1)],axis=-1)
    ratio = average-ref[:,None]
    spatial = average-np.median(average,axis=0,keepdims=True)
    p0 = np.r_[average.ravel(), spatial.ravel()]
    p1 = np.r_[p0, ratio.ravel()]
    # Standardized correlations from a subsampled, continuous band waveform.
    # A single full-trial covariance per band is independent of channel adjacency.
    v = centered(band_data[:,REF::4])
    v /= np.maximum(np.sqrt(np.mean(v*v,axis=-1,keepdims=True)),1e-8)
    triu = np.triu_indices(len(CHANNELS),k=1)
    p2 = np.r_[p0, (v@v.T/v.shape[-1])[triu]]
    return p0,p1,p2


def features(clean, noise):
    by_case = [[[],[],[]] for _ in range(1+len(CASES))]
    beta_metrics = []
    for lo,hi in BANDS:
        sos = filter_design((lo,hi))
        x = centered(sosfilt(sos,spans(clean)))
        xb = centered(sosfilt(sos,clean[:,:REF]))
        xc = sosfilt(sos,clean)
        vx = np.mean(x*x,axis=-1)
        vb = np.mean(xb*xb,axis=-1)
        de = half_logvar(vx)
        base = half_logvar(vb)
        for j,v in enumerate(descriptors(de,base,xc)):
            by_case[0][j].append(v)
        for ix,raw_noise in enumerate(noise):
            z = centered(sosfilt(sos,spans(raw_noise)))
            zb = centered(sosfilt(sos,raw_noise[:,:REF]))
            zc = sosfilt(sos,raw_noise)
            vz = np.mean(z*z,axis=-1)
            vnb = np.mean(zb*zb,axis=-1)
            cov = np.einsum("cwt,cwt->cw",x,z)/WINDOW
            covb = np.einsum("ct,ct->c",xb,zb)/REF
            seed = REALIZATIONS[ix//2]
            mode = ("direct","antialias")[ix%2]
            for db in LEVELS:
                k = amount(clean,raw_noise,db)
                a = half_logvar(vx+2*k*cov+k*k*vz)
                b = half_logvar(vb+2*k*covb+k*k*vnb)
                case = CASES.index((seed,db,mode))
                for j,v in enumerate(descriptors(a,b,xc+k*zc)):
                    by_case[1+case][j].append(v)
                if lo==14:
                    beta_metrics.append([seed,db,mode,
                                         float(np.abs(a-b[:,None]-de+base[:,None]).mean()),
                                         float(10*np.log10(vx.mean()/(k*k*vz.mean())))])
    return [np.stack([np.concatenate(b[j]) for b in by_case]) for j in range(3)], beta_metrics


def load_raw_trials(root):
    cases = sorted((x for x in root.glob("*.mat") if re.fullmatch(r"\d+_\d+\.mat",x.name)),
                   key=lambda x:(int(x.stem.split("_")[0]),x.name))
    assert len(cases)==45
    labels = loadmat(root/"label.mat")["label"].ravel().astype(int)
    assert len(labels)==15 and set(labels)=={-1,0,1}
    for si,path in enumerate(cases):
        subject = int(path.stem.split("_")[0])
        session = si%3+1
        raw = loadmat(path)
        for trial,label in enumerate(labels,1):
            x = next(v for k,v in raw.items() if re.search(rf"eeg{trial}$",k))
            x = resample_poly(x[:,:12000].astype("float32"),16,25,axis=-1)
            assert x.shape==(62,T)
            yield (subject,session,trial,int(label)),x


def extract(seeddir, ctl, epochs, out):
    items = [[],[],[]]
    records, diagnostics = [],[]
    for row,x in load_raw_trials(seeddir):
        subject,session,trial,label = row
        noise = [wave for seed in REALIZATIONS for wave in paired_noise(ctl,epochs,seed,subject,session,trial)]
        f,delta = features(x,noise)
        records.append(row)
        for k,v in enumerate(f):
            items[k].append(v.astype("float32"))
        diagnostics.extend([[*row,*v] for v in delta])
        if trial==15:
            print(f"Features: subject {subject:02d}, session {session}/3",flush=True)
    assert len(records)==675
    write_csv(out/"trial_beta_metrics.csv",["subject","session","trial","label","realization","snr_db","synthesis","abs_delta_bde","beta_snr_db"],diagnostics)
    np.savez(out/"trial_features.npz",meta=np.asarray(records),**{f"x{i}":np.asarray(v) for i,v in enumerate(items)})


def relative_to_calibration(f, subject, session):
    result = f.copy()
    for s in np.unique(subject):
        pool=f[(subject==s)&(session==1),0]
        center=np.median(pool,axis=0)
        # Subtract only the location. Test-session inputs do not set the scale.
        result[subject==s]-=center
    return result


CANDIDATES = [(feature,kernel,c) for feature in range(3)
              for kernel,c in (("linear",.01),("rbf",.3),("rbf",3),("rbf",30))]


def kernel_matrices(train, test, kernel):
    scaler=StandardScaler().fit(train)
    fit=scaler.transform(train).astype("float64")
    evaluate=scaler.transform(test).astype("float64")
    gfit=fit@fit.T
    gtest=evaluate@fit.T
    if kernel=="linear":
        return gfit,gtest
    diagonal=np.diag(gfit)
    dist_fit=np.maximum(diagonal[:,None]+diagonal[None,:]-2*gfit,0.)
    dist_test=np.maximum(np.einsum("ij,ij->i",evaluate,evaluate)[:,None]+diagonal[None,:]-2*gtest,0.)
    gamma=1/(fit.shape[1]*fit.var()) if fit.var() else 1.
    return np.exp(-gamma*dist_fit),np.exp(-gamma*dist_test)


def fit_kernel(data, labels, c):
    return SVC(kernel="precomputed",C=c,class_weight="balanced").fit(data,labels)


def decode(cache,out):
    subject,session,trial,y=cache["meta"].T
    data=[cache[f"x{i}"] for i in range(3)]
    for x in data:
        assert np.isfinite(x).all()
    results=[]; selections=[]; accuracy={}
    for protocol in ("calibrated","strict_loso"):
        allowed=session!=1 if protocol=="calibrated" else np.ones(len(y),dtype=bool)
        X=[relative_to_calibration(x,subject,session) for x in data] if protocol=="calibrated" else data
        for s in np.unique(subject):
            train=np.flatnonzero(allowed&(subject!=s))
            test=np.flatnonzero(allowed&(subject==s))
            assert not set(subject[train])&set(subject[test])
            cv=list(GroupKFold(n_splits=4).split(train,y[train],subject[train]))
            scores=[[] for _ in CANDIDATES]
            for fit,valid in cv:
                idx_train=train[fit]
                idx_val=train[valid]
                # Each kernel is computed once per feature set and inner split.
                kernels={}
                for idx,(feat,kernel,c) in enumerate(CANDIDATES):
                    key=feat,kernel
                    if key not in kernels:
                        kernels[key]=kernel_matrices(X[feat][idx_train,0],X[feat][idx_val,0],kernel)
                    x0,x1=kernels[key]
                    guesses=fit_kernel(x0,y[idx_train],c).predict(x1)
                    scores[idx].extend(balanced_accuracy_score(y[idx_val][subject[idx_val]==v],
                                                               guesses[subject[idx_val]==v])
                                       for v in np.unique(subject[idx_val]))
            means=[float(np.mean(sc)) for sc in scores]
            best=int(np.argmax(means))
            feat,kernel,c=CANDIDATES[best]
            x0,x1=kernel_matrices(X[feat][train,0],X[feat][test].reshape(-1,X[feat].shape[-1]),kernel)
            yhat=fit_kernel(x0,y[train],c).predict(x1).reshape(len(test),-1)
            selections.append([protocol,int(s),feat,kernel,c,means[best],len(train),len(test)])
            accuracy[protocol,int(s)]=[balanced_accuracy_score(y[test],yhat[:,k]) for k in range(yhat.shape[1])]
            for j,i in enumerate(test):
                for idx,key in enumerate([(0,0,"clean"),*CASES]):
                    results.append([protocol,int(s),int(session[i]),int(trial[i]),int(y[i]),*key,int(yhat[j,idx])])
            print(f"Decoding: {protocol} subject {s:02d}, clean={accuracy[protocol,int(s)][0]:.3f}",flush=True)
    write_csv(out/"trial_predictions.csv",["protocol","subject","session","trial","label","realization","snr_db","synthesis","prediction"],results)
    write_csv(out/"clean_inner_selection.csv",["protocol","outer_subject","feature","kernel","C","mean_inner_clean_ba","train_trials","test_trials"],selections)
    return accuracy


def paired_summary(matrix):
    rng=np.random.default_rng(BASE_SEED)
    sizes=rng.integers(0,len(matrix),size=(20000,len(matrix)))
    mean=matrix.mean(axis=0)
    return {
        "clean":float(mean[0]),
        "clean_ci":[float(v) for v in np.quantile(matrix[sizes,0].mean(axis=1),[.025,.975])],
        "snr":{str(db):summarize_level(matrix,sizes,db) for db in LEVELS},
    }


def summarize_level(m,samples,db):
    d=m[:,[1+CASES.index((seed,db,"direct")) for seed in REALIZATIONS]].mean(axis=1)
    a=m[:,[1+CASES.index((seed,db,"antialias")) for seed in REALIZATIONS]].mean(axis=1)
    delta=a-d
    return {"direct":float(d.mean()),"antialias":float(a.mean()),"paired_difference":float(delta.mean()),
            "ci95":np.quantile(delta[samples].mean(axis=1),[.025,.975]).tolist(),
            "paired_sd":float(delta.std(ddof=1)),
            "wilcoxon_p":float(wilcoxon(delta).pvalue) if np.any(delta) else 1.,
            "mde_80":sensitivity(delta,.05)["mde_80"],
            "realization_differences":[float(np.mean(m[:,1+CASES.index((seed,db,"antialias"))]-
                                                     m[:,1+CASES.index((seed,db,"direct"))])) for seed in REALIZATIONS]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-data",type=Path,required=True)
    p.add_argument("--control",type=Path,required=True)
    p.add_argument("--epochs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--reuse-features",action="store_true")
    p.add_argument("--features-cache",type=Path)
    opt=p.parse_args()
    opt.out.mkdir(exist_ok=True,parents=True)
    paths=[opt.control,opt.epochs/"EOG_all_epochs.mat",opt.epochs/"EMG_all_epochs.mat"]
    manifest={"seed":BASE_SEED,"primary":{"protocol":"calibrated","global_snr_db":0},
              "sensitivity":{"protocol":"strict_loso","snr_cuts_db":[5,0,-5]},
              "simulation_seeds":list(REALIZATIONS),"candidates":CANDIDATES,
              "source_hashes":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
              "split":"LOSO, nested four-group clean-only selection; first-session clean calibration excluded from fitting and testing in primary"}
    (opt.out/"protocol.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    if not opt.reuse_features:
        extract(opt.seed_data,np.load(opt.control,allow_pickle=False)["control_sequence"],
                load_artifacts(opt.epochs),opt.out)
    with np.load(opt.features_cache or opt.out/"trial_features.npz",allow_pickle=False) as cache:
        a=decode(cache,opt.out)
    summary={}
    for name in ("calibrated","strict_loso"):
        scores=np.array([a[name,i] for i in range(1,16)])
        summary[name]=paired_summary(scores)
    (opt.out/"results.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()

