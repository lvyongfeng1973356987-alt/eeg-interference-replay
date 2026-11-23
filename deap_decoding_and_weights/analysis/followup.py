"""Prespecified follow-up to the Sensors EEG line-propagation audit."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import pickle
import json
from zipfile import ZipFile

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import loadmat
from scipy.optimize import brentq
from scipy.signal import butter, iirnotch, resample_poly, sosfilt, sosfiltfilt, tf2sos
from scipy.stats import nct, t
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from paired_audit import ACTIVE, BANDS, FILTERS, FS, INDEX_GAINS, comparisons, components, holm_family, windows, write_csv

ALL_BANDS = [*BANDS, (31,45)]
SEED = 20261121


def power(x):
    return np.mean(np.square(x), dtype="float64")


def filter_design(band, order=5):
    return butter(order,band,btype="bandpass",fs=FS,output="sos").astype("float32")


def de(x, sos):
    return .5*np.log2(np.maximum(np.var(sosfilt(sos,x),axis=-1),1e-10))


def retention(control, output):
    rows = []
    accum = defaultdict(lambda: np.zeros(2))
    notch = tf2sos(*iirnotch(50,30,fs=FS))
    rng = np.random.default_rng(SEED)
    for block, envelope in enumerate(control.reshape(-1,7680),1):
        env = np.pad(envelope,(384,0),mode="edge")
        phases = rng.uniform(0,2*np.pi,3)
        direct = components(env, phases).sum(axis=0)
        aa = components(env,phases,True).sum(axis=0)
        for path, x in (("direct",direct),("antialias",aa)):
            for protect in ("none","notch50"):
                z = sosfiltfilt(notch,x) if protect=="notch50" else x
                for band in ALL_BANDS:
                    sos = filter_design(band).astype("float64")
                    for design,f in [
                        ("causal5",lambda y: sosfilt(sos,y)),
                        ("bidirectional5",lambda y: sosfiltfilt(sos,y)),
                        ("causal10",lambda y: sosfilt(filter_design(band,10).astype("float64"),y)),
                        ("causal_cascade5x2",lambda y: sosfilt(np.vstack([sos,sos]),y)),
                    ]:
                        for span in ("window","full"):
                            filtered=f(windows(z)) if span=="window" else windows(f(z))
                            numerator=np.sum(np.var(filtered,axis=-1))
                            denominator=np.sum(np.var(windows(x),axis=-1))
                            key=path,protect,design,span,*band
                            accum[key] += numerator,denominator
                            rows.append((block,*key,numerator,denominator))
    cols=["path","protection","filter","span","low_hz","high_hz"]
    write_csv(output/"retention_blocks.csv",["block",*cols,"filtered_variance_sum","original_variance_sum"],rows)
    write_csv(output/"retention_summary.csv",cols+["retention_pct"],
              [(*k,100*v[0]/v[1]) for k,v in accum.items()])


def drift_and_pulse(env, rng, width=.1, threshold=.6):
    tgrid = np.arange(len(env))/FS
    drift = np.zeros(len(env),dtype="float32")
    nseg=max(1,len(env)//(10*FS))
    bounds = np.linspace(0,len(env),nseg+1,dtype=int)
    for start,end in zip(bounds[:-1],bounds[1:]):
        f=rng.uniform(.1,.5,5)
        a=rng.uniform(.2,1,5)
        ph=rng.uniform(0,2*np.pi,5)
        drift[start:end]=(0.5+1.5*env[start:end].mean())*(5*a/a.sum())@np.sin(2*np.pi*f[:,None]*tgrid[start:end]+ph[:,None])
    ix=np.flatnonzero(env>threshold)
    pulse=np.zeros_like(drift)
    if len(ix):
        for group in np.split(ix,np.flatnonzero(np.diff(ix)>.1*FS)+1):
            peak=group[np.argmax(env[group])]
            w=env[peak]
            sigma=width/(3*(1+w))
            carrier=sosfiltfilt(butter(4,min(.3/sigma,63.9),fs=FS,output="sos"),
                                rng.standard_normal(len(env)))
            pulse += w*w*np.exp(-.5*((tgrid-tgrid[peak])/sigma)**2)*carrier
    return drift,pulse


def controls_from_wav(wav, out):
    from scipy.io import wavfile
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import filtfilt
    fs,audio=wavfile.read(wav,mmap=True)
    assert fs==48000
    low=np.empty(int(round(len(audio)/fs*128)),np.float32)
    for sec in range(0,len(audio)//fs,60):
        end=min(sec+60,len(audio)//fs)
        a=max(0,sec-1)
        b=min(len(audio)//fs,end+1)
        raw=np.asarray(audio[a*fs:b*fs],dtype=np.float32)
        if raw.ndim==2:
            raw=(raw[:,0]+raw[:,1])*.5
        z=resample_poly(raw/32768.,1,375)
        start=(sec-a)*128
        low[sec*128:end*128]=z[start:start+(end-sec)*128]
    b,a=butter(4,1.5/64,btype="low")
    env=filtfilt(b,a,np.abs(low),padlen=15)
    variants={}
    for n in [32,64,128]:
        v=uniform_filter1d(env,size=n,mode="nearest")
        v=np.clip(v,0,np.percentile(v,99))
        variants[f"s{n}"]=((v-v.min())/(v.max()-v.min())).astype("float32")
    np.savez_compressed(out/"control_variants.npz",**variants)


def event_summary(env,threshold,width):
    ix=np.flatnonzero(env>threshold)
    mask=np.zeros(len(env),dtype=bool)
    groups=[] if not len(ix) else np.split(ix,np.flatnonzero(np.diff(ix)>.1*FS)+1)
    for group in groups:
        peak=group[np.argmax(env[group])]
        radius=int(np.ceil(width/(1+float(env[peak]))*FS))
        mask[max(0,peak-radius):min(len(env),peak+radius+1)]=True
    return len(groups),float(mask.mean()),float((env>threshold).mean())


def config_audit(clean,control,variants,subject,trial,artifacts):
    env,phase,rng=draw_control(control,subject,trial)
    # Recover the identical slice index for all smoothing variants.
    start=np.random.default_rng(SEED+100*subject+trial).integers(0,len(control)-8064+1)
    a,b=(unit_noise(artifact(epochs,profile,rng)) for epochs,profile in zip(artifacts,[EOG_PROFILE,EMG_PROFILE]))
    gain=(INDEX_GAINS*rng.uniform(.7,1,32)).astype("float32")[:,None]
    white=(.1*rng.standard_normal(clean.shape)).astype("float32")
    specs=[("default",64,.6,.1,.25),("threshold04",64,.4,.1,.25),
           ("threshold08",64,.8,.1,.25),("width005",64,.6,.05,.25),
           ("width02",64,.6,.20,.25),("smooth32",32,.6,.1,.25),("smooth128",128,.6,.1,.25),
           ("simulated015",64,.6,.1,.15),("simulated035",64,.6,.1,.35)]
    sos=FILTERS[2]
    d0,b0=de(windows(clean),sos),de(clean[:,:384],sos)
    results=[]
    for name,smooth,threshold,width,frac in specs:
        w=env if smooth==64 else variants[f"s{smooth}"][start:start+8064]
        line=components(w,phase).sum(axis=0)
        rr=np.random.default_rng(SEED+100000+100*subject+trial)
        drift,pulse=drift_and_pulse(w,rr,width,threshold)
        sim=gain*(line+drift+pulse+white)
        m=np.sqrt(frac)*unit_noise(sim)+np.sqrt((1-frac)*.4/.75)*a+np.sqrt((1-frac)*.35/.75)*b
        event,duty,above=event_summary(w,threshold,width)
        results.append([subject,trial,name,event,duty,above,power(gain*pulse)/power(sim),
                        feature_delta(clean,m*amount(clean,m,-5),sos,d0,b0)])
    return results

def draw_control(control, subject, trial):
    rng=np.random.default_rng(SEED+100*subject+trial)
    start=rng.integers(0,len(control)-8064+1)
    env=control[start:start+8064]
    phases=rng.uniform(0,2*np.pi,3)
    return env,phases,rng


def amount(clean,n,db):
    return np.float32(np.sqrt(power(clean)/power(n)*10**(-db/10)))


def line_metrics(clean, lines, sos):
    xv=sosfilt(sos,windows(clean))
    xb=sosfilt(sos,clean[:,:384])
    xv-=xv.mean(axis=-1,keepdims=True)
    xb-=xb.mean(axis=-1,keepdims=True)
    v=np.mean(xv*xv,axis=-1)
    b=np.mean(xb*xb,axis=-1)
    d=.5*np.log2(np.maximum(v,1e-10))
    r=.5*np.log2(np.maximum(b,1e-10))
    clean_feat=d-r[:,None]
    x_center=clean_feat-clean_feat.mean(axis=-1,keepdims=True)
    var0=np.mean(x_center*x_center,axis=-1)
    output=[]
    for line, gain in lines:
        z=sosfilt(sos,windows(line))
        zb=sosfilt(sos,line[:384])
        z-=z.mean(axis=-1,keepdims=True)
        zb-=zb.mean()
        k=gain[:,None]
        cross=np.einsum("cwt,wt->cw",xv,z)/512
        base_cross=np.einsum("ct,t->c",xb,zb)/384
        vp=v+2*k*cross+k*k*np.mean(z*z,axis=-1)
        bp=b+2*gain*base_cross+gain*gain*np.mean(zb*zb)
        noisy=.5*np.log2(np.maximum(vp,1e-10))
        ref=.5*np.log2(np.maximum(bp,1e-10))
        feature=noisy-ref[:,None]
        centered=feature-feature.mean(axis=-1,keepdims=True)
        v1=np.mean(centered*centered,axis=-1)
        corr=np.mean(x_center*centered,axis=-1)/np.maximum(np.sqrt(var0*v1),1e-20)
        output.append([float(np.abs(noisy-d).mean()),float(np.abs(feature-clean_feat).mean()),
                       float(np.median(v1/np.maximum(var0,1e-20))),float(np.mean(corr)),
                       float(10*np.log10(v.mean()/np.mean(k*k*np.mean(z*z,axis=-1)))),
                       float(np.mean(np.abs(feature))),float(np.mean(np.abs(clean_feat)))])
    return output


def deap_files(archive):
    with ZipFile(archive) as z:
        for subject in range(1,33):
            name=next(n for n in z.namelist() if n.split("/")[-1]==f"s{subject:02}.dat")
            with z.open(name) as f:
                yield subject,pickle.load(f,encoding="latin1")


def unit_noise(noise):
    return noise/np.float32(np.sqrt(power(noise)))


def feature_delta(clean, noise, sos, d0, b0):
    x=clean+noise
    return float(np.abs(de(windows(x),sos)-de(x[:,:384],sos)[:,None]-(d0-b0[:,None])).mean())


def artifact(epochs,profile,rng):
    fade=10
    stream=np.empty(8064+epochs.shape[1],dtype="float32")
    filled=0
    while filled<8064:
        piece=epochs[rng.integers(len(epochs))]*(-1 if rng.random()<.5 else 1)
        if filled==0:
            stream[:len(piece)]=piece
            filled=len(piece)
            continue
        ramp=np.linspace(0,1,fade,dtype="float32")
        stream[filled-fade:filled]*=1-ramp
        stream[filled-fade:filled]+=ramp*piece[:fade]
        stream[filled:filled+len(piece)-fade]=piece[fade:]
        filled+=len(piece)-fade
    result=np.zeros((32,8064),dtype="float32")
    for idx,weight in profile.items():
        shift=rng.integers(-2,3)
        gain=rng.uniform(.92,1.08)
        result[idx]=np.roll(stream[:8064],shift)*weight*gain
    return result


EOG_PROFILE={0:1,1:.88,2:.58,3:.78,4:.43,5:.28,16:1,17:.88,18:.58,19:.78,20:.43,21:.28,30:.58}
EMG_PROFILE={4:.55,6:.4,7:1,8:.72,9:.45,20:.55,22:.4,23:1,24:.72,25:.45}


def load_artifacts(folder):
    result=[]
    for kind,expected in (("EOG",(3400,512)),("EMG",(5598,512))):
        name=f"{kind}_all_epochs"
        if (folder/(name+".npy")).exists():
            x=np.load(folder/(name+".npy"),allow_pickle=False)
        else:
            md=loadmat(folder/(name+".mat"))
            assert int(md["fs"].item())==256
            x=md[name]
        if x.T.shape==expected:
            x=x.T
        assert x.shape==expected,(name,x.shape)
        x=resample_poly(x.astype("float32"),1,2,axis=-1)
        x-=x.mean(axis=-1,keepdims=True)
        result.append(x)
    return result


def deap(archive,control,output,artifacts=None, mixtures_only=False, variants=None):
    filters=[filter_design(b) for b in ALL_BANDS]
    names=["direct_m5","antialias_m5","direct_m20"]
    rows,power_rows,mixed_rows,cfg_rows=[],[],[],[]
    for subject,dataset in deap_files(archive):
        for trial,raw in enumerate(dataset["data"],1):
            clean=np.asarray(raw[:32],dtype="float32")
            assert windows(clean).shape==(32,62,512)
            if variants is not None and trial in (1,11,21,31):
                cfg_rows.extend(config_audit(clean,control,variants,subject,trial,artifacts))
            env,phases,rng=draw_control(control,subject,trial)
            line=components(env,phases).sum(axis=0)
            aa=components(env,phases,True).sum(axis=0)
            gains=(INDEX_GAINS*rng.uniform(.7,1,32)).astype("float32")
            if not mixtures_only:
                lines=[(z,gains*amount(clean,gains[:,None]*z,db)) for z,db in [(line,-5),(aa,-5),(line,-20)]]
                for band,sos in zip(ALL_BANDS,filters):
                    rows.extend([subject,trial,name,*band,*vals] for name,vals in zip(names,line_metrics(clean,lines,sos)))
            drift,pulse=drift_and_pulse(env,rng)
            white=(.1*rng.standard_normal(clean.shape)).astype("float32")
            weighting=gains[:,None]
            background=weighting*(drift+pulse+white)
            orig=weighting*line+background
            aa_sim=weighting*aa+background
            powers=[power(z) for z in (weighting*line,weighting*drift,weighting*pulse,weighting*white,orig)]
            power_rows.append([subject,trial,*powers])
            mixes = {"synthetic_direct":orig,"synthetic_antialias":aa_sim}
            if artifacts is not None:
                a,b=(unit_noise(artifact(epochs,profile,rng)) for epochs,profile in zip(artifacts,[EOG_PROFILE,EMG_PROFILE]))
                for frac in [.15,.25,.35]:
                    fixed=np.sqrt((1-frac)*.4/.75)*a+np.sqrt((1-frac)*.35/.75)*b
                    mixes[f"joint_{int(frac*100):02}_direct"]=np.sqrt(frac)*unit_noise(orig)+fixed
                    if frac==.25:
                        mixes["joint_25_antialias"]=np.sqrt(frac)*unit_noise(aa_sim)+fixed
            for band,sos in zip([BANDS[2],BANDS[3]],filters[2:4]):
                d0,b0=de(windows(clean),sos),de(clean[:,:384],sos)
                for k,n in mixes.items():
                    mixed_rows.append([subject,trial,k,*band,feature_delta(clean,n*amount(clean,n,-5),sos,d0,b0)])
        print(f"DEAP: subject {subject}/32",flush=True)
    if rows:
        write_csv(output/"deap_reference_trials.csv",
                  ["subject","trial","condition","low_hz","high_hz","abs_de","abs_bde","temporal_variance_ratio","temporal_r","band_snr","abs_noisy_bde","abs_clean_bde"],rows)
    if cfg_rows:
        write_csv(output/"control_sensitivity_trials.csv",["subject","trial","condition","events","duty","above","pulse_over_sim","beta_abs_bde"],cfg_rows)
    write_csv(output/("joint_component_power.csv" if mixtures_only else "simulated_component_power.csv"),["subject","trial","line","drift","pulse","white","mixture"],power_rows)
    write_csv(output/("joint_trials.csv" if artifacts is not None else "synthetic_trials.csv"),["subject","trial","condition","low_hz","high_hz","abs_bde"],mixed_rows)


def seed_features(x,filters):
    windows = sliding_window_view(x,512,axis=-1)[...,::256,:]
    return np.concatenate([np.stack([de(w,sos).mean(axis=-1)
                                      for w in np.array_split(windows,3,axis=1)],axis=-1).ravel()
                           for sos in filters])


def seed_loso(folder,control,output):
    import re
    data=[]
    y=[]
    g=[]
    names=("clean","direct","antialias")
    paths=sorted(p for p in folder.glob("*.mat") if re.fullmatch(r"\d+_\d+\.mat",p.name))
    assert len(paths)==45
    label=loadmat(folder/"label.mat")["label"].ravel()
    assert len(label)==15
    filters=[filter_design(b) for b in BANDS]
    for j,path in enumerate(paths):
        data_mat=loadmat(path)
        for trial in range(1,16):
            raw=next(x for name,x in data_mat.items() if re.search(rf"eeg{trial}$",name))
            x=resample_poly(raw[:,:12000].astype("float32"),16,25,axis=-1)
            assert x.shape==(62,7680)
            env,ph,rng=draw_control(control,j+1,trial)
            gain=(np.exp(-.2*np.abs(np.arange(62)-31))*rng.uniform(.7,1.,62)).astype("float32")
            variants=[x]
            for high_rate in [False,True]:
                noise=gain[:,None]*components(env[:7680],ph,high_rate).sum(axis=0)
                variants.append(x+amount(x,noise,-5)*noise)
            data.append([seed_features(z,filters) for z in variants])
            y.append(label[trial-1])
            g.append(int(path.stem.split("_")[0]))
        print(f"SEED: session {j+1}/45",flush=True)
    f,y,g=np.asarray(data),np.asarray(y),np.asarray(g)
    session=np.repeat(np.asarray([int(p.stem.split("_")[1]) for p in paths]),15)
    rows=[]
    for protocol in ("uncalibrated","calibrated"):
        selected=np.ones(len(g),dtype=bool)
        features=f.copy()
        if protocol=="calibrated":
            for sid in np.unique(g):
                calibration=(g==sid)&(session==session[g==sid].min())
                selected[calibration]=False
                mean=f[calibration,0].mean(axis=0)
                sd=f[calibration,0].std(axis=0)
                features[g==sid]=(f[g==sid]-mean)/np.maximum(sd,0.1)
        results_rows=[]
        for sid in np.unique(g):
            train=selected&(g!=sid)
            test=selected&(g==sid)
            model=make_pipeline(StandardScaler(),SVC(C=10,kernel="rbf",gamma="scale",class_weight="balanced"))
            model.fit(features[train,0],y[train])
            for j,key in enumerate(names):
                results_rows.append([sid,key,balanced_accuracy_score(y[test],model.predict(features[test,j]))])
        rows.extend([[protocol,*r] for r in results_rows])
    for sid in np.unique(g):
        pooled=[[],[],[]]
        label_test=[]
        for sess in np.unique(session[g==sid]):
            train=(g==sid)&(session!=sess)
            test=(g==sid)&(session==sess)
            model=make_pipeline(StandardScaler(),SVC(C=10,kernel="rbf",gamma="scale",class_weight="balanced"))
            model.fit(f[train,0],y[train])
            label_test.extend(y[test])
            for j in range(3):
                pooled[j].extend(model.predict(f[test,j]))
        rows.extend([["within_subject_session_holdout",sid,key,balanced_accuracy_score(label_test,pred)]
                     for key,pred in zip(names,pooled)])
    write_csv(output/"seed_loso_scores.csv",["protocol","subject","condition","balanced_accuracy"],rows)
    result={}
    for protocol in ("uncalibrated","calibrated","within_subject_session_holdout"):
        values=np.asarray([v for p,_,_,v in rows if p==protocol]).reshape(15,3)
        result[protocol]=comparisons(values,names)
        result[protocol]["same_snr"]=comparisons(values[:,1:],names[1:])
        result[protocol]["paired_sensitivity"]=sensitivity(values[:,2]-values[:,1],.05)
    (output/"seed_loso_results.json").write_text(json.dumps(result,indent=2),encoding="utf8")


def sensitivity(paired, alpha):
    n=len(paired)
    sem=float(np.std(paired,ddof=1)/np.sqrt(n))
    cutoff=t.ppf(1-alpha/2,n-1)
    def pwr(standardized):
        return nct.cdf(-cutoff,n-1,standardized)+nct.sf(cutoff,n-1,standardized)
    return {"n":n,"alpha":alpha,"paired_se":sem,"critical":cutoff*sem,
            "mde_80":brentq(lambda d:pwr(d)-.8,0,8)*sem if sem>0 else None}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode",choices=["retention","deap","seed","joint","controls"])
    cli.add_argument("--control",type=Path,required=True)
    cli.add_argument("--data",type=Path)
    cli.add_argument("--epochs",type=Path)
    cli.add_argument("--variants",type=Path)
    cli.add_argument("--out",type=Path,default=Path(__file__).parent)
    opt=cli.parse_args()
    control=np.load(opt.control,allow_pickle=False)["control_sequence"]
    if opt.mode=="controls":
        controls_from_wav(opt.data,opt.out)
    elif opt.mode=="retention":
        retention(control,opt.out)
    elif opt.mode=="seed":
        seed_loso(opt.data,control,opt.out)
    elif opt.mode=="joint":
        deap(opt.data,control,opt.out,load_artifacts(opt.epochs),mixtures_only=True,variants=np.load(opt.variants) if opt.variants else None)
    else:
        deap(opt.data,control,opt.out)


if __name__=="__main__":
    main()

