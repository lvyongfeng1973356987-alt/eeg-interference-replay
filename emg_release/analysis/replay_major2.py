"""SEED/DEAP synthesis x spatial support x reference replay for Major 2."""
import argparse
from pathlib import Path
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import sosfilt

import followup as f
import joint_decode as j
from emg_source import release_epochs

SOS=f.filter_design((14,31))
LEVELS=(0,-5)
L648=640
LN2=np.log(2.)


def seed_draw(control,epochs,si,session,t,draw):
    r=np.random.default_rng(np.random.SeedSequence([j.BASE_SEED,draw,si,session,t]))
    start=r.integers(len(control)-j.T+1)
    env=control[start:start+j.T]
    phase=r.uniform(0,2*np.pi,3)
    drift,pulse=f.drift_and_pulse(env,r)
    white=.1*r.standard_normal((62,j.T),dtype="float32")
    gain=(np.exp(-.2*np.abs(np.arange(62)-31))*r.uniform(.7,1,62)).astype("float32")[:,None]
    eye=j.artifact(epochs[0],j.EOG,r)
    muscle=j.artifact(epochs[1],j.EMG,r)
    return env,phase,gain,drift+pulse+white,eye,muscle


def deap_draw(control,epochs,si,t):
    env,phase,r=f.draw_control(control,si,t)
    gain=(f.INDEX_GAINS*r.uniform(.7,1,32)).astype("float32")[:,None]
    drift,pulse=f.drift_and_pulse(env,r)
    white=(.1*r.standard_normal((32,8064))).astype("float32")
    eye=f.unit_noise(f.artifact(epochs[0],f.EOG_PROFILE,r))
    muscle=f.unit_noise(f.artifact(epochs[1],f.EMG_PROFILE,r))
    return env,phase,gain,drift+pulse+white,eye,muscle


def spread(x):
    """Copy the matched epoch streams and jitter cyclically into inactive channels.
    Normalize channel power for this equal-power placement intervention."""
    active=np.flatnonzero(np.any(x,axis=1))
    v=x.copy()
    inactive=np.setdiff1d(np.arange(len(x)),active)
    v[inactive]=x[np.resize(active,len(inactive))]
    v/=np.sqrt(np.mean(v*v,axis=-1,keepdims=True))
    return f.unit_noise(v)


def cepstrum(v):
    return .5*np.log2(np.maximum(v,1e-10))


def filt(v):
    w=sosfilt(SOS,v)
    w-=w.mean(axis=-1,keepdims=True)
    return w


def component_waves(env,phase,gain,bg,cohort):
    old=f.components(env,phase,False)
    new=f.components(env,phase,True)
    d=gain*old.sum(axis=0)+gain*bg if cohort=="DEAP" else gain*(old.sum(axis=0)+bg)
    a=gain*new.sum(axis=0)+gain*bg if cohort=="DEAP" else gain*(new.sum(axis=0)+bg)
    h=gain*old[1:].sum(axis=0)*np.float32(.5/np.sqrt(f.power(d)))
    return f.unit_noise(d),f.unit_noise(a),h


def window(v,start):
    return sliding_window_view(v[:,start:],512,axis=-1)[:,::117]


def window_pair(x,y,stim):
    start=stim
    vx=filt(window(x,start))
    late=((640+116)//117) if stim else 0
    return vx,filt(window(y,start)),late


def mean_abs(x):
    return float(np.mean(np.abs(x),dtype=np.float64))


def per_trial(cohort,meta,x,noise):
    env,phase,gain,bg,e,m=noise
    d,a,h=component_waves(env,phase,gain,bg,cohort)
    stim=384 if cohort=="DEAP" else 0
    start=stim if cohort=="DEAP" else 640
    original5=(0,640)
    original3=(0,384)
    bounds={"early3":(stim,stim+384),"early5":(stim,stim+640)}
    if cohort=="DEAP":
        bounds["pre3"]=original3
    full_x=filt(window(x,start))
    # Align the scoring grid to the original DEAP stimulus start and drop all
    # windows that overlap any candidate early reference.
    drop=math.ceil(640/117) if cohort=="DEAP" else 0
    vx=np.mean(full_x**2,axis=-1)
    clean_de=cepstrum(vx)
    xr={k:filt(x[:,l:r]) for k,(l,r) in bounds.items()}
    ref0={k:np.mean(z*z,axis=-1) for k,z in xr.items()}
    harm=filt(window(h,start))
    harmonic=np.var(harm[:,drop:],axis=-1).mean(axis=-1)
    rows=[]
    for placement in ("native","even"):
        qe,qm=(e,m) if placement=="native" else (spread(e),spread(m))
        p=np.sqrt(.4)*qe+np.sqrt(.35)*qm
        pe=np.mean(p*p,axis=-1)
        effective=float(pe.sum()**2/np.sum(pe**2))
        n_active=int(np.count_nonzero(pe))
        cases=[]
        for signal in (d,a):
            n=.5*signal+p
            z=filt(window(n,start))
            zn=(z*z).mean(axis=-1)
            cc=np.einsum("cwt,cwt->cw",full_x,z)/512
            rb={}
            for name,(l,r) in bounds.items():
                rz=filt(n[:,l:r])
                rb[name]=(np.mean(rz*rz,axis=-1),np.mean(rz*xr[name],axis=-1))
            cases.append((n,zn,cc,rb))
        for db in LEVELS:
            result=[]
            for n,zn,cc,rb in cases:
                k=f.amount(x,n,db)
                v=vx+2*k*cc+k*k*zn
                rvars={name:ref0[name]+2*k*cv+k*k*nv for name,(nv,cv) in rb.items()}
                result.append((k,v,rvars))
            old,new=result
            for ref in [*bounds,"nominal"]:
                rn=("pre3" if cohort=="DEAP" else "early5") if ref=="nominal" else ref
                sl=slice(0 if ref=="nominal" else drop,None)
                k0,vd,r0=old
                k1,va,r1=new
                de_d=cepstrum(vd[:,sl])-clean_de[:,sl]
                de_a=cepstrum(va[:,sl])-clean_de[:,sl]
                ref_d=cepstrum(r0[rn])-cepstrum(ref0[rn])
                ref_a=cepstrum(r1[rn])-cepstrum(ref0[rn])
                bde_d=de_d-ref_d[:,None]; bde_a=de_a-ref_a[:,None]
                # Mechanism intervention: add the measured direct harmonic
                # window variance to both AA variances; no waveform re-scaling.
                L=harmonic*k0*k0
                injected=cepstrum(va[:,sl]+L[:,None])-clean_de[:,sl]
                injected_ref=cepstrum(r1[rn]+L)-cepstrum(ref0[rn])
                rows.append([*meta,placement,db,ref,n_active,effective,
                    mean_abs(de_d),mean_abs(de_a),mean_abs(ref_d),mean_abs(ref_a),
                    mean_abs(bde_d),mean_abs(bde_a),mean_abs(injected-injected_ref[:,None]),
                    float(np.mean(L,dtype=np.float64)),
                    float(np.mean(de_d*ref_d[:,None],dtype=np.float64)),
                    float(np.mean(de_a*ref_a[:,None],dtype=np.float64)),
                    *[float(10*np.log10(np.mean(vx[:,sl])/(k*k*np.mean(zn[:,sl]))))
                      for (k, _, _), (_, zn, _, _) in zip(result, cases)]])
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus",choices=["seed","deap"])
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--epochs",type=Path,required=True)
    p.add_argument("--emg512",type=Path)
    p.add_argument("--epoch-map",type=Path)
    p.add_argument("--control",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    o=p.parse_args()
    control=np.load(o.control,allow_pickle=False)["control_sequence"]
    epochs=release_epochs(o.epochs,o.emg512,o.epoch_map)
    o.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    if o.corpus=="seed":
        for (s,session,t,label),x in j.load_raw_trials(o.data):
            if session==1:
                continue
            for draw in j.REALIZATIONS:
                rows.extend(per_trial("SEED",(s,session,t,draw),x,seed_draw(control,epochs,s,session,t,draw)))
            if t==15:
                print("SEED",s,session,flush=True)
    else:
        for s,d in f.deap_files(o.data):
            for t,u in enumerate(d["data"],1):
                x=np.asarray(u[:32],dtype=np.float32)
                rows.extend(per_trial("DEAP",(s,0,t,1),x,deap_draw(control,epochs,s,t)))
            print("DEAP",s,flush=True)
    f.write_csv(o.out/f"{o.corpus}_factorial.csv",["subject","session","trial","draw","placement","snr","reference","active",
        "effective_channels","de_direct","de_aa","ref_direct","ref_aa","bde_direct","bde_aa","bde_aa_floor","mean_line_floor",
        "de_ref_product_direct","de_ref_product_aa","beta_snr_direct","beta_snr_aa"],rows)


if __name__=="__main__":
    main()
