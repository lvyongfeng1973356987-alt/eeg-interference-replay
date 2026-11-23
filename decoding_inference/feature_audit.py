"""Replay input, reference, and finely resolved spectral diagnostics."""
import argparse
import csv
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy.fft import rfft
from scipy.signal import sosfilt

import followup as f
import joint_decode as j
from emg_source import release_epochs


MODES = ("direct", "antialias")
PSD_LEVELS = (5, 10)
PSD_CASES = [(0, 0, "clean"), *((d, db, m) for d in j.REALIZATIONS
                               for db in PSD_LEVELS for m in MODES)]
SOS = f.filter_design((14, 31))
BINS = np.arange(4, 49)
HANN = np.hanning(j.WINDOW).astype("float32")


def seed_parts(control, epochs, draw, subject, session, trial):
    r = np.random.default_rng(np.random.SeedSequence([j.BASE_SEED, draw, subject, session, trial]))
    offset = r.integers(len(control) - j.T + 1)
    env = control[offset:offset + j.T]
    phase = r.uniform(0, 2*np.pi, 3)
    drift, pulse = f.drift_and_pulse(env, r)
    white = .1*r.standard_normal((62, j.T), dtype="float32")
    gains = (np.exp(-.2*np.abs(np.arange(62)-31))*r.uniform(.7, 1, 62)).astype("float32")[:, None]
    eye = np.sqrt(.4)*j.artifact(epochs[0], j.EOG, r)
    muscle = np.sqrt(.35)*j.artifact(epochs[1], j.EMG, r)
    background = drift + pulse + white
    for aa in (False, True):
        line = f.components(env, phase, aa).sum(axis=0)
        yield .5*f.unit_noise(gains*(line+background)), eye, muscle


def deap_parts(clean, control, epochs, subject, trial):
    env, phase, r = f.draw_control(control, subject, trial)
    gains = (f.INDEX_GAINS*r.uniform(.7, 1, 32)).astype("float32")[:, None]
    drift, pulse = f.drift_and_pulse(env, r)
    white = (.1*r.standard_normal(clean.shape)).astype("float32")
    background = gains*(drift+pulse+white)
    eye = np.sqrt(.4)*f.unit_noise(f.artifact(epochs[0], f.EOG_PROFILE, r))
    muscle = np.sqrt(.35)*f.unit_noise(f.artifact(epochs[1], f.EMG_PROFILE, r))
    for aa in (False, True):
        line = gains*f.components(env, phase, aa).sum(axis=0)
        yield .5*f.unit_noise(line+background), eye, muscle


def ledger(part):
    sim, eye, muscle = part
    wave = (sim+(eye+muscle)).astype("float32")
    total = f.power(wave)
    fractions = [f.power(x)/total for x in part]
    return wave, (*fractions, 1-sum(fractions))


def beta_signal(x):
    return (j.centered(sosfilt(SOS, j.spans(x))),
            j.centered(sosfilt(SOS, x[:, :j.REF])))


def beta_diagnostic(clean, clean_beta, noise, noise_beta):
    gain = f.amount(clean, noise, 0)
    changes = [j.half_logvar(np.mean((a+gain*b)**2, axis=-1)) -
               j.half_logvar(np.mean(a*a, axis=-1))
               for a,b in zip(clean_beta, noise_beta)]
    d, r = changes
    return float(np.abs(d).mean()), float(np.abs(r).mean()), float(np.abs(d-r[:, None]).mean())


def spectrum(x):
    return rfft(j.centered(j.spans(x))*HANN, axis=-1)


def spectral_features(spectral_windows):
    # One-Hz bins centred on integer frequencies, four 0.25-Hz ordinates each.
    idx = 4*BINS[:, None] + np.arange(-2, 2)[None, :]
    psd = (np.abs(spectral_windows)**2).mean(axis=1)
    narrow = psd[:, idx].sum(axis=-1)
    pooled = np.stack([narrow[:, (BINS >= lo)&(BINS < hi)].sum(axis=-1)
                       for lo,hi in f.BANDS], axis=-1)
    out = []
    for values in (narrow, pooled):
        power = np.log(np.maximum(values, 1e-10))
        out.append(np.r_[power.ravel(), (power-np.median(power, axis=0)).ravel()])
    return out


def seed_audit(data, control, epochs, output):
    rows, power_rows, meta, cached = [], [], [], []
    for subject, group in groupby(j.load_raw_trials(data), key=lambda item: item[0][0]):
        bank = {(session,trial):(label,x) for (_,session,trial,label),x in group}
        assert len(bank)==45
        for trial in range(1,16):
            clean = {session:bank[session,trial][1] for session in (1,2,3)}
            clean_beta = {s:beta_signal(x) for s,x in clean.items()}
            fft = {s:spectrum(x) for s,x in clean.items()}
            trial_features = {s:{(0,0,"clean"):spectral_features(spec)} for s,spec in fft.items()}
            for s,x in clean.items():
                rows.append((subject,s,trial,0,"clean",float(np.mean(clean_beta[s][0]**2)),
                             float(np.mean(clean_beta[s][1]**2)),0.,0.,0.))
            for draw in j.REALIZATIONS:
                for source,x in clean.items():
                    for mode,part in zip(MODES,seed_parts(control,epochs,draw,subject,source,trial)):
                        noise, shares = ledger(part)
                        noise_beta = beta_signal(noise)
                        noise_fft = spectrum(noise)
                        power_rows.append((subject,source,trial,draw,mode,*shares))
                        for db in PSD_LEVELS:
                            gain = f.amount(x,noise,db)
                            trial_features[source][draw,db,mode] = spectral_features(fft[source]+gain*noise_fft)
                        for dest,case in ((source,"matched"),):
                            rows.append((subject,dest,trial,draw,case+"_"+mode,0.,0.,
                                         *beta_diagnostic(clean[dest],clean_beta[dest],noise,noise_beta)))
                        for dest in (1,2,3) if source==1 else ():
                            rows.append((subject,dest,trial,draw,"fixed_noise_"+mode,0.,0.,
                                         *beta_diagnostic(clean[dest],clean_beta[dest],noise,noise_beta)))
                        rows.append((subject,source,trial,draw,"fixed_clean_"+mode,0.,0.,
                                     *beta_diagnostic(clean[1],clean_beta[1],noise,noise_beta)))
            for s in (1,2,3):
                meta.append((subject,s,trial,bank[s,trial][0]))
                cached.append([trial_features[s][case] for case in PSD_CASES])
        print(f"SEED features and session replay {subject:02d}",flush=True)
    assert len(cached)==675
    for i in (0,1):
        assert all(np.isfinite(pair[i]).all() for trial in cached for pair in trial)
    np.savez(output/"psd_cache.npz",meta=np.asarray(meta),
             **{f"x{i}":np.asarray([[p[i] for p in trial] for trial in cached]) for i in (0,1)})
    f.write_csv(output/"seed_session_replay.csv",
                ["subject","session","trial","draw","case","clean_stim_power",
                 "clean_ref_power","abs_de","abs_ref","abs_bde"],rows)
    write_ledger(output/"seed_power_ledger.csv",power_rows)


def write_ledger(path, rows):
    f.write_csv(path,["subject","session","trial","draw","synthesis","sim","eog","emg","cross"],rows)


def deap_audit(data,control,epochs,output):
    rows=[]
    for subject,ds in f.deap_files(data):
        for trial,raw in enumerate(ds["data"],1):
            x=np.asarray(raw[:32],dtype="float32")
            for mode,part in zip(MODES,deap_parts(x,control,epochs,subject,trial)):
                _, shares=ledger(part)
                rows.append((subject,0,trial,1,mode,*shares))
        print(f"DEAP component ledger {subject:02d}",flush=True)
    assert len(rows)==2560
    write_ledger(output/"deap_power_ledger.csv",rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus",choices=("seed","deap"))
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--epochs",type=Path,required=True)
    p.add_argument("--control",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    audit=seed_audit if a.corpus=="seed" else deap_audit
    audit(a.data,np.load(a.control,allow_pickle=False)["control_sequence"],
          release_epochs(a.epochs),a.out)
