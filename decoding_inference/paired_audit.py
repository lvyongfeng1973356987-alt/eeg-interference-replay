"""Paired power-line audit using the supplied control envelope and licensed DEAP ZIP."""
from pathlib import Path
import argparse
import csv
import json
import pickle
from zipfile import ZipFile

import numpy as np
from scipy.signal import butter, resample_poly, resample, sosfilt, sosfiltfilt, sosfreqz
from scipy.stats import wilcoxon
from numpy.lib.stride_tricks import sliding_window_view

FS = 128
POINTS = 8064
BANDS = [(4, 8), (8, 14), (14, 31), (31, 49)]
FILTERS = [butter(5, band, fs=FS, btype="bandpass", output="sos").astype("float32") for band in BANDS]
CONDITIONS = ("direct_global", "antialias_global", "direct_active", "direct_uniform")
ACTIVE = np.array([0,1,2,3,4,5,6,7,8,9,16,17,18,19,20,21,22,23,24,25,30])
INDEX_GAINS = np.exp(-0.2 * np.abs(np.arange(32)-16))
FREQ = np.array([50,100,150])
AMP = np.array([1., .3, .15])
SNR = -5


def windows(x):
    return sliding_window_view(x[..., 384:], 512, axis=-1)[..., ::117, :]


def components(control, phase, high_rate=False):
    if high_rate:
        envelope = resample_poly(0.8+0.4*control, 125, 16, padtype="line")
        t = np.arange(len(envelope))/1000
        x = AMP[:, None] * np.sin(2*np.pi*FREQ[:,None]*t + phase[:,None]) * envelope
        return resample_poly(x, 16, 125, axis=-1, padtype="line")[..., :len(control)].astype("float32")
    t = np.arange(len(control))/FS
    return (AMP[:,None]*np.sin(2*np.pi*FREQ[:,None]*t + phase[:,None])*(0.8+0.4*control)).astype("float32")


def retention(x, mode, sos):
    w = windows(x)
    y = sosfilt(sos,w) if mode=="window" else windows(sosfiltfilt(sos,x))
    return np.sum(np.var(y,axis=-1),axis=-1), np.sum(np.var(w,axis=-1),axis=-1)


def line_audit(control):
    rng = np.random.default_rng(20261121)
    rows = []
    for k in range(len(control)//(60*FS)):
        w = np.pad(control[k*7680:(k+1)*7680], (384,0), mode="edge")
        phase = rng.uniform(0,2*np.pi,3)
        original = components(w,phase)
        for path in ("direct", "antialias"):
            x = original if path=="direct" else components(w,phase,True)
            signals = np.stack([x[0],x[1]+x[2],x.sum(axis=0)])
            for mode in ("window","full"):
                for band,sos in zip(BANDS,FILTERS):
                    numerator, denominator = retention(signals,mode,sos)
                    rows.append((path,mode,*band,*numerator,denominator[-1],retention(original.sum(axis=0),mode,sos)[1]))
    cols = ["path","filter","low_hz","high_hz","fundamental","harmonics","combined","input_power","direct_input_power"]
    summary = []
    for path in ("direct","antialias"):
        for mode in ("window","full"):
            for band in BANDS:
                selected = np.array([r[4:] for r in rows if r[:4]==(path,mode,*band)])
                total = selected.sum(axis=0)
                summary.append([path,mode,*band,*(100*total[:3]/total[3]),100*total[2]/total[4]])
    return rows,cols,summary


def tone_resampling():
    fs = 200
    t = np.arange(60*fs)/fs
    n = 60*128
    out = []
    for f in [50, 90]:
        x = np.sin(2*np.pi*f*t)
        for name,y in [
            ("integer", x[np.linspace(0,len(x)-1,n).astype(int)]),
            ("polyphase", resample_poly(x,16,25)),
            ("fft", resample(x,n)),
        ]:
            z = sosfilt(FILTERS[2],y)
            out.append([f,name,float(np.var(y)/np.var(x)),float(np.var(z)/np.var(x))])
    return out


def scales(clean, direct, antialias, gains):
    def k(noise, gain, idx):
        return np.sqrt(np.mean(clean[idx]**2) / np.mean(noise**2) / np.mean(gain[idx]**2) * 10**(-SNR/10))
    all_ch = np.arange(len(clean))
    equal = np.ones_like(gains)
    return np.stack([
        gains * k(direct,gains,all_ch), gains * k(antialias,gains,all_ch),
        gains * k(direct,gains,ACTIVE), equal * k(direct,equal,all_ch),
    ]).astype("float32")


def band_features(clean, line, amplitudes, sos):
    def segment(x,z):
        xf = sosfilt(sos,x)
        zf = sosfilt(sos,z)
        xf -= xf.mean(axis=-1,keepdims=True)
        zf -= zf.mean(axis=-1,keepdims=True)
        vx = np.mean(xf**2,axis=-1)
        vz = np.mean(zf**2,axis=-1)
        cross = np.einsum("cwt,kwt->kcw",xf,zf,optimize=False)/xf.shape[-1]
        a = amplitudes[...,None]
        v = vx + 2*a*cross + a*a*vz[:,None,:]
        return .5*np.log2(np.maximum(vx,1e-10)), .5*np.log2(np.maximum(v,1e-10)), vx, a*a*vz[:,None,:]
    base, nbase, _, _ = segment(clean[:,None,:384],line[:,None,:384])
    feat, noisy, vx, vn = segment(windows(clean), windows(line))
    return feat-base, noisy-nbase, 10*np.log10(vx.mean(axis=(0,1))/vn.mean(axis=(1,2)))


def run_deap(archive, control, output):
    features, labels, rows = [], [], []
    with ZipFile(archive) as z:
        files = sorted(n for n in z.namelist() if n.split("/")[-1] in {f"s{i:02d}.dat" for i in range(1,33)})
        assert len(files)==32
        for si,path in enumerate(files,1):
            with z.open(path) as stream:
                subject = pickle.load(stream,encoding="latin1")
            for ti,(trial,ratings) in enumerate(zip(subject["data"],subject["labels"]),1):
                clean = np.asarray(trial[:32],dtype="float32")
                assert clean.shape==(32,POINTS)
                rng = np.random.default_rng(20261121 + 100*si+ti)
                start = rng.integers(0,len(control)-POINTS+1)
                env = control[start:start+POINTS]
                phase = rng.uniform(0,2*np.pi,3)
                direct = components(env,phase).sum(axis=0)
                aa = components(env,phase,True).sum(axis=0)
                gains = INDEX_GAINS*rng.uniform(.7,1,32)
                line = np.stack([direct,aa,direct,direct])
                a = scales(clean,direct,aa,gains)
                clean_feat, perturbed = [], []
                for band,sos in zip(BANDS,FILTERS):
                    f, nf, band_snr = band_features(clean,line,a,sos)
                    rows.extend([si,ti,name,*band,float(np.abs(n-f).mean()),float(s)] for name,n,s in zip(CONDITIONS,nf,band_snr))
                    clean_feat.append(f.mean(axis=-1))
                    perturbed.append(nf.mean(axis=-1))
                features.append(np.concatenate([np.stack(clean_feat,axis=-1)[None,...],np.stack(perturbed,axis=-1)]))
                labels.append([si,ti,int(ratings[0]>=5)])
            print(f"Finished DEAP subject {si}/32",flush=True)
    write_csv(output/"deap_trial_metrics.csv",["subject","trial","condition","low_hz","high_hz","mean_abs_delta_bits","band_snr_db"],rows)
    return np.asarray(features),np.asarray(labels),rows


def write_csv(path,header,rows):
    with path.open("w",newline="",encoding="utf8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def comparisons(values, labels):
    """Paired analysis at the participant level. Percentile bootstrap 95% CIs."""
    rng = np.random.default_rng(222)
    resample_ids = rng.integers(0,len(values),size=(10000,len(values)))
    out = {}
    for k,label in enumerate(labels):
        x = values[:,k]
        lo,hi = np.quantile(x[resample_ids].mean(axis=1), [.025,.975])
        out[label] = dict(mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),
                          iqr=np.quantile(x,[.25,.75]).tolist(),ci95=[float(lo),float(hi)])
        if k:
            delta = x-values[:,0]
            lo,hi=np.quantile(delta[resample_ids].mean(axis=1),[.025,.975])
            out[label].update(paired_diff=float(delta.mean()),diff_ci95=[float(lo),float(hi)],
                              p_wilcoxon=float(wilcoxon(delta).pvalue) if np.any(delta) else 1.)
    return out


def holm_family(records):
    ranked = sorted(records,key=lambda row: row["p_wilcoxon"])
    adjusted = 0.
    for i,row in enumerate(ranked):
        adjusted = max(adjusted,min(1.,(len(ranked)-i)*row["p_wilcoxon"]))
        row["p_holm"] = adjusted


def decode(x, label, output):
    import torch
    from torch import nn
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    torch.set_num_threads(1)
    y = label[:,2]
    groups = label[:,0]
    all_rows = []
    for subject in np.unique(groups):
        test = groups==subject
        normalizer = StandardScaler().fit(x[~test,0].reshape((~test).sum(),-1))
        train = normalizer.transform(x[~test,0].reshape((~test).sum(),-1)).astype("float32")
        tests = [normalizer.transform(x[test,k].reshape(test.sum(),-1)).astype("float32") for k in range(x.shape[1])]
        svm = LinearSVC(C=.1,class_weight="balanced",random_state=0).fit(train,y[~test])
        torch.manual_seed(20261121+int(subject))
        model = nn.Sequential(nn.Conv1d(4,16,3,padding=1),nn.ReLU(),nn.AvgPool1d(2),nn.Flatten(),nn.Linear(256,2))
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        counts = np.bincount(y[~test])
        loss = nn.CrossEntropyLoss(weight=torch.tensor(len(train)/(2*counts),dtype=torch.float32))
        def tensor(a):
            return torch.tensor(a.reshape(-1,32,4).transpose(0,2,1))
        tx,ty = tensor(train),torch.tensor(y[~test],dtype=torch.long)
        for _ in range(15):
            for ix in torch.randperm(len(tx)).split(64):
                optimizer.zero_grad()
                loss(model(tx[ix]),ty[ix]).backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            for name,xx in zip(("clean",*CONDITIONS),tests):
                for method,pred in [("linear_svm",svm.predict(xx)),("cnn",model(tensor(xx)).argmax(1).numpy())]:
                    all_rows.append([int(subject),method,name,float(balanced_accuracy_score(y[test],pred)),float((y[test]==pred).mean())])
    write_csv(output/"deap_loso_scores.csv",["subject","model","condition","balanced_accuracy","accuracy"],all_rows)
    result = {}
    for method in ("linear_svm","cnn"):
        data = [r[3] for r in all_rows if r[1]==method]
        result[method] = comparisons(np.asarray(data).reshape(32,5),("clean",*CONDITIONS))
    for method in result:
        data = [r[3] for r in all_rows if r[1]==method and r[2]!="clean"]
        result[method]["same_snr"] = comparisons(np.asarray(data).reshape(32,4),CONDITIONS)
    holm_family([r for method in result.values() for r in method["same_snr"].values() if "p_wilcoxon" in r])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--deap",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent)
    options = parser.parse_args()
    control = np.load(options.control)["control_sequence"]
    assert len(control)==921600
    raw,cols,summary = line_audit(control)
    write_csv(options.output/"line_retention_blocks.csv",["block",*cols],[(i//16+1,*r) for i,r in enumerate(raw)])
    write_csv(options.output/"line_retention_summary.csv",cols[:4]+["fundamental_pct","harmonics_pct","combined_pct","combined_pct_direct_input"],summary)
    write_csv(options.output/"tone_resampling.csv",["tone_hz","resampler","output_over_input","beta_over_input"],tone_resampling())
    x,y,rows = run_deap(options.deap,control,options.output)
    matrix = np.array([r[5] for r in rows]).reshape(32,40,4,4).mean(axis=1)
    result = {"band_features":{f"{a}-{b}":comparisons(matrix[:,i,:],CONDITIONS) for i,(a,b) in enumerate(BANDS)},
              "decode":decode(x,y,options.output)}
    holm_family([r for band in result["band_features"].values() for r in band.values() if "p_wilcoxon" in r])
    (options.output/"deap_results.json").write_text(json.dumps(result,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
