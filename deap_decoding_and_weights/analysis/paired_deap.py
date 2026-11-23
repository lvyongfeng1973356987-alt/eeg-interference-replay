"""Extract matched DEAP valence features and audit component power shares."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt, sosfiltfilt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import followup as f
from calibration_replay import BETA, HP
from emg_source import release_epochs


SNR_LEVELS = (0, -5)
DRIFT = (0., 1/6, 1/3, .5, 2/3, 5/6, 11/12)
CASES = ("clean", "direct_0", "aa_0", "direct_m5", "aa_m5")
GRID = [(features, kind, c) for features in (0, 1)
        for kind, c in (("linear", .01), ("linear", .1), ("rbf", 1.), ("rbf", 10.))]


def components(x, control, subject, trial):
    env, phase, rng = f.draw_control(control, subject, trial)
    gains = (f.INDEX_GAINS * rng.uniform(.7, 1, 32)).astype("float32")[:, None]
    drift, pulse = f.drift_and_pulse(env, rng)
    white = (.1 * rng.standard_normal(x.shape)).astype("float32")
    direct, aa = (gains * f.components(env, phase, path).sum(axis=0) for path in (False, True))
    background = gains * (drift + pulse + white)
    return direct, aa, gains * drift, gains * (pulse + white), background, rng


def center(x):
    return x - x.mean(axis=-1, keepdims=True)


def log_variance(x, minimum=1e-10):
    return .5 * np.log2(np.maximum(x, minimum))


def valence_features(x, mixtures):
    by_case = [[[], []] for _ in CASES]
    gains = [[f.amount(x, n, db) for n in mixtures] for db in SNR_LEVELS]
    for band in f.BANDS:
        sos = f.filter_design(band)
        stim = center(sosfilt(sos, f.windows(x)))
        ref = center(sosfilt(sos, x[:, :384]))
        clean_variance = (np.mean(stim**2, axis=-1), np.mean(ref**2, axis=-1))
        moments = []
        for n in mixtures:
            ns = center(sosfilt(sos, f.windows(n)))
            nr = center(sosfilt(sos, n[:, :384]))
            moments.append(((np.mean(ns**2, axis=-1), np.mean(nr**2, axis=-1)),
                            (np.einsum("cwt,cwt->cw", stim, ns) / stim.shape[-1],
                             np.mean(ref * nr, axis=-1))))
        variances = [clean_variance]
        for scales in gains:
            for scale, (power, covariance) in zip(scales, moments):
                variances.append(tuple(c + 2*scale*r + scale**2*n
                                       for c, r, n in zip(clean_variance, covariance, power)))
        for block, (vs, vr) in zip(by_case, variances):
            temporal = np.stack([m.mean(axis=-1) for m in
                                 np.array_split(log_variance(vs), 3, axis=-1)], axis=1)
            scalp_centered = temporal - np.median(temporal, axis=0, keepdims=True)
            block[0].extend((temporal.ravel(), scalp_centered.ravel()))
            block[1].append((temporal - log_variance(vr)[:, None]).ravel())
    absolute = np.stack([np.concatenate(block[0]) for block in by_case])
    referenced = np.stack([np.concatenate(block[1]) for block in by_case])
    return absolute, np.concatenate((absolute, referenced), axis=-1)


def gram_power(waves):
    v = waves.reshape(len(waves), -1).astype("float64")
    return (v @ v.T) / v.shape[1]


def mixture_scan(x, parts, subject, trial):
    direct, aa, drift, wide, background, _ = parts
    waves = np.stack([f.unit_noise(v) for v in (drift, direct, aa, wide)])
    matrices = [gram_power(waves), gram_power(sosfiltfilt(HP, waves, axis=-1)),
                gram_power(sosfilt(BETA, waves, axis=-1))]
    clean_in = f.power(x)
    clean_hp = f.power(sosfiltfilt(HP, x, axis=-1))
    clean_beta = f.power(sosfilt(BETA, x, axis=-1))
    weights = np.array([[np.sqrt(d), np.sqrt((1-d)/2), 0., np.sqrt((1-d)/2)]
                        for d in DRIFT])
    weights_aa = weights[:, [0, 2, 1, 3]]
    powers = [np.einsum("ni,kij,nj->nk", w, matrices, w)
              for w in (weights, weights_aa)]
    rows = []
    for i, d in enumerate(DRIFT):
        rows.append(scan_row(subject, trial, str(d), clean_in, clean_hp, clean_beta,
                             powers[0][i], powers[1][i]))
    native_powers = []
    for n in (direct + background, aa + background):
        native_powers.append((f.power(n), f.power(sosfiltfilt(HP, n, axis=-1)),
                              f.power(sosfilt(BETA, n, axis=-1))))
    rows.append(scan_row(subject, trial, "native", clean_in, clean_hp, clean_beta, *native_powers))
    return rows


def scan_row(subject, trial, condition, clean, clean_hp, clean_beta, direct, aa):
    rx = clean_hp / clean
    result = []
    for total, filtered, beta in (direct, aa):
        rn = filtered / total
        beta_snr = -5 + 10*np.log10((clean_beta/clean)/(beta/total))
        result.extend((rn, 10*np.log10(rn/rx), beta_snr))
    return subject, trial, condition, rx, *result, result[-1] - result[2]


def kernels(train, test):
    scaler = StandardScaler().fit(train)
    a = scaler.transform(train).astype("float64")
    b = scaler.transform(test).astype("float64")
    g, h = a @ a.T, b @ a.T
    norm = np.einsum("ij,ij->i", b, b)
    d = np.diag(g)
    gamma = 1 / (a.shape[1] * max(float(a.var()), 1e-10))
    return {"linear": (g, h),
            "rbf": (np.exp(-gamma*np.maximum(d[:, None] + d - 2*g, 0)),
                    np.exp(-gamma*np.maximum(norm[:, None] + d - 2*h, 0)))}


def predict(kernel, labels, c):
    fit, test = kernel
    return SVC(kernel="precomputed", C=c, class_weight="balanced").fit(fit, labels).predict(test)


def decode(data, meta):
    subjects, _, labels = meta.T
    output, selected = [], []
    for subject in np.unique(subjects):
        train, test = np.flatnonzero(subjects != subject), np.flatnonzero(subjects == subject)
        scores = [[] for _ in GRID]
        for it, iv in GroupKFold(n_splits=4).split(train, labels[train], subjects[train]):
            fit, valid = train[it], train[iv]
            bank = [kernels(x[fit, 0], x[valid, 0]) for x in data]
            for i, (feat, kind, c) in enumerate(GRID):
                guess = predict(bank[feat][kind], labels[fit], c)
                for s in np.unique(subjects[valid]):
                    use = subjects[valid] == s
                    scores[i].append(balanced_accuracy_score(labels[valid][use], guess[use]))
        means = np.asarray([np.mean(a) for a in scores])
        best = int(np.argmax(means))
        feat, kind, c = GRID[best]
        vals = data[feat]
        hold = vals[test].reshape(-1, vals.shape[-1])
        guess = predict(kernels(vals[train, 0], hold)[kind], labels[train], c).reshape(len(test), -1)
        selected.append((int(subject), feat, kind, c, float(means[best])))
        for i, predictions in zip(test, guess):
            output.extend((*meta[i], name, int(y)) for name, y in zip(CASES, predictions))
        print(f"DEAP LOSO {subject:02d}", flush=True)
    return output, selected


def save_csv(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["extract", "decode"])
    p.add_argument("--data", type=Path)
    p.add_argument("--epochs", type=Path)
    p.add_argument("--control", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / "deap_decoder_cache.npz"
    if a.mode == "decode":
        with np.load(cache, allow_pickle=False) as obj:
            rows, selection = decode([obj["x0"], obj["x1"]], obj["meta"])
        save_csv(a.out / "deap_predictions.csv", ("subject", "trial", "label", "case", "prediction"), rows)
        save_csv(a.out / "deap_selection.csv", ("subject", "family", "kernel", "c", "inner_clean_ba"), selection)
        return
    control = np.load(a.control, allow_pickle=False)["control_sequence"]
    epochs = release_epochs(a.epochs)
    features, meta, scan = [], [], []
    for subject, dataset in f.deap_files(a.data):
        labels = (dataset["labels"][:, 0] >= 5).astype(int)
        assert set(labels) == {0, 1}
        for trial, (raw, label) in enumerate(zip(dataset["data"], labels), 1):
            x = np.asarray(raw[:32], dtype="float32")
            parts = components(x, control, subject, trial)
            direct, aa, _, _, background, rng = parts
            eog, emg = (f.unit_noise(f.artifact(bank, profile, rng))
                        for bank, profile in zip(epochs, (f.EOG_PROFILE, f.EMG_PROFILE)))
            physiological = np.sqrt(.4)*eog + np.sqrt(.35)*emg
            noise = [.5*f.unit_noise(v + background) + physiological for v in (direct, aa)]
            features.append(valence_features(x, noise))
            scan.extend(mixture_scan(x, parts, subject, trial))
            meta.append((subject, trial, label))
        print(f"DEAP features and weight scan {subject:02d}", flush=True)
    assert len(meta) == 1280
    np.savez(cache, x0=np.stack([a[0] for a in features]), x1=np.stack([a[1] for a in features]),
             meta=np.asarray(meta))
    save_csv(a.out / "component_scan_trials.csv",
             ("subject", "trial", "drift_target", "clean_retained", "noise_retained_direct",
              "order_db_direct", "beta_snr_direct", "noise_retained_aa", "order_db_aa",
              "beta_snr_aa", "aa_minus_direct_beta_db"), scan)


if __name__ == "__main__":
    main()
