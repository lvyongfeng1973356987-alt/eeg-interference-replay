"""Measure power retention and calibration order on the paired DEAP mixtures."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

import followup as f
from emg_source import release_epochs


TARGET = -5
HP = butter(4, 1, fs=f.FS, btype="highpass", output="sos")
BETA = f.filter_design((14, 31))
OPERATORS = ("identity", "hp", "car", "hp_car")


def car(x):
    return x - x.mean(axis=0, keepdims=True)


def powers(x):
    hp = sosfiltfilt(HP, x, axis=-1)
    band = sosfilt(BETA, x, axis=-1)
    band_hp = sosfilt(BETA, hp, axis=-1)
    return np.array([
        [f.power(x), f.power(band)],
        [f.power(hp), f.power(band_hp)],
        [f.power(car(x)), f.power(car(band))],
        [f.power(car(hp)), f.power(car(band_hp))],
    ])


def mixtures(clean, control, epochs, subject, trial):
    env, phase, rng = f.draw_control(control, subject, trial)
    line = f.components(env, phase).sum(axis=0)
    gains = (f.INDEX_GAINS * rng.uniform(.7, 1, 32)).astype("float32")[:, None]
    drift, pulse = f.drift_and_pulse(env, rng)
    white = (.1 * rng.standard_normal(clean.shape)).astype("float32")
    weighted_line = gains * line
    synthetic = weighted_line + gains * (drift + pulse + white)
    eye, muscle = (
        f.unit_noise(f.artifact(bank, profile, rng))
        for bank, profile in zip(epochs, (f.EOG_PROFILE, f.EMG_PROFILE))
    )
    yield "line", weighted_line
    yield "synthetic", synthetic
    yield "joint", .5 * f.unit_noise(synthetic) + np.sqrt(.4) * eye + np.sqrt(.35) * muscle
    yield "drift", gains * drift


def snr(clean, noise, scale):
    return 10 * np.log10(clean / noise) - 20 * np.log10(scale)


def summarize(rows):
    group = defaultdict(lambda: defaultdict(list))
    for subject, trial, mix, operator, *values in rows:
        group[mix, operator][subject].append(values)
    draws = np.random.default_rng(20261121).integers(32, size=(20_000, 32))
    results = {}
    for (mix, operator), subjects in group.items():
        assert sorted(subjects) == list(range(1, 33))
        assert all(len(values) == 40 for values in subjects.values())
        means = np.array([np.mean(subjects[s], axis=0) for s in sorted(subjects)])
        low, high = np.percentile(means[draws].mean(axis=1), (2.5, 97.5), axis=0)
        results[f"{mix}/{operator}"] = {
            field: {"mean": float(mean), "ci95": [float(lo), float(hi)]}
            for field, mean, lo, hi in zip(FIELDS, means.mean(axis=0), low, high)
        }
    return results


FIELDS = ("clean_retained", "noise_retained", "broadband_pre_db",
          "beta_pre_db", "beta_post_db", "beta_post_minus_pre_db")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=Path, required=True)
    parser.add_argument("--emg512", type=Path)
    parser.add_argument("--epoch-map", type=Path)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    epochs = release_epochs(args.epochs, args.emg512, args.epoch_map)
    control = np.load(args.control, allow_pickle=False)["control_sequence"]
    rows, drift_rows = [], []
    for subject, data in f.deap_files(args.data):
        for trial, raw in enumerate(data["data"], start=1):
            clean = np.asarray(raw[:32], dtype="float32")
            cp = powers(clean)
            for mix, noise in mixtures(clean, control, epochs, subject, trial):
                npow = powers(noise)
                assert np.all(cp > 0) and np.all(npow > 0)
                if mix == "drift":
                    drift_rows.append((subject, trial, *(npow[:, 0] / npow[0, 0])))
                    continue
                gains = np.sqrt(cp[:, 0] / npow[:, 0] * 10**(-TARGET / 10))
                wide_pre = snr(cp[:, 0], npow[:, 0], gains[0])
                beta_pre = snr(cp[:, 1], npow[:, 1], gains[0])
                beta_post = snr(cp[:, 1], npow[:, 1], gains)
                np.testing.assert_allclose(
                    beta_post - beta_pre,
                    10 * np.log10((npow[:, 0] / npow[0, 0]) / (cp[:, 0] / cp[0, 0])),
                    atol=1e-10,
                )
                rows.extend(
                    (subject, trial, mix, op, cp[i, 0] / cp[0, 0],
                     npow[i, 0] / npow[0, 0], wide_pre[i], beta_pre[i], beta_post[i],
                     beta_post[i] - beta_pre[i])
                    for i, op in enumerate(OPERATORS)
                )
        print(f"DEAP {subject}/32", flush=True)
    assert len(rows) == 1280 * 3 * len(OPERATORS)
    assert len(drift_rows) == 1280
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "calibration_trials.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject", "trial", "mixture", "operator", *FIELDS))
        writer.writerows(rows)
    with (args.out / "drift_retention.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject", "trial", *OPERATORS))
        writer.writerows(drift_rows)
    summary = {
        "participant_count": 32, "trial_count": 1280, "target_db": TARGET,
        "drift_retained": dict(zip(OPERATORS, np.mean(np.array(drift_rows)[:, 2:], axis=0))),
        "conditions": summarize(rows),
    }
    (args.out / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
