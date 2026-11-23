"""Verify and compare the EEGdenoiseNet artifact releases."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import periodogram, resample_poly


def load_bank(path, kind, rate, n_epochs):
    mat = loadmat(path)
    key = "EMG_epochs_512hz_random_5598" if rate == 512 else f"{kind}_all_epochs"
    if key not in mat or int(np.asarray(mat["fs"]).item()) != rate:
        raise ValueError(f"Unexpected variable or sampling rate in {path}")
    x = mat[key]
    if x.shape != (n_epochs, 2 * rate) or not np.isfinite(x).all():
        raise ValueError(f"Unexpected samples in {path}: {x.shape}")
    return x


def to_128(x, rate):
    y = resample_poly(x.astype("float32"), 1, rate // 128, axis=-1)
    y -= y.mean(axis=-1, keepdims=True)
    return y


def release_epochs(folder, official_emg=None, epoch_map=None):
    eye = load_bank(folder / "EOG_all_epochs.mat", "EOG", 256, 3400)
    if official_emg is None:
        muscle = load_bank(folder / "EMG_all_epochs.mat", "EMG", 256, 5598)
        return to_128(eye, 256), to_128(muscle, 256)
    muscle = load_bank(official_emg, "EMG", 512, 5598)
    if epoch_map is not None:
        order = np.load(epoch_map, allow_pickle=False)
        np.testing.assert_array_equal(np.sort(order), np.arange(len(muscle)))
        muscle = muscle[order]
    return to_128(eye, 256), to_128(muscle, 512)


def spectrum(values, rate):
    f, p = periodogram(values, fs=rate, window="hann", axis=-1)
    p = p.mean(axis=0, dtype=np.float64)
    probability = p / p.sum()
    summary = {
        "f95_hz": float(np.interp(.95, np.cumsum(probability), f)),
        "f99_hz": float(np.interp(.99, np.cumsum(probability), f)),
        "relative_power": {
            label: float(probability[condition].sum())
            for label, condition in {
                "above_10": f > 10, "14_to_31": (f >= 14) & (f < 31),
                "31_to_49": (f >= 31) & (f < 49), "at_or_above_64": f >= 64,
                "64_to_120": (f >= 64) & (f < 120), "above_120": f > 120,
            }.items()
        },
    }
    return summary, zip(f, p, probability)


def correlation(a, b):
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    return np.einsum("ij,ij->i", a, b) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def match_epochs(old, official):
    a, b = to_128(old, 256), to_128(official, 512)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    order = np.concatenate([np.argmax(a[i:i+160] @ b.T, axis=1)
                            for i in range(0, len(a), 160)])
    counts = np.bincount(order, minlength=len(a))
    missing = set(np.flatnonzero(counts == 0))
    for index in np.flatnonzero(counts > 1):
        duplicates = np.flatnonzero(order == index)
        # Identical official rows may tie. Assign each to a distinct index.
        for source in duplicates[1:]:
            alternative = next(k for k in sorted(missing)
                               if np.array_equal(official[index], official[k]))
            order[source] = alternative
            missing.remove(alternative)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(a)))
    assert np.min(correlation(a, b[order])) > .99
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=Path, required=True)
    parser.add_argument("--emg512", type=Path, required=True)
    parser.add_argument("--npy512", type=Path)
    parser.add_argument("--gin256", type=Path)
    parser.add_argument("--map-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    descriptions = [
        ("github_eog", args.epochs / "EOG_all_epochs.mat", "EOG", 256, 3400),
        ("github_emg", args.epochs / "EMG_all_epochs.mat", "EMG", 256, 5598),
        ("gin_emg_512", args.emg512, "EMG", 512, 5598),
    ]
    banks, report, psd_rows = {}, {}, []
    for label, path, kind, rate, count in descriptions:
        x = load_bank(path, kind, rate, count)
        banks[label] = x
        native, native_bins = spectrum(x, rate)
        output, output_bins = spectrum(to_128(x, rate), 128)
        for stage, bins in (("native", native_bins), ("output", output_bins)):
            psd_rows.extend((label, stage, *row) for row in bins)
        report[label] = {
            "samples_per_epoch": x.shape[1], "epochs": count, "fs": rate,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "native_psd": native, "output_psd": output,
        }
    old, official = banks["github_emg"], banks["gin_emg_512"]
    order = match_epochs(old, official)
    np.save(args.map_out, order)
    matched = official[order]
    poly = resample_poly(matched, 1, 2, axis=-1)
    agreement = correlation(old, poly)
    output_agreement = correlation(to_128(old, 256).astype("float64"),
                                   to_128(matched, 512).astype("float64"))
    report["source_checks"] = {
        "matched_epochs": len(order),
        "matched_polyphase_512_to_256_correlation": {
            "median": float(np.median(agreement)), "min": float(np.min(agreement))},
        "matched_128_hz_correlation": {
            "median": float(np.median(output_agreement)), "min": float(np.min(output_agreement))},
        "matched_256_median_relative_rms": float(np.median(
            np.linalg.norm(old - poly, axis=1) / np.linalg.norm(old, axis=1))),
        "median_row_correlation": {
            method: float(np.median(correlation(old, candidate)))
            for method, candidate in (
                ("even_sample_selection", official[:, ::2]),
                ("odd_sample_selection", official[:, 1::2]),
                ("polyphase_512_to_256", resample_poly(official, 1, 2, axis=-1)),
            )
        }
    }
    if args.npy512:
        y = np.load(args.npy512, allow_pickle=False)
        np.testing.assert_array_equal(y, official)
        report["source_checks"]["gin_mat_and_npy_equal"] = True
        report["source_checks"]["gin_npy_sha256"] = hashlib.sha256(args.npy512.read_bytes()).hexdigest()
    if args.gin256:
        secondary = load_bank(args.gin256, "EMG", 256, 5598)
        report["source_checks"]["github_and_gin_256_equal"] = bool(np.array_equal(old, secondary))
        report["source_checks"]["gin_256_sha256"] = hashlib.sha256(args.gin256.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.out.with_name("emg_psd.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("bank", "stage", "frequency_hz", "density", "relative_power"))
        writer.writerows(psd_rows)


if __name__ == "__main__":
    main()
