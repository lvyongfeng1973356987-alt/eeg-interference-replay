"""Run a line-only replay with an independently generated AR(1) timing control."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "front_end"))
import audit_front_end as front


def control():
    rng = np.random.default_rng(20261122)
    fs = 128
    innovations = rng.standard_normal(fs * (7200 + 10))
    autoregression = np.exp(-1 / (fs * 0.508))
    correlated = lfilter([1], [1, -autoregression], innovations)[fs * 10:]
    normalized = (correlated - correlated.mean()) / correlated.std()
    return np.clip(0.766 + 0.097 * normalized, 0, 1).astype(np.float32)


def stats(values):
    x = values.astype(np.float64)
    x -= x.mean()
    power = x @ x
    autocorrelation = np.array([
        (x[t:] @ x[:-t]) / power for t in (list(range(1, 129)) + [256])
    ])
    decay = np.flatnonzero(autocorrelation[:128] <= np.exp(-1))[0] + 1
    return {"mean": float(values.mean(dtype=np.float64)), "sd": float(values.std(dtype=np.float64)),
            "first_crossing_below_1_over_e_seconds": float(decay / 128),
            "autocorrelation_at_2_seconds": float(autocorrelation[-1]),
            "fraction_above_0_60": float(np.mean(values > .60)),
            "fraction_above_0_90": float(np.mean(values > .90))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    envelope = control()
    np.savez_compressed(args.out / "synthetic_control.npz", control_sequence=envelope)
    front.validate()
    rows = front.summaries(front.audit(envelope))
    results = {"control_statistics": stats(envelope), "tone_gain_db": front.tone_data(),
               "retained_power": rows}
    (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
