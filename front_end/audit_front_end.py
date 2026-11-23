"""Calculate reference transfer functions and replay them on matched line blocks."""
from pathlib import Path
import csv
import json

import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import resample_poly, sosfilt

from paired_audit import FS, FREQ, AMP, FILTERS, components, windows

SOURCE_RATE = 2048
R = SOURCE_RATE // FS
RC_HZ = 83
BLOCK = 60 * FS
GUARD = 3 * FS
MODES = ("direct", "unfiltered_decimation", "rc1", "rc2",
         "cic3", "cic5", "rc1_cic5", "polyphase")
HERE = Path(__file__).resolve().parent


def cic(f, order, source_rate, output_rate):
    """Complex unity-DC transfer of a cascade of normalized moving averages."""
    ratio = source_rate // output_rate
    magnitude_with_sign = np.sinc(f/output_rate) / np.sinc(f/source_rate)
    delay_s = order*(ratio-1)/(2*source_rate)
    return magnitude_with_sign**order * np.exp(-2j*np.pi*f*delay_s)


def responses(f):
    rc = 1/(1+1j*f/RC_HZ)
    sinc3 = cic(f, 3, SOURCE_RATE, FS)
    sinc5 = cic(f, 5, SOURCE_RATE, FS)
    return {"rc1": rc, "rc2": rc**2, "cic3": sinc3,
            "cic5": sinc5, "rc1_cic5": rc*sinc5}


def tone_data():
    transfer = {"direct": np.ones(3), **responses(FREQ),
                "ads1299_250_cic3": cic(FREQ, 3, 1024000, 250)}
    rows = {}
    for name, h in transfer.items():
        rows[name] = {str(f): float(20*np.log10(abs(x))) for f, x in zip(FREQ, h)}
    return rows


def energy(x):
    return float(np.sum(np.var(windows(x), axis=-1)))


def metrics(parts):
    mixed = parts.sum(axis=0, dtype="float64")
    fundamental = parts[0]
    harmonics = parts[1:].sum(axis=0, dtype="float64")
    filtered = sosfilt(FILTERS[2], windows(mixed))
    return (energy(fundamental), energy(harmonics), energy(mixed),
            float(np.sum(np.var(filtered, axis=-1))))


def audit(control):
    if control.shape != (120*BLOCK,) or not np.all(np.isfinite(control)):
        raise ValueError("Expected the frozen 120-minute control array.")
    t = np.arange((BLOCK+2*GUARD)*R)/SOURCE_RATE
    carrier = 2*np.pi*FREQ[:, None]*t
    transfer = responses(rfftfreq(t.size, 1/SOURCE_RATE))
    rng = np.random.default_rng(20261121)
    rows = []
    for i, segment in enumerate(control.reshape(-1, BLOCK), start=1):
        phase = rng.uniform(0, 2*np.pi, 3)
        window_input = np.pad(segment, (GUARD, 0), mode="edge")
        original = components(window_input, phase)
        original_aa = components(window_input, phase, True)
        env = resample_poly(0.8+0.4*np.pad(segment, (GUARD, GUARD), mode="edge"),
                            R, 1, padtype="line")
        wave = AMP[:, None]*np.sin(carrier+phase[:, None])*env
        spectral = rfft(wave, axis=-1)
        cases = {"direct": original, "unfiltered_decimation": wave[:, ::R]}
        for name, h in transfer.items():
            cases[name] = irfft(spectral*h, n=t.size, axis=-1)[:, ::R]
        cases["polyphase"] = original_aa
        for name in MODES:
            rows.append((i, name, *metrics(cases[name][:, :BLOCK+GUARD])))
    return rows


def summaries(rows):
    totals = {name: np.sum([row[2:] for row in rows if row[1] == name], axis=0)
              for name in MODES}
    direct = totals["direct"]
    return {name: {
        "fundamental_relative_pct": float(100*v[0]/direct[0]),
        "folded_harmonics_pct_of_direct_line": float(100*v[1]/direct[2]),
        "folded_harmonics_relative_db": float(10*np.log10(v[1]/direct[1])),
        "windowed_beta_pct_of_direct_line": float(100*v[3]/direct[2]),
    } for name, v in totals.items()}


def validate():
    # Compare the exact response with ordinary finite convolution at the
    # integer rate ratio and verify sinusoidal DC gain and output folding.
    for order in (3, 5):
        boxcar = np.ones(R)/R
        kernel = boxcar
        for _ in range(order-1):
            kernel = np.convolve(kernel, boxcar)
        expected = np.exp(-2j*np.pi*FREQ[:, None]*np.arange(len(kernel))/SOURCE_RATE) @ kernel
        np.testing.assert_allclose(cic(FREQ, order, SOURCE_RATE, FS), expected, atol=1e-14)
    np.testing.assert_allclose(cic(np.array([0.]), 5, SOURCE_RATE, FS), 1.)
    folded = np.abs((FREQ+FS//2) % FS-FS//2)
    np.testing.assert_array_equal(folded, [50, 28, 22])


def main():
    validate()
    with np.load(HERE/"S1_Data_control_sequence.npz", allow_pickle=False) as archive:
        rows = audit(archive["control_sequence"])
    estimates = summaries(rows)
    # The first and last arms reproduce the original line audit.
    assert abs(estimates["direct"]["windowed_beta_pct_of_direct_line"]-9.77) < .01
    assert abs(estimates["polyphase"]["windowed_beta_pct_of_direct_line"]-.023) < .001
    with (HERE/"front_end_blocks.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["block", "path", "fundamental_variance_sum",
                         "harmonic_variance_sum", "line_variance_sum", "beta_variance_sum"])
        writer.writerows(rows)
    result = {"tone_gain_db": tone_data(), "blocks": estimates}
    (HERE/"front_end_summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
