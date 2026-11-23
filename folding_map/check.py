"""Map mains harmonics to sampled frequencies and check against sampled tones."""
import csv
from pathlib import Path

import numpy as np


def mappings():
    for mains in (50, 60):
        for fs in (128, 200, 250):
            tones = tuple(mains * order for order in (1, 2, 3))
            mapped = tuple(abs((2 * f + fs) % (2 * fs) - fs) // 2 for f in tones)
            time = np.arange(fs) / fs
            # A nonzero phase leaves a visible tone at exactly Nyquist.
            peaks = [int(np.argmax(np.abs(np.fft.rfft(np.sin(2*np.pi*f*time + .41))))) for f in tones]
            assert tuple(peaks) == mapped, (mains, fs, mapped, peaks)
            yield mains, fs, *mapped


if __name__ == "__main__":
    output = Path(__file__).with_name("mains_folding.csv")
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("mains_hz", "sampling_hz", "fundamental_hz", "second_harmonic_hz", "third_harmonic_hz"))
        writer.writerows(mappings())
