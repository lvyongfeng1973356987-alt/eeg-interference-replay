"""Extract the frozen-control recipe from a local 48 kHz PCM soundtrack."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, resample_poly


def envelope(wav):
    fs, audio = read(wav, mmap=True)
    if fs != 48000 or audio.dtype != np.int16 or audio.shape != (7200*fs, 2):
        raise ValueError("Expected a 120-minute 48 kHz stereo signed 16-bit PCM WAV.")
    low = np.empty(7200*128, dtype=np.float32)
    for start in range(0,7200,60):
        stop = start+60
        lo, hi = max(0,start-1), min(7200,stop+1)
        local = audio[lo*fs:hi*fs].astype(np.float32)
        mono = (local[:,0]+local[:,1])*.5/32768
        down = resample_poly(mono,1,375)
        j = (start-lo)*128
        low[start*128:stop*128] = down[j:j+(stop-start)*128]
    b,a = butter(4,1.5/64,btype="low")
    smooth = uniform_filter1d(filtfilt(b,a,np.abs(low),padlen=15),size=64,mode="nearest")
    clipped = np.clip(smooth,0,np.percentile(smooth,99))
    scale = clipped.max()-clipped.min()
    if scale <= 0:
        raise ValueError("The clipped envelope has zero range.")
    return ((clipped-clipped.min())/scale).astype(np.float32)


if __name__ == "__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("wav",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--compare",type=Path,help="Check against a supplied frozen NPZ.")
    args=parser.parse_args()
    data=envelope(args.wav)
    if args.compare:
        with np.load(args.compare,allow_pickle=False) as original:
            error=np.max(np.abs(data-original["control_sequence"]))
        print(f"Maximum absolute difference: {error:.8g}")
    np.savez_compressed(args.out,control_sequence=data,sampling_rate_hz=128,duration_s=7200)
