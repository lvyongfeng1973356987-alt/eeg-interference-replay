"""Re-extract PSD inputs with exact band boundaries; refit the paired decoders."""
from pathlib import Path
import argparse
import numpy as np

import feature_audit as a
import decode_psd as decoder


ROOT = Path(__file__).resolve().parent
LOWER_EDGES = np.arange(4, 49)
ORDINATES = 4*LOWER_EDGES[:, None] + np.arange(5)
WEIGHTS = np.array([.5, 1., 1., 1., .5], dtype="float32")


def aligned_features(spectral_windows, focused=False):
    """Composite trapezoidal integration of the same Fourier ordinates."""
    psd = (np.abs(spectral_windows)**2).mean(axis=1)
    narrow = (psd[:, ORDINATES]*WEIGHTS).sum(axis=-1)
    pooled = np.stack([narrow[:, (LOWER_EDGES>=lo)&(LOWER_EDGES<hi)].sum(axis=-1)
                       for lo, hi in a.f.BANDS], axis=-1)
    representations = (
        (narrow[:, (LOWER_EDGES >= 14) & (LOWER_EDGES < 31)],
         narrow[:, (LOWER_EDGES >= 20) & (LOWER_EDGES < 30)])
        if focused else (narrow, pooled)
    )
    features = []
    for representation in representations:
        power = np.log(np.maximum(representation, 1e-10))
        features.append(np.r_[power.ravel(), (power-np.median(power, axis=0)).ravel()])
    return features


def validate():
    np.testing.assert_allclose((ORDINATES/4)@WEIGHTS/WEIGHTS.sum(), LOWER_EDGES+.5)
    grid = np.arange(257, dtype="float32")**2
    integrated = (grid[ORDINATES]*WEIGHTS).sum(axis=-1)
    for lo,hi in a.f.BANDS:
        weights = np.ones(4*(hi-lo)+1)
        weights[[0,-1]] = .5
        reference = grid[4*lo:4*hi+1]@weights
        np.testing.assert_allclose(integrated[(LOWER_EDGES>=lo)&(LOWER_EDGES<hi)].sum(),reference)


def run(seed, artifact_dir, focused=False):
    control = np.load(ROOT/"S1_Data_control_sequence.npz", allow_pickle=False)["control_sequence"]
    epochs = a.release_epochs(artifact_dir)
    meta, cache = [], []
    for identifiers, clean in a.j.load_raw_trials(seed):
        subject, session, trial, _ = identifiers
        s0 = a.spectrum(clean)
        features = {(0,0,"clean"):aligned_features(s0, focused)}
        if session==1:
            # The first session contributes only the clean calibration median.
            features.update({key:features[0,0,"clean"] for key in a.PSD_CASES[1:]})
        else:
            for draw in a.j.REALIZATIONS:
                for mode, parts in zip(a.MODES,a.seed_parts(control,epochs,draw,subject,session,trial)):
                    noise,_ = a.ledger(parts)
                    sn = a.spectrum(noise)
                    for db in a.PSD_LEVELS:
                        features[draw,db,mode] = aligned_features(s0+a.f.amount(clean,noise,db)*sn, focused)
        cache.append([features[key] for key in a.PSD_CASES])
        meta.append(identifiers)
        if session==3 and trial==15:
            print(f"Aligned spectra, participant {subject}",flush=True)
    assert len(cache)==675
    meta = np.asarray(meta)
    # Match the archived ordering within each participant and fold.
    order = np.lexsort((meta[:,1],meta[:,2],meta[:,0]))
    rows = []
    families = ("beta_1hz", "fold_1hz") if focused else ("narrow", "pooled")
    for i, family in enumerate(families):
        x = np.asarray([[values[i] for values in trial] for trial in cache])
        assert np.isfinite(x).all()
        rows.extend(decoder.evaluate(meta[order],x[order],family))
    output = "focused_psd_predictions.csv" if focused else "aligned_psd_predictions.csv"
    a.j.write_csv(ROOT/output,
                  ["family","subject","session","trial","label","draw","snr_db",
                   "synthesis","prediction","kernel","c"],rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=Path,required=True)
    parser.add_argument("--epochs",type=Path,required=True)
    parser.add_argument("--focused", action="store_true", help="Fit 14–31 Hz and 20–30 Hz 1 Hz PSD decoders.")
    args=parser.parse_args()
    validate()
    run(args.seed,args.epochs,args.focused)
