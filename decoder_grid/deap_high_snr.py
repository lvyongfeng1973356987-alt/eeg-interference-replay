"""Evaluate higher-SNR DEAP mixtures using the original clean-only decoders."""
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"deap_decoding_and_weights/analysis"))
import paired_deap as deap
import blocked_deap as blocked

LEVELS=(5,10,20)
CASES=("clean",*(f"{mode}_{level}" for level in LEVELS for mode in ("direct","aa")))


def run(data, epochs, control):
    deap.SNR_LEVELS=LEVELS
    deap.CASES=blocked.CASES=CASES
    meta, rows = [], []
    for subject, dataset in deap.f.deap_files(data):
        labels=(dataset["labels"][:,0]>=5).astype(int)
        for trial,(raw,label) in enumerate(zip(dataset["data"],labels,strict=True),1):
            x=np.asarray(raw[:32],dtype="float32")
            direct,aa,_,_,background,rng=deap.components(x,control,subject,trial)
            eye, muscle = (deap.f.unit_noise(deap.f.artifact(bank, profile, rng))
                           for bank, profile in zip(epochs,(deap.f.EOG_PROFILE,deap.f.EMG_PROFILE)))
            physiology=np.sqrt(.4)*eye+np.sqrt(.35)*muscle
            waves=[.5*deap.f.unit_noise(wave+background)+physiology for wave in (direct,aa)]
            rows.append(deap.valence_features(x,waves))
            meta.append((subject,trial,label))
        print(f"DEAP higher-SNR features: {subject}/32",flush=True)
    assert len(meta)==1280
    meta=np.asarray(meta)
    features=[np.stack([x[i] for x in rows]) for i in range(2)]
    predictions, selection = deap.decode(features,meta)
    keys=("subject","trial","label","case","prediction")
    deap.save_csv(HERE/"deap_lopo.csv",keys,predictions)
    deap.save_csv(HERE/"deap_lopo_selection.csv",("subject","family","kernel","c","inner_clean_ba"),selection)
    deap.save_csv(HERE/"deap_blocked.csv",(*keys,"block","family","kernel","c"),
                  blocked.blocked_predictions(meta,features))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--epochs",type=Path,required=True)
    a=p.parse_args()
    control_path=HERE.parent/"decoding_inference/S1_Data_control_sequence.npz"
    with np.load(control_path,allow_pickle=False) as source:
        run(a.data,deap.release_epochs(a.epochs),source["control_sequence"])
