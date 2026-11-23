# EEG interference replay

Source code and numerical evidence for *Spectral Propagation of Controlled EEG Interference Under Nominal SNR Calibration* by Yongbin Zhang, Yongfeng Lv, Xiuli Fu, and Yuyang Du.

## Start here

Use Python 3.12 and run from the repository root.

```sh
python -m pip install -r requirements.txt
python folding_map/check.py
python open_replay/demo.py --out synthetic_replay
python snr_audit/check_public.py
```

The [original server synthesizer](PrintShop_EEG_Synthesizer) supplies the historical waveform generator. The manuscript-specific [front-end audit](front_end), [matched joint replays](emg_release/analysis), [SEED PSD](decoder_grid/seed_psd.py), and [DEAP classifier](deap_decoding_and_weights/analysis/paired_deap.py) are separate entry points, with options listed under `--help`. The [calibration audit](snr_audit/audit_summary.json) records the original study's gain and ledger checks. The [decoder summaries](decoder_grid/decoder_grid_summary.json) and [feature-dynamics summaries](feature_dynamics/feature_dynamics_summary.json) report the corresponding numerical outcomes.

The exact soundtrack-based replays use a 128 Hz timing sequence derived from the cited [BlackAtom Productions recording on YouTube](https://www.youtube.com/watch?v=-udI5DCSQQA). Researchers with authorized access to the matching audio can rebuild the input using `python methods_audit/extract_control.py path/to/source.wav control.npz`. Compare the contiguous little-endian float32 array digest in the [line-replay comparison](line-replay-comparison.json). Scripts with an explicit `--control` option accept that NPZ. Scripts with a fixed control path expect the same file at `S1_Data_control_sequence.npz` in the location shown in the source.

The independent seeded AR(1) demo uses no soundtrack and tests the line-only front-end comparison. The comparison JSON records both the synthetic run and the archived benchmark. Participant recordings and artifact epochs are obtained from [SEED](https://bcmi.sjtu.edu.cn/home/seed/), [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), and [EEGdenoiseNet](https://gin.g-node.org/NCClab/EEGdenoiseNet) under their providers' terms. The MIT License covers the author-owned code.

The 256-to-512 Hz EMG match is the second, zero-based index column in `emg_release/results/epoch_map_256_to_512hz.csv`. To supply it to `--epoch-map`, save that column as an integer NumPy array using `numpy.save`.
