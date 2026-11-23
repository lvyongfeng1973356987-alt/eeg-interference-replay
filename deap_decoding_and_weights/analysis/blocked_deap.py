"""Fit clean-only, within-participant DEAP decoders on whole trial blocks."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from paired_deap import CASES, GRID, kernels, predict, save_csv


def blocked_predictions(meta, features):
    subjects, trials, labels = meta.T
    for subject in np.unique(subjects):
        participant = np.flatnonzero(subjects == subject)
        participant = participant[np.argsort(trials[participant])]
        np.testing.assert_array_equal(trials[participant], np.arange(1, 41))
        for block, test in enumerate(np.split(participant, 5), 1):
            train = np.setdiff1d(participant, test)
            splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=20261122)
            clean_guesses = np.empty((len(GRID), len(train)), dtype=int)
            for fit_index, valid_index in splitter.split(train, labels[train]):
                fit, valid = train[fit_index], train[valid_index]
                bank = [kernels(x[fit, 0], x[valid, 0]) for x in features]
                for i, (family, kind, c) in enumerate(GRID):
                    clean_guesses[i, valid_index] = predict(bank[family][kind], labels[fit], c)
            scores = [balanced_accuracy_score(labels[train], row) for row in clean_guesses]
            family, kind, c = GRID[int(np.argmax(scores))]
            x = features[family]
            test_matrix = x[test].reshape(-1, x.shape[-1])
            guess = predict(kernels(x[train, 0], test_matrix)[kind], labels[train], c)
            for index, row in zip(test, guess.reshape(len(test), -1), strict=True):
                for case, y in zip(CASES, row, strict=True):
                    yield (*meta[index], case, int(y), block, family, kind, c)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="Feature cache from paired_deap.py extract")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as data:
        rows = list(blocked_predictions(data["meta"], [data["x0"], data["x1"]]))
    save_csv(args.output, ("subject", "trial", "label", "case", "prediction", "block", "family", "kernel", "c"), rows)
