# -*- coding: utf-8 -*-
"""Reviewer 6 comment 37: permutation test of the COMPLETE nested PLS-DA development procedure.

For each permutation the category labels of the 58 samples are shuffled, and the whole
procedure is rerun: outer stratified 5-fold, inner stratified 5-fold selection over
24 preprocessing options x 1-8 LVs, refit, outer prediction. The statistic is the
outer-OOF accuracy of one complete repeat. The observed statistic is the same
procedure on the true labels (repeat 1, seed 42, as in the R1 nested analysis).

Labels are permuted at the sample level: no documented higher-level unit (lot) is
available for the 58 materials, so the test assumes exchangeable samples (see R6-3/26).
p = (1 + #{perm >= observed}) / (1 + n_perm).
Run: python r6_37_full_procedure_permutation.py --n-perm 200 --jobs 4
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'R2_revision' / 'R6-37_full_procedure_permutation'
SEED = 42


def load_r1_functions():
    """Import preprocessing and PLS-DA helpers from the R1 nested script without running its loop."""
    source = (ROOT / 'R1_revision' / 'scripts' / 'comment03_nested_cv.py').read_text(encoding='utf-8')
    head = source.split('# ------------------------------------------------------------------ nested CV')[0]
    head = head.replace('ROOT = Path(__file__).resolve().parents[2]', f'ROOT = Path(r"{ROOT}")')
    head = head.replace('TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)', '')
    namespace = {'__name__': 'r1_nested'}
    exec(compile(head, 'comment03_nested_cv_head', 'exec'), namespace)
    return namespace


_CACHE = {}


def r1():
    # Loaded lazily inside each worker process; exec-defined helpers cannot be pickled.
    if not _CACHE:
        _CACHE.update(load_r1_functions())
    return _CACHE


def nested_accuracy(y, seed=SEED):
    """One complete repeat of the R1 nested procedure on labels y; returns outer-OOF accuracy."""
    R1 = r1()
    X_ALL, COMBOS, LV_GRID = R1['X_ALL'], R1['COMBOS'], R1['LV_GRID']
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    for fold_id, (tr, te) in enumerate(outer.split(X_ALL, y), 1):
        Xtr, ytr = X_ALL[tr], y[tr]
        inner = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 1000 + fold_id).split(Xtr, ytr))
        best = None
        for combo in COMBOS:
            transformed = [R1['transform_pair'](Xtr[a], Xtr[b], combo) for a, b in inner]
            for lv in LV_GRID:
                acc = np.mean([accuracy_score(ytr[b], R1['plsda_fit_predict'](A, ytr[a], B, lv))
                               for (a, b), (A, B) in zip(inner, transformed)])
                key = (-acc, lv, combo[0])
                if best is None or key < best[0]:
                    best = (key, combo, lv)
        _, combo, lv = best
        A, B = R1['transform_pair'](Xtr, X_ALL[te], combo)
        pred[te] = R1['plsda_fit_predict'](A, ytr, B, lv)
    return float(accuracy_score(y, pred))


def permuted(i):
    """i = -1 runs the true labels; i >= 0 runs permutation i."""
    y = r1()['Y_ALL']
    if i >= 0:
        y = np.random.default_rng(10_000 + i).permutation(y)
    return i, nested_accuracy(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-perm', type=int, default=200)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(permuted)(i) for i in range(-1, args.n_perm))
    observed = dict(results)[-1]
    null = pd.DataFrame([r for r in results if r[0] >= 0], columns=['Permutation', 'Nested_accuracy'])
    null.to_csv(OUT / 'null_distribution.csv', index=False, encoding='utf-8-sig')
    p = (1 + int((null.Nested_accuracy >= observed).sum())) / (1 + len(null))
    summary = {'observed_nested_accuracy': observed, 'n_permutations': len(null), 'p_value': p,
               'null_mean': float(null.Nested_accuracy.mean()), 'null_max': float(null.Nested_accuracy.max()),
               'chance_level_majority_class': float(pd.Series(r1()['Y_ALL']).value_counts(normalize=True).max()),
               'scope': 'complete nested development procedure re-run per permutation; sample-level label exchangeability',
               'seconds': round(time.time() - start)}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
