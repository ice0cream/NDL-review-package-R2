# -*- coding: utf-8 -*-
"""Reviewer 6 comment 12: replicate (within-sample) vs between-sample spectral variation.

Uses the 58 x 3 raw replicate spectra (OPUS block Abs_from_ScSm_ScRf), linked to modelling IDs by
exact spectral identity of their means. For raw, SNV and SNV + second-derivative (SG 9/3) spectra,
reports per wavenumber the pooled within-sample SD and the between-sample SD of sample means within
each category, averaged over the spectrum.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'NIR' / 'botanical_samples' / 'converted_csv' / 'batch_spectra'
LINK = ROOT / 'R2_revision' / 'R6-3_sample_traceability' / 'raw_batch_to_modelling_ID_by_spectrum.csv'
OUT = ROOT / 'R2_revision' / 'R6-12_15_spectral_repeatability'
link = pd.read_csv(LINK)


def snv(X):
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


PREP = {'raw': lambda X: X, 'SNV': snv,
        'SNV_SD_SG': lambda X: savgol_filter(snv(X), 9, 3, deriv=2, axis=1)}
reps = {}
for _, r in link.iterrows():
    f = pd.read_csv(SRC / f'{r.batch_id}.csv')
    reps[r.modelling_ID] = (r.category, f[[c for c in f if c.startswith('rep_')]].to_numpy(float).T)

rows = []
for name, fn in PREP.items():
    within, means, cats = [], [], []
    for mid, (cat, X) in reps.items():
        P = fn(X)
        within.append(P.var(0, ddof=1))
        means.append(P.mean(0)); cats.append(cat)
    within, means, cats = np.array(within), np.array(means), np.array(cats)
    for cat in ['FTR', 'WTR', 'SFTR', 'AST']:
        k = cats == cat
        w = np.sqrt(within[k].mean(0))           # pooled within-sample SD per wavenumber
        b = means[k].std(0, ddof=1)               # between-sample SD of means per wavenumber
        # between-sample variance expected from replicate noise alone = w^2 / 3
        rows.append({'Preprocessing': name, 'Category': cat, 'n_samples': int(k.sum()),
                     'within_SD_mean': w.mean(), 'between_SD_mean': b.mean(),
                     'ratio_within_to_between': (w / b).mean(),
                     'share_of_between_variance_from_replicate_noise': np.mean((w ** 2 / 3) / b ** 2)})
res = pd.DataFrame(rows)
res.to_csv(OUT / 'within_vs_between_by_preprocessing.csv', index=False, encoding='utf-8-sig')
print(res.round(4).to_string(index=False))
