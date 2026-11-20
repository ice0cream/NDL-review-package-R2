# -*- coding: utf-8 -*-
"""Can any model predict TFC/TPC/TSS WITHIN categories? (follow-up to Reviewer 6 comment 35)

Nested repeated 5-fold CV (seeds 42-46, same KFold as the R1 nested PLSR), inner 5-fold
selects preprocessing (6 options) and hyperparameters on outer-training samples only.
Models:
  PLSR_global      ordinary PLSR on spectra (R1 approach, reduced preprocessing grid)
  PLSR_within      category-centred PLSR: training category means are removed from spectra
                   and targets; prediction = training category mean + PLSR on the centred spectrum.
                   Uses the recorded category (NIR PLS-DA recovers it with 99.5% nested accuracy).
  SVR_global       RBF SVR on autoscaled spectra, C in {1,10,100} (y standardized)
  Category_mean    training mean of the sample's category (baseline)
Reported: overall R2, and pooled within-category R2 = 1 - SSE / sum over categories of SS around
the true category mean (0 = no better than knowing the category mean exactly).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'R2_revision' / 'R6-35_phytochem_baselines'
spec = pd.read_csv(ROOT / 'script' / 'data' / 'nir_spectra.csv')
ref = pd.read_csv(ROOT / 'phytochemical_assays' / 'DZ.CSV')
targets = {'TFC': 'TFC (μg RE/g)', 'TPC': 'TPC (μg GAE/g)', 'TSS': 'TSS (mg Glc/g)'}
d = spec.merge(ref, on='sample_id', suffixes=('', '_ref'))
wn_cols = sorted([c for c in spec if c.startswith('wn_')], key=lambda c: float(c[3:]))
WN = np.array([float(c[3:]) for c in wn_cols])
X_RAW = d[wn_cols].to_numpy(float)
CAT = d.group.to_numpy(str)
DELTA = float(np.mean(np.diff(WN)))


def snv(X):
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


PREPS = {'SNV': lambda X: snv(X),
         'SNV_FD_SG': lambda X: savgol_filter(snv(X), 9, 3, deriv=1, delta=DELTA, axis=1),
         'SNV_SD_SG': lambda X: savgol_filter(snv(X), 9, 3, deriv=2, delta=DELTA, axis=1),
         'FD_SG': lambda X: savgol_filter(X, 9, 3, deriv=1, delta=DELTA, axis=1),
         'SD_SG': lambda X: savgol_filter(X, 9, 3, deriv=2, delta=DELTA, axis=1),
         'SG': lambda X: savgol_filter(X, 9, 3, axis=1)}
XP = {k: f(X_RAW) for k, f in PREPS.items()}  # row-wise transforms only: no training statistics involved


def cat_center(Xtr, ytr, ctr, Xte, cte):
    means_x = {c: Xtr[ctr == c].mean(0) for c in np.unique(ctr)}
    means_y = {c: ytr[ctr == c].mean() for c in np.unique(ctr)}
    return (Xtr - np.array([means_x[c] for c in ctr]), ytr - np.array([means_y[c] for c in ctr]),
            Xte - np.array([means_x[c] for c in cte]), np.array([means_y[c] for c in cte]))


def fit_predict(model, prep, h, tr, te, y):
    Xtr, Xte = XP[prep][tr], XP[prep][te]
    if model == 'PLSR_within':
        Xtr, ytr, Xte, offset = cat_center(Xtr, y[tr], CAT[tr], Xte, CAT[te])
    else:
        ytr, offset = y[tr], 0.0
    sx = StandardScaler().fit(Xtr)
    A, B = sx.transform(Xtr), sx.transform(Xte)
    if model == 'SVR_global':
        sy = StandardScaler().fit(ytr[:, None])
        m = SVR(C=h, gamma='scale', epsilon=0.1).fit(A, sy.transform(ytr[:, None]).ravel())
        return sy.inverse_transform(m.predict(B)[:, None]).ravel() + offset
    m = PLSRegression(n_components=min(h, len(tr) - 2), scale=False, max_iter=2000).fit(A, ytr)
    return m.predict(B).ravel() + offset


GRID = {'PLSR_global': range(1, 13), 'PLSR_within': range(1, 11), 'SVR_global': [1, 10, 100]}


def nested(model, y, seed):
    pred = np.empty(len(y))
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(y)):
        inner = list(KFold(5, shuffle=True, random_state=seed * 100 + k).split(tr))
        best = None
        for prep in PREPS:
            for h in GRID[model]:
                sse = sum(np.sum((fit_predict(model, prep, h, tr[a], tr[b], y) - y[tr[b]]) ** 2) for a, b in inner)
                if best is None or sse < best[0]:
                    best = (sse, prep, h)
        pred[te] = fit_predict(model, best[1], best[2], tr, te, y)
    return pred


def category_mean(y, seed):
    pred = np.empty(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(y):
        means = {c: y[tr][CAT[tr] == c].mean() for c in np.unique(CAT[tr])}
        pred[te] = [means[c] for c in CAT[te]]
    return pred


def scores(y, p):
    ss_within = sum(np.sum((y[CAT == c] - y[CAT == c].mean()) ** 2) for c in np.unique(CAT))
    sse = np.sum((p - y) ** 2)
    return {'R2': 1 - sse / np.sum((y - y.mean()) ** 2), 'RMSE': np.sqrt(sse / len(y)),
            'R2_within_category': 1 - sse / ss_within}


rows = []
for name, col in targets.items():
    y = d[col].to_numpy(float)
    for seed in range(42, 47):
        rows.append({'Target': name, 'Seed': seed, 'Model': 'Category_mean', **scores(y, category_mean(y, seed))})
        for model in GRID:
            rows.append({'Target': name, 'Seed': seed, 'Model': model, **scores(y, nested(model, y, seed))})
    print(name, 'done', flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / 'alternative_models_per_repeat.csv', index=False, encoding='utf-8-sig')
summary = res.groupby(['Target', 'Model'])[['R2', 'RMSE', 'R2_within_category']].agg(['mean', 'std']).round(3)
summary.to_csv(OUT / 'alternative_models_summary.csv', encoding='utf-8-sig')
print(summary.to_string())
