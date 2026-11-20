# -*- coding: utf-8 -*-
"""Reviewer 6, comment 9: association between moisture and the NIR spectra.

Moisture: loss on drying, one determination per sample, November 2026 (R2_revision/R6-9_moisture). Spectra: the 58 mean
spectra recorded in May 2026 (NIR/mean-1.csv, apparent absorbance). The two were measured about six months apart, so a weak
association does not show that water has no effect on the spectra.

1. Correlation spectrum: Pearson r between moisture and the spectrum at every wavenumber (SNV, and SNV + second derivative),
   over all samples and after removing category means from both (within-category).
2. Water bands: r at the O-H combination band (about 5,160 cm-1) and first overtone (about 6,900 cm-1).
3. Nested PLSR of moisture on NIR: the nested procedure of the manuscript (nested_plsr.py: 24 preprocessing options, 1-15
   LVs, 10 x 5 outer folds, inner 5-fold), with the category-mean baseline and the within-category R2 as for Comment 35.
4. How much of the category labels moisture alone carries: LDA on moisture, repeated stratified 5-fold CV.

  python r6_9_moisture_nir_association.py run     (about 3 min; without "run" the saved nested predictions are reused)
Outputs: R2_revision/R6-9_moisture/nir_association/, Figures_R2/FigS26_moisture_NIR.(png|pdf)
"""
import importlib.util
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
MO = ROOT / 'R2_revision' / 'R6-9_moisture'
OUT = MO / 'nir_association'
OUT.mkdir(exist_ok=True)
FIG = ROOT / 'R2_revision' / 'Figures_R2'
NESTED = ROOT / 'R1_revision' / 'comment03_validation' / 'nested_PLSR_20260910'
spec_ = importlib.util.spec_from_file_location('nested_plsr', NESTED / 'nested_plsr.py')
npl = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(npl)

dd, wn_cols, _ = npl.load_data(None)
mo = pd.read_csv(MO / 'moisture_loss_on_drying.csv', encoding='utf-8-sig').set_index('Modelling_ID')
dd['moisture'] = dd.sample_id.map(mo.moisture_percent)
assert dd.moisture.notna().all() and len(dd) == 58
wn = np.array([float(c[3:]) for c in wn_cols])
X = dd[wn_cols].to_numpy(float)
y = dd.moisture.to_numpy(float)
cat = dd.group.to_numpy(str)
ids, groups = dd.sample_id.to_numpy(str), dd.parent_batch_id.to_numpy(str)


def snv(a):
    return (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)


def centre(a):
    out = np.array(a, float, copy=True)
    for c in np.unique(cat):
        out[cat == c] -= out[cat == c].mean(0)
    return out


def corr(A, v):
    A0, v0 = A - A.mean(0), v - v.mean()
    return (A0 * v0[:, None]).sum(0) / np.sqrt((A0 ** 2).sum(0) * (v0 ** 2).sum())


XS = snv(X)
XD = savgol_filter(XS, 9, 3, deriv=2, delta=float(np.mean(np.diff(wn))), axis=1)
cs = pd.DataFrame({'wavenumber': wn, 'r_SNV_all': corr(XS, y), 'r_SNV_within_category': corr(centre(XS), centre(y)),
                   'r_SNV_2nd_derivative_all': corr(XD, y), 'r_SNV_2nd_derivative_within_category': corr(centre(XD), centre(y))})
cs.to_csv(OUT / 'correlation_spectrum.csv', index=False, encoding='utf-8-sig')
n = len(y)
r_crit = stats.t.ppf(0.975, n - 2) / np.sqrt(n - 2 + stats.t.ppf(0.975, n - 2) ** 2)          # |r| for p = 0.05, one test
r_crit_w = stats.t.ppf(0.975, n - 5) / np.sqrt(n - 5 + stats.t.ppf(0.975, n - 5) ** 2)        # within category: n - 4 - 1 df


def at(col, centre_wn, half=60):
    w = (wn > centre_wn - half) & (wn < centre_wn + half)
    k = np.abs(cs.loc[w, col]).idxmax()
    return float(cs.loc[k, col]), float(wn[k])


bands = []
for label, c0 in (('O-H combination, about 5,160 cm-1', 5160), ('O-H first overtone, about 6,900 cm-1', 6900)):
    row = {'band': label}
    for col in cs.columns[1:]:
        r, w_ = at(col, c0)
        row[col] = r
        row[col + '_at'] = w_
    bands.append(row)
bands = pd.DataFrame(bands)
bands.to_csv(OUT / 'correlation_at_water_bands.csv', index=False, encoding='utf-8-sig')
print(bands.round(3).T.to_string())
mx = {col: (float(cs[col].abs().max()), float(cs.loc[cs[col].abs().idxmax(), 'wavenumber']),
            float((cs[col].abs() > (r_crit_w if 'within' in col else r_crit)).mean())) for col in cs.columns[1:]}
print('largest |r|, its wavenumber, share of wavenumbers above the single-test 5 % limit:', mx)

# ---------------------------------------------------------------- nested PLSR of moisture on NIR
RUN = OUT / 'nested_plsr_moisture'
if len(sys.argv) > 1 and sys.argv[1] == 'run':
    from threadpoolctl import threadpool_limits
    Y = np.column_stack([y, y, y])        # nested_plsr.py expects three responses; the same response is given three times
    tasks = [(rep, f, tr, va) for rep in range(1, 11) for f, (tr, va) in enumerate(npl.splits(groups, 5, 42 + rep - 1), 1)]
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(npl.run_fold, t, X, Y, ids, groups, cat, wn, RUN) for t in tasks]
            results = [f.result() for f in as_completed(futures)]
        results.sort(key=lambda r: (r['metrics'][0]['Repeat'], r['metrics'][0]['Outer_fold']))
        npl.summarize(RUN, results)
pred = pd.read_csv(RUN / 'tables' / 'outer_OOF_predictions.csv', encoding='utf-8-sig')
pred = pred[pred.Target == 'TFC'].copy()         # first copy of the response (see above)
rows = []
for rep, s in pred.groupby('Repeat'):
    s = s.copy()
    s['base'] = np.nan
    for fold, va in s.groupby('Outer_fold'):
        tr = s[s.Outer_fold != fold]
        s.loc[va.index, 'base'] = va.Category.map(tr.groupby('Category').Observed.mean()).to_numpy()
    sst = ((s.Observed - s.Observed.mean()) ** 2).sum()
    ssw = ((s.Observed - s.groupby('Category').Observed.transform('mean')) ** 2).sum()
    sse = ((s.Predicted - s.Observed) ** 2).sum()
    rows.append({'Repeat': rep, 'R2': 1 - sse / sst, 'RMSE': np.sqrt(sse / len(s)),
                 'R2_category_mean': 1 - ((s.base - s.Observed) ** 2).sum() / sst, 'RMSE_category_mean': np.sqrt(((s.base - s.Observed) ** 2).mean()),
                 'R2_within_category': 1 - sse / ssw})
per = pd.DataFrame(rows)
per.to_csv(OUT / 'nested_plsr_moisture_per_repeat.csv', index=False, encoding='utf-8-sig')
print(per.mean().round(3).to_string())

# ---------------------------------------------------------------- category from moisture alone
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=50, random_state=2026)
acc = cross_val_score(LinearDiscriminantAnalysis(), y.reshape(-1, 1), cat, cv=cv).reshape(50, 5).mean(1)
majority = pd.Series(cat).value_counts(normalize=True).max()
print('LDA on moisture alone: accuracy %.3f (SD %.3f); majority class %.3f' % (acc.mean(), acc.std(), majority))

summary = {'n': n, 'r_single_test_5pct_all': round(float(r_crit), 2), 'r_single_test_5pct_within': round(float(r_crit_w), 2),
           'max_abs_r': {k: {'r': round(v[0], 2), 'wavenumber': round(v[1]), 'share_above_limit_percent': round(100 * v[2])}
                         for k, v in mx.items()},
           'water_bands': {('combination' if 'combination' in b.band else 'overtone'):
                           {'r_SNV_all': round(b.r_SNV_all, 2), 'r_SNV_within': round(b.r_SNV_within_category, 2),
                            'r_D2_all': round(b.r_SNV_2nd_derivative_all, 2), 'r_D2_within': round(b.r_SNV_2nd_derivative_within_category, 2)}
                           for _, b in bands.iterrows()},
           'plsr': {'R2': round(per.R2.mean(), 3), 'R2_sd': round(per.R2.std(), 3), 'RMSE': round(per.RMSE.mean(), 2),
                    'R2_category_mean': round(per.R2_category_mean.mean(), 3), 'RMSE_category_mean': round(per.RMSE_category_mean.mean(), 2),
                    'R2_within_category': round(per.R2_within_category.mean(), 2)},
           'lda_moisture_only': {'accuracy_percent': round(100 * acc.mean(), 1), 'sd_percent': round(100 * acc.std(), 1),
                                 'majority_class_percent': round(100 * majority, 1)}}
json.dump(summary, io.open(OUT / 'moisture_nir_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))

# ---------------------------------------------------------------- figure
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
fig, ax = plt.subplots(1, 3, figsize=(17 / 2.54, 6.2 / 2.54), gridspec_kw={'width_ratios': [1.25, 1.25, 1]})
for a, cols, title, lim in ((ax[0], ('r_SNV_all', 'r_SNV_within_category'), 'SNV spectra', None),
                            (ax[1], ('r_SNV_2nd_derivative_all', 'r_SNV_2nd_derivative_within_category'), 'SNV + second derivative', None)):
    a.plot(wn, cs[cols[0]], color='#1f6fb4', lw=0.8, label='All samples')
    a.plot(wn, cs[cols[1]], color='#c0392b', lw=0.8, label='Within category')
    for v in (r_crit, -r_crit):
        a.axhline(v, color='0.5', lw=0.5, ls='--')
    for b0 in (5160, 6900):
        a.axvline(b0, color='0.75', lw=0.5, ls=':')
    a.axhline(0, color='0.3', lw=0.5)
    a.set_xlim(wn.max(), wn.min())
    a.set_ylim(-1, 1)
    a.set_xlabel('Wavenumber (cm$^{-1}$)')
    a.set_title(title, fontsize=8)
ax[0].set_ylabel('Correlation with moisture (r)')
ax[0].legend(frameon=False, fontsize=7, loc='upper left')
m = pred.groupby('sample_id').agg(Observed=('Observed', 'first'), Predicted=('Predicted', 'mean'), Category=('Category', 'first'))
colors = {'FTR': '#1f6fb4', 'WTR': '#e08a1e', 'AST': '#2e8b57', 'SFTR': '#c0392b'}
for c, col in colors.items():
    s = m[m.Category == c]
    ax[2].scatter(s.Observed, s.Predicted, s=10, facecolor='none', edgecolor=col, lw=0.7, label=c)
lo, hi = m.Observed.min() - 0.2, m.Observed.max() + 0.2
ax[2].plot([lo, hi], [lo, hi], color='0.3', lw=0.6)
ax[2].set_xlim(lo, hi)
ax[2].set_ylim(lo, hi)
ax[2].set_xlabel('Moisture, loss on drying (%)')
ax[2].set_ylabel('Moisture predicted from NIR (%)')
ax[2].legend(frameon=False, fontsize=6.5, loc='upper left', handletextpad=0.2)
ax[2].set_title('Nested PLSR', fontsize=8)
for a, lab in zip(ax, 'ABC'):
    a.spines[['top', 'right']].set_visible(False)
    a.tick_params(width=0.6, length=2.5)
    a.text(-0.18, 1.06, lab, transform=a.transAxes, fontsize=9, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / 'FigS26_moisture_NIR.png', dpi=600)
fig.savefig(FIG / 'FigS26_moisture_NIR.pdf')
print('figure saved')
