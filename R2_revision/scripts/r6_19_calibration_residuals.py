# -*- coding: utf-8 -*-
"""Reviewer 6, comment 19: calibration levels, residuals and range of the three colorimetric assays.

Input: phytochemical_assays/1/calibration_curves.csv (standard concentration, three absorbance readings per level; blocks TPC, TSS, TFC), the data
behind Table S4. The working equations of Table S4 (coefficients to four decimals, zero standard included as a level) are
the ones with which the sample values were calculated (workbooks TPC_assay.xlsx, TFC_assay.xlsx, TSS_assay.xlsx).
Checks: (1) the working equation and R2 of Table S4 are reproduced; (2) residuals of every reading; (3) back-calculated
concentration of every level; (4) lack-of-fit F test (pure error from the three readings) and test of a quadratic term;
(5) position of the sample absorbances within the calibrated absorbance range.
Outputs: R2_revision/R6-19_calibration/ and Figures_R2/FigS22_calibration_residuals.(png|pdf)
"""
import io
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'phytochemical_assays' / 'raw_records'
OUT = ROOT / 'R2_revision' / 'R6-19_calibration'
OUT.mkdir(exist_ok=True)
FIG = ROOT / 'R2_revision' / 'Figures_R2'
NAMES = {'总黄酮': 'TFC', '总多酚': 'TPC', '总多糖': 'TSS'}
UNIT = {'TFC': 'µg/mL', 'TPC': 'µg/mL', 'TSS': 'mg/mL'}
STD = {'TFC': 'rutin', 'TPC': 'gallic acid', 'TSS': 'glucose'}
WORKING = {'TFC': (0.0028, 0.0552, 0.999220), 'TPC': (0.0028, 0.0572, 0.995902), 'TSS': (2.7391, 0.0673, 0.997667)}   # Table S4
SAMPLES = {'TFC': ('TFC_assay.xlsx', 'Sheet1', 2, 59), 'TPC': ('TPC_assay.xlsx', 'Sheet1', 9, 66), 'TSS': ('TSS_assay.xlsx', '原始计算表', 2, 59)}

# ---------------------------------------------------------------- read
raw = io.open(SRC / 'calibration_curves.csv', encoding='utf-8-sig').read()
rows, name = [], None
block = []
for line in raw.splitlines() + [',,,,']:
    p = line.split(',')
    if not p[0].strip():
        for c, a1, a2, a3 in block:
            for k, a in enumerate((a1, a2, a3), 1):
                rows.append({'assay': NAMES[name], 'concentration': c, 'reading': k, 'absorbance': a})
        block = []
        continue
    if len(p) > 4 and p[4].strip():
        name = p[4].strip()
    block.append([float(x) for x in p[:4]])
d = pd.DataFrame(rows)
d.to_csv(OUT / 'calibration_readings.csv', index=False, encoding='utf-8-sig')

summary, levels_out, fig_data = {}, [], {}
for assay in ('TFC', 'TPC', 'TSS'):
    s = d[d.assay == assay]
    x, y = s.concentration.to_numpy(), s.absorbance.to_numpy()
    lv = s.groupby('concentration').absorbance.agg(['mean', 'std']).reset_index()
    m_fit, b_fit = np.polyfit(lv.concentration, lv['mean'], 1)
    m, b, r2_s4 = WORKING[assay]
    assert round(m_fit, 4) == m and round(b_fit, 4) == b, (assay, m_fit, b_fit)      # Table S4 equation reproduced
    res = y - (m * x + b)
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert abs(r2 - r2_s4) < 5e-7, (assay, r2)                                        # Table S4 R2 reproduced
    r2_exact = 1 - ((lv['mean'] - m_fit * lv.concentration - b_fit) ** 2).sum() / ((lv['mean'] - lv['mean'].mean()) ** 2).sum()
    # lack of fit of the straight line (least squares on all readings)
    mm, bb = np.polyfit(x, y, 1)
    sse = ((y - mm * x - bb) ** 2).sum()
    spe = ((y - s.groupby('concentration').absorbance.transform('mean')) ** 2).sum()
    k, n = lv.shape[0], len(y)
    f_lof = ((sse - spe) / (k - 2)) / (spe / (n - k))
    p_lof = stats.f.sf(f_lof, k - 2, n - k)
    # quadratic term, tested on the level means (levels, not readings, are the independent units)
    q = np.polyfit(lv.concentration, lv['mean'], 2)
    sse1 = ((lv['mean'] - m_fit * lv.concentration - b_fit) ** 2).sum()
    sse2 = ((lv['mean'] - np.polyval(q, lv.concentration)) ** 2).sum()
    f_q = (sse1 - sse2) / (sse2 / (k - 3))
    p_q = stats.f.sf(f_q, 1, k - 3)
    lv['fitted_working_equation'] = m * lv.concentration + b
    lv['residual'] = lv['mean'] - lv.fitted_working_equation
    lv['back_calculated'] = (lv['mean'] - b) / m
    lv['back_calculated_error_percent'] = np.where(lv.concentration > 0, 100 * (lv.back_calculated - lv.concentration) / lv.concentration.replace(0, np.nan), np.nan)
    lv.insert(0, 'assay', assay)
    levels_out.append(lv)
    nz = lv[lv.concentration > 0]
    # sample absorbances
    f, sh, r0, r1 = SAMPLES[assay]
    tab = pd.read_csv(SRC / 'sample_absorbances.csv', float_precision='round_trip')   # extracted from the workbook rows r0-r1
    samp = tab.loc[tab.assay == assay, ['A1', 'A2', 'A3']].to_numpy(float).mean(1)
    assert len(samp) == 58 and np.isfinite(samp).all()
    a_lo, a_hi = lv['mean'].iloc[1], lv['mean'].iloc[-1]             # lowest non-zero standard, highest standard
    summary[assay] = {
        'standard': STD[assay], 'unit': UNIT[assay], 'levels_including_zero': int(k), 'nonzero_levels': int(k - 1),
        'lowest': float(nz.concentration.min()), 'highest': float(nz.concentration.max()), 'readings_per_level': 3,
        'slope_exact': round(float(m_fit), 6), 'intercept_exact': round(float(b_fit), 6), 'slope_working': m, 'intercept_working': b,
        'R2_readings_working_equation': round(float(r2), 6), 'R2_level_means_exact_fit': round(float(r2_exact), 6),
        'max_abs_residual': round(float(np.abs(res).max()), 4),
        'max_abs_residual_percent_of_top': round(float(100 * np.abs(res).max() / lv['mean'].iloc[-1]), 1),
        'back_calc_error_percent_min': round(float(nz.back_calculated_error_percent.min()), 1),
        'back_calc_error_percent_max': round(float(nz.back_calculated_error_percent.max()), 1),
        'back_calc_error_lowest_level_percent': round(float(nz.back_calculated_error_percent.iloc[0]), 1),
        'back_calc_abs_error_max_excluding_lowest': round(float(nz.back_calculated_error_percent.iloc[1:].abs().max()), 1),
        'lack_of_fit_F': round(float(f_lof), 1), 'lack_of_fit_p': float(p_lof),
        'quadratic_F': round(float(f_q), 2), 'quadratic_p': round(float(p_q), 3),
        'sample_absorbance_min': round(float(samp.min()), 3), 'sample_absorbance_max': round(float(samp.max()), 3),
        'calibrated_absorbance_low': round(float(a_lo), 3), 'calibrated_absorbance_high': round(float(a_hi), 3),
        'samples_below_lowest_standard': int((samp < a_lo).sum()), 'samples_above_highest_standard': int((samp > a_hi).sum())}
    fig_data[assay] = (x, y, res, m, b, lv)
levels = pd.concat(levels_out)
levels.to_csv(OUT / 'calibration_levels_residuals.csv', index=False, encoding='utf-8-sig')
json.dump(summary, io.open(OUT / 'calibration_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))
print(levels.round(4).to_string())

# ---------------------------------------------------------------- figure
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
fig, ax = plt.subplots(2, 3, figsize=(17 / 2.54, 10.5 / 2.54), sharex='col')
for j, assay in enumerate(('TFC', 'TPC', 'TSS')):
    x, y, res, m, b, lv = fig_data[assay]
    xx = np.linspace(0, x.max(), 50)
    ax[0, j].plot(xx, m * xx + b, color='0.3', lw=0.8)
    ax[0, j].scatter(x, y, s=10, facecolor='none', edgecolor='#1f6fb4', lw=0.7)
    ax[0, j].set_title('{} ({})'.format(assay, STD[assay]), fontsize=8)
    ax[0, j].text(0.04, 0.95, 'A = {}c + {}\nR² = {:.4f}'.format(m, b, summary[assay]['R2_readings_working_equation']),
                  transform=ax[0, j].transAxes, va='top', fontsize=7)
    ax[1, j].axhline(0, color='0.3', lw=0.8)
    ax[1, j].scatter(x, res, s=10, facecolor='none', edgecolor='#1f6fb4', lw=0.7)
    ax[1, j].plot(lv.concentration, lv.residual, color='#c0392b', lw=0.8)
    lim = np.abs(res).max() * 1.25
    ax[1, j].set_ylim(-lim, lim)
    ax[1, j].set_xlabel('Standard concentration ({})'.format(UNIT[assay]))
    for a in ax[:, j]:
        a.spines[['top', 'right']].set_visible(False)
        a.tick_params(width=0.6, length=2.5)
ax[0, 0].set_ylabel('Absorbance')
ax[1, 0].set_ylabel('Residual (absorbance)')
for a, lab in zip(ax.ravel(), 'ABCDEF'):
    a.text(-0.2, 1.05, lab, transform=a.transAxes, fontsize=9, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / 'FigS22_calibration_residuals.png', dpi=600)
fig.savefig(FIG / 'FigS22_calibration_residuals.pdf')
print('figure saved')
