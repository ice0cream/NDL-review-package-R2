# -*- coding: utf-8 -*-
"""Read-only audit of the re-run adulteration data (NIR OPUS files + Excel record sheet).

Nothing in repeated_experiment_raw_spectra or the record sheet is changed. Outputs (this folder):
  file_inventory.csv      one row per OPUS file: parent, step, replicate, time, apparent starch fraction
  state_summary.csv       one row per parent x step: file count, time span, record-sheet values
  weighing_checks.csv     per step: mass continuity and recomputed concentration
  colour_checks.csv       colour replicate outliers and L* direction
Apparent starch fraction: each SNV spectrum is fitted as (1-f)*parent S0 mean + f*starch reference mean
(least squares on SNV spectra). It is a rough label check, not a calibration.
"""
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'repeated_experiment_raw_spectra'
SHEET = ROOT / 'repeated_experiment_record_sheet.csv'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'adulteration' / 'NIR' / 'R1_cassava_starch_spectra'))
from opus_mean_spectra_cassava_FINAL_v2 import read_opus_spectrum  # noqa: E402

PAT = re.compile(r'^F1_(?P<label>.+?)-(?P<rep>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.001$')
STATE = re.compile(r'^(?P<parent>F\d+)-S(?P<step>\d)$')
NOMINAL = {0: 0, 1: 5, 2: 15, 3: 30, 4: 50}


def snv(a):
    return (a - a.mean(axis=-1, keepdims=True)) / a.std(axis=-1, keepdims=True)


def inventory():
    rows, spectra, axis = [], [], None
    for f in sorted(DATA.iterdir()):
        m = PAT.match(f.name)
        row = {'file': f.name, 'bytes': f.stat().st_size}
        if not m:
            row['name_issue'] = 'unparsed name'
            rows.append(row)
            spectra.append(None)
            continue
        label = m['label']
        s = STATE.match(label)
        row.update(label=label, rep=int(m['rep']),
                   time=pd.to_datetime(m['date'] + m['time'], format='%Y%m%d%H%M%S'))
        if s:
            row.update(parent=s['parent'], step=int(s['step']))
        elif label.startswith('MSDF'):
            row.update(parent='MSDF', step=-1)
        else:
            row['name_issue'] = f'label "{label}" is not Fxx-Sk'
        x, y, _ = read_opus_spectrum(f)
        if axis is None:
            axis = x
        if len(x) != len(axis) or not np.allclose(x, axis):
            y = np.interp(axis[::-1], x[::-1], y[::-1])[::-1]
            row['axis_note'] = 'interpolated'
        rows.append(row)
        spectra.append(y)
    inv = pd.DataFrame(rows)
    X = np.vstack(spectra)
    return inv, X, axis


def apparent_fraction(inv, X):
    Z = snv(X)
    starch = Z[(inv.parent == 'MSDF').to_numpy()].mean(axis=0)
    frac = np.full(len(inv), np.nan)
    resid = np.full(len(inv), np.nan)
    for p, g in inv[inv.parent.str.match(r'F\d+$', na=False)].groupby('parent'):
        s0 = g[(g.step == 0) & ~g.file.isin(EXCLUDE_FROM_S0)]
        base = Z[s0.index].mean(axis=0)
        d = starch - base
        for i in g.index:
            f = float(np.dot(Z[i] - base, d) / np.dot(d, d))
            frac[i] = f
            resid[i] = float(np.sqrt(np.mean((Z[i] - base - f * d) ** 2)))
    inv['apparent_starch_frac'] = frac
    inv['fit_rmse'] = resid
    # replicate consistency: distance of each SNV spectrum to its label's median
    inv['dist_to_label_median'] = np.nan
    for _, g in inv.groupby('label'):
        med = np.median(Z[g.index], axis=0)
        inv.loc[g.index, 'dist_to_label_median'] = np.sqrt(((Z[g.index] - med) ** 2).mean(axis=1))
    return inv


EXCLUDE_FROM_S0 = set()


def read_sheet():
    """The record sheet as extracted to a table (one row per state); plain numbers are read as numbers."""
    sh = pd.read_csv(SHEET, float_precision='round_trip', keep_default_na=False, na_values=[''])
    for col in sh.columns[sh.dtypes == object]:
        sh[col] = [float(v) if isinstance(v, str) and re.match(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$', v) else v for v in sh[col]]
    return sh


def read_sheet_workbook():
    wv = load_workbook(SHEET, data_only=True)
    wf = load_workbook(SHEET)
    rows = []
    for d, name in enumerate(['第1天', '第2天', '第3天'], 1):
        sv, sf = wv[name], wf[name]
        for r in range(3, sv.max_row + 1):
            parent = sv.cell(r, 2).value
            if not parent:
                continue
            g = lambda c: sv.cell(r, c).value
            formula = lambda c: isinstance(sf.cell(r, c).value, str) and str(sf.cell(r, c).value).startswith('=')
            rows.append({'sheet_day': d, 'row': r, 'block': g(1), 'parent': parent, 'step': g(3), 'nominal': g(4),
                         'tare': g(5), 'before': g(6), 'add_target': g(7), 'after': g(9),
                         'after_is_formula': formula(9), 'add_is_typed': not formula(7) and g(3) != 0,
                         'conc_sheet': g(10),
                         **{f'c{k}': g(14 + k) for k in range(1, 10)}})
    return pd.DataFrame(rows)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def weighing_checks(sh):
    out = []
    for p, g in sh.groupby('parent', sort=False):
        g = g.sort_values('step')
        tare = num(g.iloc[0]['tare'])
        c_prev, after_prev = 0.0, num(g.iloc[0]['after'])
        for _, r in g.iterrows():
            if r.step == 0:
                out.append({'parent': p, 'step': 0, 'parent_mass_g': after_prev - tare, 'conc_recomputed': 0.0})
                continue
            before, after = num(r.before), num(r.after)
            net_before = before - tare
            c = (c_prev / 100 * net_before + (after - before)) / (after - tare) * 100
            out.append({'parent': p, 'step': int(r.step), 'nominal': r.nominal,
                        'mass_change_since_last_g': round(before - after_prev, 3),
                        'starch_added_g': round(after - before, 3), 'after_is_formula': r.after_is_formula,
                        'conc_sheet': num(r.conc_sheet), 'conc_recomputed': round(c, 3)})
            c_prev, after_prev = c, after
    return pd.DataFrame(out)


def colour_checks(sh):
    out = []
    for _, r in sh.iterrows():
        vals = {}
        for k, comp in enumerate(['L', 'a', 'b']):
            reps = [r[f'c{1 + k + 3 * j}'] for j in range(3)]
            nums = [num(v) for v in reps]
            vals[comp] = nums
            bad_type = [str(v) for v in reps if not isinstance(v, (int, float))]
            med = np.nanmedian(nums)
            dev = [abs(v - med) for v in nums]
            limit = {'L': 0.5, 'a': 0.1, 'b': 0.3}[comp]
            if bad_type or max(dev) > limit:
                out.append({'parent': r.parent, 'step': r.step, 'component': comp + '*',
                            'readings': reps, 'issue': 'non-numeric' if bad_type else f'reading off median by {max(dev):.2f}'})
        r_mean = {c: np.nanmedian(v) for c, v in vals.items()}
        sh.loc[_, ['L_med', 'a_med', 'b_med']] = [r_mean['L'], r_mean['a'], r_mean['b']]
    return pd.DataFrame(out), sh


def main():
    inv, X, axis = inventory()
    # the second F3 S0 set and the duplicate F2 S0-1 are kept out of the S0 baseline until resolved
    EXCLUDE_FROM_S0.update(inv.loc[(inv.label == 'F3-S0') & (inv.time.dt.hour < 12), 'file'])
    inv = apparent_fraction(inv, X)
    inv.to_csv(OUT / 'file_inventory.csv', index=False, encoding='utf-8-sig')

    sh = read_sheet()
    wc = weighing_checks(sh)
    wc.to_csv(OUT / 'weighing_checks.csv', index=False, encoding='utf-8-sig')
    cc, sh = colour_checks(sh)
    cc.to_csv(OUT / 'colour_checks.csv', index=False, encoding='utf-8-sig')

    st = (inv[inv.step >= 0].groupby(['parent', 'step'])
          .agg(n_files=('file', 'size'), reps=('rep', lambda s: ','.join(map(str, sorted(s)))),
               first=('time', 'min'), last=('time', 'max'),
               apparent_frac=('apparent_starch_frac', 'median')).reset_index())
    st = st.merge(sh[['parent', 'step', 'sheet_day', 'block', 'L_med', 'a_med', 'b_med']], on=['parent', 'step'], how='outer')
    st = st.merge(wc[['parent', 'step', 'conc_recomputed']], on=['parent', 'step'], how='left')
    st['span_min'] = (st['last'] - st['first']).dt.total_seconds() / 60
    st.sort_values(['first']).to_csv(OUT / 'state_summary.csv', index=False, encoding='utf-8-sig')
    print('files', len(inv), 'name issues', inv.name_issue.notna().sum())


if __name__ == '__main__':
    main()
