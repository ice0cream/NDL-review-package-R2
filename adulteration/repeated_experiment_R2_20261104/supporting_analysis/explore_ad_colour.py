# -*- coding: utf-8 -*-
"""EXPLORATORY: can the T2 / Q / replicate-SD applicability checks be applied to the colour (L*a*b*) model?

Same nested scheme as explore_ad.py: outer leave-one-parent-out; colour PLS on autoscaled L*, a*, b*
(LV 1-3 by inner leave-one-parent-out RMSECV); limits = 95th / 99th percentile of inner
leave-one-parent-out projections of the training states.
  T2     Hotelling T2 on the colour-PLS scores
  Q      residual after k LV (identically 0 when k = 3, because there are only 3 variables)
  repSD  SD of the colour-model predictions of the 3 individual readings. Entry errors (a reading more
         than 5x the screening tolerance from the state median; tolerance L 0.5, a 0.1, b 0.3) set to NaN.
Flags are checked against the colour-model errors and, as an external domain check, the NIR-model errors
(results_ad_explore/ad_per_state.csv).
Outputs: ./results_ad_explore/colour_*. Run: python explore_ad_colour.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import redo_analysis as ra
import explore_fusion as ef
from explore_ad import pls_fit, project, predict

OUT = Path(__file__).resolve().parent / 'results_ad_explore'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data_checks'))
import check_redo_data as sheetio  # noqa: E402

TOL = {'L': 0.5, 'a': 0.1, 'b': 0.3}


def replicate_colour(st):
    sh = sheetio.read_sheet()
    sh['label'] = sh.parent + '-S' + sh.step.astype(int).astype(str)
    sh = sh.set_index('label').loc[st.label]
    R = np.full((len(st), 3, 3), np.nan)          # state, reading, component
    for k, comp in enumerate('Lab'):
        v = np.array([[sheetio.num(x) for x in sh[f'c{1 + k + 3 * j}']] for j in range(3)]).T
        med = np.nanmedian(v, 1, keepdims=True)
        v[np.abs(v - med) > 5 * TOL[comp]] = np.nan
        R[:, :, k] = v
    return R


def main():
    files, S, axis = ra.load_spectra()
    st, X, C = ra.load_states(files, S)
    y = st.conc_recomputed.to_numpy(float)
    groups = st.parent.to_numpy()
    Rc = replicate_colour(st)
    n_typo = int(np.isnan(Rc).sum())

    def scaled(tr, A):
        m, s = C[tr].mean(0), C[tr].std(0, ddof=1)
        return (A - m) / s

    def stats(m, tr_scale, idx):
        t2, q = project(m, scaled(tr_scale, C[idx]))
        sd = np.array([np.nanstd(predict(m, scaled(tr_scale, r[~np.isnan(r).any(1)])), ddof=1) for r in Rc[idx]])
        return t2, q, sd

    design = ef.make_design('col', None, C)
    rows = []
    for g in np.unique(groups):
        te, tr = np.where(groups == g)[0], np.where(groups != g)[0]
        _, k, _ = ef.select(design, [None], 3, y, tr, groups)
        inner = {'T2': [], 'Q': [], 'repSD': []}
        for h in np.unique(groups[tr]):
            ite, itr = tr[groups[tr] == h], tr[groups[tr] != h]
            mi = pls_fit(scaled(itr, C[itr]), y[itr], k)
            for key, v in zip(inner, stats(mi, itr, ite)):
                inner[key].append(v)
        lim = {key: np.nanpercentile(np.concatenate(v), [95, 99]) for key, v in inner.items()}
        m = pls_fit(scaled(tr, C[tr]), y[tr], k)
        t2, q, sd = stats(m, tr, te)
        p = predict(m, scaled(tr, C[te]))
        for j, i in enumerate(te):
            r = {'state': st.label[i], 'nominal': st.nominal[i], 'y': y[i], 'pred_colour': p[j], 'LV': k,
                 'T2': t2[j], 'Q': q[j], 'repSD': sd[j]}
            for key in lim:
                for n, pc in enumerate([95, 99]):
                    r[f'flag_{key}_{pc}'] = bool(r[key] > lim[key][n] + 1e-12)
            rows.append(r)
    d = pd.DataFrame(rows)
    nir = pd.read_csv(OUT / 'ad_per_state.csv')
    d = d.merge(nir[['state', 'pred', 'flag_any_99']].rename(columns={'pred': 'pred_NIR', 'flag_any_99': 'NIRflag_99'}),
                on='state')
    d.to_csv(OUT / 'colour_ad_per_state.csv', index=False, encoding='utf-8-sig')

    rmse = lambda e: float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan
    out = []
    for key in ['T2', 'Q', 'repSD']:
        for pc in [95, 99]:
            f = d[f'flag_{key}_{pc}']
            out.append({'rule': key, 'pct': pc, 'n': int(f.sum()), 'states': ', '.join(d.state[f]),
                        'colour_acc': rmse((d.pred_colour - d.y)[~f]), 'colour_flag': rmse((d.pred_colour - d.y)[f]),
                        'NIR_acc': rmse((d.pred_NIR - d.y)[~f]), 'NIR_flag': rmse((d.pred_NIR - d.y)[f])})
    s = pd.DataFrame(out)
    s.to_csv(OUT / 'colour_ad_summary.csv', index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250, 'display.max_colwidth', 100)
    print(f'readings set to NaN as entry errors: {n_typo}; LV chosen: {sorted(d.LV.unique())}')
    print('colour all %.2f   NIR all %.2f' % (rmse(d.pred_colour - d.y), rmse(d.pred_NIR - d.y)))
    print(s.round(2).to_string())
    e = (d.pred_colour - d.y)
    print('largest colour errors:', ', '.join(f'{a}:{b:+.1f}' for a, b in zip(d.state[e.abs().nlargest(8).index], e[e.abs().nlargest(8).index])))
    print('colour T2 percentile rank of F1 states:', d.loc[d.state.str.startswith('F1-'), ['state', 'T2']].round(2).values.tolist())


if __name__ == '__main__':
    main()
