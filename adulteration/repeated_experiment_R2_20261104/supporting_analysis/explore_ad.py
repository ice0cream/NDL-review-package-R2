# -*- coding: utf-8 -*-
"""EXPLORATORY: applicability-domain flags for the NIR-PLSR model (Hotelling T2, Q residuals, replicate SD).

Model: SNV (fixed), PLS1, LV chosen by inner leave-one-parent-out RMSECV, outer leave-one-parent-out.
For each outer fold, everything below uses the training parents only:
  T2, Q   limits = 95th / 99th percentile of the T2 and Q values of the training states, each obtained
          when its parent was held out of an inner model (inner leave-one-parent-out projection), so the
          limits describe 'new parent' states, as the test parent is.
  rep SD  SD of the predictions of the 3 repacked replicate spectra of a state by the fold model;
          limit = 95th / 99th percentile of the same statistic over the training states (inner LOPO models).
The held-out parent's states are then flagged against those limits. Prediction error is reported for
accepted and flagged states. Also applied to the late-fusion predictions (NIR + Lab OLS) from
explore_fusion.py.
Outputs: ./results_ad_explore. Run: python explore_ad.py  (about 1 min)
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

import redo_analysis as ra
import explore_fusion as ef

OUT = Path(__file__).resolve().parent / 'results_ad_explore'
KMAX = 15


def pls_fit(X, y, k):
    xm, ym = X.mean(0), y.mean()
    E, f = X - xm, y - ym
    W, P, Q, T = [], [], [], []
    for _ in range(k):
        w = E.T @ f
        w /= np.linalg.norm(w)
        t = E @ w
        tt = t @ t
        p, q = E.T @ t / tt, f @ t / tt
        E = E - np.outer(t, p)
        f = f - q * t
        W.append(w), P.append(p), Q.append(q), T.append(t)
    W, P, Q, T = np.array(W).T, np.array(P).T, np.array(Q), np.array(T).T
    B = W @ np.linalg.solve(P.T @ W, Q)
    return {'xm': xm, 'ym': ym, 'W': W, 'P': P, 'B': B, 'tvar': T.var(0, ddof=1)}


def project(m, Xn):
    E = Xn - m['xm']
    T = np.zeros((len(Xn), m['W'].shape[1]))
    for a in range(T.shape[1]):
        T[:, a] = E @ m['W'][:, a]
        E = E - np.outer(T[:, a], m['P'][:, a])
    return (T ** 2 / m['tvar']).sum(1), (E ** 2).sum(1)


def predict(m, Xn):
    return m['ym'] + (Xn - m['xm']) @ m['B']


def stats(m, idx, N, R, rep_of):
    t2, q = project(m, N[idx])
    sd = np.array([predict(m, R[rep_of[i]]).std(ddof=1) for i in idx])
    return t2, q, sd


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    files, S, axis = ra.load_spectra()
    st, X, C = ra.load_states(files, S)
    y = st.conc_recomputed.to_numpy(float)
    groups = st.parent.to_numpy()
    N, R = ra.snv(X), ra.snv(S)
    rep_of = [files.index[files.label == lab].to_numpy() for lab in st.label]
    assert all(len(r) == 3 for r in rep_of)

    design = ef.make_design('nir', N, None)
    rows = []
    for g in np.unique(groups):
        te, tr = np.where(groups == g)[0], np.where(groups != g)[0]
        _, k, _ = ef.select(design, [None], KMAX, y, tr, groups)
        # limits from inner leave-one-parent-out projections of the training states
        inner = {'t2': [], 'q': [], 'sd': []}
        for h in np.unique(groups[tr]):
            ite, itr = tr[groups[tr] == h], tr[groups[tr] != h]
            mi = pls_fit(N[itr], y[itr], k)
            for key, v in zip(['t2', 'q', 'sd'], stats(mi, ite, N, R, rep_of)):
                inner[key].append(v)
        lim = {key: np.percentile(np.concatenate(v), [95, 99]) for key, v in inner.items()}
        m = pls_fit(N[tr], y[tr], k)
        t2, q, sd = stats(m, te, N, R, rep_of)
        p = predict(m, N[te])
        for j, i in enumerate(te):
            rows.append({'state': st.label[i], 'parent': g, 'nominal': st.nominal[i], 'y': y[i], 'pred': p[j],
                         'LV': k, 'T2': t2[j], 'Q': q[j], 'repSD': sd[j],
                         **{f'{key}_lim{pc}': lim[key][n] for key in lim for n, pc in enumerate([95, 99])}})
    d = pd.DataFrame(rows)
    fus = pd.read_csv(Path(__file__).resolve().parent / 'results_fusion_explore' / 'fusion_screen_predictions.csv')
    d['pred_fusion'] = d.state.map(dict(zip(fus.state, fus.HL_ols_lab)))
    for pc in [95, 99]:
        for key in ['T2', 'Q', 'repSD']:
            d[f'flag_{key}_{pc}'] = d[key] > d[f'{key.replace("T2", "t2").replace("Q", "q").replace("repSD", "sd")}_lim{pc}']
        d[f'flag_TQ_{pc}'] = d[f'flag_T2_{pc}'] | d[f'flag_Q_{pc}']
        d[f'flag_any_{pc}'] = d[f'flag_TQ_{pc}'] | d[f'flag_repSD_{pc}']
    d.to_csv(OUT / 'ad_per_state.csv', index=False, encoding='utf-8-sig')

    rmse = lambda e: float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan
    summ = []
    for rule in ['T2', 'Q', 'TQ', 'repSD', 'any']:
        for pc in [95, 99]:
            f = d[f'flag_{rule}_{pc}']
            r = {'rule': rule, 'limit_pct': pc, 'n_flagged': int(f.sum()),
                 'flagged_states': ', '.join(d.state[f])}
            for col, nm in [('pred', 'NIR'), ('pred_fusion', 'fusion')]:
                e = d[col] - d.y
                r[f'RMSEP_{nm}_accepted'] = rmse(e[~f])
                r[f'RMSEP_{nm}_flagged'] = rmse(e[f])
            r['RMSE5%_NIR_accepted'] = rmse((d.pred - d.y)[~f & (d.nominal == 5)])
            summ.append(r)
    s = pd.DataFrame(summ)
    s.to_csv(OUT / 'ad_summary.csv', index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250, 'display.max_colwidth', 120)
    print('NIR all: %.2f   fusion all: %.2f' % (rmse(d.pred - d.y), rmse(d.pred_fusion - d.y)))
    print(s.round(2).to_string())
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
