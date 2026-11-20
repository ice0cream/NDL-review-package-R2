# -*- coding: utf-8 -*-
"""EXPLORATORY screen: which way of combining NIR and colour gives the lowest prediction error?

Not a pre-specified analysis. Many configurations are compared on the same 110 states, so the best
one is optimistically biased. Use it only to choose what to pre-specify and then confirm.

Fast settings: NIR preprocessing fixed to SNV (applied after region selection). Outer leave-one-parent-out
(22 folds). Inside each training set, the number of LV (1-15) and any listed tuning parameter are chosen
by inner leave-one-parent-out RMSECV. Stacking learners are fitted on the inner out-of-fold predictions.

Families
  REF   NIR alone (full range, <=9000 cm-1, starch bands); colour alone
  LL    low-level fusion: SNV-NIR and colour features, each block autoscaled, NIR weighted 1/sqrt(p_nir),
        colour weighted w/sqrt(p_col). w fixed (0.5-16) or chosen in the inner CV (nested w)
        colour features: Lab | Lab + C*, h, whiteness | Lab quadratic (squares and products)
  HL    late / decision fusion on the inner OOF predictions of the NIR and colour models:
        convex weight, OLS on both predictions, OLS on NIR prediction + Lab, SVR on NIR prediction + Lab
Outputs: ./results_fusion_explore. Run: python explore_fusion.py  (about 5 min)
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import redo_analysis as ra

OUT = Path(__file__).resolve().parent / 'results_fusion_explore'
KMAX = 15
WGRID = [0.5, 1, 2, 4, 8, 16]


# ---------------------------------------------------------------- features
def colour_feats(C, kind):
    L, a, b = C.T
    if kind == 'Lab':
        return C
    if kind == 'LabCHW':
        return np.column_stack([L, a, b, np.hypot(a, b), np.degrees(np.arctan2(b, a)),
                                100 - np.sqrt((100 - L) ** 2 + a ** 2 + b ** 2)])
    if kind == 'Labquad':
        return np.column_stack([L, a, b, L * L, a * a, b * b, L * a, L * b, a * b])
    raise ValueError(kind)


def region_mask(axis, region):
    if region == 'full':
        return np.ones_like(axis, bool)
    if region == 'le9000':
        return axis <= 9000
    if region == 'starch':          # 1st overtone / combination bands of O-H and C-H in starch
        return ((axis >= 4000) & (axis <= 5000)) | ((axis >= 5500) & (axis <= 7500))
    raise ValueError(region)


# ---------------------------------------------------------------- generic nested PLS
def make_design(kind, N, F):
    """kind: 'nir' | 'col' | 'll'. N: SNV spectra, F: colour features. param = colour weight w."""
    def f(tr, te, w):
        if kind == 'nir':
            return N[tr], N[te]
        if kind == 'col':
            return ra.autoscale(F[tr], F[te])
        a, b = ra.autoscale(N[tr], N[te])
        c, d = ra.autoscale(F[tr], F[te])
        wn, wc = 1 / np.sqrt(a.shape[1]), w / np.sqrt(c.shape[1])
        return np.hstack([a * wn, c * wc]), np.hstack([b * wn, d * wc])
    return f


def all_lv(design, w, K, y, tr, te):
    A, B = design(tr, te, w)
    return np.column_stack([B @ b + b0 for b, b0 in ra.pls1(A, y[tr], K)])


def select(design, params, K, y, idx, groups):
    """Inner LOPO over idx. Returns (w, k) and the OOF predictions for that choice."""
    best, store = None, {}
    for w in params:
        oof = np.full((len(idx), K), np.nan)
        for g in np.unique(groups[idx]):
            m = groups[idx] == g
            p = all_lv(design, w, K, y, idx[~m], idx[m])
            oof[m, :p.shape[1]] = p
        r = np.sqrt(np.nanmean((oof - y[idx, None]) ** 2, 0))
        k = int(np.nanargmin(r))
        if best is None or r[k] < best[2]:
            best = (w, k + 1, r[k])
        store[w] = oof
    return best[0], best[1], store[best[0]][:, best[1] - 1]


def nested(design, params, K, y, groups):
    pred, info = np.full(len(y), np.nan), {}
    for g in np.unique(groups):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        w, k, oof = select(design, params, K, y, tr, groups)
        pred[te] = all_lv(design, w, k, y, tr, te)[:, k - 1]
        info[g] = {'tr': tr, 'te': te, 'oof': oof, 'w': w, 'k': k}
    return pred, info


# ---------------------------------------------------------------- late fusion
def late(kind, nir, col, F, y, groups):
    pred, chosen = np.full(len(y), np.nan), []
    for g, a in nir.items():
        tr, te = a['tr'], a['te']
        zn, zc = a['oof'], col[g]['oof']                        # inner OOF predictions (training parents)
        pn = nir_pred[te]
        pc = col_pred[te]
        if kind == 'convex':
            ws = np.linspace(0, 1, 101)
            r = [np.sqrt(np.mean((w * zn + (1 - w) * zc - y[tr]) ** 2)) for w in ws]
            w = ws[int(np.argmin(r))]
            pred[te] = w * pn + (1 - w) * pc
            chosen.append(w)
        elif kind == 'ols2':
            Z = np.column_stack([np.ones(len(tr)), zn, zc])
            beta = np.linalg.lstsq(Z, y[tr], rcond=None)[0]
            pred[te] = beta[0] + beta[1] * pn + beta[2] * pc
            chosen.append(beta[2])
        elif kind == 'ols_lab':
            Z = np.column_stack([np.ones(len(tr)), zn, F[tr]])
            beta = np.linalg.lstsq(Z, y[tr], rcond=None)[0]
            pred[te] = np.column_stack([np.ones(len(te)), pn, F[te]]) @ beta
        elif kind == 'svr_lab':
            Z, Zt = np.column_stack([zn, F[tr]]), np.column_stack([pn, F[te]])
            gs = GridSearchCV(make_pipeline(StandardScaler(), SVR()),
                              {'svr__C': [10, 100, 1000], 'svr__epsilon': [0.5, 1, 2], 'svr__gamma': ['scale', 0.05]},
                              cv=LeaveOneGroupOut(), scoring='neg_root_mean_squared_error', n_jobs=-1)
            gs.fit(Z, y[tr], groups=groups[tr])
            pred[te] = gs.predict(Zt)
    return pred, chosen


# ---------------------------------------------------------------- evaluation
def evaluate(name, family, p, y, st, ref=None, note=''):
    e = p - y
    nom = st.nominal.to_numpy()
    out = {'config': name, 'family': family, 'RMSEP_all110': np.sqrt(np.mean(e ** 2)),
           'R2': 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), 'MAE': np.mean(np.abs(e)),
           'RMSE_mixtures': np.sqrt(np.mean(e[nom > 0] ** 2))}
    for lv in [0, 5, 15, 30, 50]:
        out[f'RMSE_{lv}%'] = np.sqrt(np.mean(e[nom == lv] ** 2))
    keep = ~st.after_is_formula.astype(bool).to_numpy()
    out['RMSEP_excl_day1_5%'] = np.sqrt(np.mean(e[keep] ** 2))
    neg, pos = p[nom == 0], p[nom == 5]
    out['AUC_0v5'] = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    if ref is not None:
        rng = np.random.default_rng(ra.SEED)
        par = st.parent.unique()
        by = {q: np.where(st.parent == q)[0] for q in par}
        d = []
        for _ in range(2000):
            i = np.concatenate([by[q] for q in rng.choice(par, len(par))])
            d.append(np.sqrt(np.mean((p[i] - y[i]) ** 2)) - np.sqrt(np.mean((ref[i] - y[i]) ** 2)))
        d = np.array(d)
        out.update({'dRMSEP_vs_NIR': out['RMSEP_all110'] - np.sqrt(np.mean((ref - y) ** 2)),
                    'd_low': np.quantile(d, .025), 'd_high': np.quantile(d, .975), 'share_d<0': np.mean(d < 0)})
    out['note'] = note
    return out


def main():
    global nir_pred, col_pred
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    files, S, axis = ra.load_spectra()
    st, X, C = ra.load_states(files, S)
    y = st.conc_recomputed.to_numpy(float)
    groups = st.parent.to_numpy()
    rows, preds = [], {}

    def log(r):
        rows.append(r)
        print(f"{r['config']:<34} RMSEP {r['RMSEP_all110']:.2f}  5% {r['RMSE_5%']:.2f}  "
              f"excl.d1 {r['RMSEP_excl_day1_5%']:.2f}  AUC {r['AUC_0v5']:.3f}  {r['note']}  [{time.time() - t0:.0f}s]",
              flush=True)

    # references
    Nreg = {reg: ra.snv(X[:, region_mask(axis, reg)]) for reg in ['full', 'le9000', 'starch']}
    nir_pred, nir_info = nested(make_design('nir', Nreg['full'], None), [None], KMAX, y, groups)
    preds['NIR_full'] = nir_pred
    log(evaluate('REF NIR SNV full', 'REF', nir_pred, y, st))
    ref = nir_pred
    for reg in ['le9000', 'starch']:
        p, _ = nested(make_design('nir', Nreg[reg], None), [None], KMAX, y, groups)
        preds[f'NIR_{reg}'] = p
        log(evaluate(f'REF NIR SNV {reg}', 'REF', p, y, st, ref))
    F = colour_feats(C, 'Lab')
    col_pred, col_info = nested(make_design('col', None, F), [None], 3, y, groups)
    preds['colour'] = col_pred
    log(evaluate('REF colour Lab', 'REF', col_pred, y, st, ref))

    # low-level fusion: block weight
    for w in WGRID:
        p, info = nested(make_design('ll', Nreg['full'], F), [w], KMAX, y, groups)
        preds[f'LL_Lab_w{w}'] = p
        log(evaluate(f'LL Lab w={w}', 'LL weight', p, y, st, ref,
                     f"LV median {int(np.median([v['k'] for v in info.values()]))}"))
    p, info = nested(make_design('ll', Nreg['full'], F), WGRID, KMAX, y, groups)
    preds['LL_Lab_nestedw'] = p
    ws = pd.Series([v['w'] for v in info.values()]).value_counts().to_dict()
    log(evaluate('LL Lab w nested', 'LL weight', p, y, st, ref, f'w chosen {ws}'))

    # low-level fusion: colour features and NIR region
    for kind in ['LabCHW', 'Labquad']:
        Fk = colour_feats(C, kind)
        for w in [1, 4]:
            p, _ = nested(make_design('ll', Nreg['full'], Fk), [w], KMAX, y, groups)
            preds[f'LL_{kind}_w{w}'] = p
            log(evaluate(f'LL {kind} w={w}', 'LL colour features', p, y, st, ref))
    for reg in ['le9000', 'starch']:
        for w in [1, 4]:
            p, _ = nested(make_design('ll', Nreg[reg], F), [w], KMAX, y, groups)
            preds[f'LL_{reg}_w{w}'] = p
            log(evaluate(f'LL Lab {reg} w={w}', 'LL NIR region', p, y, st, ref))

    # late fusion
    for kind in ['convex', 'ols2', 'ols_lab', 'svr_lab']:
        p, ch = late(kind, nir_info, col_info, F, y, groups)
        preds[f'HL_{kind}'] = p
        note = f'NIR weight median {np.median(ch):.2f}' if kind == 'convex' else ''
        log(evaluate(f'HL {kind}', 'late fusion', p, y, st, ref, note))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'fusion_screen.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(preds).assign(state=st.label, nominal=st.nominal, y=y).to_csv(
        OUT / 'fusion_screen_predictions.csv', index=False, encoding='utf-8-sig')
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
