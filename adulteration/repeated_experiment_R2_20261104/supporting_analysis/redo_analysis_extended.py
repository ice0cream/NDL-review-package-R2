# -*- coding: utf-8 -*-
"""Extended analyses of the repeated experiment (PLS family, nested leave-one-parent-out).

Reuses data loading, preprocessing and model designs from redo_analysis.py.
  1. Regression with and without the authentic states: positive mixtures only (88) vs all 110.
  2. Grouped vs ungrouped validation (NIR-PLSR): 10 x random state-level 5-fold CV (parent ignored,
     inner selection also state-level) against 10 x grouped 5-fold and leave-one-parent-out.
     The gap shows how much a split that lets one parent's states fall on both sides inflates accuracy.
  3. PLS-DA classification (SIMPLS PLS2 on class dummies, argmax), NIR / colour / fusion:
     binary Authentic vs Adulterated (all 110) and the manuscript's four classes
     Authentic (0) / Low (<=20) / Medium (<=40) / High. Preprocessing and LV chosen by inner
     leave-one-parent-out macro-F1 (ties to fewer LV).
Outputs in ./results_extended. Run: python redo_analysis_extended.py
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

import redo_analysis as ra

OUT = Path(__file__).resolve().parent / 'results_extended'
SEED = 2026
MODELS = ['M1_NIR', 'M2_colour', 'M3_fusion']


# ---------------------------------------------------------------- SIMPLS for multi-column Y
def simpls(X, Y, K):
    xm, ym = X.mean(0), Y.mean(0)
    X0, Y0 = X - xm, Y - ym
    K = min(K, len(X) - 2)
    S = X0.T @ Y0
    R, Q, V = [], [], []
    for _ in range(K):
        u, s, vt = np.linalg.svd(S, full_matrices=False)
        r = u[:, 0] * s[0]
        t = X0 @ r
        nt = np.linalg.norm(t)
        t, r = t / nt, r / nt
        p, q = X0.T @ t, Y0.T @ t
        v = p.copy()
        if V:
            Vm = np.array(V).T
            v -= Vm @ (Vm.T @ p)
        v /= np.linalg.norm(v)
        S = S - np.outer(v, v @ S)
        R.append(r), Q.append(q), V.append(v)
    R, Q = np.array(R).T, np.array(Q).T
    return [(R[:, :k] @ Q[:, :k].T, ym - xm @ (R[:, :k] @ Q[:, :k].T)) for k in range(1, K + 1)]


def da_scores_all_lv(model, prep, K, Xtr, Ytr, Xte, Ctr, Cte):
    A, B = ra.design(model, prep, Xtr, Xte, Ctr, Cte)
    return [B @ b + b0 for b, b0 in simpls(A, Ytr, K)]


def macro_f1(y, p, classes):
    f = []
    for c in classes:
        tp = np.sum((p == c) & (y == c))
        fp, fn = np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def class_labels(conc, task):
    if task == 'binary':
        return np.where(conc == 0, 'Authentic', 'Adulterated'), np.array(['Authentic', 'Adulterated'])
    lab = np.select([conc == 0, conc <= 20, conc <= 40], ['Authentic', 'Low', 'Medium'], 'High')
    return lab, np.array(['Authentic', 'Low', 'Medium', 'High'])


def nested_da(model, X, C, labels, classes, groups):
    Y = (labels[:, None] == classes[None, :]).astype(float)
    pred = np.empty(len(labels), dtype=object)
    choice = []
    for g in np.unique(groups):
        te = groups == g
        tr_groups = groups[~te]
        best = None
        for prep, K in ra.grid(model):
            oof = [np.empty(len(tr_groups), dtype=object) for _ in range(K)]
            for h in np.unique(tr_groups):
                ite = tr_groups == h
                sc = da_scores_all_lv(model, prep, K, X[~te][~ite], Y[~te][~ite], X[~te][ite], C[~te][~ite], C[~te][ite])
                for k, s in enumerate(sc):
                    oof[k][ite] = classes[s.argmax(1)]
            for k in range(len(sc)):
                f = macro_f1(labels[~te], oof[k], classes)
                if best is None or f > best[2] + 1e-12:
                    best = (prep, k + 1, f)
        prep, k, f = best
        sc = da_scores_all_lv(model, prep, k, X[~te], Y[~te], X[te], C[~te], C[te])[k - 1]
        pred[te] = classes[sc.argmax(1)]
        choice.append({'test_parent': g, 'prep': prep, 'LV': k, 'inner_macroF1': f})
    return pred, pd.DataFrame(choice)


def class_metrics(y, p, classes):
    rec = {c: float(np.mean(p[y == c] == c)) for c in classes}
    return {'n': len(y), 'accuracy': float(np.mean(p == y)), 'balanced_accuracy': float(np.mean(list(rec.values()))),
            'macro_F1': macro_f1(y, p, classes), **{f'recall_{c}': v for c, v in rec.items()}}


# ---------------------------------------------------------------- ungrouped (row-level) nested CV
def nested_rowlevel(model, X, y, C, outer_ids, inner_seed):
    pred = np.full(len(y), np.nan)
    for f in np.unique(outer_ids):
        te = outer_ids == f
        rng = np.random.default_rng(inner_seed + int(f))
        inner = rng.permutation(np.arange((~te).sum()) % 5)
        (prep, k, _), _, _ = ra.inner_select(model, X[~te], y[~te], C[~te], inner)
        pred[te] = ra.predict_all_lv(model, prep, k, X[~te], y[~te], X[te], C[~te], C[te])[:, k - 1]
    return pred


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    files, S, axis = ra.load_spectra()
    st, X, C = ra.load_states(files, S)
    y = st.conc_recomputed.to_numpy(float)
    groups = st.parent.to_numpy()
    lopo = [[p] for p in st.parent.unique()]

    # 1. with vs without authentic states
    rows, per_level = [], []
    pos = (st.nominal > 0).to_numpy()
    st_pos = st[pos].reset_index(drop=True)
    for m in MODELS:
        pr, ch = ra.nested(m, X[pos], y[pos], C[pos], st_pos, lopo)
        rows.append({'model': m, 'set': 'positive mixtures only (88)', **ra.metrics(y[pos], pr.pred.to_numpy())})
        for lv, g in st_pos.assign(p=pr.pred.to_numpy()).groupby('nominal'):
            e = g.p - g.conc_recomputed
            per_level.append({'model': m, 'set': 'positive only', 'nominal_%': lv,
                              'RMSE': float(np.sqrt((e ** 2).mean())), 'bias': float(e.mean())})
        print(m, 'positive-only done', round(time.time() - t0), flush=True)
    prev = pd.read_csv(Path(__file__).resolve().parent / 'results' / 'state_predictions.csv')
    for m in MODELS:
        p = prev[f'pred_{m}'].to_numpy()
        rows.append({'model': m, 'set': 'all 110 incl. authentic', **ra.metrics(y, p)})
        rows.append({'model': m, 'set': 'all-110 model, scored on mixtures only', **ra.metrics(y[pos], p[pos])})
    pd.DataFrame(rows).to_csv(OUT / 'regression_with_without_authentic.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(per_level).to_csv(OUT / 'per_level_positive_only.csv', index=False, encoding='utf-8-sig')

    # 2. grouped vs ungrouped (NIR-PLSR, all 110)
    gv = []
    rng = np.random.default_rng(SEED)
    for r in range(10):
        ids = rng.permutation(np.arange(len(y)) % 5)
        p = nested_rowlevel('M1_NIR', X, y, C, ids, SEED + 100 * r)
        gv.append({'scheme': 'ungrouped state-level 5-fold', 'repeat': r + 1, **ra.metrics(y, p)})
    rep = pd.read_csv(Path(__file__).resolve().parent / 'results' / 'repeated_grouped5fold_M1.csv')
    for _, r in rep.iterrows():
        gv.append({'scheme': 'parent-grouped 5-fold', **r.to_dict()})
    gv.append({'scheme': 'leave-one-parent-out', 'repeat': 1, **ra.metrics(y, prev.pred_M1_NIR.to_numpy())})
    gv = pd.DataFrame(gv)
    gv.to_csv(OUT / 'grouped_vs_ungrouped_M1.csv', index=False, encoding='utf-8-sig')
    print('grouped vs ungrouped done', round(time.time() - t0), flush=True)

    # 3. PLS-DA
    cls_rows, conf = [], []
    for task in ['binary', 'four_class']:
        labels, classes = class_labels(st.nominal.to_numpy(), task)
        for m in MODELS:
            pred, ch = nested_da(m, X, C, labels, classes, groups)
            ch.to_csv(OUT / f'plsda_choices_{task}_{m}.csv', index=False)
            cls_rows.append({'task': task, 'model': m, **class_metrics(labels, pred, classes)})
            cm = pd.crosstab(pd.Categorical(labels, classes), pd.Categorical(pred, classes),
                             rownames=['true'], colnames=['pred'], dropna=False)
            cm.insert(0, 'model', m)
            cm.insert(0, 'task', task)
            conf.append(cm.reset_index())
            st[f'da_{task}_{m}'] = pred
            print(task, m, 'done', round(time.time() - t0), flush=True)
    pd.DataFrame(cls_rows).to_csv(OUT / 'plsda_summary.csv', index=False, encoding='utf-8-sig')
    pd.concat(conf).to_csv(OUT / 'plsda_confusion.csv', index=False, encoding='utf-8-sig')
    # binary errors by nominal level
    lev = []
    for m in MODELS:
        for lv, g in st.groupby('nominal'):
            truth = 'Authentic' if lv == 0 else 'Adulterated'
            lev.append({'model': m, 'nominal_%': lv, 'n': len(g),
                        'correct': int((g[f'da_binary_{m}'] == truth).sum())})
    pd.DataFrame(lev).to_csv(OUT / 'plsda_binary_by_level.csv', index=False, encoding='utf-8-sig')
    st.to_csv(OUT / 'state_predictions_extended.csv', index=False, encoding='utf-8-sig')
    (OUT / 'run_manifest.json').write_text(json.dumps({'seed': SEED, 'runtime_s': round(time.time() - t0)}, indent=2))
    print('all done', round(time.time() - t0))


if __name__ == '__main__':
    main()
