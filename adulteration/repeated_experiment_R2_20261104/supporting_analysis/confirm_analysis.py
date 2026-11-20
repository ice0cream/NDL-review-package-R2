# -*- coding: utf-8 -*-
"""Repeated adulteration experiment: partition, exclusion and uncertainty analyses for Revision 2.

Builds on redo_analysis.py (same data loading, preprocessing grid, PLS code and seed); nothing in that
script or its outputs is changed. Models: M1 NIR-PLSR, M2 colour PLSR, M3 NIR + colour block-scaled PLSR.
The unit of partitioning is always the parent powder: all five states of a parent (0, 5, 15, 30, 50 %)
stay on the same side of every outer and inner split.

Stages (outputs in ./results_confirm)
  1 lopo           Nested leave-one-parent-out, recomputed and checked against results/state_predictions.csv.
                   split_audit.csv lists every outer and inner split with the number of shared parents (must be 0).
  2 domain         Applicability-domain statistics of M1 for every held-out state, using the preprocessing and
                   LV chosen in that outer fold: Hotelling T2, Q residual, SD of the predictions of the 3 repacked
                   readings. Limits come from the training parents only: 95th / 99th percentile of the same
                   statistics obtained by inner leave-one-parent-out projection (primary), and the classical
                   F-based T2 and Jackson-Mudholkar Q limits of the fold model (for comparison).
                   A state is flagged, never deleted: results are given for all states and for accepted states.
  3 exclusions     Sensitivity to exclusions: accepted states only; calibration refitted without flagged states;
                   refit without parent F1; F2 mixtures excluded; day-1 5 % states (unweighed) excluded.
  4 partitions     10 x parent-grouped 5-fold (same partitions for the three models; identical to those used for M1
                   in redo_analysis.py), leave-one-measurement-batch-out (3 folds) and leave-one-supplier-out (2 folds).
                   Fold-wise errors and the parents of each fold are saved.
  5 parents        Error by parent (leave-one-parent-out).
  6 clipping       Predictions outside 0-100 % and the effect of clipping them.
  7 weighing       Monte-Carlo propagation of the 0.01 g balance resolution into the calculated concentration.
  8 order          Residuals against acquisition order within a batch; starch-reference predictions by session.
  9 detection      0 % vs 5 % and higher: cross-fitted threshold on the regression output, AUC with parent-cluster
                   bootstrap interval, and the PLS-DA results of redo_analysis_extended.py.
 10 bootstrap      Parent bootstrap WITH model redevelopment (B = 500): preprocessing and LV are reselected in every
                   resample and the model is scored on the out-of-bag parents. Copies of a resampled parent share
                   one inner fold. Gives intervals that include model selection and fitting variability.
 11 permutation     Permutation test of the complete M1 procedure (199 permutations): concentrations are shuffled among the
                   five states within each parent, then preprocessing and LV are reselected (inner leave-one-parent-out)
                   and the model is scored on the fixed parent-grouped 5-fold plan of primary_pipeline/inputs/outer_fold_plan_redo.csv.
Run: python confirm_analysis.py [--skip-bootstrap] [--jobs 3]     (about 15 min; stages 10-11 add about 45 min)
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy import stats

import redo_analysis as ra

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_confirm'
SEED = ra.SEED
MODELS = ['M1_NIR', 'M2_colour', 'M3_fusion']
LEVELS = [0, 5, 15, 30, 50]
SUPPLIER_MAP = ra.ROOT / 'R2_revision' / 'R6-3_sample_traceability' / 'mapping_author_statement.csv'
BALANCE_RESOLUTION_G = 0.01
B_REFIT = 500


# ---------------------------------------------------------------- small helpers
def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')
    return df


def cluster_boot(y, preds, parents, B=2000, seed=SEED):
    """Parent-cluster bootstrap of RMSEP, conditional on the saved predictions (no refitting)."""
    rng = np.random.default_rng(seed)
    uniq = pd.unique(parents)          # order of appearance, as in redo_analysis.cluster_boot, so the intervals coincide
    by = {p: np.where(parents == p)[0] for p in uniq}
    rec = []
    for _ in range(B):
        i = np.concatenate([by[p] for p in rng.choice(uniq, len(uniq), replace=True)])
        rec.append({m: rmse(p[i] - y[i]) for m, p in preds.items()})
    return pd.DataFrame(rec)


def metric_row(label, y, p, parents=None, **extra):
    row = {**extra, 'set': label, **ra.metrics(y, p)}
    if parents is not None:
        b = cluster_boot(y, {'m': p}, parents).m
        row.update({'RMSEP_ci_low': b.quantile(.025), 'RMSEP_ci_high': b.quantile(.975)})
    return row


def auc(neg, pos):
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


# ---------------------------------------------------------------- PLS1 with scores, for T2 / Q
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
    return {'xm': xm, 'ym': ym, 'W': W, 'P': P, 'B': W @ np.linalg.solve(P.T @ W, Q),
            'tvar': T.var(0, ddof=1), 'E': E, 'n': len(y), 'k': k}


def project(m, A):
    E = A - m['xm']
    T = np.zeros((len(A), m['k']))
    for a in range(m['k']):
        T[:, a] = E @ m['W'][:, a]
        E = E - np.outer(T[:, a], m['P'][:, a])
    return (T ** 2 / m['tvar']).sum(1), (E ** 2).sum(1)


def predict(m, A):
    return m['ym'] + (A - m['xm']) @ m['B']


def t2_limit_F(n, k, alpha):
    return k * (n ** 2 - 1) / (n * (n - k)) * stats.f.ppf(alpha, k, n - k)


def q_limit_JM(E, alpha):
    """Jackson-Mudholkar (1979) limit; Box approximation when h0 <= 0."""
    s = np.linalg.svd(E - E.mean(0), compute_uv=False)
    lam = s ** 2 / (len(E) - 1)
    lam = lam[lam > 1e-12 * lam.max()]
    t1, t2, t3 = lam.sum(), (lam ** 2).sum(), (lam ** 3).sum()
    h0 = 1 - 2 * t1 * t3 / (3 * t2 ** 2)
    base = stats.norm.ppf(alpha) * np.sqrt(2 * t2 * h0 ** 2) / t1 + 1 + t2 * h0 * (h0 - 1) / t1 ** 2
    if h0 > 0 and np.isfinite(base) and base > 0:
        return float(t1 * base ** (1 / h0))
    return float(t2 / t1 * stats.chi2.ppf(alpha, t1 ** 2 / t2))


# ---------------------------------------------------------------- stage 1: LOPO and split audit
def audit_rows(scheme, repeat, folds, st):
    groups = st.parent.to_numpy()
    rows = []
    for i, test_parents in enumerate(folds, 1):
        te = np.isin(groups, test_parents)
        tr_p, te_p = set(groups[~te]), set(groups[te])
        rows.append({'scheme': scheme, 'repeat': repeat, 'outer_fold': i, 'level': 'outer', 'held_out': ','.join(test_parents),
                     'n_train_states': int((~te).sum()), 'n_test_states': int(te.sum()),
                     'n_train_parents': len(tr_p), 'n_test_parents': len(te_p), 'shared_parents': len(tr_p & te_p)})
        for h in sorted(tr_p, key=lambda p: int(p[1:])):          # inner leave-one-parent-out
            ite = groups[~te] == h
            a, b = set(groups[~te][~ite]), set(groups[~te][ite])
            rows.append({'scheme': scheme, 'repeat': repeat, 'outer_fold': i, 'level': 'inner', 'held_out': h,
                         'n_train_states': int((~ite).sum()), 'n_test_states': int(ite.sum()),
                         'n_train_parents': len(a), 'n_test_parents': len(b),
                         'shared_parents': len(a & b) + len(b & te_p) + len(a & te_p)})
    return rows


def run_scheme(scheme, repeat, folds, X, y, C, st, audit, models=MODELS):
    """Nested validation of each model on one outer partition. Returns predictions and choices."""
    audit.extend(audit_rows(scheme, repeat, folds, st))
    preds, choices = {}, []
    for m in models:
        pr, ch = ra.nested(m, X, y, C, st, folds)
        preds[m] = pr
        choices.append(ch.assign(model=m, scheme=scheme, repeat=repeat))
    return preds, pd.concat(choices)


# ---------------------------------------------------------------- stage 2: applicability domain (M1)
def domain_stats(st, X, y, S, rep_of, choices):
    groups = st.parent.to_numpy()
    ch = choices.set_index('test_parents')
    rows = []
    for g in st.parent.unique():
        prep, k = ch.loc[g, 'prep'], int(ch.loc[g, 'LV'])
        te, tr = np.where(groups == g)[0], np.where(groups != g)[0]

        def fit_eval(fit_idx, ev_idx):
            Ntr, Nev = ra.preprocess(prep, X[fit_idx], X[ev_idx])
            m = pls_fit(Ntr, y[fit_idx], k)
            t2, q = project(m, Nev)
            reps = np.concatenate([rep_of[i] for i in ev_idx])
            pr = predict(m, ra.preprocess(prep, X[fit_idx], S[reps])[1]).reshape(len(ev_idx), 3)
            return m, t2, q, pr.std(1, ddof=1), predict(m, Nev)

        inner = [[], [], []]
        for h in np.unique(groups[tr]):
            ite, itr = tr[groups[tr] == h], tr[groups[tr] != h]
            for store, v in zip(inner, fit_eval(itr, ite)[1:4]):
                store.append(v)
        lim = {name: np.percentile(np.concatenate(v), [95, 99]) for name, v in zip(['T2', 'Q', 'repSD'], inner)}
        m, t2, q, sd, p = fit_eval(tr, te)
        for j, i in enumerate(te):
            rows.append({'state': st.label[i], 'parent': g, 'nominal': st.nominal[i], 'prep': prep, 'LV': k,
                         'pred_check': p[j], 'T2': t2[j], 'Q': q[j], 'repSD': sd[j],
                         'T2_cv95': lim['T2'][0], 'T2_cv99': lim['T2'][1], 'Q_cv95': lim['Q'][0], 'Q_cv99': lim['Q'][1],
                         'repSD_cv95': lim['repSD'][0], 'repSD_cv99': lim['repSD'][1],
                         'T2_F95': t2_limit_F(len(tr), k, .95), 'T2_F99': t2_limit_F(len(tr), k, .99),
                         'Q_JM95': q_limit_JM(m['E'], .95), 'Q_JM99': q_limit_JM(m['E'], .99)})
    d = pd.DataFrame(rows)
    for pc in (95, 99):
        d[f'out_T2_{pc}'] = d.T2 > d[f'T2_cv{pc}']
        d[f'out_Q_{pc}'] = d.Q > d[f'Q_cv{pc}']
        d[f'out_rep_{pc}'] = d.repSD > d[f'repSD_cv{pc}']
        d[f'flag_domain_{pc}'] = d[f'out_T2_{pc}'] | d[f'out_Q_{pc}']
        d[f'flag_any_{pc}'] = d[f'flag_domain_{pc}'] | d[f'out_rep_{pc}']
        d[f'classical_T2_{pc}'] = d.T2 > d[f'T2_F{pc}']
        d[f'classical_Q_{pc}'] = d.Q > d[f'Q_JM{pc}']
    return d


# ---------------------------------------------------------------- stage 7: weighing uncertainty
def weighing_uncertainty(st, draws=5000):
    """Two scenarios. (i) Resolution only: each reading uniform within +-0.005 g. (ii) Upper bound: the whole
    discrepancy between a post-addition weighing and the next pre-addition weighing of the same tube (which also
    contains real handling loss) is treated as weighing error, normal with SD = SD(discrepancy) / sqrt(2) per reading."""
    sh = ra.sheetio.read_sheet()
    tubes, checks = {}, []
    for p, g in sh.groupby('parent', sort=False):
        g = g.sort_values('step')
        tare = ra.sheetio.num(g.iloc[0]['tare'])
        before = np.array([ra.sheetio.num(v) for v in g.before])[1:]
        after = np.array([ra.sheetio.num(v) for v in g.after])
        tubes[p] = (tare, before, after)
        for s in range(1, len(after) - 1):
            checks.append({'label': f'{p}-S{s}', 'after_entered_as_target': bool(g.iloc[s].after_is_formula),
                           'next_weighing_minus_this_g': round(float(before[s] - after[s]), 3)})
    checks = pd.DataFrame(checks)
    sd_disc = float(checks.loc[~checks.after_entered_as_target, 'next_weighing_minus_this_g'].std(ddof=1))

    def mc(noise):
        out = {}
        for p, (tare, before, after) in tubes.items():
            t_s, b_s, a_s = tare + noise(draws), before + noise((draws, len(before))), after + noise((draws, len(after)))
            c = np.zeros(draws)
            for s in range(1, len(after)):
                c = (c / 100 * (b_s[:, s - 1] - t_s) + (a_s[:, s] - b_s[:, s - 1])) / (a_s[:, s] - t_s) * 100
                out[f'{p}-S{s}'] = float(c.std(ddof=1))
        return out

    rng = np.random.default_rng(SEED + 7)
    half = BALANCE_RESOLUTION_G / 2
    u_res = mc(lambda shape: rng.uniform(-half, half, shape))
    u_up = mc(lambda shape: rng.normal(0, sd_disc / np.sqrt(2), shape))
    u = pd.DataFrame({'label': list(u_res), 'u_resolution_pp': list(u_res.values()), 'u_upper_bound_pp': [u_up[k] for k in u_res]})
    u = u.merge(st[['label', 'parent', 'nominal', 'conc_recomputed']], on='label')
    return u, checks, sd_disc


# ---------------------------------------------------------------- stage 10: bootstrap with refitting
def boot_one(b, X, y, C, groups):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        rng = np.random.default_rng([SEED, 10, b])
        uniq = np.unique(groups)
        draw = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == p)[0] for p in draw])
        oob = np.where(~np.isin(groups, draw))[0]
        rec, preds = {'b': b, 'n_unique_train_parents': len(set(draw)), 'n_oob_parents': len(set(groups[oob]))}, {}
        for m in MODELS:
            (prep, k, _), _, _ = ra.inner_select(m, X[idx], y[idx], C[idx], groups[idx])
            p = ra.predict_all_lv(m, prep, k, X[idx], y[idx], X[oob], C[idx], C[oob])[:, k - 1]
            rec.update({f'RMSE_{m}': rmse(p - y[oob]), f'prep_{m}': prep, f'LV_{m}': k})
            preds[m] = p
    return rec, oob, preds


def bootstrap_refit(X, y, C, groups, jobs):
    import os
    from joblib import Parallel, delayed
    os.environ['PYTHONPATH'] = os.pathsep.join([str(HERE), os.environ.get('PYTHONPATH', '')])   # workers import redo_analysis
    res = Parallel(n_jobs=jobs)(delayed(boot_one)(b, X, y, C, groups) for b in range(B_REFIT))
    rec = pd.DataFrame([r[0] for r in res])
    sq = {m: [[] for _ in y] for m in MODELS}
    for _, oob, preds in res:
        for m in MODELS:
            for i, p in zip(oob, preds[m]):
                sq[m][i].append((p - y[i]) ** 2)
    summ = []
    for m in MODELS:
        v = rec[f'RMSE_{m}']
        summ.append({'quantity': f'RMSEP {m}', 'oob_pooled_by_state': float(np.sqrt(np.mean([np.mean(s) for s in sq[m] if s]))),
                     'median': v.median(), 'low_2.5': v.quantile(.025), 'high_97.5': v.quantile(.975)})
    for a, b in [('M3_fusion', 'M1_NIR'), ('M2_colour', 'M1_NIR')]:
        d = rec[f'RMSE_{a}'] - rec[f'RMSE_{b}']
        summ.append({'quantity': f'dRMSEP {a} - {b}', 'median': d.median(), 'low_2.5': d.quantile(.025),
                     'high_97.5': d.quantile(.975), 'share_below_0': float((d < 0).mean())})
    sel = []
    for m in ['M1_NIR', 'M3_fusion']:
        f = rec.groupby([f'prep_{m}']).size().rename('n_resamples').reset_index().rename(columns={f'prep_{m}': 'prep'})
        f['LV_median'] = f.prep.map(rec.groupby(f'prep_{m}')[f'LV_{m}'].median())
        f['LV_min'] = f.prep.map(rec.groupby(f'prep_{m}')[f'LV_{m}'].min())
        f['LV_max'] = f.prep.map(rec.groupby(f'prep_{m}')[f'LV_{m}'].max())
        sel.append(f.assign(model=m))
    return rec, pd.DataFrame(summ), pd.concat(sel)


def perm_one(b, X, y, C, st, folds):
    """b = 0 is the observed arrangement; b >= 1 shuffles concentration within each parent."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        st_p, yp = st.copy(), y.copy()
        if b > 0:
            rng = np.random.default_rng([SEED, 11, b])
            nominal = st.nominal.to_numpy().copy()
            for p in st.parent.unique():
                i = np.where(st.parent.to_numpy() == p)[0]
                j = rng.permutation(i)
                yp[i], nominal[i] = y[j], st.nominal.to_numpy()[j]
            st_p['nominal'] = nominal
        pr, _ = ra.nested('M1_NIR', X, yp, C, st_p, folds)
        m = ra.metrics(yp, pr.pred.to_numpy())
    return {'permutation': b, 'RMSEP': m['RMSEP'], 'R2': m['R2']}


def permutation_test(X, y, C, st, jobs, n_perm=199):
    import os
    from joblib import Parallel, delayed
    os.environ['PYTHONPATH'] = os.pathsep.join([str(HERE), os.environ.get('PYTHONPATH', '')])
    plan = pd.read_csv(HERE.parent / 'primary_pipeline' / 'inputs' / 'outer_fold_plan_redo.csv')
    folds = [list(g.group_id) for _, g in plan.groupby('Outer_fold')]
    res = pd.DataFrame(Parallel(n_jobs=jobs)(delayed(perm_one)(b, X, y, C, st, folds) for b in range(n_perm + 1)))
    obs, null = res.iloc[0], res.iloc[1:]
    summ = {'n_permutations': n_perm, 'observed_RMSEP': float(obs.RMSEP), 'observed_R2': float(obs.R2),
            'null_RMSEP_min': float(null.RMSEP.min()), 'null_RMSEP_median': float(null.RMSEP.median()),
            'null_R2_max': float(null.R2.max()), 'null_R2_median': float(null.R2.median()),
            'p_value': float((1 + (null.RMSEP <= obs.RMSEP).sum()) / (n_perm + 1))}
    return res, summ


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-bootstrap', action='store_true', help='skip the resampling stages 10 and 11')
    ap.add_argument('--jobs', type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    tick = lambda s: print(f'[{time.time() - t0:6.0f}s] {s}', flush=True)

    files, S, axis = ra.load_spectra()
    st, X, C = ra.load_states(files, S)
    assert len(st) == 110 and st.parent.nunique() == 22 and (st.groupby('parent').size() == 5).all()
    y = st.conc_recomputed.to_numpy(float)
    groups = st.parent.to_numpy()
    parents = st.parent.unique()
    nom = st.nominal.to_numpy()
    rep_of = [files.index[files.label == lab].to_numpy() for lab in st.label]
    assert all(len(r) == 3 for r in rep_of)
    audit = []

    # ---- 1 LOPO
    lopo = [[p] for p in parents]
    pr, choices = run_scheme('leave-one-parent-out', 1, lopo, X, y, C, st, audit)
    P = {m: pr[m].pred.to_numpy() for m in MODELS}
    THR = {m: pr[m].threshold.to_numpy() for m in MODELS}
    saved = pd.read_csv(HERE / 'results' / 'state_predictions.csv')
    assert (saved.label == st.label).all()
    max_dev = max(float(np.abs(saved[f'pred_{m}'] - P[m]).max()) for m in MODELS)
    assert max_dev < 1e-6, max_dev
    save(choices, 'lopo_fold_choices.csv')
    tick(f'LOPO done; max deviation from saved predictions {max_dev:.2e}')

    # ---- 2 applicability domain (M1)
    ad = domain_stats(st, X, y, S, rep_of, choices[choices.model == 'M1_NIR'])
    assert np.abs(ad.pred_check - P['M1_NIR']).max() < 1e-6
    for m in MODELS:
        ad[f'pred_{m}'] = P[m]
    ad.insert(3, 'conc', y)
    save(ad, 'domain_flags_per_state.csv')
    flag99, flag95 = ad.flag_any_99.to_numpy(), ad.flag_any_95.to_numpy()
    fl = []
    for rule in ['out_T2', 'out_Q', 'out_rep', 'flag_domain', 'flag_any', 'classical_T2', 'classical_Q']:
        for pc in (95, 99):
            f = ad[f'{rule}_{pc}'].to_numpy()
            fl.append({'rule': rule, 'limit_pct': pc, 'n_flagged': int(f.sum()), 'flagged_states': ', '.join(ad.state[f]),
                       **{f'RMSEP_{m}_accepted': rmse((P[m] - y)[~f]) for m in MODELS},
                       **{f'RMSEP_{m}_flagged': rmse((P[m] - y)[f]) for m in MODELS}})
    save(pd.DataFrame(fl), 'domain_flag_summary.csv')
    tick(f'domain flags: 99 % -> {list(ad.state[flag99])}')

    # ---- 3 exclusion sensitivity
    ex = []
    sets = {'all 110 states (primary)': np.ones(len(y), bool),
            'accepted states, 99 % limits (no refit)': ~flag99,
            'accepted states, 95 % limits (no refit)': ~flag95,
            'excluding F2 mixtures (no refit)': ~((st.parent == 'F2') & (st.step > 0)).to_numpy(),
            'excluding day-1 5 % states, unweighed (no refit)': ~st.after_is_formula.astype(bool).to_numpy()}
    def paired(name, yy, pp, gg):
        b = cluster_boot(yy, pp, gg)
        out = []
        for a_, b_ in [('M3_fusion', 'M1_NIR'), ('M2_colour', 'M1_NIR')]:
            d = b[a_] - b[b_]
            out.append({'set': name, 'comparison': f'{a_} - {b_}', 'n': len(yy), 'difference': rmse(pp[a_] - yy) - rmse(pp[b_] - yy),
                        'ci_low': d.quantile(.025), 'ci_high': d.quantile(.975), 'share_below_0': float((d < 0).mean())})
        return out

    diffs = []
    for name, mask in sets.items():
        for m in MODELS:
            ex.append(metric_row(name, y[mask], P[m][mask], groups[mask], model=m))
        diffs += paired(name, y[mask], {m: P[m][mask] for m in MODELS}, groups[mask])
    level_rows = []
    for name, mask in list(sets.items())[:2]:
        for m in MODELS:
            for lv in LEVELS:
                k = mask & (nom == lv)
                e = (P[m] - y)[k]
                level_rows.append({'set': name, 'model': m, 'nominal_%': lv, 'n': int(k.sum()), 'RMSE': rmse(e),
                                   'bias': float(e.mean()), 'SD': float(e.std(ddof=1))})
    refits = {'calibration and test without flagged states, 99 % (refit)': ~flag99,
              'parent F1 removed (refit)': (st.parent != 'F1').to_numpy()}
    refit_pred = {}
    for name, mask in refits.items():
        sub = st[mask].reset_index(drop=True)
        for m in MODELS:
            p2, _ = ra.nested(m, X[mask], y[mask], C[mask], sub, [[p] for p in sub.parent.unique()])
            refit_pred[(name, m)] = p2.pred.to_numpy()
            ex.append(metric_row(name, y[mask], p2.pred.to_numpy(), groups[mask], model=m))
            if name.startswith('calibration'):
                for lv in LEVELS:
                    k = sub.nominal.to_numpy() == lv
                    e = (p2.pred.to_numpy() - y[mask])[k]
                    level_rows.append({'set': name, 'model': m, 'nominal_%': lv, 'n': int(k.sum()), 'RMSE': rmse(e),
                                       'bias': float(e.mean()), 'SD': float(e.std(ddof=1))})
        diffs += paired(name, y[mask], {m: refit_pred[(name, m)] for m in MODELS}, groups[mask])
    save(pd.DataFrame(ex), 'exclusion_sensitivity.csv')
    save(pd.DataFrame(diffs), 'exclusion_sensitivity_paired_differences.csv')
    save(pd.DataFrame(level_rows), 'per_level_all_vs_accepted.csv')
    tick('exclusion sensitivity done')

    # ---- 4 partitions
    rng = np.random.default_rng(SEED)                       # same generator as redo_analysis.py
    rep_rows, fold_rows = [], []
    for r in range(1, 11):
        perm = rng.permutation(parents)
        folds = [list(perm[i::5]) for i in range(5)]
        pr_r, _ = run_scheme('parent-grouped 5-fold', r, folds, X, y, C, st, audit)
        for m in MODELS:
            p = pr_r[m].pred.to_numpy()
            rep_rows.append({'scheme': 'parent-grouped 5-fold', 'repeat': r, 'model': m, **ra.metrics(y, p),
                             'RMSE_5%': rmse((p - y)[nom == 5]), 'RMSE_mixtures': rmse((p - y)[nom > 0])})
            for f, tp in enumerate(folds, 1):
                k = np.isin(groups, tp)
                fold_rows.append({'repeat': r, 'fold': f, 'parents': ','.join(sorted(tp, key=lambda q: int(q[1:]))),
                                  'model': m, 'n': int(k.sum()), 'RMSE': rmse((p - y)[k]), 'bias': float((p - y)[k].mean())})
        tick(f'grouped 5-fold repeat {r} done')
    rep = pd.DataFrame(rep_rows)
    old = pd.read_csv(HERE / 'results' / 'repeated_grouped5fold_M1.csv')
    dev = float(np.abs(old.RMSEP.to_numpy() - rep[rep.model == 'M1_NIR'].RMSEP.to_numpy()).max())
    assert dev < 1e-6, dev
    day = st.drop_duplicates('parent').set_index('parent').sheet_day
    sup = pd.read_csv(SUPPLIER_MAP).set_index('Modelling_ID').Supplier_TableS1.reindex(parents)
    assert sup.notna().all()
    extra = {'leave-one-measurement-batch-out': [list(day.index[day == d]) for d in sorted(day.unique())],
             'leave-one-supplier-out': [list(sup.index[sup == s]) for s in sorted(sup.unique())]}
    for scheme, folds in extra.items():
        pr_s, _ = run_scheme(scheme, 1, folds, X, y, C, st, audit)
        for m in MODELS:
            p = pr_s[m].pred.to_numpy()
            rep_rows.append({'scheme': scheme, 'repeat': 1, 'model': m, **ra.metrics(y, p),
                             'RMSE_5%': rmse((p - y)[nom == 5]), 'RMSE_mixtures': rmse((p - y)[nom > 0])})
            for f, tp in enumerate(folds, 1):
                k = np.isin(groups, tp)
                fold_rows.append({'repeat': scheme, 'fold': f, 'parents': ','.join(tp), 'model': m, 'n': int(k.sum()),
                                  'RMSE': rmse((p - y)[k]), 'bias': float((p - y)[k].mean())})
        tick(f'{scheme} done')
    for m in MODELS:
        rep_rows.append({'scheme': 'leave-one-parent-out', 'repeat': 1, 'model': m, **ra.metrics(y, P[m]),
                         'RMSE_5%': rmse((P[m] - y)[nom == 5]), 'RMSE_mixtures': rmse((P[m] - y)[nom > 0])})
    rep = save(pd.DataFrame(rep_rows), 'partition_results_by_repeat.csv')
    save(pd.DataFrame(fold_rows), 'partition_fold_errors.csv')
    summ = rep.groupby(['scheme', 'model']).agg(n_partitions=('RMSEP', 'size'), RMSEP_mean=('RMSEP', 'mean'),
                                                RMSEP_sd=('RMSEP', 'std'), RMSEP_min=('RMSEP', 'min'),
                                                RMSEP_max=('RMSEP', 'max'), R2_mean=('R2', 'mean'),
                                                bias_mean=('bias', 'mean'), RMSE_5pct_mean=('RMSE_5%', 'mean')).reset_index()
    save(summ, 'partition_summary.csv')
    g5 = rep[rep.scheme == 'parent-grouped 5-fold'].pivot(index='repeat', columns='model', values='RMSEP')
    paired = pd.DataFrame({'repeat': g5.index, 'fusion_minus_NIR': (g5.M3_fusion - g5.M1_NIR).to_numpy(),
                           'colour_minus_NIR': (g5.M2_colour - g5.M1_NIR).to_numpy()})
    save(paired, 'partition_paired_differences.csv')
    audit_df = save(pd.DataFrame(audit), 'split_audit.csv')
    assert (audit_df.shared_parents == 0).all()
    tick(f'partitions done; {len(audit_df)} splits audited, shared parents in any split: {int(audit_df.shared_parents.sum())}')

    # ---- 5 error by parent
    par = []
    for p in parents:
        k = groups == p
        row = {'parent': p, 'supplier': sup[p], 'batch': st.batch[k].iloc[0]}
        for m in MODELS:
            e = (P[m] - y)[k]
            row.update({f'RMSE_{m}': rmse(e), f'bias_{m}': float(e.mean())})
        row['states_flagged_99'] = ', '.join(ad.state[k & flag99])
        par.append(row)
    save(pd.DataFrame(par).sort_values('RMSE_M1_NIR', ascending=False), 'error_by_parent.csv')

    # ---- 6 clipping
    clip = []
    for m in MODELS:
        p, pc = P[m], np.clip(P[m], 0, 100)
        clip.append({'model': m, 'n_below_0': int((p < 0).sum()), 'n_below_0_authentic': int(((p < 0) & (nom == 0)).sum()),
                     'n_below_0_at_5%': int(((p < 0) & (nom == 5)).sum()), 'n_above_100': int((p > 100).sum()),
                     'min_pred': float(p.min()), 'max_pred': float(p.max()),
                     'RMSEP_raw': rmse(p - y), 'RMSEP_clipped': rmse(pc - y),
                     'RMSE_0%_raw': rmse((p - y)[nom == 0]), 'RMSE_0%_clipped': rmse((pc - y)[nom == 0]),
                     'RMSE_5%_raw': rmse((p - y)[nom == 5]), 'RMSE_5%_clipped': rmse((pc - y)[nom == 5])})
    save(pd.DataFrame(clip), 'clipping_effect.csv')

    # ---- 7 weighing
    wu, wchk, sd_disc = weighing_uncertainty(st)
    save(wu, 'weighing_uncertainty_by_state.csv')
    save(wchk, 'weighing_consistency_checks.csv')
    ws = wu.groupby('nominal').agg(n=('label', 'size'), u_resolution_mean_pp=('u_resolution_pp', 'mean'),
                                   u_resolution_max_pp=('u_resolution_pp', 'max'), u_upper_bound_mean_pp=('u_upper_bound_pp', 'mean'),
                                   u_upper_bound_max_pp=('u_upper_bound_pp', 'max'), actual_mean=('conc_recomputed', 'mean'),
                                   actual_sd=('conc_recomputed', 'std'), actual_min=('conc_recomputed', 'min'),
                                   actual_max=('conc_recomputed', 'max')).reset_index()
    ws['sd_between_step_discrepancy_g'] = sd_disc
    save(ws, 'weighing_uncertainty_summary.csv')

    # ---- 8 order and reference
    o = st[['label', 'parent', 'nominal', 'batch', 'time']].copy()
    o['time'] = pd.to_datetime(o.time)
    o['res_M1'] = P['M1_NIR'] - y
    o['res_level_centred'] = o.res_M1 - o.groupby('nominal').res_M1.transform('mean')
    o['minutes_in_round'] = (o.time - o.groupby(['batch', 'nominal']).time.transform('min')).dt.total_seconds() / 60
    o['run_order_in_batch'] = o.groupby('batch').time.rank()
    save(o, 'order_residuals.csv')
    oc = []
    for name, sub in [('all batches', o)] + list(o.groupby('batch')):
        r1 = stats.spearmanr(sub.minutes_in_round, sub.res_level_centred)
        r2 = stats.spearmanr(sub.run_order_in_batch, sub.res_M1)
        oc.append({'batch': name, 'n': len(sub), 'rho_within_round_time_vs_level_centred_residual': r1.statistic, 'p1': r1.pvalue,
                   'rho_run_order_vs_residual': r2.statistic, 'p2': r2.pvalue})
    save(pd.DataFrame(oc), 'order_correlations.csv')
    rank = o.assign(pos=o.groupby(['batch', 'nominal']).time.rank()).pivot(index='parent', columns='nominal', values='pos')
    same = [{'batch': b, 'level_pair': f'{a_} vs {b_}', 'spearman_of_parent_order': stats.spearmanr(r[a_], r[b_]).statistic}
            for b, r in rank.groupby(o.drop_duplicates('parent').set_index('parent').batch) for a_, b_ in zip(LEVELS[:-1], LEVELS[1:])]
    save(pd.DataFrame(same), 'order_parent_sequence_between_rounds.csv')
    (prep, k, _), _, _ = ra.inner_select('M1_NIR', X, y, C, groups)
    ms = files[files.label == 'MSDF'].sort_values('time')
    Ptr, Pms = ra.preprocess(prep, X, S[ms.index])
    Bk, b0 = ra.pls1(Ptr.copy(), y, k)[k - 1]
    ref = pd.DataFrame({'file': ms.file.to_numpy(), 'time': ms.time.to_numpy(), 'pred_%': Pms @ Bk + b0,
                        'session': ms.time.diff().dt.total_seconds().div(60).gt(10).cumsum().to_numpy() + 1})
    save(ref, 'starch_reference_predictions.csv')
    rs = ref.groupby('session').agg(start=('time', 'min'), n=('pred_%', 'size'), mean_pred=('pred_%', 'mean'),
                                    sd_pred=('pred_%', 'std')).reset_index()
    rs['shift_vs_first_pp'] = rs.mean_pred - rs.mean_pred.iloc[0]
    save(rs, 'starch_reference_by_session.csv')
    bt = o.groupby('batch').agg(first=('time', 'min'), last=('time', 'max'), n_states=('label', 'size'),
                                mean_residual=('res_M1', 'mean')).reset_index()
    save(bt, 'batch_times.csv')

    # ---- 9 detection
    det = []
    for m in MODELS:
        called = P[m] > THR[m]
        for lv in LEVELS:
            k = nom == lv
            lo, hi = ra.wilson(int(called[k].sum()), int(k.sum()))
            det.append({'method': 'regression + cross-fitted Youden threshold', 'model': m, 'nominal_%': lv, 'n': int(k.sum()),
                        'called_adulterated': int(called[k].sum()), 'rate': float(called[k].mean()), 'wilson_low': lo, 'wilson_high': hi})
    ext = HERE / 'results_extended' / 'plsda_binary_by_level.csv'
    if ext.exists():
        for _, r in pd.read_csv(ext).iterrows():
            n_call = int(r['n'] - r['correct']) if r['nominal_%'] == 0 else int(r['correct'])
            lo, hi = ra.wilson(n_call, int(r['n']))
            det.append({'method': 'PLS-DA (authentic vs adulterated)', 'model': r['model'], 'nominal_%': int(r['nominal_%']),
                        'n': int(r['n']), 'called_adulterated': n_call, 'rate': n_call / r['n'], 'wilson_low': lo, 'wilson_high': hi})
    save(pd.DataFrame(det), 'detection_by_level.csv')
    au = []
    rng_a = np.random.default_rng(SEED + 9)
    for m in MODELS:
        row = {'model': m, 'threshold_min': float(THR[m].min()), 'threshold_median': float(np.median(THR[m])),
               'threshold_max': float(THR[m].max())}
        for lv in (5, 15):
            est = auc(P[m][nom == 0], P[m][nom == lv])
            bs = []
            for _ in range(2000):
                d = rng_a.choice(parents, len(parents), replace=True)
                i = np.concatenate([np.where(groups == p)[0] for p in d])
                bs.append(auc(P[m][i][nom[i] == 0], P[m][i][nom[i] == lv]))
            row.update({f'AUC_0_vs_{lv}': est, f'AUC_0_vs_{lv}_low': float(np.quantile(bs, .025)),
                        f'AUC_0_vs_{lv}_high': float(np.quantile(bs, .975))})
        au.append(row)
    save(pd.DataFrame(au), 'detection_auc_thresholds.csv')
    tick('diagnostics done')

    # ---- 10 bootstrap with refitting
    boot_summary = None
    if not args.skip_bootstrap:
        rec, boot_summary, sel = bootstrap_refit(X, y, C, groups, args.jobs)
        save(rec, 'bootstrap_refit_resamples.csv')
        save(boot_summary, 'bootstrap_refit_summary.csv')
        save(sel, 'bootstrap_refit_selection_frequency.csv')
        tick('bootstrap with refitting done')
        perm, perm_summary = permutation_test(X, y, C, st, args.jobs)
        save(perm, 'permutation_full_procedure.csv')
        (OUT / 'permutation_full_procedure_summary.json').write_text(json.dumps(perm_summary, indent=2), encoding='utf-8')
        tick(f"permutation test done: p = {perm_summary['p_value']:.3f}")

    data_hash = hashlib.sha256()
    for f in [ra.SPECTRA_CSV]:
        data_hash.update(f.name.encode('utf-8'))
        data_hash.update(hashlib.sha256(f.read_bytes()).digest())
    manifest = {'script': Path(__file__).name, 'python': platform.python_version(), 'platform': platform.platform(),
                'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
                'seed': SEED, 'B_conditional': 2000, 'B_refit': None if args.skip_bootstrap else B_REFIT,
                'n_permutations': None if args.skip_bootstrap else 199,
                'n_states': len(st), 'n_parents': len(parents), 'n_spectra_files': len(files),
                'inputs': {'cleaned_spectra_folder_sha256': data_hash.hexdigest(), 'record_sheet_sha256': sha256_file(ra.SHEET),
                           'supplier_map_sha256': sha256_file(SUPPLIER_MAP)},
                'code_sha256': {f: sha256_file(HERE / f) for f in ['redo_analysis.py', 'confirm_analysis.py']},
                'checks': {'lopo_max_abs_deviation_from_saved_predictions': max_dev,
                           'grouped5fold_M1_max_abs_deviation_from_saved': dev,
                           'splits_audited': int(len(audit_df)), 'splits_with_shared_parents': int((audit_df.shared_parents > 0).sum())},
                'flagged_states_99': list(ad.state[flag99]), 'flagged_states_95': list(ad.state[flag95]),
                'runtime_s': round(time.time() - t0)}
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    tick('all done')


if __name__ == '__main__':
    sys.exit(main())
