# -*- coding: utf-8 -*-
"""Re-run cassava-starch adulteration experiment: reproducible analysis (Revision 2).

Inputs (read-only)
  repeated_experiment_cleaned_spectra/                 cleaned OPUS files (see data_checks/correction_log.csv)
  repeated_experiment_record_sheet.csv      weighings and CIELAB readings
Design: 22 FTR parents x {0, 5, 15, 30, 50 % w/w nominal} = 110 states, 3 repacked readings each.

Pre-specified analysis
  y         actual starch % from the weighings (sequential-addition formula), not the nominal level
  X (NIR)   mean of the 3 replicate absorbance spectra per state, full range 11545-3954 cm-1
  colour    median of the 3 replicate L*, a*, b* readings per state (robust to single entry errors)
  models    M1 NIR-PLSR (primary); M2 colour-only PLSR (autoscaled, 1-3 LV);
            M3 NIR+colour PLSR, block-scaled feature fusion (each block autoscaled, weighted 1/sqrt(p))
  prep      none, SNV, MSC, SG1, SG2, SNV+SG1, SNV+SG2 (Savitzky-Golay window 15, order 2); 1-15 LV
  validation  outer leave-one-parent-out (22 folds); inside each training set, preprocessing and LV
            are chosen by inner leave-one-parent-out RMSECV (minimum). Nothing from the test parent
            is used for selection, scaling or the detection threshold.
  detection  0 % vs 5 %: threshold on the predicted %, chosen per outer fold by Youden's J on the inner
            out-of-fold predictions of the training parents, then applied to the test parent.
  uncertainty  parent-cluster bootstrap (B = 2000, seed 2026) for RMSEP, R2 and paired model differences;
            Wilson 95 % intervals for detection proportions.
  sensitivity  (a) F2 mixtures excluded (weighing after its 5 % addition inconsistent with next weighing);
            (b) the 8 day-1 5 % states excluded (post-addition mass entered as target, not weighed);
            (c) 10 x repeated grouped 5-fold outer CV instead of leave-one-parent-out.
  diagnostics  replicate-level predictions (repacking repeatability), residuals by measurement batch,
            starch-reference drift expressed in predicted-% units by the final model.
Run: python redo_analysis.py   (about 1-3 min; outputs in ./results)
"""
from pathlib import Path
import json
import platform
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'repeated_experiment_cleaned_spectra'   # cleaned instrument files (not in the reviewer package)
SPECTRA_CSV = HERE.parent / 'spectra_dataset' / 'NIR_spectra_by_reading.csv'   # the same spectra, one row per reading
SHEET = ROOT / 'repeated_experiment_record_sheet.csv'
OUT = HERE / 'results'
sys.path.insert(0, str(ROOT / 'adulteration' / 'NIR' / 'R1_cassava_starch_spectra'))
sys.path.insert(0, str(HERE.parent / 'data_checks'))
from opus_mean_spectra_cassava_FINAL_v2 import read_opus_spectrum  # noqa: E402
import check_redo_data as sheetio  # noqa: E402

SEED = 2026
MAX_LV = 15
PREPS = ['none', 'SNV', 'MSC', 'SG1', 'SG2', 'SNV+SG1', 'SNV+SG2']
NAME = re.compile(r'^F1_(?P<label>F\d+-S\d|MSDF)-(?P<rep>\d)_(?P<date>\d{8})_(?P<time>\d{6})\.001$')


# ---------------------------------------------------------------- data
def load_spectra():
    """Readings of the cleaned file set, from the spectral dataset (values identical to the instrument files)."""
    d = pd.read_csv(SPECTRA_CSV, float_precision='round_trip')
    wn = [c for c in d.columns if c.startswith('wn_')]
    files = d[['file', 'label', 'rep']].copy()
    files['rep'] = files['rep'].astype(int)
    files['time'] = pd.to_datetime(d['time'], format='%Y-%m-%d %H:%M:%S')
    return files, d[wn].to_numpy(float), np.array([float(c[3:]) for c in wn])


def load_states(files, S):
    sh = sheetio.read_sheet()
    _, sh = sheetio.colour_checks(sh)
    w = sheetio.weighing_checks(sh)
    st = sh[['sheet_day', 'parent', 'step', 'nominal', 'after_is_formula', 'L_med', 'a_med', 'b_med']].merge(
        w[['parent', 'step', 'conc_recomputed']], on=['parent', 'step'])
    st['label'] = st.parent + '-S' + st.step.astype(int).astype(str)
    smp = files[files.label != 'MSDF']
    means = {lab: S[g.index].mean(axis=0) for lab, g in smp.groupby('label')}
    times = smp.groupby('label').time.mean()
    st = st[st.label.isin(means)].copy()
    st['time'] = st.label.map(times)
    st['batch'] = st.sheet_day.map({1: 'B1 (6 Nov day)', 2: 'B2 (6 Nov evening)', 3: 'B3 (7 Nov)'})
    st = st.sort_values(['parent', 'step'], key=lambda s: s.map(
        lambda v: int(v[1:]) if isinstance(v, str) and v[0] == 'F' else v)).reset_index(drop=True)
    X = np.vstack([means[lab] for lab in st.label])
    C = st[['L_med', 'a_med', 'b_med']].to_numpy(float)
    return st, X, C


# ---------------------------------------------------------------- preprocessing
def snv(A):
    return (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)


def sg(A, d):
    return savgol_filter(A, 15, 2, deriv=d, axis=1)


def msc_fit(A):
    return A.mean(0)


def msc_apply(A, ref):
    out = np.empty_like(A)
    for i, a in enumerate(A):
        b, c = np.polyfit(ref, a, 1)
        out[i] = (a - c) / b
    return out


def preprocess(prep, Atr, Ate):
    """Row-wise transforms need no fitting; MSC uses the training mean as reference."""
    if prep == 'MSC':
        ref = msc_fit(Atr)
        return msc_apply(Atr, ref), msc_apply(Ate, ref)
    f = {'none': lambda A: A, 'SNV': snv, 'SG1': lambda A: sg(A, 1), 'SG2': lambda A: sg(A, 2),
         'SNV+SG1': lambda A: sg(snv(A), 1), 'SNV+SG2': lambda A: sg(snv(A), 2)}[prep]
    return f(Atr), f(Ate)


# ---------------------------------------------------------------- PLS1 (NIPALS), coefficients for 1..K LV
def pls1(X, y, K):
    xm, ym = X.mean(0), y.mean()
    E, f = X - xm, y - ym
    K = min(K, len(y) - 2)
    W, P, Q = [], [], []
    for _ in range(K):
        w = E.T @ f
        w /= np.linalg.norm(w)
        t = E @ w
        tt = t @ t
        p, q = E.T @ t / tt, f @ t / tt
        E -= np.outer(t, p)
        f -= q * t
        W.append(w), P.append(p), Q.append(q)
    W, P, Q = np.array(W).T, np.array(P).T, np.array(Q)
    coefs = []
    for k in range(1, K + 1):
        B = W[:, :k] @ np.linalg.solve(P[:, :k].T @ W[:, :k], Q[:k])
        coefs.append((B, ym - xm @ B))
    return coefs


def autoscale(Atr, Ate):
    m, s = Atr.mean(0), Atr.std(0, ddof=1)
    s = np.where(s < 1e-12, 1, s)
    return (Atr - m) / s, (Ate - m) / s


def design(model, prep, Xtr, Xte, Ctr, Cte):
    """Return training / test matrices for one model after training-only fitting."""
    if model == 'M2_colour':
        return autoscale(Ctr, Cte)
    Ntr, Nte = preprocess(prep, Xtr, Xte)
    if model == 'M1_NIR':
        return Ntr, Nte
    a, b = autoscale(Ntr, Nte)
    c, d = autoscale(Ctr, Cte)
    wn, wc = 1 / np.sqrt(a.shape[1]), 1 / np.sqrt(c.shape[1])
    return np.hstack([a * wn, c * wc]), np.hstack([b * wn, d * wc])


def grid(model):
    if model == 'M2_colour':
        return [('none', 3)]
    return [(p, MAX_LV) for p in PREPS]


def predict_all_lv(model, prep, K, Xtr, ytr, Xte, Ctr, Cte):
    A, B = design(model, prep, Xtr, Xte, Ctr, Cte)
    return np.column_stack([B @ b + b0 for b, b0 in pls1(A, ytr, K)])


# ---------------------------------------------------------------- nested validation
def inner_select(model, X, y, C, groups):
    """Leave-one-parent-out inside the training set; returns choice and its OOF predictions."""
    best, scores = None, []
    oof_store = {}
    for prep, K in grid(model):
        oof = np.full((len(y), K), np.nan)
        for g in np.unique(groups):
            te = groups == g
            pr = predict_all_lv(model, prep, K, X[~te], y[~te], X[te], C[~te], C[te])
            oof[te, :pr.shape[1]] = pr
        rmse = np.sqrt(np.nanmean((oof - y[:, None]) ** 2, axis=0))
        for k, r in enumerate(rmse, 1):
            scores.append((prep, k, r))
            if np.isfinite(r) and (best is None or r < best[2]):
                best = (prep, k, r)
        oof_store[prep] = oof
    return best, oof_store[best[0]][:, best[1] - 1], scores


def youden_threshold(pred, y_nominal):
    neg, pos = pred[y_nominal == 0], pred[y_nominal == 5]
    cand = np.unique(np.concatenate([neg, pos]))
    mids = (cand[:-1] + cand[1:]) / 2
    j = np.array([(pos > t).mean() + (neg <= t).mean() for t in mids])
    best = mids[j == j.max()]
    return float(np.median(best))


def nested(model, X, y, C, st, folds):
    rows, choices = [], []
    groups = st.parent.to_numpy()
    for i, test_parents in enumerate(folds, 1):
        te = st.parent.isin(test_parents).to_numpy()
        (prep, k, rcv), oof, _ = inner_select(model, X[~te], y[~te], C[~te], groups[~te])
        thr = youden_threshold(oof, st.nominal.to_numpy()[~te])
        pred = predict_all_lv(model, prep, k, X[~te], y[~te], X[te], C[~te], C[te])[:, k - 1]
        choices.append({'fold': i, 'test_parents': ','.join(test_parents), 'prep': prep, 'LV': k,
                        'inner_RMSECV': rcv, 'threshold': thr})
        for idx, p in zip(np.where(te)[0], pred):
            rows.append({'idx': idx, 'fold': i, 'pred': p, 'threshold': thr, 'prep': prep, 'LV': k})
    out = pd.DataFrame(rows).set_index('idx').sort_index()
    return out, pd.DataFrame(choices)


def metrics(y, p):
    e = p - y
    sd = y.std(ddof=1)
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return {'n': len(y), 'RMSEP': rmse, 'R2': 1 - float(np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
            'bias': float(e.mean()), 'SEP': float(e.std(ddof=1)), 'MAE': float(np.abs(e).mean()), 'RPD': float(sd / rmse)}


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(0.975)
    c = (k + z * z / 2) / (n + z * z)
    h = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return c - h, c + h


def cluster_boot(st, preds, B=2000):
    rng = np.random.default_rng(SEED)
    parents = st.parent.unique()
    y = st.conc_recomputed.to_numpy()
    idx_by = {p: np.where(st.parent == p)[0] for p in parents}
    rec = []
    for _ in range(B):
        idx = np.concatenate([idx_by[p] for p in rng.choice(parents, len(parents), replace=True)])
        r = {m: float(np.sqrt(np.mean((p[idx] - y[idx]) ** 2))) for m, p in preds.items()}
        r.update({f'R2_{m}': 1 - float(np.sum((p[idx] - y[idx]) ** 2) / np.sum((y[idx] - y[idx].mean()) ** 2))
                  for m, p in preds.items()})
        rec.append(r)
    return pd.DataFrame(rec)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    files, S, axis = load_spectra()
    st, X, C = load_states(files, S)
    assert len(st) == 110 and st.parent.nunique() == 22
    y = st.conc_recomputed.to_numpy(float)
    st.drop(columns=[]).to_csv(OUT / 'states_used.csv', index=False, encoding='utf-8-sig')

    lopo = [[p] for p in st.parent.unique()]
    models = ['M1_NIR', 'M2_colour', 'M3_fusion']
    preds, summary, per_level, choices_all = {}, [], [], []
    for m in models:
        pr, ch = nested(m, X, y, C, st, lopo)
        preds[m] = pr.pred.to_numpy()
        ch.insert(0, 'model', m)
        choices_all.append(ch)
        st[f'pred_{m}'] = preds[m]
        st[f'thr_{m}'] = pr.threshold.to_numpy()
        summary.append({'model': m, 'scheme': 'LOPO', 'set': 'all 110', **metrics(y, preds[m])})
        for lv, g in st.groupby('nominal'):
            e = g[f'pred_{m}'] - g.conc_recomputed
            per_level.append({'model': m, 'nominal_%': lv, 'n': len(g), 'RMSE': float(np.sqrt((e ** 2).mean())),
                              'bias': float(e.mean()), 'SD': float(e.std(ddof=1))})
        print(f'{m} done {time.time() - t0:.0f}s', flush=True)
    pd.concat(choices_all).to_csv(OUT / 'fold_choices.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(per_level).to_csv(OUT / 'per_level_errors.csv', index=False, encoding='utf-8-sig')

    # detection 0 % vs 5 %
    det = []
    for m in models:
        flag = st[f'pred_{m}'] > st[f'thr_{m}']
        for lv, g in st.groupby('nominal'):
            k, n = int(flag[g.index].sum()), len(g)
            lo, hi = wilson(k, n)
            det.append({'model': m, 'nominal_%': lv, 'called_adulterated': k, 'n': n, 'rate': k / n,
                        'wilson_low': lo, 'wilson_high': hi})
    pd.DataFrame(det).to_csv(OUT / 'detection_by_level.csv', index=False, encoding='utf-8-sig')

    # bootstrap and paired differences
    bt = cluster_boot(st, preds)
    boot_rows = []
    for m in models:
        boot_rows.append({'quantity': f'RMSEP {m}', 'estimate': metrics(y, preds[m])['RMSEP'],
                          'low': bt[m].quantile(.025), 'high': bt[m].quantile(.975)})
        boot_rows.append({'quantity': f'R2 {m}', 'estimate': metrics(y, preds[m])['R2'],
                          'low': bt[f'R2_{m}'].quantile(.025), 'high': bt[f'R2_{m}'].quantile(.975)})
    for a, b in [('M3_fusion', 'M1_NIR'), ('M2_colour', 'M1_NIR')]:
        d = bt[a] - bt[b]
        boot_rows.append({'quantity': f'dRMSEP {a} - {b}', 'estimate': metrics(y, preds[a])['RMSEP'] - metrics(y, preds[b])['RMSEP'],
                          'low': d.quantile(.025), 'high': d.quantile(.975), 'share_below_0': float((d < 0).mean())})
    pd.DataFrame(boot_rows).to_csv(OUT / 'bootstrap_intervals.csv', index=False, encoding='utf-8-sig')

    # sensitivity (a), (b): same out-of-fold predictions, restricted sets; (c) repeated grouped 5-fold for M1
    sets = {'excl. F2 mixtures': ~((st.parent == 'F2') & (st.step > 0)),
            'excl. day-1 5% states (unweighed)': ~(st.after_is_formula.astype(bool))}
    for name, mask in sets.items():
        for m in models:
            summary.append({'model': m, 'scheme': 'LOPO', 'set': name, **metrics(y[mask], preds[m][mask])})
    rng = np.random.default_rng(SEED)
    rep_rows = []
    parents = st.parent.unique()
    for r in range(10):
        perm = rng.permutation(parents)
        folds = [list(perm[i::5]) for i in range(5)]
        pr, _ = nested('M1_NIR', X, y, C, st, folds)
        rep_rows.append({'repeat': r + 1, **metrics(y, pr.pred.to_numpy())})
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(OUT / 'repeated_grouped5fold_M1.csv', index=False, encoding='utf-8-sig')
    summary.append({'model': 'M1_NIR', 'scheme': 'grouped 5-fold x10 (mean)', 'set': 'all 110',
                    **rep.drop(columns='repeat').mean().to_dict()})
    pd.DataFrame(summary).to_csv(OUT / 'summary_metrics.csv', index=False, encoding='utf-8-sig')
    print(f'sensitivity done {time.time() - t0:.0f}s', flush=True)

    # final M1 model (selection by LOPO on all 22 parents) for diagnostics
    (prep, k, _), _, scores = inner_select('M1_NIR', X, y, C, st.parent.to_numpy())
    pd.DataFrame(scores, columns=['prep', 'LV', 'RMSECV']).to_csv(OUT / 'final_model_cv_grid.csv', index=False)
    Ptr, _ = preprocess(prep, X, X)
    Bk, b0 = pls1(Ptr, y, k)[k - 1]

    # replicate-level predictions from each outer-fold model (repacking repeatability)
    smp = files[files.label != 'MSDF']
    rep_pred = []
    m1 = pd.concat(choices_all).query("model == 'M1_NIR'").set_index('test_parents')
    for p in parents:
        te = (st.parent == p).to_numpy()
        prep_f, k_f = m1.loc[p, 'prep'], int(m1.loc[p, 'LV'])
        g = smp[smp.label.str.startswith(p + '-')]
        pr = predict_all_lv('M1_NIR', prep_f, k_f, X[~te], y[~te], S[g.index], C[~te], C[:len(g)])[:, k_f - 1]
        rep_pred.append(g.assign(pred=pr)[['file', 'label', 'rep', 'pred']])
    rep_pred = pd.concat(rep_pred)
    rep_pred.to_csv(OUT / 'replicate_predictions_M1.csv', index=False, encoding='utf-8-sig')
    rsd = rep_pred.groupby('label').pred.std(ddof=1).rename('rep_SD').reset_index()
    rsd = rsd.merge(st[['label', 'nominal']], on='label')
    rep_summary = rsd.groupby('nominal').rep_SD.agg(['median', 'max', 'count']).reset_index()
    pooled = float(np.sqrt((rsd.rep_SD ** 2).mean()))
    rep_summary.to_csv(OUT / 'replicate_repeatability_M1.csv', index=False)

    # batch residuals and starch-reference drift in predicted-% units
    st['res_M1'] = st.pred_M1_NIR - st.conc_recomputed
    batch = st.groupby(['batch', 'nominal']).res_M1.mean().unstack().round(2)
    batch.to_csv(OUT / 'residual_by_batch_level_M1.csv')
    ms = files[files.label == 'MSDF'].sort_values('time')
    session = ms.time.diff().dt.total_seconds().div(60).gt(10).cumsum()
    Pms = preprocess(prep, X, S[ms.index])[1]
    ref = pd.DataFrame({'session': session.to_numpy(), 'time': ms.time.to_numpy(), 'proj': Pms @ Bk})
    drift = ref.groupby('session').agg(time=('time', 'min'), proj=('proj', 'mean'))
    drift['shift_vs_first_%'] = drift.proj - drift.proj.iloc[0]
    drift.to_csv(OUT / 'reference_drift_M1.csv')

    manifest = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'seed': SEED, 'n_states': len(st), 'n_parents': int(st.parent.nunique()),
                'n_spectra_files': len(files), 'wavenumber_range': [float(axis[0]), float(axis[-1])],
                'n_points': len(axis), 'final_M1': {'prep': prep, 'LV': k},
                'pooled_replicate_SD_pct': pooled, 'runtime_s': round(time.time() - t0, 1)}
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    st.to_csv(OUT / 'state_predictions.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
