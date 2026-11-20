# -*- coding: utf-8 -*-
"""One file with every number quoted for the primary adulteration analysis (manuscript pipeline, all 110 states).

Reads the saved runs and writes summary/primary_numbers.json and summary/primary_tables/*.csv.
Sections that depend on runs still in progress are written with "status": "pending" and filled on the next call.
Run: python collect_primary.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
SUM = HERE / 'summary'
TAB = SUM / 'primary_tables'
TAB.mkdir(parents=True, exist_ok=True)
TASK = 'quantitative_with_zero105'
VARIANTS = ['color_only', 'nir_full', 'feature_fusion', 'nir_stack', 'fusion_stack']
SEED, B = 2026, 2000


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def oof(variant, root='runs', task=TASK):
    path = HERE / root / task / variant / task / 'outer_OOF_predictions.csv'
    return pd.read_csv(path) if path.exists() else None


def wilson(k, n, z=norm.ppf(0.975)):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(round(100 * (centre - half), 2)), float(round(100 * (centre + half), 2))]


def cluster_ci(frames, stat):
    """Parent-cluster bootstrap, conditional on saved predictions. frames: dict name -> residual Series indexed by state."""
    first = next(iter(frames.values()))
    parents = first.index.str.split(';').str[0]
    uniq = pd.unique(parents)
    rows = {p: np.flatnonzero(parents == p) for p in uniq}
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(B):
        idx = np.concatenate([rows[p] for p in rng.choice(uniq, len(uniq))])
        values.append(stat({k: v.to_numpy()[idx] for k, v in frames.items()}))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)], float(np.mean(np.array(values) < 0))


def main():
    out = {}
    res = {}
    for v in VARIANTS:
        p = oof(v).set_index('sample_state_id')
        res[v] = p.Residual
        obs = p.Observed
    states = res['fusion_stack'].index
    res = {v: r.loc[states] for v, r in res.items()}
    obs = obs.loc[states]
    nominal = pd.Series(np.select([obs < 1, obs < 10, obs < 22, obs < 40], [0, 5, 15, 30], 50), index=states)

    # 1. fixed-plan metrics, all states and mixtures scored within the all-state run
    m = pd.read_csv(SUM / 'metrics_all.csv')
    out['fixed_plan'] = {}
    for task, key in ((TASK, 'all110'), ('quantitative_positive83', 'mixtures88_trained_on_mixtures')):
        t = m[m.task == task]
        out['fixed_plan'][key] = {v: {r.Metric: [round(r.Estimate, 6), round(r.CI95_low, 6), round(r.CI95_high, 6)]
                                       for r in t[t.Variant == v].itertuples()} for v in VARIANTS}

    # 2. per level and detection
    lvl = []
    for v in VARIANTS:
        for level in (0, 5, 15, 30, 50):
            e = res[v][nominal == level]
            lvl.append({'variant': v, 'level': level, 'n': len(e), 'RMSE': round(rmse(e), 2), 'bias': round(float(e.mean()), 2)})
    pd.DataFrame(lvl).to_csv(TAB / 'rmse_by_level.csv', index=False, encoding='utf-8-sig')
    out['by_level'] = {v: {str(r['level']): [r['RMSE'], r['bias']] for r in lvl if r['variant'] == v} for v in VARIANTS}
    det = pd.read_csv(SUM / 'binary_by_level.csv')
    det['wilson95'] = [wilson(k, n) for k, n in zip(det.called_adulterated, det.n)]
    det['wilson_low'] = [w[0] for w in det.wilson95]
    det['wilson_high'] = [w[1] for w in det.wilson95]
    det.drop(columns='wilson95').to_csv(TAB / 'detection_by_level.csv', index=False, encoding='utf-8-sig')
    out['detection'] = {v: {str(r['nominal_%']): [int(r.called_adulterated), int(r.n), r.wilson95]
                            for _, r in det[det.variant == v].iterrows()} for v in VARIANTS}

    # 3. paired differences on the fixed plan (conditional parent bootstrap)
    pairs = [('fusion_stack', 'nir_stack'), ('fusion_stack', 'nir_full'), ('feature_fusion', 'nir_full'),
             ('nir_stack', 'nir_full'), ('color_only', 'nir_full')]
    out['paired_fixed_plan'] = {}
    for a, b in pairs:
        ci, share = cluster_ci({a: res[a], b: res[b]}, lambda d, a=a, b=b: rmse(d[a]) - rmse(d[b]))
        out['paired_fixed_plan'][a + ' - ' + b] = {'difference': round(rmse(res[a]) - rmse(res[b]), 2), 'ci95': ci,
                                                    'share_below_0': round(share, 3)}
    for level_set, key in (((5,), 'at_5pct'), ((0, 5), 'at_0_and_5pct'), ((15, 30, 50), 'at_15pct_and_above')):
        mask = nominal.isin(level_set).to_numpy()
        sub = {v: res[v][mask] for v in ('fusion_stack', 'nir_stack', 'feature_fusion', 'nir_full')}
        entry = {}
        for a, b in (('fusion_stack', 'nir_stack'), ('feature_fusion', 'nir_full')):
            ci, share = cluster_ci({a: sub[a], b: sub[b]}, lambda d, a=a, b=b: rmse(d[a]) - rmse(d[b]))
            entry[a + ' - ' + b] = {'difference': round(rmse(sub[a]) - rmse(sub[b]), 2), 'ci95': ci}
        out['paired_fixed_plan'][key + ' (post hoc by level)'] = entry

    # 4. states flagged by the NIR domain diagnostic (limits from training parents; PLSR supporting analysis)
    flags = pd.read_csv(HERE.parent / 'supporting_analysis' / 'results_confirm' / 'domain_flags_per_state.csv')
    flags['sid'] = flags.parent + ';S' + flags.state.astype(str).str.extract(r'S(\d)')[0]
    flags = flags.set_index('sid').loc[states]
    sens = []
    for label, keep in (('all states', np.ones(len(states), bool)),
                        ('accepted, 99 % limits', ~flags.flag_any_99.astype(bool).to_numpy()),
                        ('accepted, 95 % limits', ~flags.flag_any_95.astype(bool).to_numpy()),
                        ('parent F1 not scored', ~states.str.startswith('F1;'))):
        for v in VARIANTS:
            ci, _ = cluster_ci({v: res[v][keep]}, lambda d, v=v: rmse(d[v]))
            sens.append({'set': label, 'n': int(keep.sum()), 'variant': v, 'RMSE': round(rmse(res[v][keep]), 2),
                         'ci_low': ci[0], 'ci_high': ci[1]})
    pd.DataFrame(sens).to_csv(TAB / 'flagged_state_sensitivity.csv', index=False, encoding='utf-8-sig')
    out['flag_sensitivity'] = {s['set'] + ' | ' + s['variant']: [s['n'], s['RMSE'], s['ci_low'], s['ci_high']] for s in sens}

    # 5. error by parent, primary model
    bp = pd.DataFrame({'parent': states.str.split(';').str[0], 'res': res['fusion_stack'].to_numpy()}).groupby('parent').res.agg(
        RMSE=rmse, bias='mean').round(2).sort_values('RMSE', ascending=False)
    bp.to_csv(TAB / 'error_by_parent_fusion_stack.csv', encoding='utf-8-sig')
    out['by_parent_fusion_stack'] = {'top5': bp.head(5).reset_index().values.tolist(),
                                     'others_RMSE_range': [float(bp.RMSE.iloc[5:].min()), float(bp.RMSE.iloc[5:].max())]}

    # 6. random partitions
    rep = pd.read_csv(SUM / 'repeats_by_partition.csv')
    rep = rep[rep.task == TASK]
    out['random_partitions'] = {}
    wide = rep.pivot(index='partition', columns='variant', values='RMSE')
    for v in VARIANTS:
        y = wide[v].dropna()
        out['random_partitions'][v] = {'n': int(len(y)), 'mean': round(float(y.mean()), 2), 'sd': round(float(y.std(ddof=1)), 2),
                                       'min': round(float(y.min()), 2), 'max': round(float(y.max()), 2)}
    for a, b in pairs:
        d = (wide[a] - wide[b]).dropna()
        out['random_partitions'][a + ' - ' + b] = {'n': int(len(d)), 'mean': round(float(d.mean()), 2),
                                                   'min': round(float(d.min()), 2), 'max': round(float(d.max()), 2),
                                                   'n_first_lower': int((d < 0).sum())}
    wide.round(3).to_csv(TAB / 'random_partitions_wide.csv', encoding='utf-8-sig')

    # 7. batch and supplier hold-out
    out['holdout'] = {}
    for plan in ('batch', 'supplier'):
        entry = {}
        for v in VARIANTS:
            path = HERE / 'runs_holdout' / plan / v / TASK / 'outer_OOF_predictions.csv'
            p = pd.read_csv(path) if path.exists() else None
            if p is None or len(p) < 110:
                entry[v] = {'status': 'pending'}
                continue
            entry[v] = {'RMSE': round(rmse(p.Residual), 2),
                        'by_fold': {int(f): round(rmse(s.Residual), 2) for f, s in p.groupby('Outer_fold')},
                        'by_fold_without_F1': {int(f): round(rmse(s[s.group_id != 'F1'].Residual), 2)
                                               for f, s in p.groupby('Outer_fold')}}
        out['holdout'][plan] = entry

    # 8. permutation and refit bootstrap of the stack
    path = HERE / 'stack_confirm' / 'stack_confirm_summary.json'
    done = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for key in ('permutation', 'bootstrap'):
        out[key] = done.get(key, {'status': 'pending'})
        if key in done and done[key]['n_completed'] < done[key]['n_planned']:
            out[key]['status'] = 'incomplete'
    check = HERE / 'stack_confirm' / 'driver_check.json'
    out['driver_check'] = json.loads(check.read_text(encoding='utf-8')) if check.exists() else {'status': 'pending'}

    (SUM / 'primary_numbers.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('paired_fixed_plan', 'flag_sensitivity', 'by_parent_fusion_stack', 'random_partitions',
                                           'holdout', 'permutation', 'bootstrap', 'driver_check')}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
