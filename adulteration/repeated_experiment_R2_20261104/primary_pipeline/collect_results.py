# -*- coding: utf-8 -*-
"""Collects the manuscript-pipeline runs into summary tables (./summary).

Fixed partition (runs/<task>/<variant>; inputs/outer_fold_plan_redo.csv)
  metrics_all.csv              every metric with its parent-bootstrap 95 % CI, 5 variants x 4 tasks
  locked_by_fold.csv           configuration locked in each outer fold before outer evaluation
  regression_by_level.csv      RMSE and bias by nominal level; all-110 models also scored on the mixtures only
  paired_differences.csv       paired parent-cluster bootstrap (B = 2000, seed 2026) of RMSE differences between variants,
                               conditional on the saved out-of-fold predictions
  clipping_counts.csv          predictions at the 0 % / 100 % bounds (the pipeline clips base and meta predictions to 0-100)
  split_audit_summary.csv      number of outer / inner / stacking splits per run and how many share a parent (must be 0)
  binary_by_level.csv          share called 'Adulterated' by nominal level (0 % = false-positive rate)
  fourclass_recall.csv         recall of each class and the confusion counts
Repeated partitions (runs_repeats/<task>/<variant>; run_paper_pipeline_repeats.sh), when present
  repeats_by_partition.csv     RMSE, R2, bias of each variant in each random parent-grouped partition
  repeats_summary.csv          mean, SD, min, max over partitions and paired differences on shared partitions
Run after run_paper_pipeline.sh: python collect_results.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS, REPS, OUT = HERE / 'runs', HERE / 'runs_repeats', HERE / 'summary'
VARIANTS = ['color_only', 'nir_full', 'feature_fusion', 'nir_stack', 'fusion_stack']
TASKS = ['quantitative_positive83', 'quantitative_with_zero105', 'Task2_binary', 'Task1_4class']
REG = TASKS[:2]
PAIRS = [('fusion_stack', 'nir_stack'), ('feature_fusion', 'nir_full'), ('fusion_stack', 'nir_full'),
         ('nir_stack', 'nir_full'), ('color_only', 'nir_full')]
NOMINAL = {0: 0, 1: 5, 2: 15, 3: 30, 4: 50}
CLASSES = ['Authentic', 'Low', 'Medium', 'High']
SEED = 2026


def oof(task, v, root=RUNS):
    d = pd.read_csv(root / task / v / task / 'outer_OOF_predictions.csv', encoding='utf-8-sig')
    d['nominal'] = d.sample_state_id.str.split(';S').str[1].astype(int).map(NOMINAL)
    return d


def rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))


def paired_boot(a, b, B=2000):
    """a, b: OOF tables of two variants on the same rows. Parent-cluster bootstrap of RMSE(a) - RMSE(b)."""
    a, b = a.sort_values('sample_state_id'), b.sort_values('sample_state_id')
    assert (a.sample_state_id.to_numpy() == b.sample_state_id.to_numpy()).all()
    ea, eb, g = a.Residual.to_numpy(), b.Residual.to_numpy(), a.group_id.to_numpy()
    rng = np.random.default_rng(SEED)
    uniq = np.unique(g)
    by = {p: np.where(g == p)[0] for p in uniq}
    d = []
    for _ in range(B):
        i = np.concatenate([by[p] for p in rng.choice(uniq, len(uniq), replace=True)])
        d.append(rmse(ea[i]) - rmse(eb[i]))
    d = np.array(d)
    return {'difference': rmse(ea) - rmse(eb), 'ci_low': float(np.quantile(d, .025)), 'ci_high': float(np.quantile(d, .975)),
            'share_below_0': float((d < 0).mean())}


def main():
    OUT.mkdir(exist_ok=True)
    met, lock, reg, binl, four, clip, aud, pairs = [], [], [], [], [], [], [], []
    for task in TASKS:
        for v in VARIANTS:
            m = pd.read_csv(RUNS / task / v / 'metrics_summary.csv', encoding='utf-8-sig')
            met.append(m.assign(task=task)[['task', 'Variant', 'Metric', 'Estimate', 'CI95_low', 'CI95_high']])
            for js in sorted((RUNS / task / v / task).glob('repeat_01/fold_*/locked_selection_before_outer_evaluation.json')):
                j = json.loads(js.read_text(encoding='utf-8'))
                lock.append({'task': task, 'variant': v, 'fold': j['Outer_fold'], 'preprocessing_no': j['Number'],
                             'spectra': j.get('Spectra'), 'smoothing': j.get('Smoothing'),
                             'pathlength': j.get('Pathlength'), 'base_model': j['Base_model'],
                             'fusion': j['Fusion_type']})
            a = pd.read_csv(RUNS / task / v / task / 'all_nested_split_audits.csv', encoding='utf-8-sig')
            lv = a.Level.map(lambda s: 'outer' if s == 'outer' else 'stacking' if 'stack' in s else 'inner')
            aud.append({'task': task, 'variant': v, 'n_outer': int((lv == 'outer').sum()), 'n_inner': int((lv == 'inner').sum()),
                        'n_stacking': int((lv == 'stacking').sum()), 'splits_sharing_a_parent': int((a.overlap_count > 0).sum())})
            d = oof(task, v)
            if task in REG:
                for lvl, g in d.groupby('nominal'):
                    reg.append({'task': task, 'variant': v, 'nominal_%': lvl, 'n': len(g),
                                'RMSE': rmse(g.Residual), 'bias': float(g.Residual.mean())})
                if task == 'quantitative_with_zero105':
                    g = d[d.nominal > 0]
                    reg.append({'task': task, 'variant': v, 'nominal_%': 'mixtures only (scored)', 'n': len(g),
                                'RMSE': rmse(g.Residual), 'bias': float(g.Residual.mean())})
                clip.append({'task': task, 'variant': v, 'n': len(d), 'n_at_0': int((d.Predicted <= 0).sum()),
                             'n_at_0_authentic': int(((d.Predicted <= 0) & (d.nominal == 0)).sum()),
                             'n_at_0_at_5%': int(((d.Predicted <= 0) & (d.nominal == 5)).sum()),
                             'n_at_100': int((d.Predicted >= 100).sum()), 'min_pred': float(d.Predicted.min()),
                             'max_pred': float(d.Predicted.max())})
            elif task == 'Task2_binary':
                for lvl, g in d.groupby('nominal'):
                    binl.append({'variant': v, 'nominal_%': lvl, 'n': len(g),
                                 'called_adulterated': int((g.Predicted == 'Adulterated').sum())})
            else:
                cm = pd.crosstab(pd.Categorical(d.Observed, CLASSES), pd.Categorical(d.Predicted, CLASSES), dropna=False)
                row = {'variant': v}
                for c in CLASSES:
                    row[f'recall_{c}'] = float(cm.loc[c, c] / cm.loc[c].sum())
                    for c2 in CLASSES:
                        row[f'n_{c}->{c2}'] = int(cm.loc[c, c2])
                s5 = d[d.nominal == 5]
                row['5%_not_called_authentic'] = int((s5.Predicted != 'Authentic').sum())
                four.append(row)
    for task in REG:
        for a, b in PAIRS:
            pairs.append({'task': task, 'comparison': f'{a} - {b}', **paired_boot(oof(task, a), oof(task, b))})
    # positive-only model vs all-110 model, both scored on the same 88 mixtures (Reviewer 6, comment 33)
    for v in VARIANTS:
        z = oof('quantitative_with_zero105', v)
        pairs.append({'task': 'same 88 mixtures', 'comparison': f'{v}: trained on all 110 - trained on mixtures only',
                      **paired_boot(z[z.nominal > 0], oof('quantitative_positive83', v))})
    pd.concat(met).to_csv(OUT / 'metrics_all.csv', index=False, encoding='utf-8-sig')
    for df, name in [(lock, 'locked_by_fold'), (reg, 'regression_by_level'), (binl, 'binary_by_level'), (four, 'fourclass_recall'),
                     (clip, 'clipping_counts'), (aud, 'split_audit_summary'), (pairs, 'paired_differences')]:
        pd.DataFrame(df).to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 200, 'display.max_colwidth', 70)
    print(pd.DataFrame(pairs).round(3).to_string())
    print(pd.DataFrame(clip).to_string())
    a = pd.DataFrame(aud)
    print('split audits:', a[['n_outer', 'n_inner', 'n_stacking']].sum().to_dict(), 'sharing a parent:', int(a.splits_sharing_a_parent.sum()))

    # repeated partitions
    rows = []
    for task in REG:
        for v in VARIANTS:
            f = REPS / task / v / task / 'outer_metrics_by_repeat.csv'
            if f.exists() and (REPS / task / v / 'metrics_summary.csv').exists():
                d = oof(task, v, REPS)
                for r, g in d.groupby('Repeat'):
                    rows.append({'task': task, 'variant': v, 'partition': int(r), 'RMSE': rmse(g.Residual),
                                 'R2': 1 - float((g.Residual ** 2).sum() / ((g.Observed - g.Observed.mean()) ** 2).sum()),
                                 'bias': float(g.Residual.mean()), 'RMSE_0%': rmse(g.Residual[g.nominal == 0]) if (g.nominal == 0).any() else np.nan,
                                 'RMSE_5%': rmse(g.Residual[g.nominal == 5]), 'RMSE_mixtures': rmse(g.Residual[g.nominal > 0]),
                                 'n_at_0': int((g.Predicted <= 0).sum())})
    if rows:
        rp = pd.DataFrame(rows)
        rp.to_csv(OUT / 'repeats_by_partition.csv', index=False, encoding='utf-8-sig')
        s = rp.groupby(['task', 'variant']).agg(n_partitions=('RMSE', 'size'), RMSE_mean=('RMSE', 'mean'), RMSE_sd=('RMSE', 'std'),
                                                RMSE_min=('RMSE', 'min'), RMSE_max=('RMSE', 'max'), R2_mean=('R2', 'mean'),
                                                RMSE_5pct_mean=('RMSE_5%', 'mean'), RMSE_mixtures_mean=('RMSE_mixtures', 'mean')).reset_index()
        dd = []
        for task in REG:
            w = rp[rp.task == task].pivot(index='partition', columns='variant', values='RMSE')
            for a_, b_ in PAIRS:
                if a_ in w and b_ in w:
                    d = (w[a_] - w[b_]).dropna()
                    dd.append({'task': task, 'comparison': f'{a_} - {b_}', 'n_shared_partitions': len(d), 'mean_difference': d.mean(),
                               'min': d.min(), 'max': d.max(), 'partitions_first_is_lower': int((d < 0).sum())})
        s.to_csv(OUT / 'repeats_summary.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(dd).to_csv(OUT / 'repeats_paired_differences.csv', index=False, encoding='utf-8-sig')
        print(s.round(3).to_string())
        print(pd.DataFrame(dd).round(3).to_string())


if __name__ == '__main__':
    main()
