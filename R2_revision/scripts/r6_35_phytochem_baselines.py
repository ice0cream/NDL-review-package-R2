# -*- coding: utf-8 -*-
"""Reviewer 6 comment 35: per-category errors and simple baselines for phytochemical PLSR.

Uses the saved nested PLSR outer-OOF predictions (10 repeats x 5 outer folds).
For every outer fold, two baselines are fitted on that fold's outer-training samples only:
  Train_mean     predicts the training-set mean for every held-out sample
  Category_mean  predicts the training-set mean of the held-out sample's recorded category
The category baseline is given the category label, which PLSR is not, so it is a strong
reference: PLSR that only reproduced category differences would not beat it.
Metrics are computed per repeat and summarized as mean +/- SD over the 10 repeats.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'R1_revision' / 'comment03_validation' / 'nested_PLSR_20260910' / 'results' / 'tables' / 'outer_OOF_predictions.csv'
OUT = ROOT / 'R2_revision' / 'R6-35_phytochem_baselines'
OUT.mkdir(parents=True, exist_ok=True)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sse = np.sum((p - y) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean((p - y) ** 2))
    return {'n': len(y), 'RMSE': rmse, 'MAE': np.mean(np.abs(p - y)), 'Bias': np.mean(p - y),
            'R2': 1 - sse / sst if sst > 0 else np.nan}


pred = pd.read_csv(SRC)
rows = []
for (target, repeat), sub in pred.groupby(['Target', 'Repeat']):
    sub = sub.copy()
    sub['Train_mean'] = np.nan
    sub['Category_mean'] = np.nan
    for fold, va in sub.groupby('Outer_fold'):
        tr = sub[sub.Outer_fold != fold]
        sub.loc[va.index, 'Train_mean'] = tr.Observed.mean()
        cat_means = tr.groupby('Category').Observed.mean()
        sub.loc[va.index, 'Category_mean'] = va.Category.map(cat_means).fillna(tr.Observed.mean()).to_numpy()
    for model in ['Predicted', 'Train_mean', 'Category_mean']:
        name = 'PLSR' if model == 'Predicted' else model
        rows.append({'Target': target, 'Repeat': repeat, 'Model': name, 'Category': 'All',
                     **metrics(sub.Observed, sub[model])})
        for cat, part in sub.groupby('Category'):
            rows.append({'Target': target, 'Repeat': repeat, 'Model': name, 'Category': cat,
                         **metrics(part.Observed, part[model])})
    # Paired per-sample comparison: does PLSR beat the category-mean baseline sample by sample?
    rows.append({'Target': target, 'Repeat': repeat, 'Model': 'PLSR_better_than_Category_mean_fraction',
                 'Category': 'All', 'n': len(sub),
                 'RMSE': float(np.mean(np.abs(sub.Predicted - sub.Observed) < np.abs(sub.Category_mean - sub.Observed)))})

per_repeat = pd.DataFrame(rows)
per_repeat.to_csv(OUT / 'per_repeat_metrics.csv', index=False, encoding='utf-8-sig')
summary = (per_repeat.groupby(['Target', 'Category', 'Model'])
           .agg(n=('n', 'first'), RMSE_mean=('RMSE', 'mean'), RMSE_sd=('RMSE', 'std'),
                MAE_mean=('MAE', 'mean'), Bias_mean=('Bias', 'mean'),
                R2_mean=('R2', 'mean'), R2_sd=('R2', 'std'))
           .reset_index())
order = {'PLSR': 0, 'Category_mean': 1, 'Train_mean': 2}
summary['_o'] = summary.Model.map(order).fillna(3)
summary = summary.sort_values(['Target', 'Category', '_o']).drop(columns='_o')
summary.to_csv(OUT / 'summary_by_target_category_model.csv', index=False, encoding='utf-8-sig')
with pd.option_context('display.width', 200, 'display.max_rows', 200, 'display.float_format', '{:.3f}'.format):
    print(summary)
