# -*- coding: utf-8 -*-
"""Reviewer 6, comment 9: moisture of the retained powders and dry-matter sensitivity of the phytochemical results.

Moisture: loss on drying (Chinese Pharmacopoeia general rule 0832, method 2), ONE determination per sample, taken from the
first block (columns A-E) of the laboratory workbook 牛大力水分测定.xlsx. moisture % = (W1 - W3) / (W1 - W0) x 100, where
W0 = weighing bottle, W1 = bottle + sample, W2 = after 5 h at 105 °C, W3 = after one further hour.
The powders were dried in November 2026, about six months after the spectra and assays (May 2026), so the dry-basis values
are a sensitivity analysis and not a correction.

  python r6_9_moisture_dry_basis.py          tidy table, summaries, dry-basis values
  python r6_9_moisture_dry_basis.py run      also repeat the nested PLSR of the manuscript on the dry-basis values
                                             (same code, same splits and seeds as the original analysis; about 1 h)
Outputs: R2_revision/R6-9_moisture/
"""
import importlib.util
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'R2_revision' / 'R6-9_moisture'
OUT.mkdir(exist_ok=True)
NESTED = ROOT / 'R1_revision' / 'comment03_validation' / 'nested_PLSR_20260910'
TARGETS = {'TFC': 'TFC (μg RE/g)', 'TPC': 'TPC (μg GAE/g)', 'TSS': 'TSS (mg Glc/g)'}
ORDER = ['FTR', 'WTR', 'AST', 'SFTR']

# ---------------------------------------------------------------- 1. tidy table (first block only)
w = pd.read_csv(ROOT / 'moisture_weighings.csv', float_precision='round_trip')   # first block of the moisture sheet
assert list(w.columns) == ['ID', 'W0(g)', 'W1(g)', 'W2(g)', 'W3(g)']
rows = w.values.tolist()
m = pd.DataFrame(rows, columns=['Modelling_ID', 'W0_g', 'W1_g', 'W2_g', 'W3_g'])
mp = pd.read_csv(ROOT / 'R2_revision' / 'R6-3_sample_traceability' / 'mapping_author_statement.csv', encoding='utf-8-sig')
m = m.merge(mp[['Modelling_ID', 'Manuscript_ID', 'Category']], on='Modelling_ID', how='left', validate='one_to_one')
assert len(m) == 58 and m.Category.notna().all()
m['sample_mass_g'] = m.W1_g - m.W0_g
m['second_drying_loss_mg'] = (m.W2_g - m.W3_g) * 1000
m['moisture_percent'] = (m.W1_g - m.W3_g) / m.sample_mass_g * 100
assert (m.second_drying_loss_mg.between(-0.05, 5)).all()          # constant-weight criterion of the method: <= 5 mg
m['n'] = m.Manuscript_ID.str[1:].astype(int)
m = m.sort_values('n').drop(columns='n')[['Manuscript_ID', 'Modelling_ID', 'Category', 'W0_g', 'W1_g', 'W2_g', 'W3_g',
                                          'sample_mass_g', 'second_drying_loss_mg', 'moisture_percent']]
m.to_csv(OUT / 'moisture_loss_on_drying.csv', index=False, encoding='utf-8-sig')

by = m.groupby('Category').moisture_percent.agg(['size', 'mean', 'std', 'min', 'max']).loc[ORDER].reset_index()
by.columns = ['Category', 'n', 'mean', 'sd', 'min', 'max']
alln = pd.DataFrame([{'Category': 'All', 'n': len(m), 'mean': m.moisture_percent.mean(), 'sd': m.moisture_percent.std(),
                      'min': m.moisture_percent.min(), 'max': m.moisture_percent.max()}])
by = pd.concat([by, alln], ignore_index=True)
by.to_csv(OUT / 'moisture_by_category.csv', index=False, encoding='utf-8-sig')
kw = stats.kruskal(*[m.loc[m.Category == c, 'moisture_percent'] for c in ORDER])
print(by.round(2).to_string())
print('Kruskal-Wallis H = %.2f, p = %.2g' % (kw.statistic, kw.pvalue))

# ---------------------------------------------------------------- 2. dry-basis reference values
ref = pd.read_csv(ROOT / 'phytochemical_assays' / 'DZ.CSV', encoding='utf-8-sig')
ref['sample_id'] = ref.sample_id.astype(str).str.strip()
d = ref.merge(m[['Modelling_ID', 'Category', 'moisture_percent']], left_on='sample_id', right_on='Modelling_ID',
              validate='one_to_one')
assert len(d) == 58
d['factor'] = 1 / (1 - d.moisture_percent / 100)
rows, dry = [], d[['sample_id', 'Category', 'moisture_percent', 'factor']].copy()
for tag, col in TARGETS.items():
    dry[tag + '_as_is'] = d[col]
    dry[tag + '_dry'] = d[col] * d.factor
    r_all = stats.spearmanr(d.moisture_percent, d[col])
    centred_m = d.moisture_percent - d.groupby('Category').moisture_percent.transform('mean')
    centred_y = d[col] - d.groupby('Category')[col].transform('mean')
    r_within = stats.spearmanr(centred_m, centred_y)
    rows.append({'Target': tag, 'pearson_as_is_vs_dry': np.corrcoef(d[col], d[col] * d.factor)[0, 1],
                 'spearman_moisture_vs_value': r_all.statistic, 'p': r_all.pvalue,
                 'spearman_moisture_vs_value_within_category': r_within.statistic, 'p_within': r_within.pvalue})
    for c in ORDER:
        s = d[d.Category == c]
        rows[-1]['mean_as_is_' + c] = s[col].mean()
        rows[-1]['mean_dry_' + c] = (s[col] * s.factor).mean()
dry.to_csv(OUT / 'reference_values_dry_basis.csv', index=False, encoding='utf-8-sig')
val = pd.DataFrame(rows)
val.to_csv(OUT / 'dry_basis_effect_on_values.csv', index=False, encoding='utf-8-sig')
print(val.round(3).T.to_string())
print('factor range: %.3f-%.3f' % (d.factor.min(), d.factor.max()))

# ---------------------------------------------------------------- 3. nested PLSR on dry-basis values
spec = importlib.util.spec_from_file_location('nested_plsr', NESTED / 'nested_plsr.py')
npl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(npl)
RUN = OUT / 'nested_plsr_dry_basis'


def run_nested(workers=6):
    from threadpoolctl import threadpool_limits
    dd, wn_cols, _ = npl.load_data(None)
    fac = dd.sample_id.map(d.set_index('sample_id').factor)
    assert fac.notna().all()
    X = dd[wn_cols].to_numpy(float)
    Y = dd[npl.TARGETS].to_numpy(float) * fac.to_numpy(float)[:, None]
    ids, groups, cat = dd.sample_id.to_numpy(str), dd.parent_batch_id.to_numpy(str), dd.group.to_numpy(str)
    wn = np.array([float(c[3:]) for c in wn_cols])
    tasks = [(rep, f, tr, va) for rep in range(1, 11) for f, (tr, va) in enumerate(npl.splits(groups, 5, 42 + rep - 1), 1)]
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(npl.run_fold, t, X, Y, ids, groups, cat, wn, RUN) for t in tasks]
            results = [f.result() for f in as_completed(futures)]
        results.sort(key=lambda r: (r['metrics'][0]['Repeat'], r['metrics'][0]['Outer_fold']))
        npl.summarize(RUN, results)


if len(sys.argv) > 1 and sys.argv[1] == 'run':
    run_nested()


def metric_table(pred):
    """Per repeat: R2 and RMSE of PLSR, R2 of the category-mean baseline, pooled within-category R2 of PLSR."""
    out = []
    for (tag, rep), s in pred.groupby(['Target', 'Repeat']):
        s = s.copy()
        s['base'] = np.nan
        for fold, va in s.groupby('Outer_fold'):
            tr = s[s.Outer_fold != fold]
            s.loc[va.index, 'base'] = va.Category.map(tr.groupby('Category').Observed.mean()).to_numpy()
        sst = ((s.Observed - s.Observed.mean()) ** 2).sum()
        ssw = ((s.Observed - s.groupby('Category').Observed.transform('mean')) ** 2).sum()
        sse = ((s.Predicted - s.Observed) ** 2).sum()
        out.append({'Target': tag, 'Repeat': rep, 'R2': 1 - sse / sst, 'RMSE': np.sqrt(sse / len(s)),
                    'relative_RMSE_percent': 100 * np.sqrt(sse / len(s)) / s.Observed.mean(),
                    'R2_category_mean': 1 - ((s.base - s.Observed) ** 2).sum() / sst, 'R2_within_category': 1 - sse / ssw})
    return pd.DataFrame(out)


summary = {'moisture': {r.Category: {'n': int(r.n), 'mean': round(r['mean'], 2), 'sd': round(r.sd, 2), 'min': round(r['min'], 2),
                                     'max': round(r['max'], 2)} for _, r in by.iterrows()},
           'kruskal_H': round(float(kw.statistic), 2), 'kruskal_p': float(kw.pvalue),
           'factor_min': round(float(d.factor.min()), 3), 'factor_max': round(float(d.factor.max()), 3),
           'values': {r.Target: {'pearson_as_is_vs_dry': round(r.pearson_as_is_vs_dry, 4),
                                 'rho_within': round(r.spearman_moisture_vs_value_within_category, 2),
                                 'p_within': round(r.p_within, 2)} for _, r in val.iterrows()}}
done = RUN / 'tables' / 'outer_OOF_predictions.csv'
if done.exists():
    a = metric_table(pd.read_csv(NESTED / 'results' / 'tables' / 'outer_OOF_predictions.csv', encoding='utf-8-sig'))
    b = metric_table(pd.read_csv(done, encoding='utf-8-sig'))
    a['basis'], b['basis'] = 'as analysed', 'dry matter'
    both = pd.concat([a, b])
    both.to_csv(OUT / 'nested_plsr_per_repeat_both_bases.csv', index=False, encoding='utf-8-sig')
    comp = both.groupby(['Target', 'basis']).agg(R2=('R2', 'mean'), R2_sd=('R2', 'std'), RMSE=('RMSE', 'mean'),
                                                 relative_RMSE_percent=('relative_RMSE_percent', 'mean'),
                                                 R2_category_mean=('R2_category_mean', 'mean'),
                                                 R2_within_category=('R2_within_category', 'mean')).reset_index()
    comp.to_csv(OUT / 'nested_plsr_as_analysed_vs_dry_basis.csv', index=False, encoding='utf-8-sig')
    print(comp.round(3).to_string())
    summary['plsr'] = {'{}_{}'.format(r.Target, 'dry' if r.basis == 'dry matter' else 'asis'):
                       {'R2': round(r.R2, 3), 'R2_sd': round(r.R2_sd, 3), 'rel_RMSE': round(r.relative_RMSE_percent, 1),
                        'R2_cat': round(r.R2_category_mean, 3), 'R2_within': round(r.R2_within_category, 2)}
                       for _, r in comp.iterrows()}
else:
    print('nested PLSR on dry-basis values not run yet')
json.dump(summary, io.open(OUT / 'moisture_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
(OUT / 'README.txt').write_text(
    'Moisture by loss on drying (Chinese Pharmacopoeia general rule 0832, method 2): about 2 g of powder in a weighing bottle\n'
    'dried to constant weight, 105 °C for 5 h, cooled 30 min in a desiccator, weighed, dried 1 h more; successive weighings\n'
    'differed by no more than 5 mg. One determination per sample, November 2026, on the retained powders.\n'
    'moisture_loss_on_drying.csv: weighings as recorded and the calculated moisture.\n'
    'reference_values_dry_basis.csv: assay values per g of analysed powder and per g of dry matter.\n'
    'nested_plsr_dry_basis/: the nested PLSR of the manuscript repeated on the dry-matter values (same code, splits, seeds).\n'
    'Script: R2_revision/scripts/r6_9_moisture_dry_basis.py\n', encoding='utf-8')
