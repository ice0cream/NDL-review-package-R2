# -*- coding: utf-8 -*-
"""Build the revision-2 supplementary tables and figures from the saved analysis outputs.

Writes R2_revision/Supplement_R2/ :
  Supplement_R2_tables_figures.docx   all new/revised tables with captions, and the figures
  Table_*.csv                         the same tables as CSV
  Fig_S21_permutation_null.png/svg    full-procedure permutation null distribution
Tables are only built from files that exist, so the script can be rerun as analyses finish.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / 'R2_revision'
BENCH = ROOT / 'adulteration' / 'NIR-CLO' / 'REVISE2_benchmark'
OUT = R2 / 'Supplement_R2'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': ['Times New Roman', 'DejaVu Sans'], 'font.size': 10})

LABEL = {'fusion_stack': 'Decision fusion, NIR windows + CIELAB (R1 model)',
         'nir_stack': 'Decision fusion, NIR windows only (CIELAB removed)',
         'nir_full': 'NIR full spectrum, single model',
         'feature_fusion': 'Feature fusion, NIR windows + CIELAB',
         'color_only': 'CIELAB only'}
ORDER = list(LABEL)
tables = []  # (name, caption, frame, note)


def f3(x):
    return '' if pd.isna(x) else f'{x:.3f}'


# ---------- Table S12 (revised): fixed-partition benchmark
fp = BENCH / 'summary' / 'fixed_plan_metrics.csv'
dp = BENCH / 'summary' / 'fixed_plan_paired_differences.csv'
if fp.exists():
    m = pd.read_csv(fp)
    dd = pd.read_csv(dp) if dp.exists() else pd.DataFrame()
    rows = []
    for v in ORDER:
        def get(scope, task, col):
            s = m[(m.Scope == scope) & (m.Task == task) & (m.Variant == v)]
            return s[col].iloc[0] if len(s) else np.nan
        delta = dd[(dd.Scope == 'plan_primary') & (dd.Task == 'quantitative_positive83') & (dd.Variant == v)]
        rows.append({'Approach': LABEL[v],
                     'RMSE, 83 mixtures': f3(get('plan_primary', 'quantitative_positive83', 'RMSE')),
                     'R², 83 mixtures': f3(get('plan_primary', 'quantitative_positive83', 'R2')),
                     'ΔRMSE vs R1 model [95% CI]': '—' if delta.empty else
                         f"{delta.Delta_RMSE_vs_published.iloc[0]:+.2f} [{delta.CI95_low.iloc[0]:.2f}, {delta.CI95_high.iloc[0]:.2f}]",
                     'Density-weighted RMSE, 83': f3(get('plan_primary', 'quantitative_positive83', 'RMSE_density_weighted')),
                     'RMSE, 105 states': f3(get('plan_primary', 'quantitative_with_zero105', 'RMSE')),
                     '105-state model on 83 positives': f3(get('plan_primary', 'quantitative_with_zero105', 'RMSE_positive_rows')),
                     'RMSE, 64-mixture subset': f3(get('subset64', 'quantitative_positive83', 'RMSE'))})
    tables.append(('Table_S12_revised', 'Table S12 (revised). Modality comparison on the R1 parent-fold partition.',
                   pd.DataFrame(rows),
                   'All approaches use identical outer parent groups, three-fold grouped inner selection over 24 '
                   'preprocessing options and PLSR/SVR/RF base regressors, identical estimator settings and 0–100% '
                   'constraint. RMSE in percentage points. ΔRMSE: paired parent-cluster bootstrap (5,000 draws) of '
                   'identical held-out rows, conditional on saved predictions. Density-weighted RMSE gives equal weight '
                   'to the >0–10, >10–20, >20–40 and >40–60% intervals. The 64-mixture subset is the R1 '
                   'concentration-thinned set (Table S19).'))

# ---------- Table S20: repeated partitions
rp = BENCH / 'summary' / 'repeated_per_partition.csv'
if rp.exists():
    r = pd.read_csv(rp)
    wide = r.pivot(index='Repeat', columns='Variant', values='RMSE').reindex(columns=[v for v in ORDER if v in set(r.Variant)])
    body = wide.map(f3).reset_index().rename(columns={'Repeat': 'Partition'})
    body['Partition'] = body.Partition.map(lambda p: f'{p} (R1)' if p == 1 else str(p))
    summ = {'Partition': 'Mean ± SD'}
    for v in wide:
        s = wide[v].dropna()
        summ[v] = f'{s.mean():.2f} ± {s.std(ddof=1):.2f} (n={len(s)})'
    body = pd.concat([body, pd.DataFrame([summ])], ignore_index=True).rename(columns=LABEL)
    tables.append(('Table_S20', 'Table S20. RMSE (percentage points) of each approach over random parent-grouped partitions of the 83 mixtures.',
                   body,
                   'Partition 1 is the R1 partition. Partitions with the same number are identical across approaches. '
                   'Stacked approaches were run on five partitions and single-model approaches on ten, because of computing time.'))

# ---------- Table S18: phytochemical baselines and alternative models
bp = R2 / 'R6-35_phytochem_baselines' / 'summary_by_target_category_model.csv'
ap = R2 / 'R6-35_phytochem_baselines' / 'alternative_models_per_repeat.csv'
if bp.exists():
    b = pd.read_csv(bp)
    b = b[b.Model.isin(['PLSR', 'Category_mean', 'Train_mean'])]
    rows = []
    for (t, c), g in b.groupby(['Target', 'Category'], sort=False):
        row = {'Index': t, 'Category': c, 'n': int(g.n.iloc[0])}
        for mod in ['PLSR', 'Category_mean', 'Train_mean']:
            s = g[g.Model == mod]
            row[f'{mod} RMSE'] = f3(s.RMSE_mean.iloc[0]) if len(s) else ''
            row[f'{mod} R²'] = f3(s.R2_mean.iloc[0]) if len(s) else ''
        rows.append(row)
    t18 = pd.DataFrame(rows)
    t18['_o'] = t18.Category.map({'All': 0, 'FTR': 1, 'WTR': 2, 'SFTR': 3, 'AST': 4})
    t18 = t18.sort_values(['Index', '_o']).drop(columns='_o')
    tables.append(('Table_S18a', 'Table S18a. Nested PLSR compared with training-mean and category-mean baselines on identical outer folds.',
                   t18.rename(columns=lambda c: c.replace('Category_mean', 'Category mean').replace('Train_mean', 'Training mean')),
                   'Means over ten repeats of outer five-fold validation. Baselines are fitted on the outer-training samples '
                   'only; the category-mean baseline uses the recorded category of each held-out sample. Category-level '
                   'R² is computed within the category and is negative when predictions are worse than the true category '
                   'mean. Units: TFC μg RE/g; TPC μg GAE/g; TSS mg Glc/g.'))
if ap.exists():
    a = pd.read_csv(ap)
    g = a.groupby(['Target', 'Model'])[['R2', 'RMSE', 'R2_within_category']].mean().reset_index()
    g['Model'] = g.Model.map({'Category_mean': 'Category mean', 'PLSR_global': 'PLSR (global)',
                              'PLSR_within': 'PLSR (category-centred)', 'SVR_global': 'RBF-SVR'})
    tables.append(('Table_S18b', 'Table S18b. Alternative models under the same nested scheme.',
                   g.assign(**{c: g[c].map(f3) for c in ['R2', 'RMSE', 'R2_within_category']})
                    .rename(columns={'R2': 'R²', 'R2_within_category': 'Pooled within-category R²'}),
                   'Outer five-fold × five repeats; inner five-fold selection of preprocessing (six options) and '
                   'hyperparameters. Pooled within-category R² = 1 − SSE / within-category sum of squares.'))

# ---------- Table S19: 64-mixture subset distribution
sp = R2 / 'R6-29_30_32_39_40_adulteration_diagnostics' / 'R6-29_subset64_by_bin.csv'
if sp.exists():
    s = pd.read_csv(sp)
    s.columns = [str(c) for c in s.columns]
    s = s.rename(columns={'False': 'Excluded', 'True': 'Retained in 64'})
    s['_o'] = s.Bin.map({'>0-5%': 0, '>5-10%': 1, '>10-20%': 2, '>20-40%': 3, '>40-60%': 4})
    s = s.sort_values('_o').drop(columns='_o').rename(columns={'Bin': 'Concentration interval'})
    tables.append(('Table_S19', 'Table S19. Concentration distribution of the R1 64-mixture subset.', s,
                   'Mixtures were removed at random in R1, mainly from the over-represented >0–5% interval, in '
                   'response to Reviewer 3. All 22 parents remain represented; mixture-level membership is listed in Table S7.'))

# ---------- Table S21: spectral repeatability
rp2 = R2 / 'R6-12_15_spectral_repeatability' / 'within_vs_between_by_preprocessing.csv'
if rp2.exists():
    q = pd.read_csv(rp2)
    q = q.assign(**{c: q[c].map(lambda x: f'{x:.4f}') for c in ['within_SD_mean', 'between_SD_mean']},
                 **{'Replicate share of between-sample variance': q.share_of_between_variance_from_replicate_noise.map(lambda x: f'{x:.1%}')})
    q = q[['Preprocessing', 'Category', 'n_samples', 'within_SD_mean', 'between_SD_mean', 'Replicate share of between-sample variance']]
    tables.append(('Table_S21', 'Table S21. Within-sample (triplicate) versus between-sample spectral variation.',
                   q.rename(columns={'n_samples': 'n', 'within_SD_mean': 'Within-sample SD', 'between_SD_mean': 'Between-sample SD'}),
                   'Raw replicate spectra (OPUS block Abs_from_ScSm_ScRf; three readings per sample, 58 samples). SDs are '
                   'averaged over 1,843 wavenumbers. The replicate share is (within-sample variance / 3) divided by the '
                   'variance of sample means within the category; values near 100% mean that within-category differences '
                   'are indistinguishable from replicate variation.'))

# ---------- Table S22: classification on 105 states (when available)
cls_rows = []
for v in ORDER:
    for task in ['Task2_binary', 'Task1_4class']:
        base = BENCH / 'runs' / 'classification' / v / task
        f = base / 'outer_metrics_parent_bootstrap_CI95.csv'
        c = base / 'performance_by_concentration.csv'
        if not f.exists():
            continue
        met = pd.read_csv(f).set_index('Metric')
        row = {'Approach': LABEL[v], 'Task': 'Binary' if task == 'Task2_binary' else 'Four-class',
               'Accuracy': f3(met.Estimate['Accuracy']), 'Balanced accuracy': f3(met.Estimate['Balanced_accuracy']),
               'Macro-F1': f3(met.Estimate['F1_macro'])}
        if task == 'Task2_binary' and c.exists():
            pc = pd.read_csv(c)
            for lab, key in [('Authentic called adulterated', '0%'), ('Detected >0–5%', '(0,5]%')]:
                s = pc[pc.Concentration_interval == key]
                row[lab] = f"{int(round(s.Adulterated_call_rate.iloc[0] * s.n.iloc[0]))}/{int(s.n.iloc[0])}" if len(s) else ''
        cls_rows.append(row)
if cls_rows:
    tables.append(('Table_S22', 'Table S22. Authentic-versus-adulterated (binary) and four-class classification on all 105 states.',
                   pd.DataFrame(cls_rows),
                   'Same parent-fold plan and nested selection as Table S12; selection by macro-F1. Caution: the 22 '
                   'authentic states reuse the NIR spectra (acquired 1 May 2026, ≈25 g loading) and CIELAB values of the '
                   'botanical analysis, whereas mixtures were measured on 23–24 May 2026 (20 g preparations). '
                   'Authentic-versus-mixture separation is therefore confounded with measurement session and loading.'))

# ---------- Fig. S21: permutation null distribution
null = R2 / 'R6-37_full_procedure_permutation' / 'null_distribution.csv'
summary = R2 / 'R6-37_full_procedure_permutation' / 'summary.json'
fig_paths = []
if null.exists() and summary.exists():
    import json
    n = pd.read_csv(null); js = json.loads(summary.read_text(encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(n.Nested_accuracy * 100, bins=np.arange(10, 60, 1.724), color='#88A0B8', edgecolor='white')
    ax.axvline(js['observed_nested_accuracy'] * 100, color='#B03A2E', lw=1.5)
    ax.text(js['observed_nested_accuracy'] * 100 - 1, ax.get_ylim()[1] * .9, f"observed\n{js['observed_nested_accuracy']:.2%}",
            ha='right', va='top', fontsize=8, color='#B03A2E')
    ax.set(xlabel='Nested outer-OOF accuracy (%)', ylabel='Permutations', xlim=(10, 102))
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'Fig_S21_permutation_null.{ext}', dpi=600, bbox_inches='tight')
    plt.close(fig)
    fig_paths.append((OUT / 'Fig_S21_permutation_null.png',
                      f"Fig. S21. Full-procedure permutation test of nested NIR PLS-DA. Histogram: outer-OOF accuracy of "
                      f"{js['n_permutations']} label permutations, each repeating the complete nested selection (24 "
                      f"preprocessing options × 1–8 latent variables, inner five-fold). Red line: true labels "
                      f"({js['observed_nested_accuracy']:.2%}); p = {js['p_value']:.3f}. Labels were permuted at sample level."))
cm = R2 / 'R6-36_nested_confusion' / 'nested_PLSDA_confusion_pooled.png'
if cm.exists():
    fig_paths.append((cm, 'Fig. 3B (revised). Nested PLS-DA confusion matrix pooled over ten complete repeats '
                          '(580 outer predictions; accuracy 99.48%). Counts with row percentages.'))

# ---------- Write CSVs and Word document
doc = Document()
doc.styles['Normal'].font.name = 'Times New Roman'
doc.styles['Normal'].font.size = Pt(9)
doc.add_heading('Revision 2 — new and revised supplementary tables and figures', level=1)
for name, caption, frame, note in tables:
    frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    doc.add_paragraph().add_run(caption).bold = True
    t = doc.add_table(rows=1, cols=len(frame.columns))
    t.style = 'Table Grid'
    for j, col in enumerate(frame.columns):
        t.rows[0].cells[j].text = str(col)
    for _, row in frame.iterrows():
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = '' if pd.isna(val) else str(val)
    doc.add_paragraph(note).runs[0].italic = True
for path, caption in fig_paths:
    doc.add_picture(str(path), width=Inches(4.0))
    doc.add_paragraph(caption)
doc.save(OUT / 'Supplement_R2_tables_figures.docx')
print('tables:', [n for n, *_ in tables])
print('figures:', [p.name for p, _ in fig_paths])
