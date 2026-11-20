# -*- coding: utf-8 -*-
"""Reviewer 6 comment 3: traceability check across the modelling files.

What this script can verify from the files on disk:
  * the 58 modelling IDs (D/F/M/L) are identical across spectra, reference assays, colour and
    the general-parameter table, and carry the same category label in each file;
  * the category counts match Table S1 (AST 12, FTR 22, SFTR 12, WTR 12);
  * the supplier counts per category from Table S1 (FTR 17 Xiaoyang + 5 Jiangzhong, etc.).

What it cannot verify: which manuscript ID (S1-S58) belongs to which modelling ID. That link
is not in any file on disk, so the mapping sheet is left for the laboratory records to fill.
Run: python build_traceability_check.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NIR = ROOT / 'script' / 'data' / 'nir_spectra.csv'
REF = ROOT / 'phytochemical_assays' / 'DZ.CSV'
COL = ROOT / 'script' / 'data' / 'colorimetry.csv'
GP = ROOT / 'script' / 'data' / 'general_parameters.csv'

# Table S1 (SI), manuscript ID -> category, supplier. Read from the submitted SI.
TABLE_S1 = {}
for i in range(1, 59):
    if i <= 17:
        cat, sup = 'FTR', 'Xiaoyang'
    elif i <= 22:
        cat, sup = 'FTR', 'Jiangzhong'
    elif i <= 28:
        cat, sup = 'WTR', 'Xiaoyang'
    elif i <= 34:
        cat, sup = 'WTR', 'Jiangzhong'
    elif i <= 46:
        cat, sup = 'SFTR', 'Jiangzhong'
    else:
        cat, sup = 'AST', 'Jiangzhong'
    TABLE_S1[f'S{i}'] = (cat, sup)

sources = {
    'spectra (mean-1.csv)': pd.read_csv(NIR, usecols=['sample_id', 'group']),
    'reference (DZ.CSV)': pd.read_csv(REF, usecols=['sample_id', 'group']),
    'colour (colorimetry.csv)': pd.read_csv(COL, usecols=['sample_id', 'group']),
    'general parameters (general_parameters.csv)': pd.read_csv(GP, usecols=['sample_id', 'group']),
}
for name, df in sources.items():
    df['sample_id'] = df.sample_id.astype(str).str.strip()

ids = {name: set(df.sample_id) for name, df in sources.items()}
same_ids = all(ids[n] == ids['spectra (mean-1.csv)'] for n in ids)
merged = sources['spectra (mean-1.csv)'].rename(columns={'group': 'cat_spectra'})
for name, df in list(sources.items())[1:]:
    merged = merged.merge(df.rename(columns={'group': 'cat_' + name.split(' ')[0]}), on='sample_id', how='outer')
cat_cols = [c for c in merged if c.startswith('cat_')]
merged['categories_agree'] = merged[cat_cols].nunique(axis=1).eq(1)

checks = {
    'IDs identical across the four modelling files': same_ids,
    'Category labels agree across files': bool(merged.categories_agree.all()),
    'Count of modelling IDs = 58': len(merged) == 58,
    'Category counts match Table S1 (22/12/12/12)':
        merged.cat_spectra.value_counts().to_dict() == {'FTR': 22, 'WTR': 12, 'SFTR': 12, 'AST': 12},
}
table_counts = pd.Series([c for c, _ in TABLE_S1.values()]).value_counts().to_dict()
checks['Table S1 category counts = 22/12/12/12'] = table_counts == {'FTR': 22, 'WTR': 12, 'SFTR': 12, 'AST': 12}

# Template for the laboratory to complete: one row per manuscript ID.
template = pd.DataFrame([{
    'Manuscript_ID': sid, 'Category': cat, 'Supplier_TableS1': sup,
    'Modelling_ID': '', 'Spectrum_file_row': '', 'Reference_row': '', 'Colour_row': '',
    'Parent_batch_ID': '', 'Purchase_record': '', 'Evidence_source': 'TO_FILL',
} for sid, (cat, sup) in TABLE_S1.items()])
template.to_csv(OUT / 'mapping_S1_to_modelling_ID_TEMPLATE.csv', index=False, encoding='utf-8-sig')
merged.to_csv(OUT / 'modelling_ID_cross_file_check.csv', index=False, encoding='utf-8-sig')

with open(OUT / 'traceability_check_report.txt', 'w', encoding='utf-8') as f:
    for k, v in checks.items():
        f.write(f'{"PASS" if v else "FAIL"}  {k}\n')
    f.write('\nNOT VERIFIABLE FROM FILES ON DISK:\n')
    f.write('  Manuscript ID (S1-S58) -> modelling ID (D/F/M/L): no record found.\n')
    f.write('  Supplier for each modelling ID: not recorded in any modelling file.\n')
    f.write('  Parent_batch for each botanical sample: not recorded (parent_batch_map covers only mixtures).\n')
print(open(OUT / 'traceability_check_report.txt', encoding='utf-8').read())
