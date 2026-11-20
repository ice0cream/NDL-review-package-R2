# -*- coding: utf-8 -*-
"""Reviewer 6, comments 11, 15 and 44: acquisition metadata read from the raw OPUS files (read-only).

For every raw file of the botanical campaign and of the repeated adulteration experiment: number of scans, resolution,
scan duration, measurement channel, reference channel, and the date and time of the reference (background) spectrum that
the instrument stored with the sample spectrum. From these: how often the reference was renewed, the time needed per
reading, and (repeated experiment only, post hoc) whether the age of the reference is associated with the prediction
residual of a state.

Outputs: R2_revision/R6-11_15_acquisition_metadata/
"""
import datetime as dt
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from brukeropusreader import read_file
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'R2_revision' / 'R6-11_15_acquisition_metadata'
OUT.mkdir(exist_ok=True)
REDO = ROOT / 'adulteration' / 'repeated_experiment_R2_20261104'
SETS = {'botanical': ROOT / 'NIR' / 'botanical_samples' / 'DATA', 'repeated_adulteration': ROOT / 'repeated_experiment_raw_spectra'}


def ts(d, t):
    return dt.datetime.strptime(d + ' ' + t.split(' ')[0], '%d/%m/%Y %H:%M:%S.%f')


def read_set(folder):
    rows = []
    for p in sorted(folder.iterdir()):
        if not (p.is_file() and p.suffix[1:].isdigit()):
            continue
        o = read_file(str(p))
        g = lambda blk, key: (o.get(blk) or {}).get(key)
        rows.append({'file': p.name,
                     'sample_time': ts(g('AB Data Parameter', 'DAT'), g('AB Data Parameter', 'TIM')),
                     'reference_time': ts(g('ScRf Data Parameter', 'DAT'), g('ScRf Data Parameter', 'TIM')),
                     'scans': g('Acquisition', 'NSS'), 'reference_scans': g('Acquisition (Rf)', 'NSR'),
                     'resolution_cm-1': g('Acquisition', 'RES'), 'scan_duration_s': g('Instrument', 'DUR'),
                     'zero_filling': g('Fourier Transformation', 'ZFF'), 'apodization': g('Fourier Transformation', 'APF'),
                     'measurement_channel': g('Optik', 'CHN'), 'reference_channel': g('Optik (Rf)', 'RCH'),
                     'detector': g('Optik', 'DTC'), 'instrument': g('Instrument', 'INS'),
                     'result_block': g('Acquisition', 'PLF'), 'first_wavenumber': g('AB Data Parameter', 'FXV'),
                     'last_wavenumber': g('AB Data Parameter', 'LXV'), 'points': g('AB Data Parameter', 'NPT')})
    d = pd.DataFrame(rows)
    d['reference_age_s'] = (d.sample_time - d.reference_time).dt.total_seconds()
    return d


summary = {}
for name, folder in SETS.items():
    d = read_set(folder)
    d.to_csv(OUT / 'opus_metadata_{}.csv'.format(name), index=False, encoding='utf-8-sig')
    t = d.sample_time.sort_values()
    gaps = t.diff().dt.total_seconds().dropna()
    per_ref = d.groupby('reference_time').size()
    s = {'files': int(len(d)),
         'dates': {str(k.date()): int(v) for k, v in d.sample_time.dt.normalize().value_counts().sort_index().items()},
         'scans': sorted(int(v) for v in d.scans.unique()), 'reference_scans': sorted(int(v) for v in d.reference_scans.unique()),
         'resolution_cm-1': sorted(float(v) for v in d['resolution_cm-1'].unique()),
         'zero_filling': sorted(d.zero_filling.astype(str).unique()), 'apodization': sorted(d.apodization.unique()),
         'point_spacing_cm-1': float(((d.first_wavenumber - d.last_wavenumber) / (d.points - 1)).median()),
         'measurement_channel': sorted(d.measurement_channel.unique()), 'reference_channel': sorted(d.reference_channel.unique()),
         'result_block': sorted(d.result_block.unique()),
         'scan_duration_s': [float(d.scan_duration_s.min()), float(d.scan_duration_s.median()), float(d.scan_duration_s.max())],
         'distinct_references': int(d.reference_time.nunique()),
         'readings_per_reference_min_median_max': [int(per_ref.min()), float(per_ref.median()), int(per_ref.max())],
         'reference_age_s_min_median_p90_max': [float(d.reference_age_s.min()), float(d.reference_age_s.median()),
                                                float(d.reference_age_s.quantile(0.9)), float(d.reference_age_s.max())],
         'gap_between_consecutive_readings_s_median_(gaps<10min)': float(gaps[gaps < 600].median())}
    summary[name] = s

# ---- repeated experiment: reference age against residuals (post hoc)
d = pd.read_csv(OUT / 'opus_metadata_repeated_adulteration.csv', encoding='utf-8-sig')
inv = pd.read_csv(REDO / 'data_checks' / 'file_inventory.csv', encoding='utf-8-sig')
log = pd.read_csv(REDO / 'data_checks' / 'correction_log.csv', encoding='utf-8-sig')
st = pd.read_csv(REDO / 'supporting_analysis' / 'results' / 'state_predictions.csv', encoding='utf-8-sig')
rep = pd.read_csv(REDO / 'supporting_analysis' / 'results' / 'replicate_predictions_M1.csv', encoding='utf-8-sig')
back = dict(zip(log.new_file.dropna(), log.loc[log.new_file.notna(), 'file']))      # renamed file -> raw file name
rep['raw_file'] = rep.file.map(lambda f: back.get(f, f))
rep = rep.merge(d[['file', 'reference_age_s', 'sample_time']], left_on='raw_file', right_on='file', how='left', suffixes=('', '_raw'))
assert rep.reference_age_s.notna().all(), rep[rep.reference_age_s.isna()].file.tolist()[:5]
rep = rep.merge(st[['label', 'conc_recomputed', 'batch', 'nominal', 'parent']], on='label', validate='many_to_one')
rep['res_reading_M1'] = rep.pred - rep.conc_recomputed
state = rep.groupby('label').agg(reference_age_s=('reference_age_s', 'mean'), batch=('batch', 'first'),
                                 nominal=('nominal', 'first'), parent=('parent', 'first')).reset_index()
oof = pd.read_csv(REDO / 'primary_pipeline' / 'runs' / 'quantitative_with_zero105' / 'fusion_stack' / 'quantitative_with_zero105' /
                  'outer_OOF_predictions.csv', encoding='utf-8-sig')
oof['label'] = oof.sample_state_id.str.replace(';', '-', regex=False)
state = state.merge(oof[['label', 'Residual']].rename(columns={'Residual': 'res_fusion_stack'}), on='label', validate='one_to_one')
state = state.merge(st[['label', 'res_M1']], on='label', validate='one_to_one')
assert len(state) == 110
state.to_csv(OUT / 'reference_age_by_state.csv', index=False, encoding='utf-8-sig')
rows = []
for name, sub in [('all', state)] + list(state.groupby('batch')):
    for col in ('res_fusion_stack', 'res_M1'):
        centred = sub[col] - sub.groupby('nominal')[col].transform('mean')
        for kind, v in (('signed', sub[col]), ('signed, level-centred', centred), ('absolute', sub[col].abs())):
            r = stats.spearmanr(sub.reference_age_s, v)
            rows.append({'batch': name, 'n_states': len(sub), 'residual': col, 'kind': kind, 'spearman_rho': r.statistic, 'p': r.pvalue})
assoc = pd.DataFrame(rows)
assoc.to_csv(OUT / 'reference_age_vs_residual.csv', index=False, encoding='utf-8-sig')
print(assoc.round(3).to_string())
byb = state.groupby('batch').reference_age_s.describe(percentiles=[.5, .9]).round(0)
byb.to_csv(OUT / 'reference_age_by_batch.csv', encoding='utf-8-sig')
print(byb.to_string())
old = state[state.reference_age_s > 1800]
print('states with mean reference age > 30 min:', len(old), '| RMSE fusion_stack: old %.2f, others %.2f' % (
    np.sqrt((old.res_fusion_stack ** 2).mean()), np.sqrt((state[state.reference_age_s <= 1800].res_fusion_stack ** 2).mean())))
summary['repeated_adulteration']['kept_readings'] = int(len(rep))
summary['repeated_adulteration']['kept_reference_age_s_median_p90_max'] = [float(rep.reference_age_s.median()),
                                                                             float(rep.reference_age_s.quantile(.9)),
                                                                             float(rep.reference_age_s.max())]
a = assoc[(assoc.batch == 'all')].set_index(['residual', 'kind'])
summary['repeated_adulteration']['spearman_reference_age_vs_residual'] = {
    '{}_{}'.format(*k): [float(v.spearman_rho), float(v.p)] for k, v in a.iterrows()}
summary['repeated_adulteration']['states_reference_older_than_30min'] = int(len(old))
summary['repeated_adulteration']['rmse_fusion_stack_reference_older_than_30min'] = float(np.sqrt((old.res_fusion_stack ** 2).mean()))
summary['repeated_adulteration']['rmse_fusion_stack_reference_within_30min'] = float(
    np.sqrt((state[state.reference_age_s <= 1800].res_fusion_stack ** 2).mean()))
bot, redo = summary['botanical'], summary['repeated_adulteration']
b3 = state[state.batch.str.startswith('B3')].reference_age_s.median()
sp = redo['spearman_reference_age_vs_residual']
# values as quoted in the manuscript and the response letter
summary['text'] = {
    'scan_s': round(bot['scan_duration_s'][1], 1), 'bot_readings': bot['files'], 'bot_backgrounds': bot['distinct_references'],
    'bot_background_age_median_s': round(bot['reference_age_s_min_median_p90_max'][1]),
    'bot_cycle_s': round(bot['gap_between_consecutive_readings_s_median_(gaps<10min)']),
    'bot_minutes_per_sample': round(3 * bot['gap_between_consecutive_readings_s_median_(gaps<10min)'] / 60, 1),
    'redo_readings': redo['files'], 'redo_backgrounds': redo['distinct_references'],
    'redo_cycle_s': round(redo['gap_between_consecutive_readings_s_median_(gaps<10min)']),
    'redo_background_age_median_min': round(redo['reference_age_s_min_median_p90_max'][1] / 60, 1),
    'redo_background_age_max_min': round(redo['reference_age_s_min_median_p90_max'][3] / 60),
    'redo_b3_background_age_median_min': round(b3 / 60, 1),
    'redo_b1_background_age_median_min': round(state[state.batch.str.startswith('B1')].reference_age_s.median() / 60, 1),
    'redo_b2_background_age_median_min': round(state[state.batch.str.startswith('B2')].reference_age_s.median() / 60, 1),
    'rho_abs': round(sp['res_fusion_stack_absolute'][0], 2), 'p_abs': round(sp['res_fusion_stack_absolute'][1], 2),
    'rho_centred': round(sp['res_fusion_stack_signed, level-centred'][0], 2),
    'p_centred': round(sp['res_fusion_stack_signed, level-centred'][1], 2),
    'n_states_background_older_30min': redo['states_reference_older_than_30min'],
    'rmse_older_30min': round(redo['rmse_fusion_stack_reference_older_than_30min'], 2),
    'rmse_within_30min': round(redo['rmse_fusion_stack_reference_within_30min'], 2)}
json.dump(summary, io.open(OUT / 'acquisition_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))
