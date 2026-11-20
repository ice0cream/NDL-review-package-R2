# -*- coding: utf-8 -*-
"""Re-assign each spectrum's step (S0-S4) from its measurement time and build a cleaned copy.

Rule: within one parent, files are sorted by time and split into clusters wherever the gap
between consecutive files exceeds GAP_MIN minutes; the k-th cluster is step S(k-1).
A parent with a number of clusters other than 5 is reported and its clusters are not renamed.
Files removed by decision (REMOVE, with the reason) are excluded before clustering.
The raw folder repeated_experiment_raw_spectra is never changed: files are copied to repeated_experiment_cleaned_spectra under corrected names,
replicates renumbered 1..n by time within each state. Every file gets a row in correction_log.csv.
"""
from pathlib import Path
import re
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'repeated_experiment_raw_spectra'
CLEAN = ROOT / 'repeated_experiment_cleaned_spectra'
OUT = Path(__file__).resolve().parent
GAP_MIN = 10
PAT = re.compile(r'^F1_(?P<label>.+?)-(?P<rep>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.001$')
REMOVE = {  # removed files and reason
    'F1_F3-S0-1_20261106_105246.001': 'decision: F3-S0 keep evening set only',
    'F1_F3-S0-2_20261106_105349.001': 'decision: F3-S0 keep evening set only',
    'F1_F3-S0-3_20261106_105601.001': 'decision: F3-S0 keep evening set only',
    'F1_F3-S0-4_20261106_105701.001': 'decision: F3-S0 keep evening set only',
    'F1_F6-S1-1_20261107_105326.001': 'decision: later F6-S1 set removed',
    'F1_F6-S1-2_20261107_105419.001': 'decision: later F6-S1 set removed',
}


def main():
    rows = []
    for f in sorted(RAW.iterdir()):
        m = PAT.match(f.name)
        label = m['label']
        parent = 'MSDF' if label.startswith('MSDF') else re.match(r'F\d+', label).group()
        step = re.search(r'-S(\d)$', label)
        rows.append({'file': f.name, 'parent': parent, 'label': label, 'orig_step': int(step[1]) if step else None,
                     'orig_rep': int(m['rep']), 'stamp': m['date'] + '_' + m['time'],
                     'time': pd.to_datetime(m['date'] + m['time'], format='%Y%m%d%H%M%S')})
    df = pd.DataFrame(rows)
    df['action'] = ''
    df['note'] = ''
    df.loc[df.file.isin(REMOVE), 'action'] = 'removed'
    df.loc[df.file.isin(REMOVE), 'note'] = df.file.map(REMOVE)

    df['new_step'] = pd.NA
    problems = []
    for p, g in df[(df.parent != 'MSDF') & (df.action != 'removed')].groupby('parent'):
        g = g.sort_values('time')
        cluster = (g.time.diff().dt.total_seconds().div(60).gt(GAP_MIN)).cumsum()
        if cluster.max() != 4:
            spans = [f'{c}: {t.min():%m-%d %H:%M}-{t.max():%H:%M} labels {sorted(set(lab))}'
                     for c, (t, lab) in g.groupby(cluster).apply(lambda x: (x.time, x.label)).items()]
            problems.append(f'{p}: {cluster.max() + 1} time clusters -> ' + '; '.join(spans))
            df.loc[g.index, 'new_step'] = g.orig_step
            df.loc[g.index, 'note'] = 'cluster count != 5, label kept'
            continue
        df.loc[g.index, 'new_step'] = cluster.to_numpy()
    st = df[df.parent != 'MSDF']
    changed = st.new_step.notna() & (st.new_step != st.orig_step)
    df.loc[changed[changed].index, 'action'] = 'step relabelled by time'
    df.loc[(df.parent != 'MSDF') & (df.label != df.parent + '-S' + df.orig_step.astype('Int64').astype(str)), 'note'] += ' parent/step label malformed in raw name'

    keep = df[df.action != 'removed'].copy()
    keep['new_label'] = keep.apply(lambda r: 'MSDF' if r.parent == 'MSDF' else f'{r.parent}-S{int(r.new_step)}', axis=1)
    ms = keep[keep.parent == 'MSDF'].sort_values('time')
    session = ms.time.diff().dt.total_seconds().div(60).gt(GAP_MIN).cumsum()
    keep['group'] = keep.new_label
    keep.loc[session.index, 'group'] = 'MSDF_' + session.astype(str)
    # rule (samples and each starch-reference session): 4 readings drop the earliest, 5 drop the two earliest
    drop = []
    for lab, g in keep.sort_values('time').groupby('group'):
        n_drop = {4: 1, 5: 2}.get(len(g), 0)
        if len(g) > 5:
            print(f'{lab}: {len(g)} readings, rule does not cover it, all kept')
        drop += list(g.index[:n_drop])
    df.loc[df.file.isin(keep.loc[drop, 'file']), 'action'] = 'removed'
    df.loc[df.file.isin(keep.loc[drop, 'file']), 'note'] = 'rule: extra readings, earliest removed'
    keep = keep.drop(index=drop)
    keep['new_rep'] = keep.sort_values('time').groupby('group').cumcount() + 1
    keep['new_file'] = 'F1_' + keep.new_label + '-' + keep.new_rep.astype(str) + '_' + keep.stamp + '.001'
    df = df.merge(keep[['file', 'new_label', 'new_rep', 'new_file']], on='file', how='left')
    df.loc[(df.action == '') & (df.new_file != df.file), 'action'] = 'replicate renumbered'

    CLEAN.mkdir(exist_ok=True)
    for r in df[df.new_file.notna()].itertuples():
        shutil.copy2(RAW / r.file, CLEAN / r.new_file)
    df.drop(columns='time').to_csv(OUT / 'correction_log.csv', index=False, encoding='utf-8-sig')
    print('\n'.join(problems) or 'all parents have 5 clusters')
    print(df.action.value_counts().to_string())
    print(df[df.action.isin(['step relabelled by time', 'removed'])][['file', 'new_file', 'action']].to_string(index=False))


if __name__ == '__main__':
    main()
