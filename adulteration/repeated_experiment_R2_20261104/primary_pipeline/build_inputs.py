# -*- coding: utf-8 -*-
"""Inputs for running the manuscript's locked adulteration pipeline on the repeated experiment.

Writes (./inputs):
  NIR_color_merged_dataset_redo110.csv  110 states: sample_state_id, adulteration_percent (weighed),
                                        L, a, b (median of 3 readings), wn_* (mean of 3 spectra)
  parent_map_redo.csv                   sample_state_id -> parent_batch_id (the FTR parent)
  outer_fold_plan_redo.csv              parent -> outer fold 1..5 (seed 2026, fixed before any model run)
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'supporting_analysis'))
import redo_analysis as ra  # noqa: E402

OUT = HERE / 'inputs'
SEED = 2026


def main():
    OUT.mkdir(exist_ok=True)
    files, S, axis = ra.load_spectra()
    st, X, C = ra.load_states(files, S)
    ids = st.parent + ';S' + st.step.astype(int).astype(str)
    df = pd.DataFrame({'sample_state_id': ids, 'adulteration_percent': st.conc_recomputed.round(4),
                       'L': C[:, 0], 'a': C[:, 1], 'b': C[:, 2]})
    spec = pd.DataFrame(X, columns=[f'wn_{w:.4f}' for w in axis])
    pd.concat([df, spec], axis=1).to_csv(OUT / 'NIR_color_merged_dataset_redo110.csv', index=False)
    pd.DataFrame({'sample_state_id': ids, 'parent_batch_id': st.parent}).to_csv(OUT / 'parent_map_redo.csv', index=False)
    parents = np.array(sorted(st.parent.unique(), key=lambda p: int(p[1:])))
    perm = np.random.default_rng(SEED).permutation(parents)
    plan = pd.DataFrame({'group_id': perm, 'Outer_fold': np.arange(len(perm)) % 5 + 1}).sort_values('Outer_fold')
    plan.to_csv(OUT / 'outer_fold_plan_redo.csv', index=False)
    print(df.adulteration_percent.describe(), plan.groupby('Outer_fold').group_id.apply(list), sep='\n')


if __name__ == '__main__':
    main()
