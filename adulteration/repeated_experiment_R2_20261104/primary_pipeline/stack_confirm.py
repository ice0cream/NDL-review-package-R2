# -*- coding: utf-8 -*-
"""Permutation test and bootstrap-with-refitting for the manuscript's stacking pipeline
(Reviewer 6, comments 37 and 41), on all 110 states of the repeated experiment.

The modelling code is not re-implemented. Each fold calls the same functions, in the same order and with the
same seeds, as core.run_task in decision_nested_83.py (through benchmark_variants.py). The `check` mode
recomputes fold 1 of the saved fixed-plan run and stops unless the predictions agree.

  check   recompute fold 1 of runs/quantitative_with_zero105/fusion_stack and compare
  perm    full-procedure permutation test of fusion_stack: starch contents permuted within each parent,
          the complete procedure (preprocessing, window selection, base learner, meta-learner) repeated on the
          fixed five-fold parent plan. Statistic: RMSE of the 110 out-of-fold predictions.
          rng = default_rng([2026, 21, b]); N_PERM fixed at 99 before the run.
  boot    parent bootstrap with refitting: parents resampled with replacement (copies of a parent share its
          group label, so they stay together in every inner and stacking split); the complete procedure is run
          on the in-bag parents and scored on the out-of-bag parents. Same resamples for fusion_stack and
          nir_stack. rng = default_rng([2026, 22, b]); N_BOOT fixed at 100 before the run.

Results are appended after every fold, so an interrupted run resumes where it stopped.
Run:  python stack_confirm.py check|perm|boot|summary [--workers 14]
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'NIR-CLO' / 'REVISE2_benchmark' / 'code'))
import benchmark_variants as bv  # noqa: E402

core = bv.core
OUT = HERE / 'stack_confirm'
SEED = 2026
N_PERM = 99
N_BOOT = 100
ALL_SOURCES = list(core.SOURCE_NAMES)
ORIGINAL_MANIFEST = core.build_preprocessing_manifest
FIXED_RUN = HERE / 'runs' / 'quantitative_with_zero105' / 'fusion_stack' / 'quantitative_with_zero105' / 'outer_OOF_predictions.csv'


def setup(variant):
    """Same patches as benchmark_variants.main()."""
    bv.VARIANT['name'] = variant
    core.source_matrices = bv.source_matrices
    core.model_factory = bv.model_factory
    core.candidate_names = bv.candidate_names
    core.evaluate_preprocessing = bv.evaluate_preprocessing
    core.fit_selected_stack = bv.fit_selected_stack
    core.build_preprocessing_manifest = bv.build_manifest_for_variant(ORIGINAL_MANIFEST)
    core.SOURCE_NAMES = {'fusion_stack': ALL_SOURCES, 'nir_stack': ALL_SOURCES[:4]}[variant]


def one_fold(X, y, groups, tr, va, wns, config, seed):
    """Body of the fold loop of core.run_task (regression), without file output."""
    plan = core.make_nested_plan(groups[tr], config, seed + 10)
    final_stack = core.grouped_splits(groups[tr], config.stack_folds, seed + 20)
    manifest = core.build_preprocessing_manifest()
    if config.smoke:
        manifest = manifest.head(2)
    records = []
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = [pool.submit(core.evaluate_preprocessing, X[tr], y[tr], groups[tr], wns, None, prep, plan, config)
                   for prep in manifest.to_dict('records')]
        for future in futures:
            records.extend(future.result()['candidates'])
    selection, _ = core.select_candidate(records, None)
    pred, _, _, _ = core.fit_selected_stack(X[tr], y[tr], groups[tr], X[va], wns, None, selection, final_stack, config)
    return np.asarray(pred, float), selection


def load():
    data, wn_cols, wns, _ = core.read_input(HERE / 'inputs' / 'NIR_color_merged_dataset_redo110.csv',
                                            HERE / 'inputs' / 'parent_map_redo.csv')
    df = data.reset_index(drop=True)  # task quantitative_with_zero105: all states
    X = df[wn_cols + core.COLOR_COLS].to_numpy(float)
    return df, X, df.adulteration_percent.to_numpy(float), df.group_id.to_numpy(str), wns


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def label(sel):
    return 'N{:02d} {} + {}'.format(int(sel['Number']), sel['Base_model'], sel['Fusion_type'])


def append(path, rows):
    pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False, encoding='utf-8')


def check(config):
    df, X, y, groups, wns = load()
    setup('fusion_stack')
    outer = core.load_outer_plan(groups, HERE / 'inputs' / 'outer_fold_plan_redo.csv')
    tr, va = outer[0]
    pred, _ = one_fold(X, y, groups, tr, va, wns, config, config.seed + 1 * 10000 + 1 * 1000)
    saved = pd.read_csv(FIXED_RUN)
    saved = saved[saved.Outer_fold == 1].set_index('sample_state_id').Predicted
    dev = float(np.max(np.abs(pred - saved.loc[df.sample_state_id.iloc[va]].to_numpy())))
    print('check: max |driver - saved pipeline prediction| on fold 1 = {:.2e}'.format(dev), flush=True)
    if dev > 1e-6:
        raise SystemExit('Driver does not reproduce the saved pipeline run; stopping.')
    (OUT / 'driver_check.json').write_text(json.dumps({'max_abs_deviation_fold1': dev, 'n_states': int(len(va))}),
                                           encoding='utf-8')


def perm(config):
    df, X, y, groups, wns = load()
    setup('fusion_stack')
    outer = core.load_outer_plan(groups, HERE / 'inputs' / 'outer_fold_plan_redo.csv')
    path = OUT / 'permutation_fold_predictions.csv'
    done = set(map(tuple, pd.read_csv(path)[['perm', 'fold']].drop_duplicates().to_numpy())) if path.exists() else set()
    for b in range(1, N_PERM + 1):
        rng = np.random.default_rng([SEED, 21, b])
        yp = y.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            yp[idx] = y[rng.permutation(idx)]
        for fold, (tr, va) in enumerate(outer, 1):
            if (b, fold) in done:
                continue
            start = time.monotonic()
            pred, sel = one_fold(X, yp, groups, tr, va, wns, config, config.seed + 1 * 10000 + fold * 1000)
            append(path, [{'perm': b, 'fold': fold, 'sample_state_id': df.sample_state_id.iloc[i], 'permuted_y': yp[i],
                           'predicted': p} for i, p in zip(va, pred)])
            print('perm {}/{} fold {}: {:.0f}s  {}'.format(b, N_PERM, fold, time.monotonic() - start, label(sel)), flush=True)


def boot(config):
    df, X, y, groups, wns = load()
    parents = np.array(sorted(np.unique(groups), key=lambda p: int(p[1:])))
    rows_of = {p: np.flatnonzero(groups == p) for p in parents}
    path = OUT / 'bootstrap_oob_predictions.csv'
    done = set(map(tuple, pd.read_csv(path)[['boot', 'variant']].drop_duplicates().to_numpy())) if path.exists() else set()
    for b in range(1, N_BOOT + 1):
        rng = np.random.default_rng([SEED, 22, b])
        drawn = rng.choice(parents, size=len(parents), replace=True)
        oob = [p for p in parents if p not in set(drawn)]
        tr = np.concatenate([rows_of[p] for p in drawn])  # copies of a parent keep its group label
        va = np.concatenate([rows_of[p] for p in oob])
        rows = np.concatenate([tr, va])
        tr_pos, va_pos = np.arange(len(tr)), np.arange(len(tr), len(rows))
        for variant in ('fusion_stack', 'nir_stack'):
            if (b, variant) in done:
                continue
            setup(variant)
            start = time.monotonic()
            # seed pattern of run_task with repeat = 100 + b, fold = 1; identical for both variants
            pred, sel = one_fold(X[rows], y[rows], groups[rows], tr_pos, va_pos, wns, config,
                                 config.seed + (100 + b) * 10000 + 1000)
            append(path, [{'boot': b, 'variant': variant, 'n_inbag_parents': len(set(drawn)), 'n_oob_parents': len(oob),
                           'sample_state_id': df.sample_state_id.iloc[i], 'observed': y[i], 'predicted': p,
                           'selected': label(sel)} for i, p in zip(va, pred)])
            print('boot {}/{} {}: {:.0f}s  oob parents {}  RMSE {:.2f}'.format(
                b, N_BOOT, variant, time.monotonic() - start, len(oob), rmse(pred - y[va])), flush=True)


def summary():
    out = {}
    path = OUT / 'permutation_fold_predictions.csv'
    if path.exists():
        p = pd.read_csv(path)
        full = p.groupby('perm').filter(lambda s: s.fold.nunique() == 5 and len(s) == 110)
        rows = []
        for b, s in full.groupby('perm'):
            e = s.predicted - s.permuted_y
            rows.append({'perm': b, 'RMSE': rmse(e),
                         'R2': float(1 - np.sum(e ** 2) / np.sum((s.permuted_y - s.permuted_y.mean()) ** 2))})
        null = pd.DataFrame(rows)
        obs = pd.read_csv(FIXED_RUN)
        obs_rmse = rmse(obs.Predicted - obs.Observed)
        null.to_csv(OUT / 'permutation_null.csv', index=False, encoding='utf-8-sig')
        out['permutation'] = {'n_completed': int(len(null)), 'n_planned': N_PERM, 'observed_RMSE': obs_rmse,
                              'null_RMSE_min': float(null.RMSE.min()), 'null_RMSE_median': float(null.RMSE.median()),
                              'null_R2_max': float(null.R2.max()),
                              'p_value': float((1 + (null.RMSE <= obs_rmse).sum()) / (1 + len(null)))}
    path = OUT / 'bootstrap_oob_predictions.csv'
    if path.exists():
        p = pd.read_csv(path)
        p['sq'] = (p.predicted - p.observed) ** 2
        per = np.sqrt(p.groupby(['boot', 'variant']).sq.mean()).unstack('variant').dropna()
        per['difference'] = per.fusion_stack - per.nir_stack
        per.to_csv(OUT / 'bootstrap_refit_resamples.csv', encoding='utf-8-sig')
        both = p[p.boot.isin(per.index)]
        out['bootstrap'] = {'n_completed': int(len(per)), 'n_planned': N_BOOT}
        for col in ('fusion_stack', 'nir_stack', 'difference'):
            lo, med, hi = np.percentile(per[col], [2.5, 50, 97.5])
            out['bootstrap'][col] = {'median': float(med), 'p2.5': float(lo), 'p97.5': float(hi)}
        for variant in ('fusion_stack', 'nir_stack'):
            out['bootstrap'][variant]['pooled_oob_RMSE'] = float(np.sqrt(both[both.variant == variant].sq.mean()))
        out['bootstrap']['share_fusion_lower'] = float((per.difference < 0).mean())
        both.groupby(['variant', 'selected']).boot.nunique().rename('n_resamples').reset_index().to_csv(
            OUT / 'bootstrap_refit_selection_frequency.csv', index=False, encoding='utf-8-sig')
    (OUT / 'stack_confirm_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'perm', 'boot', 'summary'])
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--smoke', action='store_true', help='code-path test only: 2 preprocessings, 8 trees, 1 resample')
    args = parser.parse_args()
    config = core.Config(workers=args.workers, repeats=1)
    if args.smoke:
        OUT = HERE / 'stack_confirm_smoke'
        N_PERM = N_BOOT = 1
        config = core.Config(workers=args.workers, repeats=1, trees=8, smoke=True)
    OUT.mkdir(exist_ok=True)
    if args.mode == 'summary':
        summary()
    else:
        with threadpool_limits(limits=1):
            {'check': check, 'perm': perm, 'boot': boot}[args.mode](config)
