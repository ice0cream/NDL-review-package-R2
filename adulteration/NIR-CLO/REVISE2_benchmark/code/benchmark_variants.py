# -*- coding: utf-8 -*-
"""Revision-2 unified modality benchmark (Reviewer 6, comments 27/28/33/39).

Every variant reuses the locked protocol of decision_nested_83.py unchanged:
same 83 positive mixtures (or 105 states), same parent-to-outer-fold plan,
same 3-fold grouped inner selection, same estimator settings, same clipping.
Only the modelling sources differ:

  fusion_stack      4 training-selected NIR windows + CIELAB, stacked (published primary model)
  nir_stack         4 training-selected NIR windows, stacked, NO colour branch (ablation)
  nir_full          full preprocessed spectrum, single base model (PLSR/SVR/RF)
  color_only        L*, a*, b* only, single base model (PLSR/SVR/RF)
  feature_fusion    4 NIR windows + CIELAB concatenated, block-scaled, single base model

Single-source variants have no meta-learner (Equal_average over one source is the
base prediction itself), so they select only preprocessing and base algorithm.
Classification tasks (Task2_binary: authentic vs adulterated; Task1_4class: authentic/low/medium/high)
use the same sources with the core classifiers (PLS-DA, SVM, RF, LDA) and selection by macro-F1.
Run: python benchmark_variants.py --variant nir_stack --task quantitative_positive83 --output <dir>
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

CORE = Path(__file__).resolve().parents[2] / 'R1_primary_analysis_20260911' / 'code' / 'decision_nested_83.py'
spec = importlib.util.spec_from_file_location('core', CORE)
core = importlib.util.module_from_spec(spec)
sys.modules['core'] = core
spec.loader.exec_module(core)

VARIANTS = ['fusion_stack', 'nir_stack', 'nir_full', 'color_only', 'feature_fusion']
SINGLE_SOURCE = {'nir_full', 'color_only', 'feature_fusion'}
VARIANT = {'name': 'fusion_stack'}


def n_sources():
    return {'fusion_stack': 5, 'nir_stack': 4}.get(VARIANT['name'], 1)


def block_scale(train, valid):
    """Autoscale each block on training rows, then weight by 1/sqrt(block size)."""
    out_train, out_valid = [], []
    for a, b in zip(train, valid):
        mean, sd = a.mean(axis=0), a.std(axis=0, ddof=1)
        sd = np.where(sd < 1e-12, 1.0, sd)
        weight = 1.0 / np.sqrt(a.shape[1])
        out_train.append((a - mean) / sd * weight)
        out_valid.append((b - mean) / sd * weight)
    return [np.hstack(out_train)], [np.hstack(out_valid)]


original_source_matrices = core.source_matrices
original_model_factory = core.model_factory


def source_matrices(Xtr, Xva, ytr, groups, wns, prep, classification, config):
    name = VARIANT['name']
    p = len(wns)
    if name == 'color_only':
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        return [imputer.fit_transform(Xtr[:, p:])], [imputer.transform(Xva[:, p:])], []
    if name == 'nir_full':
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        nir_train = imputer.fit_transform(Xtr[:, :p])
        nir_valid = imputer.transform(Xva[:, :p])
        transform = core.NIRPreprocessBandTransformer(
            wns, wns.max(), wns.min(), pathlength=prep['Pathlength'],
            spectra=prep['Spectra'], smoothing=prep['Smoothing'])
        return [transform.fit_transform(nir_train, ytr)], [transform.transform(nir_valid)], []
    train, valid, bands = original_source_matrices(Xtr, Xva, ytr, groups, wns, prep, classification, config)
    if name == 'nir_stack':
        return train[:4], valid[:4], bands
    if name == 'feature_fusion':
        train, valid = block_scale(train, valid)
        return train, valid, bands
    return train, valid, bands


def model_factory(name, n_features, n_train, classification, config):
    model = original_model_factory(name, n_features, n_train, classification, config)
    # Block-scaled fusion must not be re-autoscaled, or the block weights vanish.
    if VARIANT['name'] == 'feature_fusion' and hasattr(model, 'steps'):
        return model.steps[-1][1]
    return model


def candidate_names(classes):
    bases, metas = original_candidate_names(classes)
    if VARIANT['name'] in SINGLE_SOURCE:
        return bases, ['Equal_average']
    return bases, metas


original_candidate_names = core.candidate_names


def evaluate_preprocessing(X, y, groups, wns, classes, prep, plan, config):
    """Same as core.evaluate_preprocessing, with the source count set by the variant."""
    k = n_sources()
    bases, metas = candidate_names(classes)
    needs_stack = VARIANT['name'] not in SINGLE_SOURCE
    classification = classes is not None
    shape = (len(y), len(classes)) if classification else (len(y),)
    predictions = {(base, meta): np.full(shape, np.nan) for base in bases for meta in metas}
    failures, band_records = {}, []
    for i, (tr, va, stack) in enumerate(plan, 1):
        outer_matrices = source_matrices(X[tr], X[va], y[tr], groups[tr], wns, prep, classification, config)
        stack_matrices = []
        for j, (a, b) in enumerate(stack if needs_stack else [], 1):
            matrices = source_matrices(X[tr[a]], X[tr[b]], y[tr[a]], groups[tr[a]], wns, prep, classification, config)
            stack_matrices.append((a, b, matrices))
            for rank, (high, low, association) in enumerate(matrices[2], 1):
                band_records.append({'Inner_fold': i, 'Stack_fold': j, 'Rank': rank,
                    'High_cm-1': high, 'Low_cm-1': low, 'Training_association': association})
        for base in bases:
            try:
                oof = np.full((len(tr), k, len(classes)) if classification else (len(tr), k), np.nan)
                if needs_stack:
                    for a, b, matrices in stack_matrices:
                        oof[b], _ = core.predict_sources(matrices[0], matrices[1], y[tr[a]], base, classes, config)
                    if not np.isfinite(oof).all():
                        raise ValueError('Incomplete base OOF coverage.')
                valid, _ = core.predict_sources(outer_matrices[0], outer_matrices[1], y[tr], base, classes, config)
                for meta in metas:
                    try:
                        predictions[(base, meta)][va], _ = core.fit_meta_predict(oof, y[tr], valid, meta, classes, config)
                    except Exception as exc:
                        failures[(base, meta)] = f'{type(exc).__name__}: {exc}'
            except Exception as exc:
                for meta in metas:
                    failures[(base, meta)] = f'{type(exc).__name__}: {exc}'
    records = []
    for (base, meta), pred in predictions.items():
        row = {'Number': int(prep['Number']), 'Pathlength': prep['Pathlength'],
               'Spectra': prep['Spectra'], 'Smoothing': prep['Smoothing'],
               'Base_model': base, 'Fusion_type': meta, 'error': failures.get((base, meta), '')}
        if not row['error'] and np.isfinite(pred).all():
            row.update(core.classification_metrics(y, classes[pred.argmax(axis=1)], classes)
                       if classification else core.regression_metrics(y, pred))
        elif not row['error']:
            row['error'] = 'Incomplete inner validation predictions.'
        records.append(row)
    return {'candidates': records, 'training_band_selections': band_records}


def fit_selected_stack(X, y, groups, test_X, wns, classes, selection, stack, config):
    k = n_sources()
    classification = classes is not None
    oof = np.full((len(y), k, len(classes)) if classification else (len(y), k), np.nan)
    if VARIANT['name'] not in SINGLE_SOURCE:
        for tr, va in stack:
            a, b, _ = source_matrices(X[tr], X[va], y[tr], groups[tr], wns, selection, classification, config)
            oof[va], _ = core.predict_sources(a, b, y[tr], selection['Base_model'], classes, config)
        if not np.isfinite(oof).all():
            raise AssertionError('Outer training base OOF is incomplete.')
    a, b, bands = source_matrices(X, test_X, y, groups, wns, selection, classification, config)
    test, calibration = core.predict_sources(a, b, y, selection['Base_model'], classes, config, calibration=True)
    combined, weights = core.fit_meta_predict(oof, y, np.concatenate([test, calibration]),
                                              selection['Fusion_type'], classes, config)
    return combined[:len(test_X)], combined[len(test_X):], bands, weights


def build_manifest_for_variant(original):
    def manifest():
        frame = original()
        # CIELAB input ignores spectral preprocessing; one entry avoids 24 identical searches.
        return frame.head(1) if VARIANT['name'] == 'color_only' else frame
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--task', choices=[*core.REGRESSION_TASKS, *core.CLASS_ORDERS], default='quantitative_positive83')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--group-map', type=Path, required=True)
    parser.add_argument('--outer-plan', type=Path, help='Fixed parent-to-fold plan; omit to draw --repeats random grouped partitions.')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    VARIANT['name'] = args.variant
    core.source_matrices = source_matrices
    core.model_factory = model_factory
    core.candidate_names = candidate_names
    core.evaluate_preprocessing = evaluate_preprocessing
    core.fit_selected_stack = fit_selected_stack
    core.build_preprocessing_manifest = build_manifest_for_variant(core.build_preprocessing_manifest)
    core.SOURCE_NAMES = {'fusion_stack': core.SOURCE_NAMES, 'nir_stack': core.SOURCE_NAMES[:4]}.get(
        args.variant, [args.variant])

    config = core.Config(workers=args.workers, repeats=args.repeats)
    data, wn_cols, wns, group_source = core.read_input(args.input, args.group_map)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    core.write_json(out / 'run_manifest.json', {
        'variant': args.variant, 'task': args.task, 'config': asdict(config),
        'input': str(args.input.resolve()), 'group_source': group_source,
        'outer_plan': str(args.outer_plan.resolve()) if args.outer_plan else f'random grouped partitions, seeds {config.seed}+repeat*10000',
        'core_script': str(CORE), 'started': time.strftime('%Y-%m-%dT%H:%M:%S')})
    with threadpool_limits(limits=1):
        metrics = core.run_task(data, wn_cols, wns, args.task, out / args.task, config, outer_plan=args.outer_plan)
    metrics.insert(0, 'Variant', args.variant)
    core.save_csv(metrics, out / 'metrics_summary.csv')
    print(f'Completed {args.variant} {args.task}: {out}', flush=True)


if __name__ == '__main__':
    main()
