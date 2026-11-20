# -*- coding: utf-8 -*-
"""Grouped nested validation of decision fusion, revision 2026-09-07.

Primary analysis: all preprocessing/base/fusion selection occurs inside each
outer training partition. Inner validation evaluates COMPLETE stacks; the
base OOF training features are regenerated within each inner training subset.
No KS prediction-set leaderboard and no meta-training resubstitution as CV.

Default bands are selected from a uniform full-spectrum window grid using
training labels only. --band-mode fixed is a historical/exploratory analysis.
Outputs are internal validation, not a new independent external test.
Run: python decision_layer_fusion_same_logic_24prep.py
Use --help for smoke testing, resuming, and regenerating figures.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import warnings
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy
from scipy.signal import savgol_filter
from scipy.special import softmax
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 2026
SG_WINDOW_LENGTH = 9
SG_POLYORDER = 3
NORRIS_SEGMENT_LENGTH = 5
NORRIS_GAP = 5
COLOR_COLS = ['L', 'a', 'b']
BANDS = {
    'B1': (6000, 5600), 'B2': (7152, 7057),
    'B3': (6851, 6666), 'B4': (4250, 4100),
}
SOURCE_NAMES = ['Band_rank_1', 'Band_rank_2', 'Band_rank_3', 'Band_rank_4', 'COLOR_Lab']
CLASS_ORDERS = {
    'Task1_4class': ['Authentic', 'Low', 'Medium', 'High'],
    'Task2_binary': ['Authentic', 'Adulterated'],
}

# Quantitative endpoints are deliberately separated so that the primary
# analysis (83 positive mixtures) cannot be confused with the prespecified
# sensitivity analysis that also contains the 22 authentic zero points.
REGRESSION_TASKS = ('quantitative_positive83', 'quantitative_with_zero105')
plt.rcParams.update({'font.family': ['Times New Roman', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 11})

def build_preprocessing_manifest():
    rows = []
    number = 1
    for pathlength in ["Constant", "MSC", "SNV"]:
        for spectra in ["Spectrum", "FD", "SD"]:
            smoothing_list = ["NS", "SG"] if spectra == "Spectrum" else ["NS", "SG", "ND"]
            for smoothing in smoothing_list:
                rows.append({
                    "Number": number,
                    "Pathlength": pathlength,
                    "Spectra": spectra,
                    "Smoothing": smoothing,
                    "SG_window_length": SG_WINDOW_LENGTH if smoothing == "SG" else np.nan,
                    "SG_polyorder": SG_POLYORDER if smoothing == "SG" else np.nan,
                    "Norris_segment_length": NORRIS_SEGMENT_LENGTH if smoothing == "ND" else np.nan,
                    "Norris_gap": NORRIS_GAP if smoothing == "ND" else np.nan,
                })
                number += 1
    return pd.DataFrame(rows)

def prep_name(row):
    return f"N{int(row['Number']):02d}_{row['Pathlength']}_{row['Spectra']}_{row['Smoothing']}"

class NIRPreprocessBandTransformer(BaseEstimator, TransformerMixin):
    """Preprocess full NIR spectra, then return one fixed band raw variables."""
    def __init__(
        self,
        wavenumbers,
        band_high,
        band_low,
        pathlength="MSC",
        spectra="FD",
        smoothing="SG",
        sg_window_length=9,
        sg_polyorder=3,
        norris_segment_length=5,
        norris_gap=5,
    ):
        self.wavenumbers = np.asarray(wavenumbers, dtype=float)
        self.band_high = float(band_high)
        self.band_low = float(band_low)
        self.pathlength = pathlength
        self.spectra = spectra
        self.smoothing = smoothing
        self.sg_window_length = int(sg_window_length)
        self.sg_polyorder = int(sg_polyorder)
        self.norris_segment_length = int(norris_segment_length)
        self.norris_gap = int(norris_gap)

    def _fill_nan_fit(self, X):
        self.col_median_ = np.nanmedian(X, axis=0)
        self.col_median_ = np.where(np.isfinite(self.col_median_), self.col_median_, 0.0)

    def _fill_nan(self, X):
        X = np.asarray(X, dtype=float).copy()
        inds = np.where(~np.isfinite(X))
        if len(inds[0]) > 0:
            X[inds] = np.take(self.col_median_, inds[1])
        return X

    def _snv(self, X):
        m = np.mean(X, axis=1, keepdims=True)
        s = np.std(X, axis=1, ddof=1, keepdims=True)
        s = np.where(s < 1e-12, 1.0, s)
        return (X - m) / s

    def _msc_fit_reference(self, X):
        self.msc_reference_ = np.mean(X, axis=0)

    def _msc(self, X):
        ref = self.msc_reference_
        out = np.zeros_like(X, dtype=float)
        A = np.vstack([np.ones_like(ref), ref]).T
        for i in range(X.shape[0]):
            xi = X[i]
            coef, _, _, _ = np.linalg.lstsq(A, xi, rcond=None)
            a, b = coef[0], coef[1]
            if abs(b) < 1e-12:
                b = 1.0
            out[i] = (xi - a) / b
        return out

    def _fd(self, X):
        return np.gradient(np.asarray(X, dtype=float), axis=1)

    def _sd(self, X):
        return np.gradient(np.gradient(np.asarray(X, dtype=float), axis=1), axis=1)

    def _sg(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        w = int(self.sg_window_length)
        if w % 2 == 0:
            w += 1
        if w >= n:
            w = n - 1 if n % 2 == 0 else n
        if w <= self.sg_polyorder:
            w = self.sg_polyorder + 2
            if w % 2 == 0:
                w += 1
        if w < 3 or w > n:
            return X
        return savgol_filter(X, window_length=w, polyorder=int(self.sg_polyorder), axis=1, mode="interp")

    def _norris(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        seg = max(1, int(self.norris_segment_length))
        gap = max(1, int(self.norris_gap))
        out = np.zeros_like(X, dtype=float)
        kernel = np.ones(seg) / seg
        moving_average = np.vstack([np.convolve(row, kernel, mode="same") for row in X])
        for i in range(n_features):
            left_start = i - gap - seg + 1
            left_end = i - gap + 1
            right_start = i + gap
            right_end = i + gap + seg
            if left_start >= 0 and right_end <= n_features:
                left_mean = X[:, left_start:left_end].mean(axis=1)
                right_mean = X[:, right_start:right_end].mean(axis=1)
                out[:, i] = (left_mean + right_mean) / 2.0
            else:
                out[:, i] = moving_average[:, i]
        return out

    def _preprocess(self, X):
        Xp = self._fill_nan(X)
        if self.pathlength == "Constant":
            pass
        elif self.pathlength == "MSC":
            Xp = self._msc(Xp)
        elif self.pathlength == "SNV":
            Xp = self._snv(Xp)
        else:
            raise ValueError(f"Unknown pathlength: {self.pathlength}")

        if self.spectra == "Spectrum":
            pass
        elif self.spectra == "FD":
            Xp = self._fd(Xp)
        elif self.spectra == "SD":
            Xp = self._sd(Xp)
        else:
            raise ValueError(f"Unknown spectra transform: {self.spectra}")

        if self.smoothing == "NS":
            pass
        elif self.smoothing == "SG":
            Xp = self._sg(Xp)
        elif self.smoothing == "ND":
            Xp = self._norris(Xp)
        else:
            raise ValueError(f"Unknown smoothing: {self.smoothing}")
        return Xp

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != len(self.wavenumbers):
            raise ValueError(f"X has {X.shape[1]} features, but wavenumbers has {len(self.wavenumbers)}.")
        self._fill_nan_fit(X)
        X_fill = self._fill_nan(X)
        if self.pathlength == "MSC":
            self._msc_fit_reference(X_fill)
        high = max(self.band_high, self.band_low)
        low = min(self.band_high, self.band_low)
        self.band_mask_ = (self.wavenumbers <= high) & (self.wavenumbers >= low)
        if self.band_mask_.sum() < 2:
            raise ValueError(f"Band {high}-{low} cm-1 contains fewer than 2 variables.")
        self.feature_names_out_ = [f"wn_{w:g}" for w in self.wavenumbers[self.band_mask_]]
        return self

    def transform(self, X):
        Xp = self._preprocess(np.asarray(X, dtype=float))
        return Xp[:, self.band_mask_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=5):
        self.n_components = int(n_components)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        Y = np.zeros((len(y_enc), n_classes))
        Y[np.arange(len(y_enc)), y_enc] = 1.0
        n_comp = min(int(self.n_components), X.shape[0] - 1, X.shape[1])
        n_comp = max(1, n_comp)
        self.model_ = PLSRegression(n_components=n_comp, scale=False, max_iter=5000)
        self.model_.fit(X, Y)
        return self

    def predict_proba(self, X):
        Y_hat = self.model_.predict(X)
        Y_hat = Y_hat - np.max(Y_hat, axis=1, keepdims=True)
        exp = np.exp(Y_hat)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(idx)

@dataclass(frozen=True)
class Config:
    outer_folds: int = 5
    inner_folds: int = 3
    stack_folds: int = 3
    repeats: int = 1
    trees: int = 200
    bootstrap: int = 2000
    workers: int = 4
    band_mode: str = 'train_selected'
    band_width: float = 200.0
    band_step: float = 100.0
    seed: int = RANDOM_STATE
    smoke: bool = False


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item()), encoding='utf-8')
    temp.replace(path)


def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, index=False, encoding='utf-8-sig')
    temp.replace(path)


def read_input(path, group_map=None):
    data = pd.read_csv(path)
    data.columns = data.columns.astype(str).str.strip()
    data = data.rename(columns={'L*': 'L', 'a*': 'a', 'b*': 'b'})
    required = ['sample_state_id', 'adulteration_percent'] + COLOR_COLS
    if any(c not in data for c in required):
        raise ValueError(f'Required columns: {required}')
    if data.sample_state_id.isna().any():
        raise ValueError('Missing sample IDs; fix the input before validation.')
    data['sample_state_id'] = data.sample_state_id.astype(str).str.strip()
    if data.sample_state_id.eq('').any() or data.sample_state_id.duplicated().any():
        raise ValueError('Sample IDs must be nonempty and unique; average technical readings first.')
    wn_cols = [c for c in data if c.startswith('wn_')]
    wn_cols.sort(key=lambda c: float(c[3:]), reverse=True)
    if len(wn_cols) < 20:
        raise ValueError('Expected at least 20 spectral columns named wn_<wavenumber>.')
    wns = np.array([float(c[3:]) for c in wn_cols])
    if not np.isfinite(wns).all() or len(np.unique(wns)) != len(wns):
        raise ValueError('Wavenumbers must be finite and unique.')
    data['adulteration_percent'] = pd.to_numeric(data.adulteration_percent, errors='raise')
    if not np.isfinite(data.adulteration_percent).all() or not data.adulteration_percent.between(0, 100).all():
        raise ValueError('Adulteration percentages must be finite and between 0 and 100.')
    group_source = 'sample_state_id prefix; experimental independence requires provenance records'
    if group_map is not None:
        mapping = pd.read_csv(group_map, dtype=str)
        if not {'sample_state_id', 'parent_batch_id'}.issubset(mapping):
            raise ValueError('Group map requires sample_state_id,parent_batch_id.')
        if mapping.sample_state_id.duplicated().any():
            raise ValueError('Group map has duplicate sample IDs.')
        data['group_id'] = data.sample_state_id.map(mapping.set_index('sample_state_id').parent_batch_id)
        group_source = str(Path(group_map).resolve())
    elif 'parent_batch_id' in data:
        data['group_id'] = data.parent_batch_id
        group_source = 'explicit parent_batch_id column'
    elif 'base_batch_id' in data:
        data['group_id'] = data.base_batch_id
        group_source = 'explicit base_batch_id column'
    else:
        data['group_id'] = data.sample_state_id.str.split(';').str[0].str.upper()
    if data.group_id.isna().any() or data.group_id.astype(str).str.strip().eq('').any():
        raise ValueError('Missing parent-batch mapping.')
    data['group_id'] = data.group_id.astype(str).str.strip()
    for c in wn_cols + COLOR_COLS:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data[wn_cols + COLOR_COLS] = data[wn_cols + COLOR_COLS].replace([np.inf, -np.inf], np.nan)
    # Deliberately preserve missing values. Statistics are fitted inside base training subsets.
    return data, wn_cols, wns, group_source


def grouped_splits(groups, folds, seed):
    """Randomized balanced group folds; never fall back to row-level splitting.

    Group allocation uses group IDs and row counts only, not spectral features
    or target labels. Assignment is reusable across modalities with same rows.
    """
    groups = np.asarray(groups).astype(str)
    unique, counts = np.unique(groups, return_counts=True)
    k = min(folds, len(unique))
    if k < 2:
        raise ValueError('At least two independent parent groups required; no row-level fallback.')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    # Shuffle before stable size sorting randomizes ties reproducibly.
    order = order[np.argsort(-counts[order], kind='stable')]
    assignments, loads = {}, np.zeros(k, dtype=int)
    for pos in order:
        candidates = np.flatnonzero(loads == loads.min())
        fold = int(rng.choice(candidates))
        assignments[unique[pos]] = fold
        loads[fold] += counts[pos]
    membership = np.array([assignments[g] for g in groups])
    result = [(np.flatnonzero(membership != i), np.flatnonzero(membership == i)) for i in range(k)]
    coverage = np.zeros(len(groups), dtype=int)
    for tr, va in result:
        assert set(groups[tr]).isdisjoint(groups[va])
        assert len(tr) > 0 and len(va) > 0
        coverage[va] += 1
    assert np.all(coverage == 1)
    return result


def load_outer_plan(groups, plan_path):
    """Build outer splits from a saved parent-batch-to-fold assignment.

    The CSV must contain columns 'group_id' and 'Outer_fold' with folds
    numbered 1..k. Every unique group in `groups` must be present. This lets
    the sensitivity analysis reuse the exact outer folds of the primary
    analysis so that the two are directly comparable fold-by-fold.
    """
    groups = np.asarray(groups).astype(str)
    plan = pd.read_csv(plan_path)
    plan['group_id'] = plan['group_id'].astype(str).str.strip()
    plan['Outer_fold'] = plan['Outer_fold'].astype(int)
    missing = set(np.unique(groups)) - set(plan['group_id'])
    if missing:
        raise ValueError(f'Outer plan missing groups: {sorted(missing)}')
    plan = plan.drop_duplicates(subset=['group_id']).set_index('group_id')['Outer_fold']
    k = int(plan.max())
    membership = np.array([plan[g] - 1 for g in groups])  # convert 1-indexed to 0-indexed
    if not np.all((membership >= 0) & (membership < k)):
        raise ValueError('Outer_fold values must be consecutive integers starting at 1.')
    result = [(np.flatnonzero(membership != i), np.flatnonzero(membership == i)) for i in range(k)]
    coverage = np.zeros(len(groups), dtype=int)
    for tr, va in result:
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError(f'Parent overlap in loaded outer plan: {set(groups[tr]) & set(groups[va])}')
        if len(tr) == 0 or len(va) == 0:
            raise ValueError('Loaded outer plan produced an empty train or validation fold.')
        coverage[va] += 1
    if not np.all(coverage == 1):
        raise ValueError('Loaded outer plan does not cover all groups exactly once.')
    return result


def make_nested_plan(groups, config, seed):
    inner = grouped_splits(groups, config.inner_folds, seed)
    return [(tr, va, grouped_splits(np.asarray(groups)[tr], config.stack_folds, seed + 100 + i))
            for i, (tr, va) in enumerate(inner)]


def split_audit(data, outer_tr, outer_va, plan, final_stack, repeat, fold):
    records = []
    def add(level, train, validation):
        a, b = set(data.iloc[train].group_id), set(data.iloc[validation].group_id)
        if a & b:
            raise AssertionError(f'Parent overlap at {level}: {a & b}')
        records.append({'Repeat': repeat, 'Outer_fold': fold, 'Level': level,
                        'n_train': len(train), 'n_validation': len(validation),
                        'train_groups': ';'.join(sorted(a)), 'validation_groups': ';'.join(sorted(b)),
                        'overlap_count': len(a & b)})
    add('outer', outer_tr, outer_va)
    for i, (tr, va, stack) in enumerate(plan, 1):
        add(f'inner_{i}', outer_tr[tr], outer_tr[va])
        for j, (a, b) in enumerate(stack, 1):
            add(f'inner_{i}/stack_{j}', outer_tr[tr[a]], outer_tr[tr[b]])
    for j, (a, b) in enumerate(final_stack, 1):
        add(f'outer_training/stack_{j}', outer_tr[a], outer_tr[b])
    return pd.DataFrame(records)


def band_selection(X, y, groups, wns, classification, config):
    """Four nonoverlapping windows ranked by group-balanced training association.

    The search grid covers the measured spectrum and is fixed before evaluation.
    No historical selected intervals participate in the primary search grid.
    Classification uses the between-class / total variance ratio. Regression
    uses squared Pearson association. Each parent contributes equal total weight.
    """
    if config.band_mode == 'fixed':
        ranges = list(BANDS.values())
        return [(float(high), float(low), np.nan) for high, low in ranges]
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inv]
    weights /= weights.sum()
    mean = weights @ X
    xc = X - mean
    variance = weights @ (xc * xc)
    if classification:
        between = np.zeros(X.shape[1])
        for cls in np.unique(y):
            mask = y == cls
            mass = weights[mask].sum()
            cm = (weights[mask] @ X[mask]) / mass
            between += mass * (cm - mean) ** 2
        score = between / np.maximum(variance, 1e-20)
    else:
        y = np.asarray(y, float)
        yc = y - weights @ y
        cov = (weights * yc) @ xc
        score = cov * cov / np.maximum(variance * (weights @ (yc * yc)), 1e-20)
    candidates = []
    low_axis, high_axis = float(wns.min()), float(wns.max())
    for low in np.arange(low_axis, high_axis - config.band_width + 1e-8, config.band_step):
        high = low + config.band_width
        mask = (wns >= low) & (wns <= high)
        if mask.sum() >= 3:
            candidates.append((float(np.mean(score[mask])), float(high), float(low)))
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    selected = []
    for score_value, high, low in candidates:
        if all(high <= other_low or low >= other_high for other_high, other_low, _ in selected):
            selected.append((high, low, score_value))
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError('Spectral range too narrow to select four nonoverlapping windows.')
    return selected


def source_matrices(Xtr, Xva, ytr, groups, wns, prep, classification, config):
    """Return independently fitted source matrices; never use validation labels."""
    p = len(wns)
    # Fitting raw imputation separately enables fold-local supervised interval selection.
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    nir_train = imputer.fit_transform(Xtr[:, :p])
    nir_valid = imputer.transform(Xva[:, :p])
    bands = band_selection(nir_train, ytr, groups, wns, classification, config)
    transform = NIRPreprocessBandTransformer(
        wns, wns.max(), wns.min(), pathlength=prep['Pathlength'],
        spectra=prep['Spectra'], smoothing=prep['Smoothing'])
    nir_train = transform.fit_transform(nir_train, ytr)
    nir_valid = transform.transform(nir_valid)
    train, valid = [], []
    for high, low, _ in bands:
        mask = (wns <= high) & (wns >= low)
        train.append(nir_train[:, mask])
        valid.append(nir_valid[:, mask])
    color_imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    train.append(color_imputer.fit_transform(Xtr[:, p:]))
    valid.append(color_imputer.transform(Xva[:, p:]))
    return train, valid, bands


def model_factory(name, n_features, n_train, classification, config):
    # These numerical settings are fixed protocol values, not tuned using outer results.
    n_comp = max(1, min(5, n_features, n_train - 1))
    if classification:
        if name == 'PLSDA':
            model = PLSDAClassifier(n_components=n_comp)
        elif name == 'SVM':
            # No libsvm ungrouped internal probability calibration. Softmax decision
            # scores below are fusion inputs, NOT calibrated class probabilities.
            model = SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced', probability=False)
        elif name == 'LDA':
            model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        elif name == 'RF':
            model = RandomForestClassifier(n_estimators=config.trees, max_features='sqrt',
                class_weight='balanced', random_state=config.seed, n_jobs=1)
        else:
            raise ValueError(name)
    else:
        if name == 'PLSR':
            model = PLSRegression(n_components=n_comp, scale=False, max_iter=5000)
        elif name == 'SVR':
            model = SVR(C=10, kernel='rbf', gamma='scale', epsilon=0.1)
        elif name == 'RF':
            model = RandomForestRegressor(n_estimators=config.trees, max_features='sqrt',
                random_state=config.seed, n_jobs=1)
        else:
            raise ValueError(name)
    if name == 'RF':
        return model
    return Pipeline([('scaler', StandardScaler()), ('model', model)])


def class_scores(model, X, classes):
    if hasattr(model, 'predict_proba'):
        score = model.predict_proba(X)
    else:
        decision = model.decision_function(X)
        if decision.ndim == 1:
            decision = np.column_stack([-decision / 2, decision / 2])
        score = softmax(decision, axis=1)
    learned = np.asarray(model.classes_)
    aligned = np.zeros((len(X), len(classes)))
    for j, cls in enumerate(learned):
        aligned[:, np.flatnonzero(classes == cls)[0]] = score[:, j]
    if not np.isfinite(aligned).all():
        raise ValueError('Nonfinite classification scores.')
    aligned = np.maximum(aligned, 0)
    return aligned / np.maximum(aligned.sum(axis=1, keepdims=True), 1e-12)


def predict_sources(train, valid, y, base, classes, config, calibration=False):
    classification = classes is not None
    if classification and len(np.unique(y)) < 2:
        raise ValueError('A grouped training subset has only one class; cannot fit this protocol.')
    validation, fitted = [], []
    for a, b in zip(train, valid):
        model = model_factory(base, a.shape[1], len(a), classification, config)
        model.fit(a, y)
        if classification:
            validation.append(class_scores(model, b, classes))
            if calibration:
                fitted.append(class_scores(model, a, classes))
        else:
            validation.append(np.clip(np.asarray(model.predict(b)).ravel(), 0, 100))
            if calibration:
                fitted.append(np.clip(np.asarray(model.predict(a)).ravel(), 0, 100))
    return np.stack(validation, axis=1), np.stack(fitted, axis=1) if calibration else None


def fit_meta_predict(oof, y, validation, meta_name, classes, config):
    classification = classes is not None
    if meta_name in ['Equal_average', 'Adaptive_weighted_average']:
        weights = np.ones(oof.shape[1])
        if meta_name == 'Adaptive_weighted_average':
            if classification:
                weights = np.array([max(f1_score(y, classes[np.argmax(oof[:, i], axis=1)],
                    labels=classes, average='macro', zero_division=0), 1e-6) for i in range(oof.shape[1])])
            else:
                weights = 1 / np.maximum(np.sqrt(np.mean((oof - y[:, None]) ** 2, axis=0)), 1e-6)
        weights /= weights.sum()
        pred = np.einsum('nsk,s->nk', validation, weights) if classification else validation @ weights
        return pred, weights
    a, b = oof.reshape(len(oof), -1), validation.reshape(len(validation), -1)
    if classification:
        if meta_name == 'Logistic':
            model = LogisticRegression(C=1.0, max_iter=3000, random_state=config.seed)
        elif meta_name == 'LDA':
            model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        elif meta_name == 'RF':
            model = RandomForestClassifier(n_estimators=config.trees, max_features='sqrt',
                class_weight='balanced', random_state=config.seed, n_jobs=1)
        else:
            raise ValueError(meta_name)
    else:
        if meta_name == 'Ridge':
            # Fixed alpha avoids RidgeCV's ungrouped leave-one-row-out selection.
            model = Ridge(alpha=1.0)
        elif meta_name == 'PLSR':
            model = PLSRegression(n_components=min(2, a.shape[1], len(a)-1), scale=False, max_iter=5000)
        elif meta_name == 'RF':
            model = RandomForestRegressor(n_estimators=config.trees, max_features='sqrt',
                random_state=config.seed, n_jobs=1)
        else:
            raise ValueError(meta_name)
    if meta_name != 'RF':
        model = Pipeline([('scaler', StandardScaler()), ('model', model)])
    model.fit(a, y)
    pred = class_scores(model, b, classes) if classification else np.clip(np.asarray(model.predict(b)).ravel(), 0, 100)
    return pred, np.full(oof.shape[1], np.nan)


def regression_metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    rmse = float(np.sqrt(np.mean((pred - y)**2)))
    sd = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
    return {'RMSE': rmse, 'R2': float(r2_score(y, pred)) if len(y)>1 and np.ptp(y)>0 else np.nan,
            'MAE': float(np.mean(abs(pred-y))), 'Bias': float(np.mean(pred-y)),
            'RPD': sd/rmse if rmse > 0 else np.nan}


def classification_metrics(y, pred, classes):
    cm = confusion_matrix(y, pred, labels=classes)
    recall = np.divide(np.diag(cm), cm.sum(axis=1), out=np.full(len(classes), np.nan), where=cm.sum(axis=1)>0)
    return {'Accuracy': float(accuracy_score(y, pred)), 'Balanced_accuracy': float(np.nanmean(recall)),
            'F1_macro': float(f1_score(y, pred, labels=classes, average='macro', zero_division=0)),
            'Kappa': float(cohen_kappa_score(y, pred, labels=classes)),
            'MCC': float(matthews_corrcoef(y, pred))}


def candidate_names(classes):
    if classes is None:
        return ['PLSR', 'SVR', 'RF'], ['Equal_average', 'Adaptive_weighted_average', 'Ridge', 'PLSR', 'RF']
    return ['PLSDA', 'SVM', 'RF', 'LDA'], ['Equal_average', 'Adaptive_weighted_average', 'Logistic', 'LDA', 'RF']


def evaluate_preprocessing(X, y, groups, wns, classes, prep, plan, config):
    """Joint inner validation of all base/meta combinations for one preprocessing.

    Features are shared ONLY across estimators with identical training rows.
    Every inner validation row is excluded from imputation, MSC, band selection,
    base OOF generation, meta fitting, and adaptive weight calculation.
    """
    bases, metas = candidate_names(classes)
    shape = (len(y), len(classes)) if classes is not None else (len(y),)
    predictions = {(base, meta): np.full(shape, np.nan) for base in bases for meta in metas}
    failures, band_records = {}, []
    for i, (tr, va, stack) in enumerate(plan, 1):
        outer_matrices = source_matrices(X[tr], X[va], y[tr], groups[tr], wns, prep, classes is not None, config)
        stack_matrices = []
        for j, (a, b) in enumerate(stack, 1):
            matrices = source_matrices(X[tr[a]], X[tr[b]], y[tr[a]], groups[tr[a]], wns, prep, classes is not None, config)
            stack_matrices.append((a, b, matrices))
            for rank, (high, low, association) in enumerate(matrices[2], 1):
                band_records.append({'Inner_fold': i, 'Stack_fold': j, 'Rank': rank,
                    'High_cm-1': high, 'Low_cm-1': low, 'Training_association': association})
        for base in bases:
            try:
                oof_shape = (len(tr), 5, len(classes)) if classes is not None else (len(tr), 5)
                oof = np.full(oof_shape, np.nan)
                for a, b, matrices in stack_matrices:
                    oof[b], _ = predict_sources(matrices[0], matrices[1], y[tr[a]], base, classes, config)
                if not np.isfinite(oof).all():
                    raise ValueError('Incomplete base OOF coverage.')
                valid, _ = predict_sources(outer_matrices[0], outer_matrices[1], y[tr], base, classes, config)
                for meta in metas:
                    try:
                        predictions[(base, meta)][va], _ = fit_meta_predict(oof, y[tr], valid, meta, classes, config)
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
            row.update(classification_metrics(y, classes[pred.argmax(axis=1)], classes)
                       if classes is not None else regression_metrics(y, pred))
        elif not row['error']:
            row['error'] = 'Incomplete inner validation predictions.'
        records.append(row)
    return {'candidates': records, 'training_band_selections': band_records}


def select_candidate(records, classes):
    frame = pd.DataFrame(records)
    valid = frame[frame.error.eq('')].copy()
    if valid.empty:
        raise RuntimeError(f'All candidates failed: {frame.error.value_counts().to_dict()}')
    if classes is not None:
        fields, ascending = ['F1_macro', 'Balanced_accuracy', 'Number', 'Base_model', 'Fusion_type'], [False, False, True, True, True]
    else:
        fields, ascending = ['RMSE', 'MAE', 'Number', 'Base_model', 'Fusion_type'], [True]*5
    # Ties use deterministic names, never outer validation metrics.
    valid = valid.sort_values(fields, ascending=ascending, kind='stable')
    return valid.iloc[0].to_dict(), valid


def fit_selected_stack(X, y, groups, test_X, wns, classes, selection, stack, config):
    shape = (len(y), 5, len(classes)) if classes is not None else (len(y), 5)
    oof = np.full(shape, np.nan)
    for tr, va in stack:
        a, b, _ = source_matrices(X[tr], X[va], y[tr], groups[tr], wns, selection, classes is not None, config)
        oof[va], _ = predict_sources(a, b, y[tr], selection['Base_model'], classes, config)
    if not np.isfinite(oof).all():
        raise AssertionError('Outer training base OOF is incomplete.')
    a, b, bands = source_matrices(X, test_X, y, groups, wns, selection, classes is not None, config)
    test, calibration = predict_sources(a, b, y, selection['Base_model'], classes, config, calibration=True)
    # Fit meta exactly once for both training diagnostics and untouched outer prediction.
    combined, weights = fit_meta_predict(oof, y, np.concatenate([test, calibration]),
                                         selection['Fusion_type'], classes, config)
    return combined[:len(test_X)], combined[len(test_X):], bands, weights


def bootstrap_metrics(predictions, classes, config):
    """Percentile cluster bootstrap of complete outer OOF predictions.

    Parent IDs are resampled, retaining every derivative and every repeat.
    Metrics are computed per repeat then averaged (never pool repeats as n).
    These conditional OOF intervals do not retrain the nested procedure.
    """
    groups = np.array(sorted(predictions.group_id.unique()))
    per_repeat = [sub.reset_index(drop=True) for _, sub in predictions.groupby('Repeat')]
    index = [{g: np.flatnonzero(sub.group_id.to_numpy() == g) for g in groups} for sub in per_repeat]
    rng = np.random.default_rng(config.seed + 9000)
    draws = []
    for _ in range(config.bootstrap):
        selected = rng.choice(groups, len(groups), replace=True)
        metrics = []
        for sub, lookup in zip(per_repeat, index):
            take = np.concatenate([lookup[g] for g in selected])
            boot = sub.iloc[take]
            metrics.append(classification_metrics(boot.Observed, boot.Predicted, classes) if classes is not None
                           else regression_metrics(boot.Observed, boot.Predicted))
        draws.append(pd.DataFrame(metrics).mean().to_dict())
    distribution = pd.DataFrame(draws)
    point = []
    for sub in per_repeat:
        point.append(classification_metrics(sub.Observed, sub.Predicted, classes) if classes is not None
                     else regression_metrics(sub.Observed, sub.Predicted))
    estimate = pd.DataFrame(point).mean()
    rows = []
    for metric, value in estimate.items():
        values = distribution[metric].dropna()
        rows.append({'Metric': metric, 'Estimate': value,
            'CI95_low': values.quantile(.025), 'CI95_high': values.quantile(.975),
            'Valid_bootstrap_draws': len(values), 'Bootstrap_unit': 'parent_batch',
            'Interval_scope': 'conditional_on_outer_OOF_predictions; no bootstrap refitting'})
    return pd.DataFrame(rows), distribution, pd.DataFrame(point)


def class_details(y, pred, classes):
    cm = confusion_matrix(y, pred, labels=classes)
    rows = []
    for i, label in enumerate(classes):
        tp = int(cm[i, i]); fn = int(cm[i].sum()-tp); fp = int(cm[:, i].sum()-tp)
        tn = int(cm.sum()-tp-fn-fp)
        divide = lambda a, b: a/b if b else np.nan
        rows.append({'Class': label, 'n': tp+fn, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Sensitivity_recall': divide(tp, tp+fn), 'Specificity': divide(tn, tn+fp),
            'Precision': divide(tp, tp+fp), 'F1': divide(2*tp, 2*tp+fp+fn)})
    return pd.DataFrame(rows), cm


def concentration_results(predictions, classification):
    intervals = [('0%', 0, 0), ('(0,5]%', 0, 5), ('(5,10]%', 5, 10),
                 ('(10,20]%', 10, 20), ('(20,40]%', 20, 40), ('(40,60]%', 40, 60), ('(60,100]%', 60, 100)]
    records = []
    for repeat, sub in predictions.groupby('Repeat'):
        for label, low, high in intervals:
            mask = sub.adulteration_percent.eq(0) if high == 0 else sub.adulteration_percent.gt(low) & sub.adulteration_percent.le(high)
            part = sub[mask]
            row = {'Repeat': repeat, 'Concentration_interval': label, 'n': len(part),
                   'n_parent_batches': part.group_id.nunique()}
            if len(part):
                if classification:
                    row['Accuracy'] = accuracy_score(part.Observed, part.Predicted)
                    if set(sub.Observed.unique()).issubset({'Authentic', 'Adulterated'}):
                        row['Adulterated_call_rate'] = float(part.Predicted.eq('Adulterated').mean())
                        row['Interpretation'] = 'false_positive_rate' if high == 0 else 'sensitivity'
                else:
                    row.update(regression_metrics(part.Observed, part.Predicted))
            records.append(row)
    return pd.DataFrame(records)


def save_figure(fig, stem):
    fig.tight_layout()
    fig.savefig(str(stem)+'.png', dpi=600, bbox_inches='tight')
    fig.savefig(str(stem)+'.svg', bbox_inches='tight')
    plt.close(fig)


def plot_task(out, predictions, selected, bands, classes, metrics):
    # One row per sample in each repeat; show repeat 1 to avoid duplicate counts.
    primary = predictions[predictions.Repeat == predictions.Repeat.min()]
    title_suffix = 'Outer grouped CV'
    if classes is not None:
        details, cm = class_details(primary.Observed, primary.Predicted, classes)
        for normalized in [False, True]:
            values = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True)>0) if normalized else cm
            fig, ax = plt.subplots(figsize=(5.2, 4.5))
            im = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized else max(1, cm.max()))
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f'{values[i,j]:.1%}' if normalized else str(cm[i,j]), ha='center', va='center',
                            color='white' if values[i,j] > im.norm.vmax*.55 else 'black')
            ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
                   yticklabels=classes, xlabel='Predicted class', ylabel='Observed class', title=title_suffix)
            fig.colorbar(im, ax=ax, fraction=.046)
            save_figure(fig, out / ('confusion_matrix_percent' if normalized else 'confusion_matrix_counts'))
        fig, ax = plt.subplots(figsize=(6.2, 4))
        x = np.arange(len(classes)); width = .25
        for offset, (name, label) in zip([-1,0,1], [('Sensitivity_recall','Recall'), ('Specificity','Specificity'), ('F1','F1')]):
            ax.bar(x+offset*width, details[name], width=width, label=label)
        ax.set(xticks=x, xticklabels=classes, ylim=(0,1.1), ylabel='Score', title=title_suffix)
        ax.legend(frameon=False, ncol=3)
        save_figure(fig, out/'class_performance')
    else:
        fig, ax = plt.subplots(figsize=(5.2,4.5))
        for fold, sub in primary.groupby('Outer_fold'):
            ax.scatter(sub.Observed, sub.Predicted, s=28, alpha=.8, label=f'Fold {fold}')
        limits = [min(0, primary.Observed.min()), max(primary.Observed.max(), primary.Predicted.max())*1.05]
        ax.plot(limits, limits, '--', color='0.4', lw=1)
        ax.set(xlabel='Observed adulteration (%)', ylabel='Predicted adulteration (%)', title=title_suffix)
        point = metrics.set_index('Metric').Estimate
        ax.text(.04,.95,f'RMSE = {point.RMSE:.3f} percentage points\nR² = {point.R2:.3f}',
                transform=ax.transAxes, va='top')
        ax.legend(frameon=False, fontsize=8, loc='lower right')
        save_figure(fig,out/'outer_OOF_observed_predicted')
        fig, axes = plt.subplots(1,2,figsize=(9,3.8))
        residual = primary.Predicted-primary.Observed
        axes[0].scatter(primary.Observed, residual, s=24, alpha=.75)
        axes[0].axhline(0,color='0.3',ls='--',lw=1)
        axes[0].set(xlabel='Observed adulteration (%)',ylabel='Residual (percentage points)')
        axes[1].hist(residual,bins='auto',color='#4477AA',edgecolor='white')
        axes[1].set(xlabel='Residual (percentage points)',ylabel='Sample count')
        save_figure(fig,out/'outer_OOF_residuals')
        errors = primary.assign(Absolute_error=np.abs(residual)).groupby('group_id').Absolute_error.mean().sort_values()
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(errors.index,errors.values,color='#4477AA')
        ax.set(ylabel='MAE (percentage points)',xlabel='Parent batch',title=title_suffix)
        ax.tick_params(axis='x',rotation=60)
        save_figure(fig,out/'error_by_parent_batch')
    if not bands.empty:
        fig,ax=plt.subplots(figsize=(8,4))
        for _,row in bands.iterrows():
            position=(row.Repeat-1)*selected.Outer_fold.max()+row.Outer_fold+(row.Rank-2.5)*.13
            ax.plot([row['High_cm-1'],row['Low_cm-1']],[position]*2,lw=4,color=plt.get_cmap('tab10')(int(row.Rank)-1))
        ax.invert_xaxis()
        ax.set(xlabel='Wavenumber (cm⁻¹)',ylabel='Outer split',title='Bands selected using outer training batches only')
        save_figure(fig,out/'training_selected_bands')
    frequency=selected.Fusion_type.value_counts()
    fig,ax=plt.subplots(figsize=(6,4))
    ax.barh(frequency.index,frequency.values,color='#4477AA')
    ax.set(xlabel='Number of outer training selections',title='Fusion selection frequency')
    save_figure(fig,out/'fusion_selection_frequency')
    ranges=concentration_results(primary,classes is not None)
    ranges=ranges[ranges.n>0]
    value='Accuracy' if classes is not None else 'RMSE'
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(ranges.Concentration_interval,ranges[value],color='#4477AA')
    for i,(_,row) in enumerate(ranges.iterrows()):
        ax.annotate(f'n={int(row.n)}',xy=(i,row[value]),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
    ax.set(xlabel='Observed adulteration interval',ylabel='Accuracy' if classes is not None else 'RMSE (percentage points)',
           title='Outer CV performance by concentration')
    if classes is not None:ax.set_ylim(0,1.15)
    else:ax.set_ylim(0,max(ranges[value])*1.2)
    save_figure(fig,out/'performance_by_concentration')
    metric_names=['Accuracy','Balanced_accuracy','F1_macro'] if classes is not None else ['RMSE','MAE']
    ci=metrics.set_index('Metric').loc[metric_names]
    fig,ax=plt.subplots(figsize=(5.5,3.5))
    positions=np.arange(len(ci))
    # Draw percentile bounds independently: a finite-sample estimate can lie
    # outside a percentile interval, so do not assume nonnegative errorbar spans.
    ax.hlines(positions,ci.CI95_low,ci.CI95_high,color='#4477AA',lw=2)
    ax.scatter(ci.Estimate,positions,color='#CC6677',s=35,zorder=3)
    ax.set(yticks=positions,yticklabels=metric_names,xlabel='Score' if classes is not None else 'Percentage points',
           title='95% parent-cluster bootstrap intervals')
    save_figure(fig,out/'outer_metric_confidence_intervals')


def plot_workflow(out, config):
    fig,ax=plt.subplots(figsize=(10,6))
    ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
    boxes=[(5,6.4,'Parent batches: keep all derivatives together'),
           (5,5.3,f'Outer grouped CV ({config.outer_folds} folds): reserve validation batches'),
           (5,4.2,f'Inner grouped CV ({config.inner_folds} folds): compare 24 preprocessing strategies + base + fusion'),
           (5,3.1,f'Within each inner training subset: grouped base OOF ({config.stack_folds} folds)\nFit imputation, bands, MSC, scaling and base models using training rows only'),
           (5,1.9,'Fit meta learner on base OOF; score complete stack on inner validation\nLock winning configuration; rebuild stack on outer training batches'),
           (5,.6,'Predict outer validation batches once\nPool outer OOF predictions; parent-cluster confidence intervals and figures')]
    for x,y,label in boxes:
        ax.text(x,y,label,ha='center',va='center',fontsize=10,
                bbox=dict(boxstyle='round,pad=.6',fc='#EDF3F8',ec='#4477AA'))
    for (_,y,_),(_,next_y,_) in zip(boxes,boxes[1:]):
        ax.annotate('',xy=(5,next_y+.38),xytext=(5,y-.36),arrowprops=dict(arrowstyle='->',color='#4477AA'))
    save_figure(fig,out/'nested_validation_workflow')


def export_task(out, config, classes):
    completed=sorted(out.glob('repeat_*/fold_*/completed.json'))
    if not completed:
        raise ValueError(f'No completed folds in {out}')
    payload=[json.loads(p.read_text(encoding='utf-8')) for p in completed]
    predictions=pd.concat([pd.DataFrame(p['predictions']) for p in payload],ignore_index=True)
    calibration=pd.concat([pd.DataFrame(p['calibration_diagnostics']) for p in payload],ignore_index=True)
    selected=pd.DataFrame([p['selection'] for p in payload])
    bands=pd.DataFrame([row for p in payload for row in p['outer_training_bands']])
    for repeat,sub in predictions.groupby('Repeat'):
        if sub.sample_state_id.duplicated().any():
            raise AssertionError('Duplicate outer validation sample predictions.')
        if sub.groupby('group_id').Outer_fold.nunique().max()!=1:
            raise AssertionError('Parent group split across outer validation folds.')
    save_csv(predictions,out/'outer_OOF_predictions.csv')
    save_csv(calibration,out/'calibration_diagnostics_NOT_CV.csv')
    save_csv(selected,out/'selected_configuration_by_outer_fold.csv')
    save_csv(bands,out/'selected_bands_by_outer_training_fold.csv')
    audits=pd.concat([pd.read_csv(p.parent/'split_audit.csv') for p in completed],ignore_index=True)
    assert audits.overlap_count.eq(0).all()
    save_csv(audits,out/'all_nested_split_audits.csv')
    metrics,draws,repeats=bootstrap_metrics(predictions,classes,config)
    save_csv(metrics,out/'outer_metrics_parent_bootstrap_CI95.csv')
    save_csv(draws,out/'parent_bootstrap_metric_distribution.csv')
    repeats.insert(0,'Repeat',range(1,len(repeats)+1))
    save_csv(repeats,out/'outer_metrics_by_repeat.csv')
    folds=[]; details=[]
    for (repeat,fold),sub in predictions.groupby(['Repeat','Outer_fold']):
        result=classification_metrics(sub.Observed,sub.Predicted,classes) if classes is not None else regression_metrics(sub.Observed,sub.Predicted)
        train=calibration[(calibration.Repeat==repeat)&(calibration.Outer_fold==fold)]
        train_metrics=classification_metrics(train.Observed,train.Predicted,classes) if classes is not None else regression_metrics(train.Observed,train.Predicted)
        folds.append({'Repeat':repeat,'Outer_fold':fold,'n_validation':len(sub),'n_validation_groups':sub.group_id.nunique(),
                      'n_calibration':len(train),'n_calibration_groups':train.group_id.nunique(),**result,
                      **{f'Calibration_{key}':value for key,value in train_metrics.items()}})
    save_csv(pd.DataFrame(folds),out/'outer_fold_metrics_and_calibration_diagnostics.csv')
    if classes is not None:
        for repeat,sub in predictions.groupby('Repeat'):
            detail,cm=class_details(sub.Observed,sub.Predicted,classes)
            detail.insert(0,'Repeat',repeat);details.append(detail)
            table=pd.DataFrame(cm,columns=classes);table.insert(0,'Observed_class',classes)
            save_csv(table,out/f'confusion_matrix_counts_repeat_{repeat}.csv')
        save_csv(pd.concat(details,ignore_index=True),out/'per_class_metrics.csv')
    save_csv(concentration_results(predictions,classes is not None),out/'performance_by_concentration.csv')
    plot_task(out,predictions,selected,bands,classes,metrics)
    return metrics


def run_task(data, wn_cols, wns, task, out, config, outer_plan=None):
    if task in REGRESSION_TASKS:
        if task == 'quantitative_positive83':
            df=data[data.adulteration_percent>0].reset_index(drop=True)
        else:
            df=data.reset_index(drop=True)
        classes=None; y=df.adulteration_percent.to_numpy(float)
    else:
        classes=np.array(CLASS_ORDERS[task])
        df=data[data.adulteration_percent.le(60)].reset_index(drop=True) if task=='Task1_4class' else data.reset_index(drop=True)
        ratio=df.adulteration_percent.to_numpy()
        y=np.where(ratio==0,'Authentic','Adulterated') if task=='Task2_binary' else np.select(
            [ratio==0,ratio<=20,ratio<=40],['Authentic','Low','Medium'],default='High')
    out.mkdir(parents=True,exist_ok=True)
    groups=df.group_id.to_numpy(str); X=df[wn_cols+COLOR_COLS].to_numpy(float)
    manifest=build_preprocessing_manifest()
    if config.smoke:
        manifest=manifest[manifest.Number.isin([1,13])]
    save_csv(df[['sample_state_id','group_id','adulteration_percent']],out/'sample_manifest.csv')
    save_csv(df.groupby('group_id').adulteration_percent.agg(['size','min','max']).reset_index(),out/'parent_sample_distribution.csv')
    if classes is not None:
        save_csv(pd.DataFrame({'Class':classes,'n':[int(np.sum(y==c)) for c in classes]}),out/'class_counts.csv')
    for repeat in range(1,config.repeats+1):
        if outer_plan is not None:
            outer = load_outer_plan(groups, outer_plan)
        else:
            outer=grouped_splits(groups,config.outer_folds,config.seed+repeat*10000)
        for fold,(tr,va) in enumerate(outer,1):
            folder=out/f'repeat_{repeat:02d}'/f'fold_{fold:02d}'
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/'completed.json').exists():
                print(f'{task} repeat {repeat} outer {fold}: resume completed',flush=True)
                continue
            seed=config.seed+repeat*10000+fold*1000
            plan=make_nested_plan(groups[tr],config,seed+10)
            final_stack=grouped_splits(groups[tr],config.stack_folds,seed+20)
            save_csv(split_audit(df,tr,va,plan,final_stack,repeat,fold),folder/'split_audit.csv')
            records=[]; pending=[]
            for prep in manifest.to_dict('records'):
                checkpoint=folder/f'preprocessing_{int(prep["Number"]):02d}.json'
                if checkpoint.exists():
                    records.extend(json.loads(checkpoint.read_text(encoding='utf-8'))['candidates'])
                else:
                    pending.append((prep,checkpoint))
            print(f'{task} repeat {repeat}/{config.repeats} outer {fold}/{len(outer)}: '
                  f'{len(tr)} training, {len(va)} validation; {len(pending)} preprocessing searches',flush=True)
            start=time.monotonic()
            with ThreadPoolExecutor(max_workers=config.workers) as pool:
                futures={pool.submit(evaluate_preprocessing,X[tr],y[tr],groups[tr],wns,classes,prep,plan,config):(prep,path)
                         for prep,path in pending}
                for future in as_completed(futures):
                    prep,path=futures[future]
                    payload=future.result()
                    write_json(path,payload);records.extend(payload['candidates'])
                    errors=sum(bool(row['error']) for row in payload['candidates'])
                    print(f'  N{int(prep["Number"]):02d} complete; failed candidates={errors}; elapsed={time.monotonic()-start:.0f}s',flush=True)
            selection,ranked=select_candidate(records,classes)
            save_csv(pd.DataFrame(records),folder/'inner_candidate_results.csv')
            save_csv(ranked,folder/'inner_candidate_ranking.csv')
            # Lock choices BEFORE any outer prediction or label-based metric calculation.
            selection.update({'Repeat':repeat,'Outer_fold':fold,'Selection_scope':'inner_grouped_complete_stack_CV'})
            write_json(folder/'locked_selection_before_outer_evaluation.json',selection)
            pred,cal,bands,weights=fit_selected_stack(X[tr],y[tr],groups[tr],X[va],wns,classes,selection,final_stack,config)
            def prediction_table(indices,values):
                rows=df.iloc[indices][['sample_state_id','group_id','adulteration_percent']].copy()
                rows['Repeat']=repeat;rows['Outer_fold']=fold;rows['Observed']=y[indices]
                rows['Predicted']=classes[values.argmax(axis=1)] if classes is not None else values
                if classes is not None:
                    for j,cls in enumerate(classes):rows[f'Fusion_score_{cls}']=values[:,j]
                else:rows['Residual']=values-y[indices]
                return rows.to_dict('records')
            band_rows=[{'Repeat':repeat,'Outer_fold':fold,'Rank':i,'High_cm-1':high,'Low_cm-1':low,
                        'Training_association':association} for i,(high,low,association) in enumerate(bands,1)]
            write_json(folder/'completed.json',{'selection':selection,'predictions':prediction_table(va,pred),
                'calibration_diagnostics':prediction_table(tr,cal),'outer_training_bands':band_rows,
                'adaptive_or_equal_weights':dict(zip(SOURCE_NAMES,weights.tolist()))})
            print(f'  Locked N{int(selection["Number"]):02d} {selection["Base_model"]} + {selection["Fusion_type"]}; outer predictions saved',flush=True)
    return export_task(out,config,classes)


def protocol_text(config):
    return f'''Analysis protocol — decision fusion, revision 2026-09-07

Primary estimates use {config.repeats} repeat(s) of {config.outer_folds}-fold parent-group outer CV.
Each outer training subset searches 24 preprocessing combinations, base algorithms,
and five fusion alternatives using {config.inner_folds}-fold grouped inner CV.
For EACH inner training subset, base OOF features for meta training are regenerated
using {config.stack_folds}-fold grouped CV. Inner validation groups never enter those fits.
Choices are locked to disk before outer evaluation. No outer metric selects a model.
Classification selects macro-F1, then balanced accuracy; regression selects RMSE,
then MAE. Ties are broken deterministically. No global best outer-scoring model is exported.

Band mode: {config.band_mode}. Primary train_selected uses four nonoverlapping
{config.band_width:g} cm-1 windows, from a full-spectrum grid spaced {config.band_step:g} cm-1.
Windows are scored only with the current base-training labels and spectra;
each parent has equal total weight in interval scoring. Source columns denote
training association rank, not a fixed chemical identity across folds.
The historical four fixed bands are an exploratory option and are not asserted
to have independent selection provenance.

All imputation (median), MSC reference fitting and scaling use base-training rows only.
SNV is row-wise. Processing order: Constant/MSC/SNV -> Spectrum/FD/SD -> NS/SG/ND.
FD and SD retain the original numpy.gradient index-axis implementation (edge_order=1).
SG: window 9, polynomial 3, mode interp. ND retains the original symmetric gap
smoother: left/right segment mean, segment=5, gap=5, edge moving average with
numpy.convolve(mode=same); it is not rebranded as a new derivative algorithm.
No PCA is used in this decision-layer analysis.

Fixed base settings: PLS latent variables=min(5, training n-1, source variables),
PLSR scale=False; SVR C=10, epsilon=.1, gamma=scale, RBF; SVM C=10, gamma=scale,
RBF, balanced classes, probability=False. SVM softmax decision scores are NOT
calibrated probabilities. LDA: lsqr, automatic shrinkage. RF: {config.trees} trees,
max_features=sqrt, seed={config.seed}, balanced class weights for classification.
Meta: equal average; weights from training OOF macro-F1 or inverse RMSE;
Logistic C=1; LDA lsqr/auto; Ridge alpha=1 (no ungrouped RidgeCV); PLSR 2 components;
RF as above. All numerical hyperparameters and grids are fixed protocol settings,
not changed after observing outer results. PLS components are bounded for feasibility,
not chosen by testing outer labels. Scalers fit on corresponding training matrices.
Predicted regression values are clipped to the prespecified physical range [0,100].

Primary regression task quantitative_positive83 excludes 0% and must contain exactly
83 positive mixtures. Sensitivity task quantitative_with_zero105 retains all 22
authentic 0% samples and the 83 positive mixtures (105 rows total). Classification
also retains authentic samples. Four-class labels:
Authentic=0, Low=(0,20], Medium=(20,40], High=(40,60]; binary: Authentic=0 vs >0.
Grouped allocation uses parent IDs and row counts only; no sample-level fallback.
Explicit parent mapping can be supplied with --group-map; otherwise recorded IDs
or the sample ID prefix are used. Biological independence still requires experiment records.

Intervals: {config.bootstrap} parent-cluster bootstrap resamples of complete outer OOF
predictions. All derivatives of a sampled parent stay together. With repeated CV,
metrics are computed separately per repeat then averaged. Intervals are conditional
on fitted OOF predictions; they do not include full retraining uncertainty.
Per-fold metrics, classwise counts/recall/specificity/precision/F1, low-concentration
performance, regression bias/MAE/RMSE/R2/RPD and residuals are exported.
RMSE, MAE and bias have units of percentage points. R2 is sklearn.r2_score,
not squared Pearson correlation. RPD uses sample SD (ddof=1) / RMSE.

These are INTERNAL nested-validation estimates. This reused laboratory dataset
is not a new untouched external or market validation set. Historical development
on the dataset cannot be erased retrospectively. No claim of external validation,
calibrated detection limit, or instrument/market transfer is made.
Calibration diagnostics are explicitly marked NOT_CV and excluded from final metrics.
Smoke outputs are software checks only and are stored separately.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=BASE_DIR/'NIR_color_merged_dataset.csv')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--group-map',type=Path)
    parser.add_argument('--tasks',nargs='+',choices=[*REGRESSION_TASKS,'Task1_4class','Task2_binary'],
                        default=list(REGRESSION_TASKS))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--trees',type=int,default=200)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--band-mode',choices=['train_selected','fixed'],default='train_selected')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--outer-plan',type=Path,default=None,
                        help='CSV with columns group_id and Outer_fold to fix outer fold assignment across analyses.')
    args=parser.parse_args()
    config=Config(workers=args.workers,repeats=args.repeats,trees=args.trees,bootstrap=args.bootstrap,
                  band_mode=args.band_mode,smoke=args.smoke)
    if args.smoke:
        config=Config(outer_folds=3,inner_folds=2,stack_folds=2,trees=8,bootstrap=100,
                      workers=args.workers,band_mode=args.band_mode,smoke=True)
    if min(config.workers,config.repeats,config.trees,config.bootstrap)<1:
        parser.error('workers, repeats, trees and bootstrap must be positive')
    suffix='_smoke' if args.smoke else ''
    base='决策层融合_固定4波段' if args.band_mode=='fixed' else '决策层融合_重选波段_主分析'
    out=args.output or BASE_DIR/(base+suffix)
    out.mkdir(parents=True,exist_ok=True)
    if args.plots_only:
        stored=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
        config=Config(**stored['config'])
        for task in args.tasks:
            export_task(out/task,config,None if task in REGRESSION_TASKS else np.array(CLASS_ORDERS[task]))
        return
    data,wn_cols,wns,group_source=read_input(args.input,args.group_map)
    n_positive = int(data.adulteration_percent.gt(0).sum())
    n_zero = int(data.adulteration_percent.eq(0).sum())
    if n_positive != 83 or n_zero != 22 or len(data) != 105:
        raise ValueError(f'Expected 83 positive + 22 zero samples (105 total); got '
                         f'{n_positive} positive + {n_zero} zero ({len(data)} total).')
    signature_config=asdict(config);signature_config.pop('workers')
    digest=hashlib.sha256(Path(__file__).read_bytes()+args.input.read_bytes()+json.dumps(signature_config,sort_keys=True).encode())
    if args.group_map:digest.update(args.group_map.read_bytes())
    if args.outer_plan:digest.update(args.outer_plan.read_bytes())
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
              'scipy':scipy.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__}
    digest.update(json.dumps(versions,sort_keys=True).encode())
    signature=digest.hexdigest()
    manifest_path=out/'run_manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['signature']!=signature:
            raise RuntimeError('Code/data/protocol differs from saved run. Use a new --output directory; stale checkpoints are never reused.')
    write_json(manifest_path,{'signature':signature,'config':asdict(config),'versions':versions,
               'input':str(args.input.resolve()),'group_source':group_source,'n_samples':len(data),
               'n_positive_samples':n_positive,'n_zero_samples':n_zero,
               'n_parent_batches':data.group_id.nunique(),'missing_feature_values':int(data[wn_cols+COLOR_COLS].isna().sum().sum()),
               'validation_type':'internal_grouped_nested_CV','created_or_resumed':time.strftime('%Y-%m-%dT%H:%M:%S')})
    (out/'ANALYSIS_PROTOCOL.txt').write_text(protocol_text(config),encoding='utf-8')
    plot_workflow(out,config)
    save_csv(build_preprocessing_manifest(),out/'preprocessing_manifest_24.csv')
    save_csv(data[['sample_state_id','group_id','adulteration_percent']],out/'complete_sample_group_mapping.csv')
    save_csv(data.groupby('adulteration_percent').agg(n=('sample_state_id','size'),n_parents=('group_id','nunique')).reset_index(),
             out/'concentration_distribution.csv')
    results=[]
    # Native math stays single threaded; preprocessing searches share memory in a
    # bounded thread pool. Forests also use one job, preventing oversubscription.
    with threadpool_limits(limits=1):
        for task in args.tasks:
            result=run_task(data,wn_cols,wns,task,out/task,config,outer_plan=args.outer_plan)
            result.insert(0,'Task',task);results.append(result)
    save_csv(pd.concat(results,ignore_index=True),out/'analysis_metrics_summary.csv')
    write_json(out/'analysis_completed.json',{'tasks':args.tasks,'signature':signature,'time':time.strftime('%Y-%m-%dT%H:%M:%S')})
    print(f'Completed. Results and PNG/SVG figures: {out}',flush=True)


if __name__=='__main__':
    main()
