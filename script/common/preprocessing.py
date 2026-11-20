import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from sklearn.base import BaseEstimator, TransformerMixin


SG_WINDOW = 9
SG_POLYORDER = 3
NORRIS_SEGMENT_LENGTH = 5
NORRIS_SEGMENT_SPACING = 5


def make_odd_window(window, n_points, polyorder):
    window = int(window)
    if window > n_points:
        window = n_points
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
    if window > n_points:
        window = n_points if n_points % 2 == 1 else n_points - 1
    return window


def msc_global(X):
    X_corr = np.zeros_like(X, dtype=float)
    ref = np.mean(X, axis=0)
    for i in range(X.shape[0]):
        y = X[i, :]
        slope, intercept = np.polyfit(ref, y, 1)
        if np.isclose(slope, 0):
            X_corr[i, :] = y
        else:
            X_corr[i, :] = (y - intercept) / slope
    return X_corr


def snv(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1
    return (X - mean) / std


def finite_derivative(X, wn, order=1):
    X_der = X.copy()
    for _ in range(order):
        X_der = np.gradient(X_der, wn, axis=1)
    return X_der


def sg_transform(X, wn, deriv_order=0, window=SG_WINDOW, polyorder=SG_POLYORDER):
    window = make_odd_window(window, X.shape[1], polyorder)
    delta = float(np.mean(np.diff(wn)))
    return savgol_filter(
        X,
        window_length=window,
        polyorder=polyorder,
        deriv=deriv_order,
        delta=delta,
        axis=1,
        mode="interp"
    )


def norris_first_derivative(X, wn, segment_length=NORRIS_SEGMENT_LENGTH, segment_spacing=NORRIS_SEGMENT_SPACING):
    s = int(segment_length)
    g = int(segment_spacing)
    if s < 1:
        s = 1
    if g < 1:
        g = 1
    n_points = X.shape[1]
    if n_points <= 2 * g + 1:
        return finite_derivative(X, wn, order=1)
    X_smooth = uniform_filter1d(X, size=s, axis=1, mode="nearest")
    X_der = np.zeros_like(X_smooth, dtype=float)
    for j in range(g, n_points - g):
        denom = wn[j + g] - wn[j - g]
        if np.isclose(denom, 0):
            X_der[:, j] = 0
        else:
            X_der[:, j] = (X_smooth[:, j + g] - X_smooth[:, j - g]) / denom
    X_der[:, :g] = X_der[:, g:g + 1]
    X_der[:, n_points - g:] = X_der[:, n_points - g - 1:n_points - g]
    return X_der


def norris_transform(X, wn, deriv_order=1):
    X_out = X.copy()
    for _ in range(deriv_order):
        X_out = norris_first_derivative(X_out, wn)
    return X_out


def apply_pathlength(X, method):
    if method == "Constant":
        return X.copy()
    elif method == "MSC":
        return msc_global(X)
    elif method == "SNV":
        return snv(X)
    else:
        raise ValueError(f"Unknown pathlength method: {method}")


def apply_spectral_processing(X, wn, spectra_method, smoothing_method):
    if spectra_method == "Spectrum":
        if smoothing_method == "NS":
            return X.copy()
        elif smoothing_method == "SG":
            return sg_transform(X, wn, deriv_order=0)
        elif smoothing_method == "ND":
            raise ValueError("Spectrum + ND is not a valid combination")
    elif spectra_method == "FD":
        if smoothing_method == "NS":
            return finite_derivative(X, wn, order=1)
        elif smoothing_method == "SG":
            return sg_transform(X, wn, deriv_order=1)
        elif smoothing_method == "ND":
            return norris_transform(X, wn, deriv_order=1)
    elif spectra_method == "SD":
        if smoothing_method == "NS":
            return finite_derivative(X, wn, order=2)
        elif smoothing_method == "SG":
            return sg_transform(X, wn, deriv_order=2)
        elif smoothing_method == "ND":
            return norris_transform(X, wn, deriv_order=2)
    raise ValueError(f"Unknown combination: {spectra_method} + {smoothing_method}")


def preprocess_combination(X, wn, pathlength, spectra, smoothing):
    X_step1 = apply_pathlength(X, method=pathlength)
    return apply_spectral_processing(X_step1, wn, spectra_method=spectra, smoothing_method=smoothing)


def build_preprocessing_manifest():
    pathlength_methods = ["Constant", "MSC", "SNV"]
    spectra_smoothing_pairs = [
        ("Spectrum", "NS"),
        ("Spectrum", "SG"),
        ("FD", "NS"),
        ("FD", "SG"),
        ("FD", "ND"),
        ("SD", "NS"),
        ("SD", "SG"),
        ("SD", "ND"),
    ]
    combinations = []
    num = 1
    for path_method in pathlength_methods:
        for spectra_method, smoothing_method in spectra_smoothing_pairs:
            combinations.append({
                "Number": num,
                "Pathlength": path_method,
                "Spectra": spectra_method,
                "Smoothing": smoothing_method
            })
            num += 1
    return combinations


def parse_wavenumber_columns(df, prefix="wn_"):
    wn_cols = [col for col in df.columns if str(col).startswith(prefix)]
    if len(wn_cols) == 0:
        raise ValueError(f"No columns starting with '{prefix}' found")
    wavenumbers = np.array([float(re.sub(prefix, "", str(col))) for col in wn_cols])
    return wn_cols, wavenumbers


def preprocess_24_combinations(df, wn_cols, wavenumbers, output_descending=True):
    sort_idx_asc = np.argsort(wavenumbers)
    wn_asc = wavenumbers[sort_idx_asc]
    X_raw = df[wn_cols].astype(float).values
    X_asc = X_raw[:, sort_idx_asc]
    meta_df = df.drop(columns=wn_cols).reset_index(drop=True)
    combinations = build_preprocessing_manifest()
    out_dfs = []
    for combo in combinations:
        number = combo["Number"]
        path_method = combo["Pathlength"]
        spectra_method = combo["Spectra"]
        smoothing_method = combo["Smoothing"]
        X_step1 = apply_pathlength(X_asc, method=path_method)
        X_processed_asc = apply_spectral_processing(
            X_step1, wn_asc,
            spectra_method=spectra_method,
            smoothing_method=smoothing_method
        )
        if output_descending:
            output_idx = np.argsort(wn_asc)[::-1]
        else:
            output_idx = np.argsort(wn_asc)
        wn_out = wn_asc[output_idx]
        X_out = X_processed_asc[:, output_idx]
        out_wn_cols = [f"wn_{w:.4f}" for w in wn_out]
        out_df = pd.concat([meta_df, pd.DataFrame(X_out, columns=out_wn_cols)], axis=1)
        out_dfs.append((number, path_method, spectra_method, smoothing_method, out_df))
    return out_dfs


class SNVTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return snv(np.asarray(X, dtype=float))


class MSCTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ref_ = None

    def fit(self, X, y=None):
        self.ref_ = np.mean(np.asarray(X, dtype=float), axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_corr = np.zeros_like(X)
        for i in range(X.shape[0]):
            y = X[i, :]
            slope, intercept = np.polyfit(self.ref_, y, 1)
            if np.isclose(slope, 0):
                X_corr[i, :] = y
            else:
                X_corr[i, :] = (y - intercept) / slope
        return X_corr


class SGSmoothingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, deriv_order=0, window=SG_WINDOW, polyorder=SG_POLYORDER):
        self.deriv_order = deriv_order
        self.window = window
        self.polyorder = polyorder

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        wn = np.arange(X.shape[1], dtype=float)
        return sg_transform(X, wn, deriv_order=self.deriv_order, window=self.window, polyorder=self.polyorder)


class NorrisSmoothingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, deriv_order=1):
        self.deriv_order = deriv_order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        wn = np.arange(X.shape[1], dtype=float)
        return norris_transform(X, wn, deriv_order=self.deriv_order)
