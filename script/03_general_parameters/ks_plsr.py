import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from common.preprocessing import parse_wavenumber_columns

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


TRAIN_RATIO = 0.70
CV_FOLDS = 5
MAX_LV = 15
RANDOM_STATE = 42
KS_SCALE_X = True
DPI = 600

TOP_POSITIVE_PEAKS = 2
TOP_NEGATIVE_PEAKS = 2
MIN_Y_SEPARATION = 420
MIN_WAVENUMBER_DISTANCE = 220
NEGATIVE_LABEL_X_FRACTION = 0.78
POSITIVE_LABEL_X_OFFSET = 0.05
NEGATIVE_Y_OFFSET = 180

SAMPLE_ID_COL_CANDIDATES = ["sample_id", "sample", "id", "name"]
GROUP_COL_CANDIDATES = ["group", "groups", "class", "label", "type"]


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        for enc in ["utf-8-sig", "utf-8", "gbk"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                pass
        return pd.read_csv(path)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def find_col(df, candidates, required=True):
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Column not found: {candidates}")
    return None


def safe_name(s):
    s = str(s)
    s = re.sub(r"[\\/:*?\"<>|()\[\]{}]+", "_", s)
    s = s.replace(" ", "_")
    return s


def kennard_stone(X, n_cal, scale=True):
    X = np.asarray(X, dtype=float)
    if scale:
        X_ks = StandardScaler().fit_transform(X)
    else:
        X_ks = X.copy()

    n = X_ks.shape[0]
    if n_cal >= n:
        return np.arange(n), np.array([], dtype=int)
    if n_cal < 2:
        raise ValueError("n_cal must be at least 2")

    diff = X_ks[:, None, :] - X_ks[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [i, j]
    remaining = set(range(n)) - set(selected)

    while len(selected) < n_cal:
        rem_list = list(remaining)
        min_d = dist[np.ix_(rem_list, selected)].min(axis=1)
        next_idx = rem_list[int(np.argmax(min_d))]
        selected.append(next_idx)
        remaining.remove(next_idx)

    cal_idx = np.array(selected, dtype=int)
    pred_idx = np.array(sorted(list(remaining)), dtype=int)
    return cal_idx, pred_idx


def pearson_r(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return np.corrcoef(y_true, y_pred)[0, 1]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def fit_predict_plsr(X_cal, y_cal, X_pred, lv):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_pred_s = scaler.transform(X_pred)

    lv = int(min(lv, X_cal_s.shape[0] - 1, X_cal_s.shape[1]))
    lv = max(1, lv)

    model = PLSRegression(n_components=lv, scale=False, max_iter=5000, tol=1e-07)
    model.fit(X_cal_s, y_cal)

    y_cal_hat = model.predict(X_cal_s).ravel()
    y_pred_hat = model.predict(X_pred_s).ravel()
    return model, scaler, y_cal_hat, y_pred_hat


def cv_predict_plsr(X_cal, y_cal, lv, cv_folds=5):
    n = X_cal.shape[0]
    cv_folds = min(cv_folds, n)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    y_cv_hat = np.full(n, np.nan)

    for train_idx, valid_idx in kf.split(X_cal):
        X_tr, X_va = X_cal[train_idx], X_cal[valid_idx]
        y_tr = y_cal[train_idx]
        _, _, _, y_va_hat = fit_predict_plsr(X_tr, y_tr, X_va, lv)
        y_cv_hat[valid_idx] = y_va_hat
    return y_cv_hat


def evaluate_one_model(X, y, cal_idx, pred_idx, max_lv=15):
    X_cal, X_pred = X[cal_idx], X[pred_idx]
    y_cal, y_pred = y[cal_idx], y[pred_idx]

    max_allowed_lv = min(max_lv, X_cal.shape[0] - 2, X_cal.shape[1])
    max_allowed_lv = max(1, max_allowed_lv)

    cv_records = []
    for lv in range(1, max_allowed_lv + 1):
        y_cv_hat = cv_predict_plsr(X_cal, y_cal, lv, CV_FOLDS)
        rmsecv = rmse(y_cal, y_cv_hat)
        rcv = pearson_r(y_cal, y_cv_hat)
        cv_records.append({"LV": lv, "RMSECV": rmsecv, "RCV": rcv, "RCV2": rcv ** 2 if pd.notna(rcv) else np.nan})

    cv_df = pd.DataFrame(cv_records)
    best_lv = int(cv_df.sort_values(["RMSECV", "LV"], ascending=[True, True]).iloc[0]["LV"])

    model, scaler, y_cal_hat, y_pred_hat = fit_predict_plsr(X_cal, y_cal, X_pred, best_lv)
    y_cv_hat = cv_predict_plsr(X_cal, y_cal, best_lv, CV_FOLDS)

    rc = pearson_r(y_cal, y_cal_hat)
    rp = pearson_r(y_pred, y_pred_hat)
    rcv = pearson_r(y_cal, y_cv_hat)

    rmsec = rmse(y_cal, y_cal_hat)
    rmsep = rmse(y_pred, y_pred_hat)
    rmsecv = rmse(y_cal, y_cv_hat)
    rpd = np.std(y_pred, ddof=1) / rmsep if rmsep > 0 else np.inf

    metrics = {
        "RMSEC": rmsec, "RC": rc, "RC2": rc ** 2 if pd.notna(rc) else np.nan,
        "RMSEP": rmsep, "RP": rp, "RP2": rp ** 2 if pd.notna(rp) else np.nan,
        "RMSECV": rmsecv, "RCV": rcv, "RCV2": rcv ** 2 if pd.notna(rcv) else np.nan,
        "LV": best_lv, "RPD": rpd
    }

    pred_df = pd.DataFrame({
        "Set": ["Calibration"] * len(cal_idx) + ["Validation"] * len(pred_idx),
        "Index": list(cal_idx) + list(pred_idx),
        "Actual": list(y_cal) + list(y_pred),
        "Calculated": list(y_cal_hat) + list(y_pred_hat)
    })

    return metrics, pred_df, cv_df, model, scaler


def choose_side_peaks(side_df, n_keep=2, min_dist=220, positive=True):
    if side_df.empty:
        return side_df.copy()

    ranked = side_df.sort_values("Regression_coefficient", ascending=not positive).copy()

    selected_rows = []
    selected_wn = []

    for _, row in ranked.iterrows():
        wn = float(row["Wavenumber_cm-1"])
        if all(abs(wn - prev) >= min_dist for prev in selected_wn):
            selected_rows.append(row)
            selected_wn.append(wn)
        if len(selected_rows) >= n_keep:
            break

    if len(selected_rows) == 0:
        return side_df.iloc[0:0].copy()

    return pd.DataFrame(selected_rows)


def exclude_oscillating_points(coef_df, neighbor=0):
    if len(coef_df) < 3:
        return coef_df.copy()
    df = coef_df.copy().sort_values("Wavenumber_cm-1", ascending=True).reset_index(drop=True)
    coefs = df["Regression_coefficient"].values
    diffs = np.abs(np.diff(coefs))
    threshold = np.percentile(diffs, 95)
    bad = np.zeros(len(coefs), dtype=bool)
    bad[1:] |= diffs > threshold
    bad[:-1] |= diffs > threshold
    for _ in range(neighbor):
        bad[1:] |= bad[:-1]
        bad[:-1] |= bad[1:]
    return df[~bad].copy()


def choose_bidirectional_peaks(coef_df, n_pos=2, n_neg=2, min_dist=220):
    pos_df = coef_df[coef_df["Regression_coefficient"] > 0].copy()
    neg_df = coef_df[coef_df["Regression_coefficient"] < 0].copy()

    pos_top = choose_side_peaks(pos_df, n_keep=n_pos, min_dist=min_dist, positive=True)
    neg_top = choose_side_peaks(neg_df, n_keep=n_neg, min_dist=min_dist, positive=False)

    peaks = pd.concat([pos_top, neg_top], ignore_index=True)
    peaks = peaks.sort_values("Wavenumber_cm-1", ascending=False).reset_index(drop=True)
    return peaks


def spread_y_positions(y_values, min_sep, y_min, y_max):
    y_values = list(y_values)
    if len(y_values) == 0:
        return []

    adjusted = [y_values[0]]
    for y in y_values[1:]:
        new_y = y
        if adjusted[-1] - new_y < min_sep:
            new_y = adjusted[-1] - min_sep
        adjusted.append(new_y)

    if adjusted[-1] < y_min:
        shift = y_min - adjusted[-1]
        adjusted = [yy + shift for yy in adjusted]

    if adjusted[0] > y_max:
        shift = adjusted[0] - y_max
        adjusted = [yy - shift for yy in adjusted]

    adjusted = [min(max(yy, y_min), y_max) for yy in adjusted]
    return adjusted


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "03_general_parameters"
    result_dir = base_dir / "results" / "03_general_parameters" / "ks_plsr"
    result_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_folder = data_dir / "mean-1_24_preprocessed"
    target_file = data_dir / "general_parameters.csv"

    manifest_files = list(preprocessed_folder.glob("*preprocessing_manifest.csv"))
    if len(manifest_files) == 0:
        raise FileNotFoundError("preprocessing_manifest.csv not found in preprocessed folder")

    manifest_path = manifest_files[0]
    manifest = read_table(manifest_path)
    for c in ["Number", "Pathlength", "Spectra", "Smoothing", "File"]:
        if c not in manifest.columns:
            raise ValueError(f"Manifest missing column: {c}")

    target_df = read_table(target_file)
    target_sample_col = find_col(target_df, SAMPLE_ID_COL_CANDIDATES)
    target_group_col = find_col(target_df, GROUP_COL_CANDIDATES, required=False)
    target_df[target_sample_col] = target_df[target_sample_col].astype(str).str.strip()

    target_exclude = {target_sample_col}
    if target_group_col is not None:
        target_exclude.add(target_group_col)

    target_cols = []
    for c in target_df.columns:
        if c in target_exclude:
            continue
        vals = pd.to_numeric(target_df[c], errors="coerce")
        if vals.notna().sum() > 0:
            target_df[c] = vals
            target_cols.append(c)

    if len(target_cols) == 0:
        raise ValueError("No numeric target columns found")

    all_records = []
    all_predictions = []
    all_cv_details = []
    split_records = []
    best_model_artifacts = {}

    total_tasks = len(manifest) * len(target_cols)
    pbar = tqdm(total=total_tasks, desc="PLSR modeling") if tqdm is not None else None

    for _, row in manifest.iterrows():
        number = int(row["Number"])
        pathlength = row["Pathlength"]
        spectra = row["Spectra"]
        smoothing = row["Smoothing"]
        csv_path = preprocessed_folder / str(row["File"])

        if not csv_path.exists():
            warnings.warn(f"File not found, skipped: {csv_path}")
            if pbar is not None:
                pbar.update(len(target_cols))
            continue

        spec_df = read_table(csv_path)
        spec_sample_col = find_col(spec_df, SAMPLE_ID_COL_CANDIDATES)
        spec_group_col = find_col(spec_df, GROUP_COL_CANDIDATES, required=False)
        spec_df[spec_sample_col] = spec_df[spec_sample_col].astype(str).str.strip()

        wn_cols, _ = parse_wavenumber_columns(spec_df, prefix="wn_")

        keep_cols = [spec_sample_col] + ([spec_group_col] if spec_group_col is not None else []) + wn_cols
        merged = spec_df[keep_cols].merge(target_df, left_on=spec_sample_col, right_on=target_sample_col, how="inner", suffixes=("_x", "_y"))
        if merged.shape[0] < 10:
            raise ValueError(f"Too few matched samples for {csv_path.name}: {merged.shape[0]}")

        X = merged[wn_cols].astype(float).values
        n_cal = int(round(TRAIN_RATIO * X.shape[0]))
        cal_idx, pred_idx = kennard_stone(X, n_cal=n_cal, scale=KS_SCALE_X)

        for idx in cal_idx:
            split_records.append({"Number": number, "sample_id": merged.iloc[idx][spec_sample_col], "Set": "Calibration"})
        for idx in pred_idx:
            split_records.append({"Number": number, "sample_id": merged.iloc[idx][spec_sample_col], "Set": "Validation"})

        for compound in target_cols:
            valid_mask = merged[compound].notna().values
            if valid_mask.sum() != len(valid_mask):
                X_use = X[valid_mask]
                y_use = merged.loc[valid_mask, compound].astype(float).values
                n_cal_use = int(round(TRAIN_RATIO * X_use.shape[0]))
                cal_idx_use, pred_idx_use = kennard_stone(X_use, n_cal=n_cal_use, scale=KS_SCALE_X)
                sample_ids_use = merged.loc[valid_mask, spec_sample_col].values
            else:
                X_use = X
                y_use = merged[compound].astype(float).values
                cal_idx_use, pred_idx_use = cal_idx, pred_idx
                sample_ids_use = merged[spec_sample_col].values

            metrics, pred_df, cv_detail, model, scaler = evaluate_one_model(X_use, y_use, cal_idx_use, pred_idx_use, max_lv=MAX_LV)

            record = {
                "Compounds": compound, "Number": number, "Pathlength": pathlength, "Spectra": spectra, "Smoothing": smoothing,
                **metrics, "n_total": len(y_use), "n_calibration": len(cal_idx_use), "n_prediction": len(pred_idx_use), "File": csv_path.name
            }
            all_records.append(record)

            pred_df["Compounds"] = compound
            pred_df["Number"] = number
            pred_df["Pathlength"] = pathlength
            pred_df["Spectra"] = spectra
            pred_df["Smoothing"] = smoothing
            pred_df["sample_id"] = [sample_ids_use[i] for i in pred_df["Index"].values]
            all_predictions.append(pred_df)

            cv_detail["Compounds"] = compound
            cv_detail["Number"] = number
            cv_detail["Pathlength"] = pathlength
            cv_detail["Spectra"] = spectra
            cv_detail["Smoothing"] = smoothing
            all_cv_details.append(cv_detail)

            best_model_artifacts[(compound, number)] = {
                "model": model, "scaler": scaler, "X": X_use, "y": y_use,
                "cal_idx": cal_idx_use, "pred_idx": pred_idx_use, "sample_ids": sample_ids_use,
                "wn_cols": wn_cols, "pathlength": pathlength, "spectra": spectra, "smoothing": smoothing
            }

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    results_df = pd.DataFrame(all_records)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    cv_details_df = pd.concat(all_cv_details, ignore_index=True)
    split_df = pd.DataFrame(split_records).drop_duplicates()

    results_df["meets_priority"] = (results_df["RP2"] >= 0.90) & (results_df["RPD"] >= 3) & (results_df["RCV2"] > 0.80)

    results_df = results_df.sort_values(
        ["Compounds", "meets_priority", "LV", "RMSEP"],
        ascending=[True, False, True, True]
    ).reset_index(drop=True)

    qualified_df = results_df[results_df["meets_priority"]].copy()
    qualified_df = qualified_df.drop(columns=["meets_priority"]).reset_index(drop=True)

    best_parts = []
    top5_parts = []
    for compound, group in results_df.groupby("Compounds", sort=False):
        qualified_group = group[group["meets_priority"]]
        if len(qualified_group) > 0:
            best_parts.append(qualified_group.head(1))
            top5_parts.append(qualified_group.head(5))
        else:
            fb = group.sort_values(["RP2", "RPD", "RMSEP"], ascending=[False, False, True])
            best_parts.append(fb.head(1))
            top5_parts.append(fb.head(5))

    best_df = pd.concat(best_parts, ignore_index=True) if best_parts else results_df.iloc[0:0].copy()
    top5_df = pd.concat(top5_parts, ignore_index=True) if top5_parts else results_df.iloc[0:0].copy()

    results_df = results_df.drop(columns=["meets_priority"])

    main_cols = ["Compounds", "Number", "Pathlength", "Spectra", "Smoothing", "RMSEC", "RC", "RC2", "RMSEP", "RP", "RP2",
                 "RMSECV", "RCV", "RCV2", "LV", "RPD", "n_total", "n_calibration", "n_prediction", "File"]
    results_df = results_df[main_cols]
    best_df = best_df[main_cols]
    top5_df = top5_df[main_cols]
    qualified_df = qualified_df[main_cols] if len(qualified_df) > 0 else pd.DataFrame(columns=main_cols)

    results_df.to_csv(result_dir / "PLSR_KS_all_preprocessing_results.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(result_dir / "PLSR_KS_best_model_per_compound.csv", index=False, encoding="utf-8-sig")
    top5_df.to_csv(result_dir / "PLSR_KS_top5_models_per_compound.csv", index=False, encoding="utf-8-sig")
    qualified_df.to_csv(result_dir / "PLSR_KS_qualified_models.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv(result_dir / "PLSR_KS_predictions_all_models.csv", index=False, encoding="utf-8-sig")
    cv_details_df.to_csv(result_dir / "PLSR_KS_LV_CV_details.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(result_dir / "KS_split_records.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(result_dir / "PLSR_KS_semiquant_results.xlsx", engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="All_results", index=False)
        qualified_df.to_excel(writer, sheet_name="Qualified_models", index=False)
        best_df.to_excel(writer, sheet_name="Best_per_compound", index=False)
        top5_df.to_excel(writer, sheet_name="Top5_per_compound", index=False)
        predictions_df.to_excel(writer, sheet_name="Predictions", index=False)
        cv_details_df.to_excel(writer, sheet_name="LV_CV_details", index=False)
        split_df.to_excel(writer, sheet_name="KS_split", index=False)

    plot_dir = result_dir / "prediction_plots"
    plot_dir.mkdir(exist_ok=True)

    for _, best in best_df.iterrows():
        compound = best["Compounds"]
        number = int(best["Number"])
        sub = predictions_df[(predictions_df["Compounds"] == compound) & (predictions_df["Number"] == number)].copy()

        fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=DPI)

        cal = sub[sub["Set"] == "Calibration"]
        val = sub[sub["Set"] == "Validation"]

        ax.scatter(cal["Actual"], cal["Calculated"], label="Calibration", s=28, alpha=0.70)
        ax.scatter(val["Actual"], val["Calculated"], label="Validation", s=28, alpha=0.70)

        all_vals = pd.concat([sub["Actual"], sub["Calculated"]]).astype(float)
        vmin, vmax = all_vals.min(), all_vals.max()
        pad = (vmax - vmin) * 0.08 if vmax > vmin else 1

        coef_line = np.polyfit(sub["Actual"], sub["Calculated"], 1)
        xx = np.linspace(vmin - pad, vmax + pad, 100)
        yy = np.polyval(coef_line, xx)
        ax.plot(xx, yy, linewidth=1.3)

        ax.set_xlim(vmin - pad, vmax + pad)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Calculated")

        ax.text(
            0.03, 0.97, f"{compound}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10.5, fontweight="bold"
        )
        info_text = (
            f"RMSEC: {best['RMSEC']:.3f}    Corr. Coeff.: {best['RC']:.4f}\n"
            f"RMSEP: {best['RMSEP']:.3f}    Corr. Coeff.: {best['RP']:.4f}\n"
            f"{int(best['LV'])} factors used"
        )
        ax.text(
            0.03, 0.90, info_text,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5
        )

        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        name = safe_name(compound)
        fig.savefig(plot_dir / f"{name}_best_PLSR_prediction.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(plot_dir / f"{name}_best_PLSR_prediction.pdf", bbox_inches="tight")
        fig.savefig(plot_dir / f"{name}_best_PLSR_prediction.tif", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    coef_plot_dir = result_dir / "regression_coefficients_plots"
    coef_plot_dir.mkdir(exist_ok=True)
    coef_data_dir = result_dir / "regression_coefficients_data"
    coef_data_dir.mkdir(exist_ok=True)

    all_coef_records = []

    for _, best in best_df.iterrows():
        compound = best["Compounds"]
        number = int(best["Number"])
        art = best_model_artifacts[(compound, number)]
        model = art["model"]
        wn_cols = art["wn_cols"]

        coef_vals = model.coef_.ravel()
        wn_vals = []
        for c in wn_cols:
            try:
                wn_vals.append(float(str(c).replace("wn_", "")))
            except Exception:
                wn_vals.append(np.nan)
        wn_vals = np.array(wn_vals, dtype=float)

        coef_df = pd.DataFrame({
            "Compounds": compound, "Number": number,
            "Pathlength": art["pathlength"], "Spectra": art["spectra"], "Smoothing": art["smoothing"],
            "Wavenumber_cm-1": wn_vals, "Regression_coefficient": coef_vals
        }).sort_values("Wavenumber_cm-1", ascending=False)

        coef_df.to_csv(coef_data_dir / f"{safe_name(compound)}_best_model_regression_coefficients.csv", index=False, encoding="utf-8-sig")
        all_coef_records.append(coef_df)

        coef_plot = coef_df.dropna(subset=["Wavenumber_cm-1", "Regression_coefficient"]).copy()
        coef_plot = coef_plot.sort_values("Wavenumber_cm-1", ascending=False).reset_index(drop=True)

        peaks = choose_bidirectional_peaks(
            exclude_oscillating_points(coef_plot[coef_plot["Wavenumber_cm-1"] <= 9000]),
            n_pos=TOP_POSITIVE_PEAKS,
            n_neg=TOP_NEGATIVE_PEAKS,
            min_dist=MIN_WAVENUMBER_DISTANCE
        )

        y_min = float(coef_plot["Wavenumber_cm-1"].min())
        y_max = float(coef_plot["Wavenumber_cm-1"].max())
        adjusted_y = spread_y_positions(peaks["Wavenumber_cm-1"].tolist(), MIN_Y_SEPARATION, y_min, y_max)

        x_min = float(coef_plot["Regression_coefficient"].min())
        x_max = float(coef_plot["Regression_coefficient"].max())
        x_span = x_max - x_min
        if x_span <= 0:
            x_span = max(abs(x_min), abs(x_max), 1.0)

        left_pad = 0.18 * x_span
        right_pad = 0.24 * x_span
        plot_xmin = x_min - left_pad
        plot_xmax = x_max + right_pad
        pos_x_offset = POSITIVE_LABEL_X_OFFSET * (plot_xmax - plot_xmin)

        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=DPI)
        ax.plot(coef_plot["Regression_coefficient"], coef_plot["Wavenumber_cm-1"], linewidth=1.2)
        ax.axvline(0, linestyle="--", linewidth=1)

        neg_counter = 0
        marked_rows = []

        for i, (_, rr) in enumerate(peaks.iterrows()):
            x = float(rr["Regression_coefficient"])
            y = float(rr["Wavenumber_cm-1"])
            y_text = adjusted_y[i]

            if x >= 0:
                x_text = min(x + pos_x_offset, plot_xmax - 0.02 * x_span)
                ha = "left"
            else:
                x_text = plot_xmin + NEGATIVE_LABEL_X_FRACTION * (plot_xmax - plot_xmin)
                sign = 1 if (neg_counter % 2 == 0) else -1
                y_text = np.clip(y_text + sign * NEGATIVE_Y_OFFSET, y_min, y_max)
                ha = "left"
                neg_counter += 1

            ax.annotate(
                f"{int(round(y))}cm$^{{-1}}$",
                xy=(x, y),
                xytext=(x_text, y_text),
                fontsize=7.4,
                ha=ha,
                va="center",
                arrowprops=dict(arrowstyle="-", linewidth=0.7, shrinkA=0, shrinkB=0)
            )

            marked_rows.append({
                "Compounds": compound,
                "Wavenumber_cm-1": y,
                "Regression_coefficient": x,
                "Label_x_position": x_text,
                "Label_y_position": y_text,
                "Side": "Positive" if x >= 0 else "Negative"
            })

        ax.set_xlim(plot_xmin, plot_xmax)
        ax.set_xlabel("Beta Coefficients")
        ax.set_ylabel("Wavenumber(cm$^{-1}$)")
        ax.set_title(str(compound), fontsize=12, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout(pad=0.8)

        fig.savefig(coef_plot_dir / f"{safe_name(compound)}_regression_coefficients.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(coef_plot_dir / f"{safe_name(compound)}_regression_coefficients.pdf", dpi=DPI, bbox_inches="tight")
        fig.savefig(coef_plot_dir / f"{safe_name(compound)}_regression_coefficients.tif", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        marked_df = pd.DataFrame(marked_rows)
        marked_df.to_csv(coef_data_dir / f"{safe_name(compound)}_marked_peaks.csv", index=False, encoding="utf-8-sig")

    if len(all_coef_records) > 0:
        pd.concat(all_coef_records, ignore_index=True).to_csv(result_dir / "all_best_model_regression_coefficients.csv", index=False, encoding="utf-8-sig")

    report_path = result_dir / "PLSR_KS_semiquant_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("NIR semi-quantitative PLSR modeling with KS split\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Preprocessed folder: {preprocessed_folder}\n")
        f.write(f"Target file: {target_file}\n")
        f.write(f"Manifest: {manifest_path.name}\n")
        f.write(f"Train ratio: {TRAIN_RATIO}\n")
        f.write(f"CV folds within calibration set: {CV_FOLDS}\n")
        f.write(f"Max LV: {MAX_LV}\n")
        f.write(f"KS scale X: {KS_SCALE_X}\n\n")
        f.write("Best model per compound:\n")
        for _, r in best_df.iterrows():
            f.write(
                f"{r['Compounds']} | Number {int(r['Number'])}: {r['Pathlength']} + {r['Spectra']} + {r['Smoothing']} | "
                f"LV={int(r['LV'])} | RC2={r['RC2']:.4f}, RP2={r['RP2']:.4f}, RCV2={r['RCV2']:.4f}, "
                f"RMSEP={r['RMSEP']:.4f}, RPD={r['RPD']:.3f}\n"
            )

    print("Done")
    print(result_dir)


if __name__ == "__main__":
    main()
