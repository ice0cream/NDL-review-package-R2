import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

DPI = 600
TOP_POSITIVE_PEAKS = 2
TOP_NEGATIVE_PEAKS = 2
MIN_WAVENUMBER_DISTANCE = 220
MIN_Y_SEPARATION = 420
NEGATIVE_LABEL_X_FRACTION = 0.78
POSITIVE_LABEL_X_OFFSET = 0.05
NEGATIVE_Y_OFFSET = 180
FIGSIZE = (4.2, 4.2)


def safe_name(s):
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|()\[\]{}]+', "_", s)
    s = s.replace(" ", "_")
    return s


def read_table(path):
    path = Path(path)
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


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


def plot_one_compound(coef_df, out_dir):
    compound = str(coef_df["Compounds"].iloc[0])

    coef_df = coef_df.dropna(subset=["Wavenumber_cm-1", "Regression_coefficient"]).copy()
    coef_df = coef_df.sort_values("Wavenumber_cm-1", ascending=False).reset_index(drop=True)

    peaks = choose_bidirectional_peaks(
        coef_df,
        n_pos=TOP_POSITIVE_PEAKS,
        n_neg=TOP_NEGATIVE_PEAKS,
        min_dist=MIN_WAVENUMBER_DISTANCE
    )

    y_min = float(coef_df["Wavenumber_cm-1"].min())
    y_max = float(coef_df["Wavenumber_cm-1"].max())
    adjusted_y = spread_y_positions(peaks["Wavenumber_cm-1"].tolist(), MIN_Y_SEPARATION, y_min, y_max)

    x_min = float(coef_df["Regression_coefficient"].min())
    x_max = float(coef_df["Regression_coefficient"].max())
    x_span = x_max - x_min
    if x_span <= 0:
        x_span = max(abs(x_min), abs(x_max), 1.0)

    left_pad = 0.18 * x_span
    right_pad = 0.24 * x_span
    plot_xmin = x_min - left_pad
    plot_xmax = x_max + right_pad
    pos_x_offset = POSITIVE_LABEL_X_OFFSET * (plot_xmax - plot_xmin)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(coef_df["Regression_coefficient"], coef_df["Wavenumber_cm-1"], linewidth=1.2)
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
    ax.set_title(compound, fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.8)

    fig.savefig(out_dir / f"{safe_name(compound)}_regression_coefficients.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{safe_name(compound)}_regression_coefficients.pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{safe_name(compound)}_regression_coefficients.tif", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    marked_df = pd.DataFrame(marked_rows)
    marked_df.to_csv(out_dir / f"{safe_name(compound)}_marked_peaks.csv", index=False, encoding="utf-8-sig")


def main():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "results" / "03_general_parameters" / "ks_plsr" / "all_best_model_regression_coefficients.csv"
    result_dir = base_dir / "results" / "03_general_parameters" / "plot_regression_coefficients"
    result_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(csv_path)

    required_cols = ["Compounds", "Wavenumber_cm-1", "Regression_coefficient"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    compounds = list(df["Compounds"].dropna().astype(str).unique())
    for cp in compounds:
        sub = df[df["Compounds"].astype(str) == cp].copy()
        plot_one_compound(sub, result_dir)

    print("Done")
    print(result_dir)


if __name__ == "__main__":
    main()
