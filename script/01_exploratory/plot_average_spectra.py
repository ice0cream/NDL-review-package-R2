import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.plots import set_publication_style


PEAK_POSITIONS = [8310, 6790, 5650, 5170, 4740]
LABEL_OFFSETS = {
    8310: {"dx": 0, "dy": 0.06},
    6790: {"dx": 0, "dy": 0.08},
    5650: {"dx": 0, "dy": 0.06},
    5170: {"dx": 260, "dy": 0.11},
    4740: {"dx": -260, "dy": 0.055},
}


def main():
    set_publication_style()
    data_dir = Path(__file__).resolve().parent.parent / "data"
    result_dir = Path(__file__).resolve().parent.parent / "results" / "01_exploratory"
    result_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / "nir_spectra.csv"
    df = pd.read_csv(file_path)

    wn_cols = [col for col in df.columns if str(col).startswith("wn_")]
    if len(wn_cols) == 0:
        raise ValueError("No wn_ columns found")

    wavenumbers = np.array([float(re.sub("wn_", "", str(col))) for col in wn_cols])
    X = df[wn_cols].values

    sort_idx = np.argsort(wavenumbers)[::-1]
    wavenumbers_sorted = wavenumbers[sort_idx]
    X_sorted = X[:, sort_idx]

    ref_spectrum = np.mean(X_sorted, axis=0)

    peak_indices = [np.argmin(np.abs(wavenumbers_sorted - p)) for p in PEAK_POSITIONS]
    peak_x = wavenumbers_sorted[peak_indices]
    peak_y = ref_spectrum[peak_indices]

    plt.figure(figsize=(6.2, 4.6), dpi=300)

    color_list = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#0072B2",
        "#F0E442",
        "#999999"
    ]

    if "group" in df.columns:
        groups = df["group"].values
        unique_groups = sorted(pd.unique(groups))
        group_color = {g: color_list[i % len(color_list)] for i, g in enumerate(unique_groups)}
        drawn_groups = set()
        for i in range(X_sorted.shape[0]):
            g = groups[i]
            label = str(g) if g not in drawn_groups else None
            plt.plot(wavenumbers_sorted, X_sorted[i, :], color=group_color[g], linewidth=0.8, alpha=0.55, label=label)
            drawn_groups.add(g)
    else:
        for i in range(X_sorted.shape[0]):
            plt.plot(wavenumbers_sorted, X_sorted[i, :], color="gray", linewidth=0.8, alpha=0.55)

    y_min = np.min(X_sorted)
    y_max = np.max(X_sorted)
    y_range = y_max - y_min

    for target_p, x, y in zip(PEAK_POSITIONS, peak_x, peak_y):
        dx = LABEL_OFFSETS[target_p]["dx"]
        dy = LABEL_OFFSETS[target_p]["dy"]
        text_x = x + dx
        text_y = y + dy * y_range
        plt.annotate(f"{int(target_p)} cm$^{{-1}}$", xy=(x, y), xytext=(text_x, text_y), textcoords="data",
                     ha="center", va="bottom", fontsize=5.5,
                     arrowprops=dict(arrowstyle="-", lw=0.55, color="black", shrinkA=0, shrinkB=0))

    plt.xlabel("Wavenumber/cm$^{-1}$", fontsize=10, fontweight="bold")
    plt.ylabel("Absorbance", fontsize=10, fontweight="bold")
    plt.xlim(wavenumbers_sorted.max(), wavenumbers_sorted.min())
    plt.tick_params(axis="both", labelsize=8, width=1.0, length=3)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    if "group" in df.columns:
        plt.legend(frameon=False, fontsize=8, loc="best", handlelength=1.6, borderpad=0.3, labelspacing=0.4)

    plt.tight_layout()
    plt.savefig(result_dir / "nir_spectra_marked_peaks.png", dpi=600, bbox_inches="tight")
    plt.savefig(result_dir / "nir_spectra_marked_peaks.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: {result_dir / 'nir_spectra_marked_peaks.png'}")
    print(f"Saved: {result_dir / 'nir_spectra_marked_peaks.pdf'}")


if __name__ == "__main__":
    main()
