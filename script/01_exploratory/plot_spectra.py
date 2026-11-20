import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.plots import set_publication_style


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

    plt.figure(figsize=(5.2, 3.8), dpi=300)

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
        for i in range(X_sorted.shape[0]):
            g = groups[i]
            plt.plot(wavenumbers_sorted, X_sorted[i, :], color=group_color[g], linewidth=0.8, alpha=0.70)
    else:
        for i in range(X_sorted.shape[0]):
            plt.plot(wavenumbers_sorted, X_sorted[i, :], linewidth=0.8, alpha=0.75)

    plt.xlabel("Wavenumber/cm$^{-1}$", fontsize=10, fontweight="bold")
    plt.ylabel("Absorbance", fontsize=10, fontweight="bold")
    plt.xlim(wavenumbers_sorted.max(), wavenumbers_sorted.min())
    plt.tick_params(axis="both", labelsize=8, width=1.0, length=3)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(result_dir / "nir_spectra.png", dpi=600, bbox_inches="tight")
    plt.savefig(result_dir / "nir_spectra.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: {result_dir / 'nir_spectra.png'}")
    print(f"Saved: {result_dir / 'nir_spectra.pdf'}")


if __name__ == "__main__":
    main()
