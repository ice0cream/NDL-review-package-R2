import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.plots import set_publication_style


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "script" / "data" / "colorimetry.csv"
OUT_DIR = PROJECT_ROOT / "script" / "results" / "02_colorimetry" / "color_matrix"

DPI = 600
FIG_SIZE = (6.2, 5.2)

GROUP_NAME_MAP = {
    "1": "AST",
    "2": "FTR",
    "3": "SFTR",
    "4": "WTR"
}


def main():
    set_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if INPUT_FILE.suffix.lower() == ".csv":
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)

    cols_lower = {str(c).lower(): c for c in df.columns}

    group_col = None
    for key in ["group", "groups", "class", "label", "type"]:
        if key in cols_lower:
            group_col = cols_lower[key]
            break

    if group_col is None:
        raise ValueError("No grouping column found. Name it group / class / label / type.")

    for key in ["l", "a", "b"]:
        if key not in cols_lower:
            raise ValueError(f"Missing column: {key}. Ensure L, a, b columns exist.")

    L_col = cols_lower["l"]
    a_col = cols_lower["a"]
    b_col = cols_lower["b"]

    df[group_col] = df[group_col].astype(str)
    df["Group_name"] = df[group_col].map(GROUP_NAME_MAP).fillna(df[group_col])

    for c in [L_col, a_col, b_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[L_col, a_col, b_col, "Group_name"]).copy()

    centroids = (
        df.groupby("Group_name")[[L_col, a_col, b_col]]
        .mean()
        .reset_index()
    )

    centroids = centroids.rename(columns={
        L_col: "L_mean",
        a_col: "a_mean",
        b_col: "b_mean"
    })

    groups = centroids["Group_name"].tolist()

    delta_e = pd.DataFrame(
        np.zeros((len(groups), len(groups))),
        index=groups,
        columns=groups
    )

    for i, g1 in enumerate(groups):
        L1, a1, b1 = centroids.loc[
            centroids["Group_name"] == g1,
            ["L_mean", "a_mean", "b_mean"]
        ].values[0]

        for j, g2 in enumerate(groups):
            L2, a2, b2 = centroids.loc[
                centroids["Group_name"] == g2,
                ["L_mean", "a_mean", "b_mean"]
            ].values[0]

            de = np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)
            delta_e.loc[g1, g2] = de

    centroids.to_csv(OUT_DIR / "Lab_group_centroids.csv", index=False, encoding="utf-8-sig")
    delta_e.to_csv(OUT_DIR / "DeltaE_ab_matrix.csv", encoding="utf-8-sig")

    excel_path = OUT_DIR / "DeltaE_ab_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw_data", index=False)
        centroids.to_excel(writer, sheet_name="Group_centroids", index=False)
        delta_e.to_excel(writer, sheet_name="DeltaE_matrix")

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    mask = np.triu(np.ones_like(delta_e.values, dtype=bool), k=1)
    masked_data = np.ma.masked_where(mask, delta_e.values)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        masked_data,
        cmap=cmap,
        vmin=0
    )

    ax.set_xticks(np.arange(len(groups)))
    ax.set_yticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_yticklabels(groups, fontsize=10)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title("Pairwise color difference matrix ($\\Delta E^*_{ab}$)", fontsize=13, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$\\Delta E^*_{ab}$", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    fig.savefig(OUT_DIR / "DeltaE_ab_heatmap.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / "DeltaE_ab_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "DeltaE_ab_heatmap.tif", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("Color difference matrix completed.")
    print(f"Output folder: {OUT_DIR}")
    print("Saved files:")
    print("1. Lab_group_centroids.csv")
    print("2. DeltaE_ab_matrix.csv")
    print("3. DeltaE_ab_heatmap.png / pdf / tif")
    print("4. DeltaE_ab_results.xlsx")


if __name__ == "__main__":
    main()
