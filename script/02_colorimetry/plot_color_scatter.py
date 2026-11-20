import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.plots import set_publication_style


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "script" / "data" / "colorimetry.csv"
OUT_DIR = PROJECT_ROOT / "script" / "results" / "02_colorimetry" / "color_scatter"

GROUP_NAME_MAP = {
    "1": "AST",
    "2": "FTR",
    "3": "SFTR",
    "4": "WTR"
}

SHOW_SAMPLE_LABELS = False


def main():
    set_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_cols = ["sample_id", "group", "L", "a", "b"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["group"] = df["group"].astype(str)
    df["group_name"] = df["group"].map(GROUP_NAME_MAP).fillna(df["group"])

    unique_groups = sorted(df["group_name"].unique())
    default_colors = [
        "#4C78A8", "#59A14F", "#B279A2", "#F28E2B",
        "#E15759", "#76B7B2", "#EDC948", "#9C755F"
    ]
    color_map = {g: default_colors[i % len(default_colors)] for i, g in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=300)

    for g in unique_groups:
        sub = df[df["group_name"] == g]
        color = color_map[g]

        ax.scatter(
            sub["a"],
            sub["b"],
            s=55,
            alpha=0.85,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            label=g
        )

    if SHOW_SAMPLE_LABELS:
        for _, row in df.iterrows():
            ax.text(
                row["a"] + 0.015,
                row["b"] + 0.015,
                str(row["sample_id"]),
                fontsize=7,
                alpha=0.8
            )

    x_min, x_max = df["a"].min(), df["a"].max()
    y_min, y_max = df["b"].min(), df["b"].max()

    x_pad = (x_max - x_min) * 0.15 if x_max > x_min else 0.2
    y_pad = (y_max - y_min) * 0.15 if y_max > y_min else 0.2

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel("a*", fontsize=12)
    ax.set_ylabel("b*", fontsize=12)
    ax.set_title("CIE L*a*b* color coordinates", fontsize=13, pad=10)

    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)

    ax.legend(
        title="Group",
        fontsize=9,
        title_fontsize=10,
        frameon=False,
        loc="best"
    )

    plt.tight_layout()

    fig.savefig(OUT_DIR / "Lab_ab_scatter_optimized.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Lab_ab_scatter_optimized.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "Lab_ab_scatter_optimized.tif", dpi=300, bbox_inches="tight")
    plt.close(fig)

    centroids = (
        df.groupby("group_name")[["L", "a", "b"]]
        .mean()
        .reset_index()
    )
    centroids.to_csv(OUT_DIR / "Lab_group_centroids.csv", index=False, encoding="utf-8-sig")

    print("Color scatter plot completed.")
    print(f"Output folder: {OUT_DIR}")
    print("Saved files:")
    print("1) Lab_ab_scatter_optimized.png / pdf / tif")
    print("2) Lab_group_centroids.csv")


if __name__ == "__main__":
    main()
