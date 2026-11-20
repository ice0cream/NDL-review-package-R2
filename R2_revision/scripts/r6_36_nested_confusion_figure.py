# -*- coding: utf-8 -*-
"""Reviewer 6 comment 36: main-text confusion matrix (Fig. 3B) from the NESTED PLS-DA procedure.

Pools the 10 complete nested repeats (58 samples x 10 = 580 outer-OOF predictions); mean counts per repeat come from the
R1 nested analysis table. Drawn with the plotting function and settings of the R1 Fig. 3 panels
(NIR/mean-1_preprocessed_outputs/PLSDA_建模_混淆矩阵图.py, plot_confusion_matrix, normalize=True): row percentages,
Blues colour map, Times New Roman, 4.8 x 4.2 in, 600 dpi.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'R1_revision' / 'comment03_validation' / 'tables' / 'nested_cv_confusion_matrix_mean_counts.csv'
OUT = ROOT / 'R2_revision' / 'R6-36_nested_confusion'
DPI = 600
FIG_SIZE = (4.8, 4.2)
CMAP = "Blues"


def plot_confusion_matrix(cm, labels, out_path, normalize=False):
    """Plotting function of the R1 Fig. 3 panels, unchanged except that the figure is closed instead of shown."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        plot_data = cm / row_sums * 100
        title = "PLS-DA confusion matrix (%)"
        cbar_label = "Percentage (%)"
        fmt = ".1f"
    else:
        plot_data = cm
        title = "PLS-DA confusion matrix"
        cbar_label = "Count"
        fmt = "d"

    plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    im = ax.imshow(plot_data, cmap=CMAP, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Predicted class", fontsize=10, fontweight="bold")
    ax.set_ylabel("True class", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")

    threshold = np.nanmax(plot_data) / 2 if np.nanmax(plot_data) > 0 else 0
    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            if normalize:
                text = f"{plot_data[i, j]:{fmt}}"
            else:
                text = f"{int(plot_data[i, j])}"
            color = "white" if plot_data[i, j] > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".tif"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


mean = pd.read_csv(SRC, index_col=0)
classes = [c.replace('Pred_', '') for c in mean.columns]
pooled = np.rint(mean.to_numpy() * 10).astype(int)
pd.DataFrame(pooled, index=classes, columns=classes).to_csv(OUT / 'nested_confusion_pooled_580.csv', encoding='utf-8-sig')
plot_confusion_matrix(pooled.astype(float), classes, OUT / 'nested_PLSDA_confusion_pooled', normalize=True)
print(pd.DataFrame(pooled, index=classes, columns=classes), f'\naccuracy={np.trace(pooled)/pooled.sum():.4f}')
