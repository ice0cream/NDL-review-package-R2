import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def set_publication_style():
    matplotlib.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["axes.linewidth"] = 1.0
    matplotlib.rcParams["xtick.direction"] = "out"
    matplotlib.rcParams["ytick.direction"] = "out"


def draw_confidence_ellipse(ax, x, y, n_std=2.4477, facecolor="none", edgecolor="red", **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        **kwargs
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_confusion_matrix(cm, labels, out_path, percent=True, model_name="Model"):
    set_publication_style()

    if percent:
        title = f"{model_name} confusion matrix (%)"
        cbar_label = "Percentage (%)"
        fmt = ".1f"
    else:
        title = f"{model_name} confusion matrix"
        cbar_label = "Count"
        fmt = "d"

    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=600)

    im = ax.imshow(cm, cmap="Blues", aspect="auto")

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

    threshold = np.nanmax(cm) / 2 if np.nanmax(cm) > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if percent:
                text = f"{cm[i, j]:{fmt}}"
            else:
                text = f"{int(cm[i, j])}"

            color = "white" if cm[i, j] > threshold else "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=9
            )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()

    fig.savefig(str(out_path) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path) + ".tif", dpi=600, bbox_inches="tight")

    plt.close(fig)
