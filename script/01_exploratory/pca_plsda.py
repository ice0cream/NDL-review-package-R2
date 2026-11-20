import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from common.plots import set_publication_style


def load_data(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    group_col = cols.get("group") or cols.get("class")
    if group_col is None:
        raise ValueError("Column 'group' or 'class' not found")
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in feature_cols:
        feature_cols.remove(group_col)
    X = df[feature_cols].fillna(df[feature_cols].mean()).values
    y = df[group_col].values
    return X, y


def plot_3d_confidence_ellipsoid(ax, data, color, n_std=2.795, alpha=0.2):
    if data.shape[0] < 4:
        return
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    radii = np.sqrt(np.maximum(vals, 1e-10)) * n_std

    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    n_rows, n_cols = x.shape
    transform = vecs @ np.diag(radii)
    for i in range(n_rows):
        for j in range(n_cols):
            point = np.array([x[i, j], y[i, j], z[i, j]])
            new_point = transform @ point + mean
            x[i, j], y[i, j], z[i, j] = new_point

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def main():
    set_publication_style()
    data_dir = Path(__file__).resolve().parent.parent / "data"
    result_dir = Path(__file__).resolve().parent.parent / "results" / "01_exploratory"
    result_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(data_dir / "nir_spectra.csv")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    pca_var = pca.explained_variance_ratio_

    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    if y_encoded.shape[1] < 2:
        y_encoded = np.hstack((1 - y_encoded, y_encoded))
    n_pls = min(X.shape[1], 3)
    pls = PLSRegression(n_components=n_pls)
    pls.fit(X_scaled, y_encoded)
    X_pls = pls.x_scores_
    if X_pls.shape[1] < 3:
        X_pls = np.hstack((X_pls, np.zeros((X_pls.shape[0], 3 - X_pls.shape[1]))))

    fig = plt.figure(figsize=(12, 6), dpi=300)

    unique_groups = np.unique(y)
    palette = sns.color_palette("bright", n_colors=len(unique_groups))
    group_colors = {g: palette[i] for i, g in enumerate(unique_groups)}

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_facecolor("white")
    ax1.grid(False)
    for g in unique_groups:
        idx = (y == g)
        if np.sum(idx) >= 4:
            plot_3d_confidence_ellipsoid(ax1, X_pca[idx, :], group_colors[g], alpha=0.2)
        ax1.scatter(X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2], label=g, c=[group_colors[g]], s=30, depthshade=True)

    ax1.set_title("PCA 3D", fontsize=12)
    ax1.set_xlabel(f"PC1 ({pca_var[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca_var[1]:.1%})")
    ax1.set_zlabel(f"PC3 ({pca_var[2]:.1%})")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_facecolor("white")
    ax2.grid(False)
    for g in unique_groups:
        idx = (y == g)
        if np.sum(idx) >= 4 and n_pls >= 3:
            plot_3d_confidence_ellipsoid(ax2, X_pls[idx, :], group_colors[g], alpha=0.2)
        ax2.scatter(X_pls[idx, 0], X_pls[idx, 1], X_pls[idx, 2], c=[group_colors[g]], s=30, depthshade=True)

    ax2.set_title("PLS-DA 3D", fontsize=12)
    ax2.set_xlabel("LV1")
    ax2.set_ylabel("LV2")
    ax2.set_zlabel("LV3")

    patches = [mpatches.Patch(color=group_colors[g], label=str(g), alpha=0.6) for g in unique_groups]
    fig.legend(handles=patches, loc="lower center", ncol=len(unique_groups), frameon=False, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(result_dir / "pca_plsda_3d.png", dpi=600, bbox_inches="tight")
    fig.savefig(result_dir / "pca_plsda_3d.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"PCA variance: PC1={pca_var[0]:.4f}, PC2={pca_var[1]:.4f}, PC3={pca_var[2]:.4f}")
    print(f"PLS components: {n_pls}")
    print(f"Saved: {result_dir / 'pca_plsda_3d.png'}")


if __name__ == "__main__":
    main()
