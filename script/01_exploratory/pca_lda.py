import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from common.plots import draw_confidence_ellipse, set_publication_style


def load_data(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    group_col = cols.get("group")
    if group_col is None:
        raise ValueError("Column 'group' not found")
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in feature_cols:
        feature_cols.remove(group_col)
    X = df[feature_cols]
    y = df[group_col]
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())
    return X.values, y.values, feature_cols


def main():
    set_publication_style()
    data_dir = Path(__file__).resolve().parent.parent / "data"
    result_dir = Path(__file__).resolve().parent.parent / "results" / "01_exploratory"
    result_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_data(data_dir / "nir_spectra.csv")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_sil = silhouette_score(X_pca, y)
    pca_var = pca.explained_variance_ratio_

    n_classes = len(np.unique(y))
    n_lda = min(n_classes - 1, X.shape[1])
    if n_lda < 1:
        n_lda = 1

    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    X_lda = lda.fit_transform(X_scaled, y)

    if n_lda == 1:
        X_lda_plot = np.zeros((len(X_lda), 2))
        X_lda_plot[:, 0] = X_lda.ravel()
        lda_sil = 0
    else:
        X_lda_plot = X_lda[:, :2]
        try:
            lda_sil = silhouette_score(X_lda_plot, y)
        except Exception:
            lda_sil = 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    unique_groups = np.unique(y)
    palette = sns.color_palette("viridis", n_colors=len(unique_groups))
    group_color_map = dict(zip(unique_groups, palette))

    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="viridis", s=80, ax=axes[0], alpha=0.9, legend=False)
    for grp in unique_groups:
        mask = (y == grp)
        if np.sum(mask) > 2:
            draw_confidence_ellipse(axes[0], X_pca[mask, 0], X_pca[mask, 1],
                                    n_std=2.4477, edgecolor=group_color_map[grp], linestyle="--", linewidth=2)

    axes[0].set_title(f"PCA (Silhouette: {pca_sil:.2f})")
    axes[0].set_xlabel(f"PC1 ({pca_var[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca_var[1]:.1%})")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    sns.scatterplot(x=X_lda_plot[:, 0], y=X_lda_plot[:, 1], hue=y, palette="viridis", s=80, ax=axes[1], alpha=0.9, legend=False)
    if n_lda > 1:
        for grp in unique_groups:
            mask = (y == grp)
            if np.sum(mask) > 2:
                draw_confidence_ellipse(axes[1], X_lda_plot[mask, 0], X_lda_plot[mask, 1],
                                        n_std=2.4477, edgecolor=group_color_map[grp], linestyle="--", linewidth=2)

    title_lda = f"LDA (Silhouette: {lda_sil:.2f})" if n_lda > 1 else "LDA (1D Projection)"
    axes[1].set_title(title_lda)
    axes[1].set_xlabel("LD1")
    axes[1].set_ylabel("LD2")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=group_color_map[g], markersize=8, label=str(g)) for g in unique_groups]
    fig.legend(handles=handles, loc="lower center", ncol=len(unique_groups), frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(result_dir / "pca_lda_projection.png", dpi=600, bbox_inches="tight")
    fig.savefig(result_dir / "pca_lda_projection.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"PCA variance: PC1={pca_var[0]:.4f}, PC2={pca_var[1]:.4f}")
    print(f"LDA components: {n_lda}")
    print(f"Saved: {result_dir / 'pca_lda_projection.png'}")


if __name__ == "__main__":
    main()
