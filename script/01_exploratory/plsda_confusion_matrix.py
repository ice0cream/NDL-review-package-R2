import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from common.plots import set_publication_style


N_SPLITS = 5
N_REPEATS = 50
RANDOM_STATE = 42
PLS_COMPONENTS = 3
PLS_MAX_ITER = 5000
PLS_TOL = 1e-5
DPI = 600
FIG_SIZE = (4.8, 4.2)
CMAP = "Blues"


def read_data(file_path):
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

    wn_cols = [c for c in df.columns if str(c).startswith("wn_")]
    if len(wn_cols) == 0:
        raise ValueError("No wn_ spectral columns found")

    group_col = None
    for c in df.columns:
        if str(c).lower() == "group":
            group_col = c
            break
    if group_col is None:
        raise ValueError("No group column found")

    sample_col = None
    for c in df.columns:
        if str(c).lower() in ["sample_id", "sample", "id"]:
            sample_col = c
            break

    X = df[wn_cols].astype(float).values
    y = df[group_col].astype(str).values

    if sample_col is None:
        sample_id = np.arange(len(df)).astype(str)
    else:
        sample_id = df[sample_col].astype(str).values

    return df, X, y, sample_id


def one_hot(y_train):
    classes = np.unique(y_train)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(y_train), len(classes)))
    for i, label in enumerate(y_train):
        Y[i, class_to_idx[label]] = 1
    return Y, classes


def plsda_predict(X_train, y_train, X_valid):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)

    Y_train, classes = one_hot(y_train)
    n_comp = min(PLS_COMPONENTS, X_train_s.shape[0] - 1, X_train_s.shape[1])
    n_comp = max(1, n_comp)

    model = PLSRegression(n_components=n_comp, max_iter=PLS_MAX_ITER, tol=PLS_TOL)
    model.fit(X_train_s, Y_train)

    y_score = model.predict(X_valid_s)
    pred_idx = np.argmax(y_score, axis=1)
    y_pred = classes[pred_idx]
    return y_pred


def run_repeated_cv(X, y, sample_id):
    min_class_n = pd.Series(y).value_counts().min()
    n_splits = min(N_SPLITS, int(min_class_n))
    if n_splits < 2:
        raise ValueError("Insufficient class samples for cross-validation")

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    rows = []
    fold_metrics = []
    for split_id, (train_idx, valid_idx) in enumerate(rskf.split(X, y), start=1):
        repeat_id = (split_id - 1) // n_splits + 1
        fold_id = (split_id - 1) % n_splits + 1

        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        y_pred = plsda_predict(X_train, y_train, X_valid)

        acc = accuracy_score(y_valid, y_pred)
        macro_f1 = f1_score(y_valid, y_pred, average="macro", zero_division=0)
        bacc = balanced_accuracy_score(y_valid, y_pred)
        kappa = cohen_kappa_score(y_valid, y_pred)
        mcc = matthews_corrcoef(y_valid, y_pred)

        fold_metrics.append({
            "repeat": repeat_id,
            "fold": fold_id,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "balanced_accuracy": bacc,
            "kappa": kappa,
            "mcc": mcc,
        })

        for sid, yt, yp in zip(sample_id[valid_idx], y_valid, y_pred):
            rows.append({
                "repeat": repeat_id,
                "fold": fold_id,
                "sample_id": sid,
                "true_label": yt,
                "predicted_label": yp,
                "correct": yt == yp,
            })

    pred_df = pd.DataFrame(rows)
    fold_df = pd.DataFrame(fold_metrics)
    return pred_df, fold_df, n_splits


def summarize_metrics(pred_df, fold_df):
    y_true = pred_df["true_label"].values
    y_pred = pred_df["predicted_label"].values

    summary = {
        "aggregate_accuracy": accuracy_score(y_true, y_pred) * 100,
        "aggregate_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "aggregate_balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100,
        "aggregate_kappa": cohen_kappa_score(y_true, y_pred),
        "aggregate_mcc": matthews_corrcoef(y_true, y_pred),
        "fold_accuracy_mean": fold_df["accuracy"].mean() * 100,
        "fold_accuracy_sd": fold_df["accuracy"].std(ddof=1) * 100,
        "fold_macro_f1_mean": fold_df["macro_f1"].mean() * 100,
        "fold_macro_f1_sd": fold_df["macro_f1"].std(ddof=1) * 100,
        "fold_kappa_mean": fold_df["kappa"].mean(),
        "fold_kappa_sd": fold_df["kappa"].std(ddof=1),
        "fold_mcc_mean": fold_df["mcc"].mean(),
        "fold_mcc_sd": fold_df["mcc"].std(ddof=1),
    }
    return summary


def plot_confusion_matrix(cm, labels, out_path, normalize=False):
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

    set_publication_style()
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


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    result_dir = Path(__file__).resolve().parent.parent / "results" / "01_exploratory"
    result_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / "preprocessed_nir_spectra.csv"
    df, X, y, sample_id = read_data(file_path)

    out_dir = result_dir / "plsda_confusion_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df, fold_df, actual_splits = run_repeated_cv(X, y, sample_id)

    labels = sorted(np.unique(y))
    cm = confusion_matrix(pred_df["true_label"], pred_df["predicted_label"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    summary = summarize_metrics(pred_df, fold_df)
    summary_df = pd.DataFrame([summary])
    report = classification_report(pred_df["true_label"], pred_df["predicted_label"], labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T

    pred_df.to_csv(out_dir / "PLSDA_out_of_fold_predictions.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(out_dir / "PLSDA_fold_metrics.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(out_dir / "PLSDA_confusion_matrix_counts.csv", encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "PLSDA_summary_metrics.csv", index=False, encoding="utf-8-sig")
    report_df.to_csv(out_dir / "PLSDA_classification_report.csv", encoding="utf-8-sig")

    with pd.ExcelWriter(out_dir / "PLSDA_results.xlsx", engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        fold_df.to_excel(writer, sheet_name="Fold_metrics", index=False)
        pred_df.to_excel(writer, sheet_name="Predictions", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_counts")
        report_df.to_excel(writer, sheet_name="Class_report")

    plot_confusion_matrix(cm, labels, out_dir / "PLSDA_confusion_matrix_counts", normalize=False)
    plot_confusion_matrix(cm, labels, out_dir / "PLSDA_confusion_matrix_percent", normalize=True)

    report_path = out_dir / "PLSDA_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PLS-DA repeated cross-validation report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {file_path}\n")
        f.write(f"Samples: {len(y)}\n")
        f.write(f"Classes: {', '.join(labels)}\n")
        f.write(f"N_splits: {actual_splits}\n")
        f.write(f"N_repeats: {N_REPEATS}\n")
        f.write(f"PLS_components: {PLS_COMPONENTS}\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Aggregate accuracy: {summary['aggregate_accuracy']:.2f}%")
    print(f"Fold accuracy: {summary['fold_accuracy_mean']:.2f} ± {summary['fold_accuracy_sd']:.2f}%")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
