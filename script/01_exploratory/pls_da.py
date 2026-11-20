import sys
import re
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from common.models import PLSClassifier
from common.plots import set_publication_style


warnings.filterwarnings("ignore")

MAX_LV = 10
RANDOM_STATE = 42
TEST_SAMPLE_IDS = []


def clean_columns(df):
    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["sample", "sampleid", "id", "sample_id"]:
            rename_map[c] = "sample_id"
        elif cl in ["group", "class", "category"]:
            rename_map[c] = "group"
    df = df.rename(columns=rename_map)
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].astype(str).str.strip()
    return df


def read_data(path):
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except Exception:
            df = pd.read_csv(path, encoding="gb18030")
    else:
        df = pd.read_excel(path)
    return clean_columns(df)


def extract_features(df):
    spec_cols = [c for c in df.columns if c not in ["sample_id", "group"]]
    feature_cols = [c for c in spec_cols if c.lower() not in ["dataset_role", "unnamed: 0"]]
    return feature_cols


def main():
    set_publication_style()
    data_dir = Path(__file__).resolve().parent.parent / "data"
    result_dir = Path(__file__).resolve().parent.parent / "results" / "01_exploratory"
    result_dir.mkdir(parents=True, exist_ok=True)

    df = read_data(data_dir / "nir_spectra.csv")

    if "sample_id" not in df.columns or "group" not in df.columns:
        raise ValueError("Data must contain 'sample_id' and 'group' columns")

    raw_ids = ",".join([str(s) for s in TEST_SAMPLE_IDS])
    test_ids = [x.strip() for x in re.split(r"[,\s\n]+", raw_ids) if x.strip()]

    is_test = df["sample_id"].isin(test_ids)
    df_train = df[~is_test].copy()
    df_test = df[is_test].copy()

    if len(df_train) < 5:
        raise ValueError("Insufficient training samples")

    feature_cols = extract_features(df)
    X_train = df_train[feature_cols].values
    y_train = df_train["group"].values
    X_test = df_test[feature_cols].values if len(df_test) > 0 else np.empty((0, len(feature_cols)))
    y_test = df_test["group"].values if len(df_test) > 0 else np.array([])

    min_class_count = pd.Series(y_train).value_counts().min()
    n_splits = min(5, int(min_class_count))
    if n_splits < 2:
        raise ValueError("Insufficient class samples for cross-validation")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    max_possible_lv = min(MAX_LV, X_train.shape[1], X_train.shape[0] - 1)
    lv_range = range(1, max_possible_lv + 1)
    scores_list = []
    best_score = -1
    best_lv = 1

    for n_lv in lv_range:
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", PLSClassifier(n_components=n_lv))
        ])
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
            mean_score = np.mean(cv_scores)
        except Exception:
            mean_score = 0
        scores_list.append(mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_lv = n_lv

    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", PLSClassifier(n_components=best_lv))
    ])
    final_pipeline.fit(X_train, y_train)

    y_train_pred = final_pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    cm_train = confusion_matrix(y_train, y_train_pred, labels=final_pipeline.named_steps["clf"].classes_)

    y_test_pred = None
    test_acc = 0
    cm_test = None
    if len(X_test) > 0:
        y_test_pred = final_pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        cm_test = confusion_matrix(y_test, y_test_pred, labels=final_pipeline.named_steps["clf"].classes_)

    fig = plt.figure(figsize=(15, 5), dpi=300)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(lv_range, scores_list, marker="o", color="#2196F3", linewidth=2)
    ax1.axvline(x=best_lv, color="red", linestyle="--", alpha=0.7)
    ax1.scatter([best_lv], [best_score], color="red", s=100, zorder=5, label=f"Best LV={best_lv}")
    ax1.set_title(f"LVs Optimization (CV)\nBest LV={best_lv}, Acc={best_score:.2%}", fontsize=10)
    ax1.set_xlabel("Number of Latent Variables (LVs)")
    ax1.set_ylabel("CV Accuracy")
    ax1.grid(True, linestyle=":")
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    labels = final_pipeline.named_steps["clf"].classes_
    ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=labels).plot(ax=ax2, cmap="Blues", colorbar=False)
    ax2.set_title(f"Training Set (Recall)\nN={len(X_train)}, Acc={train_acc:.2%}", fontsize=10, fontweight="bold", color="blue")

    ax3 = fig.add_subplot(1, 3, 3)
    if cm_test is not None:
        ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=labels).plot(ax=ax3, cmap="Greens", colorbar=False)
        ax3.set_title(f"Independent Test Set (Pred)\nN={len(X_test)}, Acc={test_acc:.2%}", fontsize=10, fontweight="bold", color="green")
    else:
        ax3.text(0.5, 0.5, "No Test Set", ha="center")
        ax3.set_axis_off()

    plt.tight_layout()
    fig.savefig(result_dir / "pls_da_optimization.png", dpi=600, bbox_inches="tight")
    fig.savefig(result_dir / "pls_da_optimization.pdf", bbox_inches="tight")
    plt.close(fig)

    res_train = df_train[["sample_id", "group"]].copy()
    res_train["predicted"] = y_train_pred
    res_train["dataset"] = "train"

    if len(df_test) > 0:
        res_test = df_test[["sample_id", "group"]].copy()
        res_test["predicted"] = y_test_pred
        res_test["dataset"] = "test"
        df_final = pd.concat([res_train, res_test], ignore_index=True)
    else:
        df_final = res_train

    df_final["correct"] = df_final["group"] == df_final["predicted"]
    df_final.to_csv(result_dir / "PLSDA_single_source_results.csv", index=False, encoding="utf-8-sig")

    report_path = result_dir / "PLSDA_single_source_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PLS-DA single-source modeling report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Best LV: {best_lv}\n")
        f.write(f"CV accuracy: {best_score:.4f}\n")
        f.write(f"Training accuracy: {train_acc:.4f}\n")
        if len(X_test) > 0:
            f.write(f"Test accuracy: {test_acc:.4f}\n")

    print(f"Best LV: {best_lv}, CV accuracy: {best_score:.4f}")
    print(f"Training accuracy: {train_acc:.4f}")
    if len(X_test) > 0:
        print(f"Test accuracy: {test_acc:.4f}")
    print(f"Saved: {result_dir / 'pls_da_optimization.png'}")
    print(f"Saved: {result_dir / 'PLSDA_single_source_results.csv'}")


if __name__ == "__main__":
    main()
