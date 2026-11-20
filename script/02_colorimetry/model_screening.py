import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from common.models import (
    PLSDAClassifier,
    make_lda,
    make_logistic,
    make_svm_linear,
    make_svm_rbf,
    make_rf_classifier,
    make_knn,
)
from common.metrics import classification_metrics, make_confusion_df


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "script" / "data" / "colorimetry.csv"
OUT_DIR = PROJECT_ROOT / "script" / "results" / "02_colorimetry" / "model_screening"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 50


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    sample_id = df.iloc[:, 0].astype(str)
    y_raw = df.iloc[:, 1].astype(str)
    X_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")

    valid_idx = X_df.notna().all(axis=1)
    X_df = X_df.loc[valid_idx]
    y_raw = y_raw.loc[valid_idx]
    sample_id = sample_id.loc[valid_idx]

    X = X_df.values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_

    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {list(class_names)}")

    models = {
        "PLSDA_1comp": PLSDAClassifier(n_components=1),
        "PLSDA_2comp": PLSDAClassifier(n_components=2),
        "PLSDA_3comp": PLSDAClassifier(n_components=3),
        "LDA": make_lda(),
        "Logistic": make_logistic(random_state=RANDOM_STATE),
        "SVM_linear": make_svm_linear(random_state=RANDOM_STATE),
        "SVM_rbf": make_svm_rbf(C=1, gamma="scale", random_state=RANDOM_STATE),
        "RF": make_rf_classifier(n_estimators=200, random_state=RANDOM_STATE),
        "KNN_3": make_knn(n_neighbors=3),
        "KNN_5": make_knn(n_neighbors=5),
    }

    rkf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )

    summary_records = []
    all_pred_records = []

    for model_name, model in models.items():
        print(f"\nRunning model: {model_name}")

        acc_list = []
        f1_list = []
        kappa_list = []
        mcc_list = []

        all_true = []
        all_pred = []

        fold_id = 0

        for train_idx, test_idx in rkf.split(X, y):
            fold_id += 1

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = clone(model)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            metrics = classification_metrics(y_test, y_pred)

            acc_list.append(metrics["accuracy"])
            f1_list.append(metrics["f1_macro"])
            kappa_list.append(metrics["kappa"])
            mcc_list.append(metrics["mcc"])

            all_true.extend(y_test)
            all_pred.extend(y_pred)

            for sid, true_lab, pred_lab in zip(sample_id.iloc[test_idx], y_test, y_pred):
                all_pred_records.append({
                    "Model": model_name,
                    "Fold": fold_id,
                    "sample_id": sid,
                    "True_Label": le.inverse_transform([true_lab])[0],
                    "Pred_Label": le.inverse_transform([pred_lab])[0]
                })

        cm_count_df, cm_percent_df = make_confusion_df(all_true, all_pred, class_names)

        cm_count_df.to_csv(
            OUT_DIR / f"{model_name}_confusion_matrix_count.csv",
            encoding="utf-8-sig"
        )

        cm_percent_df.to_csv(
            OUT_DIR / f"{model_name}_confusion_matrix_percent.csv",
            encoding="utf-8-sig"
        )

        report = classification_report(
            all_true,
            all_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        pd.DataFrame(report).T.to_csv(
            OUT_DIR / f"{model_name}_classification_report.csv",
            encoding="utf-8-sig"
        )

        summary_records.append({
            "Model": model_name,
            "Accuracy_mean": np.mean(acc_list),
            "Accuracy_sd": np.std(acc_list),
            "F1_macro_mean": np.mean(f1_list),
            "F1_macro_sd": np.std(f1_list),
            "Kappa_mean": np.mean(kappa_list),
            "Kappa_sd": np.std(kappa_list),
            "MCC_mean": np.mean(mcc_list),
            "MCC_sd": np.std(mcc_list)
        })

        print(
            f"{model_name} done: "
            f"Acc={np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}, "
            f"F1={np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}, "
            f"Kappa={np.mean(kappa_list):.4f} ± {np.std(kappa_list):.4f}, "
            f"MCC={np.mean(mcc_list):.4f} ± {np.std(mcc_list):.4f}"
        )

    summary_df = pd.DataFrame(summary_records)

    summary_df = summary_df.sort_values(
        by=["Accuracy_mean", "F1_macro_mean", "Kappa_mean", "MCC_mean"],
        ascending=False
    )

    summary_df.to_csv(
        OUT_DIR / "Color_multi_model_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(all_pred_records).to_csv(
        OUT_DIR / "Color_all_prediction_records.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAll models completed.")
    print(f"Results folder: {OUT_DIR}")
    print("\nModel screening summary:")
    print(summary_df)


if __name__ == "__main__":
    main()
