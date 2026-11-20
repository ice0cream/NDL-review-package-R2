import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix


N_SPLITS = 5
N_REPEATS = 50
N_PERMUTATIONS = 200
RANDOM_STATE = 42
FIXED_N_COMPONENTS = 3
PI_MODE = "macro_f1"
PLS_MAX_ITER = 5000
PLS_TOL = 1e-05


def get_X_y(df):
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

    X = df[wn_cols].astype(float).values
    y = df[group_col].astype(str).values
    return X, y


def one_hot(y):
    classes = np.unique(y)
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(y), len(classes)))
    for i, label in enumerate(y):
        Y[i, idx[label]] = 1
    return Y, classes


def plsda_fit_predict(X_train, y_train, X_valid, n_components):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    Y_train, classes = one_hot(y_train)

    n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    n_comp = max(1, n_comp)

    model = PLSRegression(n_components=n_comp, max_iter=PLS_MAX_ITER, tol=PLS_TOL)
    model.fit(X_train, Y_train)

    pred_score = model.predict(X_valid)
    pred_idx = np.argmax(pred_score, axis=1)
    y_pred = classes[pred_idx]

    return y_pred


def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)

    if PI_MODE == "macro_f1":
        pi = macro_f1
    else:
        pi = np.mean([acc, macro_f1, bacc])

    return acc, macro_f1, bacc, pi


def repeated_cv_plsda(X, y):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    acc_list, f1_list, bacc_list, pi_list = [], [], [], []
    y_true_all, y_pred_all = [], []

    for train_idx, valid_idx in rskf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        y_pred = plsda_fit_predict(X_train, y_train, X_valid, n_components=FIXED_N_COMPONENTS)

        acc, macro_f1, bacc, pi = calc_metrics(y_valid, y_pred)

        acc_list.append(acc)
        f1_list.append(macro_f1)
        bacc_list.append(bacc)
        pi_list.append(pi)

        y_true_all.extend(y_valid)
        y_pred_all.extend(y_pred)

    labels = sorted(np.unique(y))

    return {
        "acc_mean": np.mean(acc_list) * 100,
        "acc_sd": np.std(acc_list, ddof=1) * 100,
        "f1_mean": np.mean(f1_list) * 100,
        "f1_sd": np.std(f1_list, ddof=1) * 100,
        "bacc_mean": np.mean(bacc_list) * 100,
        "bacc_sd": np.std(bacc_list, ddof=1) * 100,
        "pi_mean": np.mean(pi_list) * 100,
        "pi_sd": np.std(pi_list, ddof=1) * 100,
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all, labels=labels),
        "labels": labels
    }


def single_5fold_score(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    acc_list = []
    pi_list = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        y_pred = plsda_fit_predict(X_train, y_train, X_valid, n_components=FIXED_N_COMPONENTS)

        acc, macro_f1, bacc, pi = calc_metrics(y_valid, y_pred)
        acc_list.append(acc)
        pi_list.append(pi)

    return np.mean(acc_list) * 100, np.mean(pi_list) * 100


def permutation_test(X, y, real_acc, real_pi):
    rng = np.random.default_rng(RANDOM_STATE)

    perm_acc = []
    perm_pi = []

    for _ in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)
        acc_p, pi_p = single_5fold_score(X, y_perm)
        perm_acc.append(acc_p)
        perm_pi.append(pi_p)

    perm_acc = np.array(perm_acc)
    perm_pi = np.array(perm_pi)

    p_acc = (np.sum(perm_acc >= real_acc) + 1) / (N_PERMUTATIONS + 1)
    p_pi = (np.sum(perm_pi >= real_pi) + 1) / (N_PERMUTATIONS + 1)

    return {
        "perm_acc_mean": np.mean(perm_acc),
        "perm_acc_sd": np.std(perm_acc, ddof=1),
        "perm_pi_mean": np.mean(perm_pi),
        "perm_pi_sd": np.std(perm_pi, ddof=1),
        "p_acc": p_acc,
        "p_pi": p_pi
    }


def fmt(mean, sd):
    return f"{mean:.2f} ± {sd:.2f}"


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    result_dir = Path(__file__).resolve().parent.parent / "results" / "01_exploratory"
    result_dir.mkdir(parents=True, exist_ok=True)

    folder_path = result_dir / "preprocessed_24_combinations"

    manifest_files = list(folder_path.glob("*preprocessing_manifest.csv"))
    if len(manifest_files) == 0:
        raise FileNotFoundError("preprocessing_manifest.csv not found")

    manifest_path = manifest_files[0]
    manifest = pd.read_csv(manifest_path)

    required_cols = ["Number", "Pathlength", "Spectra", "Smoothing", "File"]
    for col in required_cols:
        if col not in manifest.columns:
            raise ValueError(f"Manifest missing column: {col}")

    records = []
    confusion_matrices = {}

    for _, row in manifest.iterrows():
        number = int(row["Number"])
        pathlength = row["Pathlength"]
        spectra = row["Spectra"]
        smoothing = row["Smoothing"]
        file_name = row["File"]

        csv_path = folder_path / file_name
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        X, y = get_X_y(df)

        cv = repeated_cv_plsda(X, y)
        perm = permutation_test(X, y, cv["acc_mean"], cv["pi_mean"])

        records.append({
            "Number": number,
            "Pathlength": pathlength,
            "Spectra": spectra,
            "Smoothing": smoothing,
            "Prediction accuracy_mean(%)": cv["acc_mean"],
            "Prediction accuracy_SD(%)": cv["acc_sd"],
            "PI_mean": cv["pi_mean"],
            "PI_SD": cv["pi_sd"],
            "Permutation Acc_mean(%)": perm["perm_acc_mean"],
            "Permutation Acc_SD(%)": perm["perm_acc_sd"],
            "Permutation PI_mean": perm["perm_pi_mean"],
            "Permutation PI_SD": perm["perm_pi_sd"],
            "P_acc": perm["p_acc"],
            "P_PI": perm["p_pi"],
            "Macro_F1_mean": cv["f1_mean"],
            "Macro_F1_SD": cv["f1_sd"],
            "Balanced_Accuracy_mean": cv["bacc_mean"],
            "Balanced_Accuracy_SD": cv["bacc_sd"],
            "N_splits": N_SPLITS,
            "N_repeats": N_REPEATS,
            "N_permutations": N_PERMUTATIONS,
            "PLS_components": FIXED_N_COMPONENTS
        })

        confusion_matrices[number] = {
            "labels": cv["labels"],
            "matrix": cv["confusion_matrix"]
        }

    result_df = pd.DataFrame(records).sort_values("Number").reset_index(drop=True)

    mean_df = result_df[[
        "Number", "Pathlength", "Spectra", "Smoothing",
        "Prediction accuracy_mean(%)", "PI_mean",
        "Permutation Acc_mean(%)", "Permutation PI_mean",
        "P_acc", "P_PI"
    ]].copy()

    mean_df = mean_df.rename(columns={
        "Prediction accuracy_mean(%)": "Prediction accuracy(%)",
        "PI_mean": "PI",
        "Permutation Acc_mean(%)": "Permutation Acc(%)",
        "Permutation PI_mean": "Permutation PI"
    })

    for c in ["Prediction accuracy(%)", "PI", "Permutation Acc(%)", "Permutation PI"]:
        mean_df[c] = mean_df[c].round(2)
    mean_df["P_acc"] = mean_df["P_acc"].round(6)
    mean_df["P_PI"] = mean_df["P_PI"].round(6)

    mean_sd_df = pd.DataFrame({
        "Number": result_df["Number"],
        "Pathlength": result_df["Pathlength"],
        "Spectra": result_df["Spectra"],
        "Smoothing": result_df["Smoothing"],
        "Prediction accuracy(%)": [fmt(m, s) for m, s in zip(result_df["Prediction accuracy_mean(%)"], result_df["Prediction accuracy_SD(%)"])],
        "PI": [fmt(m, s) for m, s in zip(result_df["PI_mean"], result_df["PI_SD"])],
        "Permutation Acc(%)": [fmt(m, s) for m, s in zip(result_df["Permutation Acc_mean(%)"], result_df["Permutation Acc_SD(%)"])],
        "Permutation PI": [fmt(m, s) for m, s in zip(result_df["Permutation PI_mean"], result_df["Permutation PI_SD"])],
        "P_acc": result_df["P_acc"].round(6),
        "P_PI": result_df["P_PI"].round(6)
    })

    top_df = result_df.sort_values(["PI_mean", "Prediction accuracy_mean(%)", "P_PI"], ascending=[False, False, True]).reset_index(drop=True)

    top_mean_df = mean_df.merge(top_df[["Number"]].head(10), on="Number", how="inner")
    top_mean_sd_df = mean_sd_df.merge(top_df[["Number"]].head(10), on="Number", how="inner")

    out_dir = result_dir / "pls_screening_5fold"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_mean_csv = out_dir / "PLSDA_repeated5fold_permutation_mean.csv"
    out_mean_sd_csv = out_dir / "PLSDA_repeated5fold_permutation_mean_SD.csv"
    out_full_csv = out_dir / "PLSDA_repeated5fold_permutation_full_metrics.csv"
    out_top_mean_csv = out_dir / "PLSDA_repeated5fold_permutation_top10_mean.csv"
    out_top_mean_sd_csv = out_dir / "PLSDA_repeated5fold_permutation_top10_mean_SD.csv"
    out_excel = out_dir / "PLSDA_repeated5fold_permutation_results.xlsx"
    out_report = out_dir / "PLSDA_repeated5fold_permutation_report.txt"

    mean_df.to_csv(out_mean_csv, index=False, encoding="utf-8-sig")
    mean_sd_df.to_csv(out_mean_sd_csv, index=False, encoding="utf-8-sig")
    result_df.to_csv(out_full_csv, index=False, encoding="utf-8-sig")
    top_mean_df.to_csv(out_top_mean_csv, index=False, encoding="utf-8-sig")
    top_mean_sd_df.to_csv(out_top_mean_sd_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        mean_df.to_excel(writer, sheet_name="Mean_only", index=False)
        mean_sd_df.to_excel(writer, sheet_name="Mean_SD", index=False)
        result_df.to_excel(writer, sheet_name="Full_metrics", index=False)
        top_mean_df.to_excel(writer, sheet_name="Top10_mean", index=False)
        top_mean_sd_df.to_excel(writer, sheet_name="Top10_mean_SD", index=False)

        for number, item in confusion_matrices.items():
            labels = item["labels"]
            cm = item["matrix"]
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            cm_df.to_excel(writer, sheet_name=f"CM_{number:02d}")

    best = top_df.iloc[0]

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("PLS-DA repeated 5-fold CV and permutation test report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input folder: {folder_path}\n")
        f.write(f"Manifest: {manifest_path.name}\n")
        f.write(f"N_splits: {N_SPLITS}\n")
        f.write(f"N_repeats: {N_REPEATS}\n")
        f.write(f"N_permutations: {N_PERMUTATIONS}\n")
        f.write(f"PLS_components: {FIXED_N_COMPONENTS}\n")
        f.write("PI definition: Macro-F1 x 100\n\n")

        f.write("Best preprocessing combination:\n")
        f.write(f"Number {int(best['Number'])}: {best['Pathlength']} + {best['Spectra']} + {best['Smoothing']}\n")
        f.write(f"Accuracy mean: {best['Prediction accuracy_mean(%)']:.2f}%\n")
        f.write(f"Accuracy mean±SD: {best['Prediction accuracy_mean(%)']:.2f} ± {best['Prediction accuracy_SD(%)']:.2f}%\n")
        f.write(f"PI mean: {best['PI_mean']:.2f}\n")
        f.write(f"PI mean±SD: {best['PI_mean']:.2f} ± {best['PI_SD']:.2f}\n")
        f.write(f"P_acc: {best['P_acc']:.6f}\n")
        f.write(f"P_PI: {best['P_PI']:.6f}\n\n")

        f.write("Top 10 combinations:\n")
        for _, r in top_df.head(10).iterrows():
            f.write(f"{int(r['Number']):02d} | {r['Pathlength']} + {r['Spectra']} + {r['Smoothing']} | "
                    f"Acc {r['Prediction accuracy_mean(%)']:.2f} ± {r['Prediction accuracy_SD(%)']:.2f} | "
                    f"PI {r['PI_mean']:.2f} ± {r['PI_SD']:.2f} | "
                    f"P_PI {r['P_PI']:.6f}\n")

    print(f"Saved: {out_mean_csv}")
    print(f"Saved: {out_mean_sd_csv}")
    print(f"Saved: {out_full_csv}")
    print(f"Saved: {out_top_mean_csv}")
    print(f"Saved: {out_top_mean_sd_csv}")
    print(f"Saved: {out_excel}")
    print(f"Saved: {out_report}")
    print(f"Best combination: Number {int(best['Number'])} {best['Pathlength']}+{best['Spectra']}+{best['Smoothing']}")


if __name__ == "__main__":
    main()
