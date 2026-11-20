import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rpd = np.std(y_true) / rmse if rmse > 1e-12 else np.inf
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "RPD": rpd,
    }


def make_confusion_df(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_count_df = pd.DataFrame(
        cm,
        index=[f"True_{c}" for c in class_names],
        columns=[f"Pred_{c}" for c in class_names]
    )
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_percent_df = pd.DataFrame(
        cm_percent,
        index=[f"True_{c}" for c in class_names],
        columns=[f"Pred_{c}" for c in class_names]
    )
    return cm_count_df, cm_percent_df
