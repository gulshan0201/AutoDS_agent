# ============================================================
# AutoDS Phase 5
# Dynamic Metric Strategy
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


# ============================================================
# DETERMINE PRIMARY METRIC
# ============================================================

def choose_metric_strategy(
    problem_type,
    quality_report
):

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if problem_type == "classification":

        imbalance = quality_report.get(
            "class_imbalance_warning",
            False
        )

        if imbalance:

            return {
                "primary_metric":
                    "f1_macro",

                "cv_scoring":
                    "f1_macro",

                "direction":
                    "maximize"
            }

        return {
            "primary_metric":
                "f1_weighted",

            "cv_scoring":
                "f1_weighted",

            "direction":
                "maximize"
        }


    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    return {
        "primary_metric":
            "r2",

        "cv_scoring":
            "r2",

        "direction":
            "maximize"
    }


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    y_true,
    y_pred,
    y_probability=None
):

    metrics = {

        "accuracy":
            round(
                accuracy_score(
                    y_true,
                    y_pred
                ),
                4
            ),

        "precision_macro":
            round(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),
                4
            ),

        "recall_macro":
            round(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),
                4
            ),

        "f1_macro":
            round(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),
                4
            ),

        "f1_weighted":
            round(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),
                4
            )
    }


    # Binary ROC-AUC when probability is available

    if y_probability is not None:

        try:

            metrics[
                "roc_auc"
            ] = round(
                roc_auc_score(
                    y_true,
                    y_probability
                ),
                4
            )

        except Exception:

            pass


    return metrics


# ============================================================
# REGRESSION METRICS
# ============================================================

def regression_metrics(
    y_true,
    y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )


    return {

        "mae":
            round(
                mean_absolute_error(
                    y_true,
                    y_pred
                ),
                4
            ),

        "rmse":
            round(
                float(
                    np.sqrt(
                        mse
                    )
                ),
                4
            ),

        "r2":
            round(
                r2_score(
                    y_true,
                    y_pred
                ),
                4
            )
    }