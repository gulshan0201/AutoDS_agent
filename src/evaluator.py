from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


def evaluate_classification(
    y_true,
    y_pred
):

    return {
        "accuracy": round(
            accuracy_score(
                y_true,
                y_pred
            ),
            4
        ),

        "precision": round(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            4
        ),

        "recall": round(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            4
        ),

        "f1": round(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            4
        )
    }


def evaluate_regression(
    y_true,
    y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return {
        "mae": round(
            mean_absolute_error(
                y_true,
                y_pred
            ),
            4
        ),

        "rmse": round(
            np.sqrt(mse),
            4
        ),

        "r2": round(
            r2_score(
                y_true,
                y_pred
            ),
            4
        )
    }