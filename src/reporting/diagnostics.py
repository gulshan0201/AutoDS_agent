from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from src.advanced_ml.metric_strategy import (
    classification_metrics,
    regression_metrics
)


# ============================================================
# GET FINAL ESTIMATOR
# ============================================================

def get_final_estimator(
    pipeline
):

    if hasattr(
        pipeline,
        "named_steps"
    ):

        if "model" in pipeline.named_steps:

            return pipeline.named_steps[
                "model"
            ]

    return pipeline


# ============================================================
# PREPARE TARGET FORMAT
# ============================================================

def prepare_target_for_model(
    model,
    y,
    problem_type,
    label_encoder_path=None
):

    if problem_type != "classification":

        return y


    estimator = get_final_estimator(
        model
    )


    classes = getattr(
        estimator,
        "classes_",
        None
    )


    if classes is None:

        return y


    classes_array = np.asarray(
        classes
    )


    # --------------------------------------------------------
    # Model trained using numeric encoded target
    # --------------------------------------------------------

    if (
        np.issubdtype(
            classes_array.dtype,
            np.number
        )
        and
        not pd.api.types.is_numeric_dtype(
            y
        )
    ):

        if (
            label_encoder_path
            and
            Path(
                label_encoder_path
            ).exists()
        ):

            encoder = joblib.load(
                label_encoder_path
            )

            return pd.Series(
                encoder.transform(
                    y.astype(str)
                ),
                index=y.index
            )


    # --------------------------------------------------------
    # Model trained using string classes
    # --------------------------------------------------------

    if classes_array.dtype.kind in {
        "O",
        "U",
        "S"
    }:

        return y.astype(str)


    return y


# ============================================================
# BUILD DIAGNOSTICS
# ============================================================

def generate_diagnostics(
    dataset_path,
    target,
    problem_type,
    schema_report,
    model_path,
    label_encoder_path=None,
    output_directory="artifacts/phase6/diagnostics"
):

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    df = pd.read_csv(
        dataset_path
    )


    model = joblib.load(
        model_path
    )


    excluded_columns = (
        schema_report.get(
            "excluded_columns",
            []
        )
    )


    X = df.drop(
        columns=[
            target,
            *excluded_columns
        ],
        errors="ignore"
    )


    y = df[
        target
    ]


    valid_rows = y.notna()

    X = X.loc[
        valid_rows
    ]

    y = y.loc[
        valid_rows
    ]


    y = prepare_target_for_model(
        model,
        y,
        problem_type,
        label_encoder_path
    )


    stratify = None

    if problem_type == "classification":

        counts = y.value_counts()

        if (
            len(counts) > 1
            and
            counts.min() >= 2
        ):

            stratify = y


    (
        X_train,
        X_test,
        y_train,
        y_test

    ) = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=stratify
    )


    predictions = model.predict(
        X_test
    )


    image_paths = {}


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if problem_type == "classification":

        probability = None


        if hasattr(
            model,
            "predict_proba"
        ):

            try:

                probabilities = (
                    model.predict_proba(
                        X_test
                    )
                )

                if (
                    probabilities.ndim == 2
                    and
                    probabilities.shape[1] == 2
                ):

                    probability = (
                        probabilities[
                            :,
                            1
                        ]
                    )

            except Exception:

                probability = None


        metrics = classification_metrics(

            y_test,

            predictions,

            probability
        )


        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )


        ConfusionMatrixDisplay.from_predictions(

            y_test,

            predictions,

            ax=ax
        )


        ax.set_title(
            "Final Champion - Confusion Matrix"
        )


        fig.tight_layout()


        confusion_path = (

            output_directory

            /

            "confusion_matrix.png"
        )


        fig.savefig(

            confusion_path,

            dpi=160
        )


        plt.close(
            fig
        )


        image_paths[
            "confusion_matrix"
        ] = str(
            confusion_path
        )


        # ----------------------------------------------------
        # ROC Curve
        # ----------------------------------------------------

        if probability is not None:

            estimator = (
                get_final_estimator(
                    model
                )
            )

            classes = getattr(
                estimator,
                "classes_",
                None
            )


            if (
                classes is not None
                and
                len(classes) == 2
            ):

                try:

                    fig, ax = plt.subplots(
                        figsize=(7, 6)
                    )


                    RocCurveDisplay.from_predictions(

                        y_test,

                        probability,

                        pos_label=
                            classes[1],

                        ax=ax
                    )


                    ax.set_title(
                        "Final Champion - ROC Curve"
                    )


                    fig.tight_layout()


                    roc_path = (

                        output_directory

                        /

                        "roc_curve.png"
                    )


                    fig.savefig(

                        roc_path,

                        dpi=160
                    )


                    plt.close(
                        fig
                    )


                    image_paths[
                        "roc_curve"
                    ] = str(
                        roc_path
                    )


                except Exception:

                    pass


    # ========================================================
    # REGRESSION
    # ========================================================

    else:

        metrics = regression_metrics(

            y_test,

            predictions
        )


        # ----------------------------------------------------
        # Actual vs Predicted
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )


        ax.scatter(

            y_test,

            predictions,

            alpha=0.6
        )


        minimum = min(
            float(np.min(y_test)),
            float(np.min(predictions))
        )


        maximum = max(
            float(np.max(y_test)),
            float(np.max(predictions))
        )


        ax.plot(
            [
                minimum,
                maximum
            ],
            [
                minimum,
                maximum
            ]
        )


        ax.set_xlabel(
            "Actual"
        )

        ax.set_ylabel(
            "Predicted"
        )

        ax.set_title(
            "Actual vs Predicted"
        )


        fig.tight_layout()


        prediction_path = (

            output_directory

            /

            "actual_vs_predicted.png"
        )


        fig.savefig(

            prediction_path,

            dpi=160
        )


        plt.close(
            fig
        )


        image_paths[
            "actual_vs_predicted"
        ] = str(
            prediction_path
        )


        # ----------------------------------------------------
        # Residuals
        # ----------------------------------------------------

        residuals = (

            y_test

            -

            predictions
        )


        fig, ax = plt.subplots(
            figsize=(7, 6)
        )


        ax.scatter(

            predictions,

            residuals,

            alpha=0.6
        )


        ax.axhline(
            0
        )


        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Residual"
        )

        ax.set_title(
            "Residual Analysis"
        )


        fig.tight_layout()


        residual_path = (

            output_directory

            /

            "residual_plot.png"
        )


        fig.savefig(

            residual_path,

            dpi=160
        )


        plt.close(
            fig
        )


        image_paths[
            "residual_plot"
        ] = str(
            residual_path
        )


    return {

        "metrics":
            metrics,

        "images":
            image_paths,

        "test_rows":
            int(
                len(
                    X_test
                )
            )
    }