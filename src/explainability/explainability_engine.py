import json

from pathlib import Path


import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib.pyplot as plt


from sklearn.model_selection import (
    train_test_split
)


# ============================================================
# GET FINAL MODEL
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
# TRANSFORM FEATURES
# ============================================================

def transform_features(
    pipeline,
    X
):

    if not hasattr(
        pipeline,
        "named_steps"
    ):

        return (
            X.values,
            np.array(
                X.columns
            )
        )


    if (
        "preprocessing"
        not in pipeline.named_steps
    ):

        return (
            X.values,
            np.array(
                X.columns
            )
        )


    preprocessor = (
        pipeline.named_steps[
            "preprocessing"
        ]
    )


    transformed = (
        preprocessor.transform(
            X
        )
    )


    try:

        feature_names = np.asarray(
            preprocessor.get_feature_names_out()
        )

    except Exception:

        feature_names = np.asarray(

            [
                f"feature_{index}"

                for index in range(
                    transformed.shape[1]
                )
            ]
        )


    # --------------------------------------------------------
    # Phase 5 feature selector
    # --------------------------------------------------------

    if (
        "feature_selection"
        in pipeline.named_steps
    ):

        selector = (
            pipeline.named_steps[
                "feature_selection"
            ]
        )


        transformed = (
            selector.transform(
                transformed
            )
        )


        try:

            support = (
                selector.get_support()
            )

            feature_names = (
                feature_names[
                    support
                ]
            )

        except Exception:

            feature_names = np.asarray(

                [
                    f"selected_feature_{index}"

                    for index in range(
                        transformed.shape[1]
                    )
                ]
            )


    return (
        transformed,
        feature_names
    )


# ============================================================
# DENSE MATRIX
# ============================================================

def to_dense(
    values
):

    if hasattr(
        values,
        "toarray"
    ):

        return values.toarray()

    return np.asarray(
        values
    )


# ============================================================
# NATIVE FEATURE IMPORTANCE
# ============================================================

def native_feature_importance(
    estimator,
    feature_names
):

    importance = None


    if hasattr(
        estimator,
        "feature_importances_"
    ):

        importance = np.asarray(
            estimator.feature_importances_
        )


    elif hasattr(
        estimator,
        "coef_"
    ):

        coefficients = np.asarray(
            estimator.coef_
        )


        if coefficients.ndim == 1:

            importance = np.abs(
                coefficients
            )

        else:

            importance = np.mean(

                np.abs(
                    coefficients
                ),

                axis=0
            )


    if importance is None:

        return None


    return pd.DataFrame({

        "feature":
            feature_names,

        "importance":
            importance

    }).sort_values(

        "importance",

        ascending=False
    )


# ============================================================
# SHAP ENGINE
# ============================================================

def generate_explainability(

    dataset_path,

    target,

    schema_report,

    model_path,

    problem_type,

    output_directory=
        "artifacts/phase6/explainability",

    max_samples=250
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


    pipeline = joblib.load(
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


    valid_rows = (
        y.notna()
    )


    X = X.loc[
        valid_rows
    ]


    y = y.loc[
        valid_rows
    ]


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
        _,
        X_test,
        _,
        _

    ) = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=stratify
    )


    # --------------------------------------------------------
    # Limit SHAP workload
    # --------------------------------------------------------

    X_sample = X_test.head(
        min(
            max_samples,
            len(X_test)
        )
    )


    (
        transformed,
        feature_names

    ) = transform_features(

        pipeline,

        X_sample
    )


    transformed = to_dense(
        transformed
    )


    estimator = (
        get_final_estimator(
            pipeline
        )
    )


    # ========================================================
    # NATIVE IMPORTANCE
    # ========================================================

    native_df = (
        native_feature_importance(

            estimator,

            feature_names
        )
    )


    native_path = None


    if native_df is not None:

        native_path = (

            output_directory

            /

            "native_feature_importance.csv"
        )


        native_df.to_csv(

            native_path,

            index=False
        )


    # ========================================================
    # SHAP
    # ========================================================

    shap_status = "success"

    shap_error = None

    shap_dataframe = None

    local_explanations = []


    try:

        # ----------------------------------------------------
        # TREE MODELS
        # ----------------------------------------------------

        if hasattr(
            estimator,
            "feature_importances_"
        ):

            explainer = (
                shap.TreeExplainer(
                    estimator
                )
            )


            explanation = (
                explainer(
                    transformed
                )
            )


        # ----------------------------------------------------
        # LINEAR MODELS
        # ----------------------------------------------------

        elif hasattr(
            estimator,
            "coef_"
        ):

            background = transformed[
                :
                min(
                    100,
                    len(transformed)
                )
            ]


            explainer = (
                shap.LinearExplainer(

                    estimator,

                    background
                )
            )


            explanation = (
                explainer(
                    transformed
                )
            )


        else:

            raise RuntimeError(
                "Current estimator does not have "
                "a supported fast SHAP explainer."
            )


        values = np.asarray(
            explanation.values
        )


        # ----------------------------------------------------
        # GLOBAL SHAP IMPORTANCE
        # ----------------------------------------------------

        if values.ndim == 2:

            global_importance = (

                np.abs(
                    values
                )
                .mean(
                    axis=0
                )
            )


        elif values.ndim == 3:

            global_importance = (

                np.abs(
                    values
                )
                .mean(
                    axis=(
                        0,
                        2
                    )
                )
            )


        else:

            raise RuntimeError(
                f"Unexpected SHAP shape: {values.shape}"
            )


        shap_dataframe = pd.DataFrame({

            "feature":
                feature_names,

            "mean_abs_shap":
                global_importance

        }).sort_values(

            "mean_abs_shap",

            ascending=False
        )


        shap_csv_path = (

            output_directory

            /

            "shap_global_importance.csv"
        )


        shap_dataframe.to_csv(

            shap_csv_path,

            index=False
        )


        # ----------------------------------------------------
        # GLOBAL SHAP BAR CHART
        # ----------------------------------------------------

        top_features = (

            shap_dataframe
            .head(20)
            .sort_values(
                "mean_abs_shap"
            )
        )


        fig, ax = plt.subplots(
            figsize=(10, 7)
        )


        ax.barh(

            top_features[
                "feature"
            ],

            top_features[
                "mean_abs_shap"
            ]
        )


        ax.set_xlabel(
            "Mean absolute SHAP value"
        )


        ax.set_title(
            "Top SHAP Feature Importance"
        )


        fig.tight_layout()


        shap_plot_path = (

            output_directory

            /

            "shap_global_importance.png"
        )


        fig.savefig(

            shap_plot_path,

            dpi=160
        )


        plt.close(
            fig
        )


        # ----------------------------------------------------
        # LOCAL EXPLANATIONS
        # ----------------------------------------------------

        sample_count = min(
            3,
            len(
                transformed
            )
        )


        predictions = (
            estimator.predict(
                transformed[
                    :sample_count
                ]
            )
        )


        classes = getattr(
            estimator,
            "classes_",
            None
        )


        for row_index in range(
            sample_count
        ):

            if values.ndim == 2:

                row_values = (
                    values[
                        row_index
                    ]
                )


            else:

                predicted_class = (
                    predictions[
                        row_index
                    ]
                )


                if classes is not None:

                    class_indexes = np.where(

                        np.asarray(
                            classes
                        )
                        ==
                        predicted_class
                    )[0]


                    class_index = (

                        int(
                            class_indexes[0]
                        )

                        if len(
                            class_indexes
                        )

                        else 0
                    )

                else:

                    class_index = 0


                row_values = (

                    values[
                        row_index,
                        :,
                        class_index
                    ]
                )


            top_indexes = np.argsort(

                np.abs(
                    row_values
                )

            )[::-1][:10]


            contributions = []


            for feature_index in top_indexes:

                contributions.append({

                    "feature":
                        str(
                            feature_names[
                                feature_index
                            ]
                        ),

                    "feature_value":
                        float(
                            transformed[
                                row_index,
                                feature_index
                            ]
                        ),

                    "shap_value":
                        round(
                            float(
                                row_values[
                                    feature_index
                                ]
                            ),
                            6
                        )
                })


            local_explanations.append({

                "sample_index":
                    int(
                        row_index
                    ),

                "prediction":
                    str(
                        predictions[
                            row_index
                        ]
                    ),

                "top_contributions":
                    contributions
            })


        local_path = (

            output_directory

            /

            "shap_local_explanations.json"
        )


        with open(

            local_path,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                local_explanations,

                file,

                indent=4
            )


    except Exception as error:

        shap_status = "failed"

        shap_error = str(
            error
        )


        shap_csv_path = None

        shap_plot_path = None

        local_path = None


    # ========================================================
    # SUMMARY
    # ========================================================

    top_features_summary = []


    if shap_dataframe is not None:

        top_features_summary = [

            {
                "feature":
                    str(row["feature"]),

                "mean_abs_shap":
                    round(
                        float(
                            row[
                                "mean_abs_shap"
                            ]
                        ),
                        6
                    )
            }

            for _, row
            in shap_dataframe.head(
                15
            ).iterrows()
        ]


    elif native_df is not None:

        top_features_summary = [

            {
                "feature":
                    str(row["feature"]),

                "importance":
                    round(
                        float(
                            row[
                                "importance"
                            ]
                        ),
                        6
                    )
            }

            for _, row
            in native_df.head(
                15
            ).iterrows()
        ]


    return {

        "shap_status":
            shap_status,

        "shap_error":
            shap_error,

        "sample_size":
            int(
                len(
                    X_sample
                )
            ),

        "top_features":
            top_features_summary,

        "native_importance_path":
            (
                str(
                    native_path
                )
                if native_path
                else None
            ),

        "shap_importance_path":
            (
                str(
                    shap_csv_path
                )
                if shap_csv_path
                else None
            ),

        "shap_plot_path":
            (
                str(
                    shap_plot_path
                )
                if shap_plot_path
                else None
            ),

        "local_explanations_path":
            (
                str(
                    local_path
                )
                if local_path
                else None
            )
    }