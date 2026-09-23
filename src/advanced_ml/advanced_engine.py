# ============================================================
# AutoDS Phase 5
# Advanced ML Optimization Engine
# ============================================================

import json
import shutil

from pathlib import Path


import joblib
import numpy as np
import optuna
import pandas as pd


from sklearn.base import clone

from sklearn.feature_selection import (
    SelectPercentile,
    f_classif,
    f_regression
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    LabelEncoder
)


from src.preprocessor import (
    build_preprocessor
)


from src.advanced_ml.metric_strategy import (
    choose_metric_strategy,
    classification_metrics,
    regression_metrics
)


from src.advanced_ml.model_factory import (
    build_classifier,
    build_regressor
)


# Reduce noisy Optuna output

optuna.logging.set_verbosity(
    optuna.logging.WARNING
)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_phase5_data(
    dataset_path,
    target,
    problem_type,
    schema_report
):

    df = pd.read_csv(
        dataset_path
    )


    excluded_columns = (
        schema_report.get(
            "excluded_columns",
            []
        )
    )


    columns_to_remove = (
        [target]
        +
        excluded_columns
    )


    X = df.drop(
        columns=columns_to_remove,
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
    ].reset_index(
        drop=True
    )


    y = y.loc[
        valid_rows
    ].reset_index(
        drop=True
    )


    label_encoder = None


    if problem_type == "classification":

        label_encoder = (
            LabelEncoder()
        )

        y = pd.Series(
            label_encoder.fit_transform(
                y.astype(str)
            ),
            index=y.index
        )


    return (
        X,
        y,
        label_encoder
    )


# ============================================================
# CROSS VALIDATION STRATEGY
# ============================================================

def build_cv(
    problem_type,
    folds
):

    if problem_type == "classification":

        return StratifiedKFold(

            n_splits=folds,

            shuffle=True,

            random_state=42
        )


    return KFold(

        n_splits=folds,

        shuffle=True,

        random_state=42
    )


# ============================================================
# CLASS IMBALANCE INFORMATION
# ============================================================

def get_imbalance_information(
    y,
    problem_type,
    quality_report
):

    imbalance = False

    scale_pos_weight = 1.0


    if problem_type != "classification":

        return (
            imbalance,
            scale_pos_weight
        )


    imbalance = quality_report.get(
        "class_imbalance_warning",
        False
    )


    counts = (
        pd.Series(
            y
        )
        .value_counts()
    )


    # Only binary scale_pos_weight

    if (
        imbalance
        and len(
            counts
        ) == 2
    ):

        negative = float(
            counts.iloc[0]
        )

        positive = float(
            counts.iloc[1]
        )


        if positive > 0:

            scale_pos_weight = (
                negative
                /
                positive
            )


    return (
        imbalance,
        scale_pos_weight
    )


# ============================================================
# FEATURE SELECTOR
# ============================================================

def build_feature_selector(
    problem_type,
    percentile
):

    if problem_type == "classification":

        return SelectPercentile(

            score_func=f_classif,

            percentile=percentile
        )


    return SelectPercentile(

        score_func=f_regression,

        percentile=percentile
    )


# ============================================================
# BUILD OPTUNA PIPELINE
# ============================================================

def build_trial_pipeline(
    trial,
    model_name,
    problem_type,
    X_train,
    imbalance,
    scale_pos_weight
):

    preprocessor = build_preprocessor(
        X_train
    )


    percentile = trial.suggest_categorical(

        "feature_percentile",

        [
            60,
            80,
            100
        ]
    )


    feature_selector = (
        build_feature_selector(
            problem_type,
            percentile
        )
    )


    if problem_type == "classification":

        model, params = (
            build_classifier(

                model_name,

                trial,

                imbalance=imbalance,

                scale_pos_weight=
                    scale_pos_weight
            )
        )


    else:

        model, params = (
            build_regressor(
                model_name,
                trial
            )
        )


    pipeline = Pipeline(

        steps=[

            (
                "preprocessing",
                preprocessor
            ),

            (
                "feature_selection",
                feature_selector
            ),

            (
                "model",
                model
            )
        ]
    )


    return (
        pipeline,
        params
    )


# ============================================================
# OPTUNA MODEL OPTIMIZATION
# ============================================================

def optimize_model(
    model_name,
    problem_type,
    X_train,
    y_train,
    quality_report,
    metric_strategy,
    cv,
    n_trials
):

    (
        imbalance,
        scale_pos_weight

    ) = get_imbalance_information(

        y_train,

        problem_type,

        quality_report
    )


    def objective(
        trial
    ):

        pipeline, _ = (
            build_trial_pipeline(

                trial,

                model_name,

                problem_type,

                X_train,

                imbalance,

                scale_pos_weight
            )
        )


        scores = cross_val_score(

            pipeline,

            X_train,

            y_train,

            scoring=
                metric_strategy[
                    "cv_scoring"
                ],

            cv=cv,

            n_jobs=1
        )


        return float(
            np.mean(
                scores
            )
        )


    study = optuna.create_study(

        direction="maximize"
    )


    study.optimize(

        objective,

        n_trials=n_trials,

        show_progress_bar=False
    )


    # --------------------------------------------------------
    # REBUILD BEST PIPELINE
    # --------------------------------------------------------

    best_trial = (
        study.best_trial
    )


    final_pipeline, _ = (
        build_trial_pipeline(

            best_trial,

            model_name,

            problem_type,

            X_train,

            imbalance,

            scale_pos_weight
        )
    )


    # --------------------------------------------------------
    # CV RESULTS OF BEST PIPELINE
    # --------------------------------------------------------

    cv_scores = cross_val_score(

        final_pipeline,

        X_train,

        y_train,

        scoring=
            metric_strategy[
                "cv_scoring"
            ],

        cv=cv,

        n_jobs=1
    )


    return {

        "pipeline":
            final_pipeline,

        "best_params":
            best_trial.params,

        "cv_mean":
            round(
                float(
                    np.mean(
                        cv_scores
                    )
                ),
                4
            ),

        "cv_std":
            round(
                float(
                    np.std(
                        cv_scores
                    )
                ),
                4
            ),

        "trial_count":
            len(
                study.trials
            ),

        "study":
            study
    }


# ============================================================
# EVALUATE FITTED MODEL
# ============================================================

def evaluate_fitted_model(
    model,
    X_test,
    y_test,
    problem_type
):

    predictions = model.predict(
        X_test
    )


    if problem_type == "classification":

        probability = None


        # Binary probability

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
                    and probabilities.shape[1] == 2
                ):

                    probability = (
                        probabilities[
                            :,
                            1
                        ]
                    )

            except Exception:

                probability = None


        return classification_metrics(

            y_test,

            predictions,

            probability
        )


    return regression_metrics(

        y_test,

        predictions
    )


# ============================================================
# SAVE OPTUNA TRIALS
# ============================================================

def save_study(
    study,
    model_name,
    output_directory
):

    dataframe = (
        study.trials_dataframe()
    )


    path = (

        output_directory

        /

        f"{model_name}_optuna_trials.csv"
    )


    dataframe.to_csv(
        path,
        index=False
    )


    return str(
        path
    )


# ============================================================
# MAIN PHASE 5 ENGINE
# ============================================================

def run_advanced_ml(

    dataset_path,

    target,

    problem_type,

    schema_report,

    quality_report,

    baseline_model_path,

    n_trials=8,

    cv_folds=3,

    min_improvement=0.002
):

    print(
        "\n======================================"
    )

    print(
        "PHASE 5 — ADVANCED ML ENGINE"
    )

    print(
        "======================================"
    )


    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    X, y, label_encoder = (
        prepare_phase5_data(

            dataset_path,

            target,

            problem_type,

            schema_report
        )
    )


    # --------------------------------------------------------
    # METRIC STRATEGY
    # --------------------------------------------------------

    metric_strategy = (
        choose_metric_strategy(

            problem_type,

            quality_report
        )
    )


    primary_metric = (
        metric_strategy[
            "primary_metric"
        ]
    )


    print(
        "\nPrimary metric:",
        primary_metric
    )


    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    stratify = None


    if problem_type == "classification":

        counts = (
            y.value_counts()
        )

        if counts.min() >= 2:

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


    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    cv = build_cv(

        problem_type,

        cv_folds
    )


    # ========================================================
    # EVALUATE CURRENT CHAMPION FAIRLY
    # ========================================================

    print(
        "\nEvaluating existing champion..."
    )


    baseline_model = joblib.load(
        baseline_model_path
    )


    # clone + refit on Phase-5 training data
    # so comparison is fair

    baseline_model = clone(
        baseline_model
    )


    baseline_cv_scores = (
        cross_val_score(

            baseline_model,

            X_train,

            y_train,

            scoring=
                metric_strategy[
                    "cv_scoring"
                ],

            cv=cv,

            n_jobs=1
        )
    )


    baseline_model.fit(

        X_train,

        y_train
    )


    baseline_metrics = (
        evaluate_fitted_model(

            baseline_model,

            X_test,

            y_test,

            problem_type
        )
    )


    baseline_primary_score = float(

        baseline_metrics[
            primary_metric
        ]
    )


    print(
        "Baseline holdout",
        primary_metric,
        ":",
        baseline_primary_score
    )


    # ========================================================
    # ADVANCED MODELS
    # ========================================================

    model_names = [

        "xgboost",

        "lightgbm",

        "catboost"
    ]


    advanced_results = {}


    output_directory = Path(
        "artifacts/advanced_ml"
    )


    model_directory = (

        output_directory

        /

        "models"
    )


    study_directory = (

        output_directory

        /

        "optuna"
    )


    model_directory.mkdir(

        parents=True,

        exist_ok=True
    )


    study_directory.mkdir(

        parents=True,

        exist_ok=True
    )


    # ========================================================
    # OPTIMIZE EACH MODEL
    # ========================================================

    for model_name in model_names:

        print(
            "\n--------------------------------------"
        )

        print(
            "Optimizing:",
            model_name.upper()
        )

        print(
            "--------------------------------------"
        )


        try:

            optimization = optimize_model(

                model_name,

                problem_type,

                X_train,

                y_train,

                quality_report,

                metric_strategy,

                cv,

                n_trials
            )


            pipeline = (
                optimization[
                    "pipeline"
                ]
            )


            # Fit final candidate

            pipeline.fit(

                X_train,

                y_train
            )


            # Holdout evaluation

            test_metrics = (
                evaluate_fitted_model(

                    pipeline,

                    X_test,

                    y_test,

                    problem_type
                )
            )


            # Save candidate model

            model_path = (

                model_directory

                /

                f"{model_name}_optimized.joblib"
            )


            joblib.dump(

                pipeline,

                model_path
            )


            trial_path = (
                save_study(

                    optimization[
                        "study"
                    ],

                    model_name,

                    study_directory
                )
            )


            advanced_results[
                model_name
            ] = {

                "status":
                    "success",

                "best_params":
                    optimization[
                        "best_params"
                    ],

                "cv_mean":
                    optimization[
                        "cv_mean"
                    ],

                "cv_std":
                    optimization[
                        "cv_std"
                    ],

                "test_metrics":
                    test_metrics,

                "model_path":
                    str(
                        model_path
                    ),

                "optuna_trials":
                    trial_path
            }


            print(
                "CV score:",
                optimization[
                    "cv_mean"
                ]
            )


            print(
                "Holdout",
                primary_metric,
                ":",
                test_metrics[
                    primary_metric
                ]
            )


        except Exception as error:

            print(
                "Model failed:",
                error
            )


            advanced_results[
                model_name
            ] = {

                "status":
                    "failed",

                "error":
                    str(
                        error
                    )
            }


    # ========================================================
    # SELECT ADVANCED WINNER USING CV
    # ========================================================

    successful_models = {

        name:
            result

        for name, result
        in advanced_results.items()

        if result[
            "status"
        ] == "success"
    }


    if not successful_models:

        raise RuntimeError(
            "All Phase-5 models failed."
        )


    advanced_winner = max(

        successful_models,

        key=lambda name:
            successful_models[
                name
            ][
                "cv_mean"
            ]
    )


    winner_result = (
        successful_models[
            advanced_winner
        ]
    )


    advanced_test_score = float(

        winner_result[
            "test_metrics"
        ][
            primary_metric
        ]
    )


    improvement = round(

        advanced_test_score

        -

        baseline_primary_score,

        6
    )


    # ========================================================
    # CHAMPION / CHALLENGER DECISION
    # ========================================================

    promoted = (

        improvement
        >=
        min_improvement
    )


    final_directory = Path(
        "artifacts/advanced_ml"
    )


    final_champion_path = (

        final_directory

        /

        "phase5_final_champion.joblib"
    )


    if promoted:

        print(
            "\nNEW PHASE-5 CHAMPION FOUND"
        )

        print(
            advanced_winner
        )


        shutil.copy2(

            winner_result[
                "model_path"
            ],

            final_champion_path
        )


        final_champion_source = (
            winner_result[
                "model_path"
            ]
        )


    else:

        print(
            "\nAdvanced models did not beat "
            "the existing champion sufficiently."
        )

        print(
            "Keeping existing champion."
        )


        shutil.copy2(

            baseline_model_path,

            final_champion_path
        )


        final_champion_source = (
            baseline_model_path
        )


    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "problem_type":
            problem_type,

        "primary_metric":
            primary_metric,

        "cv_folds":
            cv_folds,

        "optuna_trials_per_model":
            n_trials,

        "baseline": {

            "source_model":
                baseline_model_path,

            "cv_mean":
                round(
                    float(
                        np.mean(
                            baseline_cv_scores
                        )
                    ),
                    4
                ),

            "cv_std":
                round(
                    float(
                        np.std(
                            baseline_cv_scores
                        )
                    ),
                    4
                ),

            "test_metrics":
                baseline_metrics
        },

        "advanced_models":
            advanced_results,

        "selected_advanced_model":
            advanced_winner,

        "advanced_test_score":
            advanced_test_score,

        "baseline_test_score":
            baseline_primary_score,

        "improvement":
            improvement,

        "promoted":
            promoted,

        "final_champion_source":
            final_champion_source,

        "final_champion_path":
            str(
                final_champion_path
            )
    }


    report_path = (

        output_directory

        /

        "phase5_advanced_ml_report.json"
    )


    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            report,

            file,

            indent=4,

            default=str
        )


    # ========================================================
    # SAVE LABEL ENCODER
    # ========================================================

    label_encoder_path = None


    if label_encoder is not None:

        label_encoder_path = (

            output_directory

            /

            "target_label_encoder.joblib"
        )


        joblib.dump(

            label_encoder,

            label_encoder_path
        )


    print(
        "\n======================================"
    )

    print(
        "PHASE 5 RESULTS"
    )

    print(
        "======================================"
    )


    print(
        "\nBaseline score:",
        baseline_primary_score
    )


    print(
        "Advanced winner:",
        advanced_winner
    )


    print(
        "Advanced score:",
        advanced_test_score
    )


    print(
        "Improvement:",
        improvement
    )


    print(
        "Promoted:",
        promoted
    )


    print(
        "\nFinal champion:"
    )

    print(
        final_champion_path
    )


    print(
        "\nReport:"
    )

    print(
        report_path
    )


    return report