# ============================================================
# AutoDS Phase 5
# Advanced Model Factory
# ============================================================

from xgboost import (
    XGBClassifier,
    XGBRegressor
)

from lightgbm import (
    LGBMClassifier,
    LGBMRegressor
)

from catboost import (
    CatBoostClassifier,
    CatBoostRegressor
)


# ============================================================
# CLASSIFICATION MODEL
# ============================================================

def build_classifier(
    model_name,
    trial,
    imbalance=False,
    scale_pos_weight=1.0
):

    # ========================================================
    # XGBOOST
    # ========================================================

    if model_name == "xgboost":

        params = {

            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    150,
                    600
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    3,
                    10
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.20,
                    log=True
                ),

            "subsample":
                trial.suggest_float(
                    "subsample",
                    0.70,
                    1.0
                ),

            "colsample_bytree":
                trial.suggest_float(
                    "colsample_bytree",
                    0.70,
                    1.0
                ),

            "min_child_weight":
                trial.suggest_int(
                    "min_child_weight",
                    1,
                    10
                ),

            "reg_alpha":
                trial.suggest_float(
                    "reg_alpha",
                    1e-4,
                    5.0,
                    log=True
                ),

            "reg_lambda":
                trial.suggest_float(
                    "reg_lambda",
                    1e-3,
                    10.0,
                    log=True
                ),

            "random_state":
                42,

            "n_jobs":
                -1,

            "eval_metric":
                "logloss"
        }


        if imbalance:

            params[
                "scale_pos_weight"
            ] = scale_pos_weight


        return (
            XGBClassifier(
                **params
            ),
            params
        )


    # ========================================================
    # LIGHTGBM
    # ========================================================

    if model_name == "lightgbm":

        params = {

            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    150,
                    600
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.20,
                    log=True
                ),

            "num_leaves":
                trial.suggest_int(
                    "num_leaves",
                    15,
                    127
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    3,
                    12
                ),

            "min_child_samples":
                trial.suggest_int(
                    "min_child_samples",
                    10,
                    60
                ),

            "subsample":
                trial.suggest_float(
                    "subsample",
                    0.70,
                    1.0
                ),

            "colsample_bytree":
                trial.suggest_float(
                    "colsample_bytree",
                    0.70,
                    1.0
                ),

            "reg_alpha":
                trial.suggest_float(
                    "reg_alpha",
                    1e-4,
                    5.0,
                    log=True
                ),

            "reg_lambda":
                trial.suggest_float(
                    "reg_lambda",
                    1e-3,
                    10.0,
                    log=True
                ),

            "random_state":
                42,

            "n_jobs":
                -1,

            "verbosity":
                -1
        }


        if imbalance:

            params[
                "class_weight"
            ] = "balanced"


        return (
            LGBMClassifier(
                **params
            ),
            params
        )


    # ========================================================
    # CATBOOST
    # ========================================================

    if model_name == "catboost":

        params = {

            "iterations":
                trial.suggest_int(
                    "iterations",
                    200,
                    700
                ),

            "depth":
                trial.suggest_int(
                    "depth",
                    4,
                    10
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.20,
                    log=True
                ),

            "l2_leaf_reg":
                trial.suggest_float(
                    "l2_leaf_reg",
                    1.0,
                    10.0
                ),

            "random_strength":
                trial.suggest_float(
                    "random_strength",
                    0.1,
                    2.0
                ),

            "random_seed":
                42,

            "verbose":
                False,

            "allow_writing_files":
                False
        }


        if imbalance:

            params[
                "auto_class_weights"
            ] = "Balanced"


        return (
            CatBoostClassifier(
                **params
            ),
            params
        )


    raise ValueError(
        f"Unknown classifier: {model_name}"
    )


# ============================================================
# REGRESSION MODEL
# ============================================================

def build_regressor(
    model_name,
    trial
):

    # ========================================================
    # XGBOOST
    # ========================================================

    if model_name == "xgboost":

        params = {

            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    150,
                    600
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    3,
                    10
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.20,
                    log=True
                ),

            "subsample":
                trial.suggest_float(
                    "subsample",
                    0.70,
                    1.0
                ),

            "colsample_bytree":
                trial.suggest_float(
                    "colsample_bytree",
                    0.70,
                    1.0
                ),

            "reg_alpha":
                trial.suggest_float(
                    "reg_alpha",
                    1e-4,
                    5.0,
                    log=True
                ),

            "reg_lambda":
                trial.suggest_float(
                    "reg_lambda",
                    1e-3,
                    10.0,
                    log=True
                ),

            "random_state":
                42,

            "n_jobs":
                -1
        }


        return (
            XGBRegressor(
                **params
            ),
            params
        )


    # ========================================================
    # LIGHTGBM
    # ========================================================

    if model_name == "lightgbm":

        params = {

            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    150,
                    600
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.20,
                    log=True
                ),

            "num_leaves":
                trial.suggest_int(
                    "num_leaves",
                    15,
                    127
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    3,
                    12
                ),

            "min_child_samples":
                trial.suggest_int(
                    "min_child_samples",
                    10,
                    60
                ),

            "subsample":
                trial.suggest_float(
                    "subsample",
                    0.70,
                    1.0
                ),

            "colsample_bytree":
                trial.suggest_float(
                    "colsample_bytree",
                    0.70,
                    1.0
                ),

            "reg_alpha":
                trial.suggest_float(
                    "reg_alpha",
                    1e-4,
                    5.0,
                    log=True
                ),

            "reg_lambda":
                trial.suggest_float(
                    "reg_lambda",
                    1e-3,
                    10.0,
                    log=True
                ),

            "random_state":
                42,

            "n_jobs":
                -1,

            "verbosity":
                -1
        }


        return (
            LGBMRegressor(
                **params
            ),
            params
        )


    # ========================================================
    # CATBOOST
    # ========================================================

    if model_name == "catboost":

        params = {

            "iterations":
                trial.suggest_int(
                    "iterations",
                    200,
                    700
                ),

            "depth":
                trial.suggest_int(
                    "depth",
                    4,
                    10
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.20,
                    log=True
                ),

            "l2_leaf_reg":
                trial.suggest_float(
                    "l2_leaf_reg",
                    1.0,
                    10.0
                ),

            "random_strength":
                trial.suggest_float(
                    "random_strength",
                    0.1,
                    2.0
                ),

            "random_seed":
                42,

            "verbose":
                False,

            "allow_writing_files":
                False
        }


        return (
            CatBoostRegressor(
                **params
            ),
            params
        )


    raise ValueError(
        f"Unknown regressor: {model_name}"
    )