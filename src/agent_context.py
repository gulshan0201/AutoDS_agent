from pathlib import Path

import pandas as pd


# ============================================================
# NUMERIC SUMMARY
# ============================================================

def build_numeric_summary(
    df,
    target,
    max_columns=15
):

    summary = {}

    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    for column in numeric_columns[
        :max_columns
    ]:

        series = df[
            column
        ]

        valid = series.dropna()

        if len(valid) == 0:
            continue

        summary[column] = {

            "count":
                int(
                    valid.count()
                ),

            "missing":
                int(
                    series.isna().sum()
                ),

            "mean":
                round(
                    float(valid.mean()),
                    4
                ),

            "median":
                round(
                    float(valid.median()),
                    4
                ),

            "std":
                round(
                    float(valid.std()),
                    4
                ),

            "min":
                round(
                    float(valid.min()),
                    4
                ),

            "max":
                round(
                    float(valid.max()),
                    4
                ),

            "skew":
                round(
                    float(valid.skew()),
                    4
                )
        }

    return summary


# ============================================================
# CATEGORICAL SUMMARY
# ============================================================

def build_categorical_summary(
    df,
    target,
    max_columns=15
):

    summary = {}

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    for column in categorical_columns[
        :max_columns
    ]:

        counts = (
            df[column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(5)
        )

        summary[column] = {

            "unique_values":
                int(
                    df[column]
                    .nunique(
                        dropna=True
                    )
                ),

            "top_values": {
                str(key):
                    int(value)

                for key, value
                in counts.items()
            }
        }

    return summary


# ============================================================
# TARGET SUMMARY
# ============================================================

def build_target_summary(
    df,
    target,
    problem_type
):

    y = df[
        target
    ].dropna()

    if problem_type == "classification":

        counts = (
            y.astype(str)
            .value_counts()
        )

        total = counts.sum()

        return {

            str(class_name): {

                "count":
                    int(count),

                "percentage":
                    round(
                        float(
                            count
                            /
                            total
                            *
                            100
                        ),
                        2
                    )

            }

            for class_name, count
            in counts.items()
        }


    return {

        "count":
            int(
                y.count()
            ),

        "mean":
            round(
                float(
                    y.mean()
                ),
                4
            ),

        "median":
            round(
                float(
                    y.median()
                ),
                4
            ),

        "std":
            round(
                float(
                    y.std()
                ),
                4
            ),

        "min":
            round(
                float(
                    y.min()
                ),
                4
            ),

        "max":
            round(
                float(
                    y.max()
                ),
                4
            )
    }


# ============================================================
# REGRESSION CORRELATIONS
# ============================================================

def build_target_correlations(
    df,
    target,
    problem_type
):

    if problem_type != "regression":

        return {}

    numeric_df = (
        df.select_dtypes(
            include=["number"]
        )
    )

    if target not in numeric_df.columns:

        return {}

    correlations = (
        numeric_df
        .corr()[target]
        .drop(
            labels=[target],
            errors="ignore"
        )
        .dropna()
        .sort_values(
            key=lambda x:
            x.abs(),
            ascending=False
        )
        .head(10)
    )

    return {

        column:
            round(
                float(value),
                4
            )

        for column, value
        in correlations.items()
    }


# ============================================================
# MAIN CONTEXT BUILDER
# ============================================================

def build_agent_context(
    dataset_path,
    target,
    goal,
    autods_result
):

    processed_path = (
        autods_result.get(
            "processed_dataset"
        )
    )

    if (
        processed_path
        and Path(
            processed_path
        ).exists()
    ):

        df = pd.read_csv(
            processed_path
        )

    else:

        df = pd.read_csv(
            dataset_path
        )


    problem_type = (
        autods_result[
            "problem_type"
        ]
    )


    eda_summary = {

        "numeric_summary":
            build_numeric_summary(
                df,
                target
            ),

        "categorical_summary":
            build_categorical_summary(
                df,
                target
            ),

        "target_summary":
            build_target_summary(
                df,
                target,
                problem_type
            ),

        "target_correlations":
            build_target_correlations(
                df,
                target,
                problem_type
            )
    }


    return {

    "goal":
        goal,

    "dataset_path":
        dataset_path,

    "target":
        target,

    "problem_type":
        problem_type,

    "profile":
        autods_result.get(
            "profile",
            {}
        ),

    "schema_report":
        autods_result.get(
            "schema_report",
            {}
        ),

    "quality_report":
        autods_result.get(
            "quality_report",
            {}
        ),

    "model_results":
        autods_result.get(
            "results",
            {}
        ),

    "best_model_name":
        autods_result.get(
            "best_model_name",
            ""
        ),

    "processed_dataset":
        autods_result.get(
            "processed_dataset",
            dataset_path
        ),

    "model_path":
        autods_result.get(
            "model_path",
            ""
        ),

    "eda_summary":
        eda_summary,

    "errors":
        []
}