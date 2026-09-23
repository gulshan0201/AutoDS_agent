import json

from pathlib import Path

import pandas as pd


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

def detect_outliers(
    df: pd.DataFrame,
    target: str,
    excluded_columns=None
):

    if excluded_columns is None:
        excluded_columns = []

    outlier_report = {}

    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    for column in numeric_columns:

        if (
            column == target
            or column in excluded_columns
        ):
            continue

        series = df[
            column
        ].dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(
            0.25
        )

        q3 = series.quantile(
            0.75
        )

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        mask = (
            (series < lower_bound)
            |
            (series > upper_bound)
        )

        count = int(
            mask.sum()
        )

        percentage = (
            count
            /
            len(series)
            *
            100
        )

        outlier_report[column] = {

            "count":
                count,

            "percentage":
                round(
                    percentage,
                    2
                ),

            "lower_bound":
                round(
                    float(lower_bound),
                    4
                ),

            "upper_bound":
                round(
                    float(upper_bound),
                    4
                )
        }

    return outlier_report


# ============================================================
# SUSPECTED LEAKAGE
# ============================================================

def detect_suspected_leakage(
    df: pd.DataFrame,
    target: str
):

    suspects = []

    target_key = (
        target
        .lower()
        .replace("_", "")
        .replace(" ", "")
    )

    for column in df.columns:

        if column == target:
            continue

        column_key = (
            str(column)
            .lower()
            .replace("_", "")
            .replace(" ", "")
        )

        # Name-based suspicion

        if target_key in column_key:

            suspects.append({
                "column":
                    column,

                "reason":
                    "Column name contains the target name."
            })

            continue

        # Exact target-copy detection

        try:

            same_values = (
                df[column]
                .fillna("__MISSING__")
                .astype(str)
                .equals(
                    df[target]
                    .fillna("__MISSING__")
                    .astype(str)
                )
            )

            if same_values:

                suspects.append({
                    "column":
                        column,

                    "reason":
                        "Column exactly duplicates target values."
                })

        except Exception:
            pass

    return suspects


# ============================================================
# MAIN QUALITY ENGINE
# ============================================================

def analyze_data_quality(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    schema_report: dict
):

    rows = len(df)

    columns = len(
        df.columns
    )

    missing_values = (
        df.isna()
        .sum()
    )

    missing_percentage = (
        df.isna()
        .mean()
        .mul(100)
    )

    missing_report = {}

    for column in df.columns:

        if missing_values[column] > 0:

            missing_report[column] = {

                "count":
                    int(
                        missing_values[column]
                    ),

                "percentage":
                    round(
                        float(
                            missing_percentage[column]
                        ),
                        2
                    )
            }


    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_percentage = (

        duplicate_rows
        /
        rows
        *
        100

        if rows > 0
        else 0
    )


    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    outliers = detect_outliers(
        df,
        target,
        schema_report.get(
            "excluded_columns",
            []
        )
    )


    # --------------------------------------------------------
    # CLASS BALANCE
    # --------------------------------------------------------

    class_balance = None

    imbalance_warning = False

    if problem_type == "classification":

        target_data = (
            df[target]
            .dropna()
        )

        counts = (
            target_data
            .value_counts()
        )

        total = counts.sum()

        class_balance = {}

        for class_name, count in counts.items():

            share = (
                count
                /
                total
            )

            class_balance[
                str(class_name)
            ] = {

                "count":
                    int(count),

                "percentage":
                    round(
                        share * 100,
                        2
                    )
            }


        if len(counts) > 1:

            smallest_share = (
                counts.min()
                /
                total
            )

            if smallest_share < 0.20:

                imbalance_warning = True


    # --------------------------------------------------------
    # LEAKAGE
    # --------------------------------------------------------

    suspected_leakage = (
        detect_suspected_leakage(
            df,
            target
        )
    )


    # --------------------------------------------------------
    # WARNINGS
    # --------------------------------------------------------

    warnings = []

    if missing_report:

        warnings.append(
            "Dataset contains missing values."
        )

    if duplicate_rows > 0:

        warnings.append(
            "Duplicate rows detected."
        )

    if schema_report.get(
        "id_columns"
    ):

        warnings.append(
            "Probable identifier columns detected and excluded from training."
        )

    if schema_report.get(
        "constant_columns"
    ):

        warnings.append(
            "Constant columns detected."
        )

    if schema_report.get(
        "high_cardinality_columns"
    ):

        warnings.append(
            "High-cardinality categorical columns detected."
        )

    if imbalance_warning:

        warnings.append(
            "Target class imbalance detected."
        )

    if suspected_leakage:

        warnings.append(
            "Potential target leakage requires review."
        )


    # --------------------------------------------------------
    # HEURISTIC DATA QUALITY SCORE
    # --------------------------------------------------------

    score = 100.0

    total_cells = (
        rows * columns
    )

    total_missing = int(
        df.isna()
        .sum()
        .sum()
    )

    if total_cells > 0:

        overall_missing_pct = (
            total_missing
            /
            total_cells
            *
            100
        )

        score -= min(
            20,
            overall_missing_pct * 0.5
        )


    score -= min(
        10,
        duplicate_percentage
    )


    score -= min(
        10,
        len(
            schema_report.get(
                "constant_columns",
                []
            )
        ) * 2
    )


    score -= min(
        10,
        len(
            schema_report.get(
                "high_cardinality_columns",
                []
            )
        ) * 2
    )


    if imbalance_warning:

        score -= 8


    if suspected_leakage:

        score -= min(
            20,
            len(
                suspected_leakage
            ) * 10
        )


    score = max(
        0,
        round(
            score,
            2
        )
    )


    report = {

        "rows":
            rows,

        "columns":
            columns,

        "quality_score":
            score,

        "score_note":
            "Heuristic project score; not a universal statistical standard.",

        "missing_values":
            missing_report,

        "duplicate_rows":
            duplicate_rows,

        "duplicate_percentage":
            round(
                duplicate_percentage,
                2
            ),

        "outliers":
            outliers,

        "class_balance":
            class_balance,

        "class_imbalance_warning":
            imbalance_warning,

        "suspected_leakage":
            suspected_leakage,

        "schema_findings":
            schema_report,

        "warnings":
            warnings
    }

    return report


# ============================================================
# SAVE REPORT
# ============================================================

def save_data_quality_report(
    report: dict,
    path
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )