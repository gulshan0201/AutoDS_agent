import re

from pathlib import Path

import matplotlib

matplotlib.use(
    "Agg"
)

import matplotlib.pyplot as plt

import pandas as pd


# ============================================================
# FILE NAME CLEANER
# ============================================================

def safe_filename(
    value
):

    value = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(value)
    )

    return value.strip(
        "_"
    )


# ============================================================
# MAIN EDA GENERATOR
# ============================================================

def generate_eda(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    output_directory="artifacts/eda",
    max_numeric_plots=8,
    max_categorical_plots=6
):

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    generated_files = []


    # ========================================================
    # 1. Missing values plot
    # ========================================================

    missing = (
        df.isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if len(missing) > 0:

        fig = plt.figure(
            figsize=(10, 6)
        )

        plt.bar(
            missing.index.astype(str),
            missing.values
        )

        plt.xticks(
            rotation=70,
            ha="right"
        )

        plt.title(
            "Missing Values by Column"
        )

        plt.ylabel(
            "Missing Count"
        )

        plt.tight_layout()

        path = (
            output_directory
            /
            "missing_values.png"
        )

        fig.savefig(
            path,
            dpi=150
        )

        plt.close(
            fig
        )

        generated_files.append(
            str(path)
        )


    # ========================================================
    # 2. Target distribution
    # ========================================================

    if target in df.columns:

        target_series = (
            df[target]
            .dropna()
        )

        if problem_type == "classification":

            counts = (
                target_series
                .astype(str)
                .value_counts()
            )

            fig = plt.figure(
                figsize=(8, 5)
            )

            plt.bar(
                counts.index,
                counts.values
            )

            plt.title(
                f"Target Distribution: {target}"
            )

            plt.ylabel(
                "Count"
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

        else:

            fig = plt.figure(
                figsize=(8, 5)
            )

            plt.hist(
                target_series,
                bins=30
            )

            plt.title(
                f"Target Distribution: {target}"
            )

            plt.xlabel(
                target
            )

            plt.ylabel(
                "Frequency"
            )


        plt.tight_layout()

        path = (
            output_directory
            /
            "target_distribution.png"
        )

        fig.savefig(
            path,
            dpi=150
        )

        plt.close(
            fig
        )

        generated_files.append(
            str(path)
        )


    # ========================================================
    # 3. Numeric feature histograms
    # ========================================================

    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    numeric_columns = [
        column
        for column in numeric_columns
        if column != target
    ]


    for column in numeric_columns[
        :max_numeric_plots
    ]:

        values = (
            df[column]
            .dropna()
        )

        if len(values) == 0:
            continue

        fig = plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            values,
            bins=30
        )

        plt.title(
            f"Distribution of {column}"
        )

        plt.xlabel(
            column
        )

        plt.ylabel(
            "Frequency"
        )

        plt.tight_layout()

        path = (
            output_directory
            /
            f"numeric_{safe_filename(column)}.png"
        )

        fig.savefig(
            path,
            dpi=150
        )

        plt.close(
            fig
        )

        generated_files.append(
            str(path)
        )


    # ========================================================
    # 4. Categorical feature plots
    # ========================================================

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


    plot_count = 0

    for column in categorical_columns:

        if (
            column == target
            or plot_count >= max_categorical_plots
        ):
            continue

        unique_count = (
            df[column]
            .nunique(
                dropna=False
            )
        )

        # Avoid huge plots for IDs/high-cardinality fields

        if unique_count > 20:
            continue

        counts = (
            df[column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
        )

        fig = plt.figure(
            figsize=(9, 5)
        )

        plt.bar(
            counts.index,
            counts.values
        )

        plt.title(
            f"Category Distribution: {column}"
        )

        plt.xticks(
            rotation=60,
            ha="right"
        )

        plt.ylabel(
            "Count"
        )

        plt.tight_layout()

        path = (
            output_directory
            /
            f"categorical_{safe_filename(column)}.png"
        )

        fig.savefig(
            path,
            dpi=150
        )

        plt.close(
            fig
        )

        generated_files.append(
            str(path)
        )

        plot_count += 1


    # ========================================================
    # 5. Correlation matrix
    # ========================================================

    numeric_df = (
        df.select_dtypes(
            include=["number"]
        )
    )

    if numeric_df.shape[1] >= 2:

        correlation = (
            numeric_df
            .corr()
        )

        fig = plt.figure(
            figsize=(10, 8)
        )

        plt.imshow(
            correlation.values,
            vmin=-1,
            vmax=1
        )

        plt.colorbar()

        plt.xticks(
            range(
                len(
                    correlation.columns
                )
            ),
            correlation.columns,
            rotation=70,
            ha="right"
        )

        plt.yticks(
            range(
                len(
                    correlation.columns
                )
            ),
            correlation.columns
        )

        plt.title(
            "Correlation Matrix"
        )

        plt.tight_layout()

        path = (
            output_directory
            /
            "correlation_matrix.png"
        )

        fig.savefig(
            path,
            dpi=150
        )

        plt.close(
            fig
        )

        generated_files.append(
            str(path)
        )


    return generated_files