import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate basic information about the dataset.
    """

    profile = {
        "rows": len(df),

        "columns": len(df.columns),

        "column_names": df.columns.tolist(),

        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },

        "missing_values": (
            df.isnull()
            .sum()
            .to_dict()
        ),

        "missing_percentage": (
            df.isnull()
            .mean()
            .mul(100)
            .round(2)
            .to_dict()
        ),

        "unique_values": (
            df.nunique(
                dropna=False
            ).to_dict()
        ),

        "duplicate_rows": int(
            df.duplicated().sum()
        )
    }

    return profile


def detect_feature_types(df: pd.DataFrame) -> dict:
    """
    Detect numerical, categorical and datetime columns.
    """

    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

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

    datetime_columns = (
        df.select_dtypes(
            include=[
                "datetime",
                "datetimetz"
            ]
        )
        .columns
        .tolist()
    )

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "datetime": datetime_columns
    }