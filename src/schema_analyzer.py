import re
import numpy as np
import pandas as pd


# ============================================================
# HELPER
# ============================================================

def normalize_column_name(name: str) -> str:
    """
    Convert a column name into a simplified form
    for rule-based analysis.
    """

    name = str(name).lower()

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    )

    return name.strip("_")


# ============================================================
# ID COLUMN DETECTION
# ============================================================

def looks_like_id_column(
    column_name: str
) -> bool:

    normalized = normalize_column_name(
        column_name
    )

    compact = normalized.replace(
        "_",
        ""
    )

    known_ids = {
        "customerid",
        "userid",
        "employeeid",
        "studentid",
        "transactionid",
        "accountid",
        "orderid",
        "productid",
        "recordid",
        "clientid",
        "patientid"
    }

    if normalized in {
        "id",
        "uuid",
        "guid",
        "identifier",
        "key"
    }:
        return True

    if (
        normalized.endswith("_id")
        or normalized.startswith("id_")
    ):
        return True

    if compact in known_ids:
        return True

    return False


# ============================================================
# MAIN SCHEMA ANALYZER
# ============================================================

def analyze_schema(
    df: pd.DataFrame,
    target: str,
    numeric_threshold: float = 0.95
):

    working_df = df.copy()

    converted_numeric_columns = []

    # --------------------------------------------------------
    # 1. Detect numeric values stored as text
    # --------------------------------------------------------

    for column in working_df.columns:

        if column == target:
            continue

        series = working_df[column]

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):

            # Convert blank strings into missing values

            cleaned = series.replace(
                r"^\s*$",
                np.nan,
                regex=True
            )

            non_missing = cleaned.dropna()

            if len(non_missing) == 0:
                continue

            converted = pd.to_numeric(
                cleaned,
                errors="coerce"
            )

            conversion_ratio = (
                converted.notna().sum()
                /
                len(non_missing)
            )

            if conversion_ratio >= numeric_threshold:

                working_df[column] = converted

                converted_numeric_columns.append(
                    column
                )


    # --------------------------------------------------------
    # 2. Detect constant columns
    # --------------------------------------------------------

    constant_columns = []

    for column in working_df.columns:

        if column == target:
            continue

        unique_count = working_df[
            column
        ].nunique(
            dropna=False
        )

        if unique_count <= 1:

            constant_columns.append(
                column
            )


    # --------------------------------------------------------
    # 3. Detect probable ID columns
    # --------------------------------------------------------

    id_columns = []

    for column in working_df.columns:

        if column == target:
            continue

        series = working_df[column]

        non_missing_count = (
            series.notna().sum()
        )

        if non_missing_count == 0:
            continue

        unique_count = series.nunique(
            dropna=True
        )

        unique_ratio = (
            unique_count
            /
            non_missing_count
        )

        if (
            looks_like_id_column(column)
            and unique_ratio >= 0.80
        ):

            id_columns.append(
                column
            )


    # --------------------------------------------------------
    # 4. High-cardinality categorical detection
    # --------------------------------------------------------

    high_cardinality_columns = []

    for column in working_df.columns:

        if column == target:
            continue

        series = working_df[column]

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(
                series.dtype,
                pd.CategoricalDtype
            )
        ):

            non_missing = series.dropna()

            if len(non_missing) == 0:
                continue

            unique_count = non_missing.nunique()

            unique_ratio = (
                unique_count
                /
                len(non_missing)
            )

            if (
                unique_count >= 50
                and unique_ratio >= 0.50
            ):

                high_cardinality_columns.append(
                    column
                )


    # --------------------------------------------------------
    # 5. Detect low-cardinality numeric features
    # --------------------------------------------------------

    discrete_numeric_columns = []

    for column in working_df.select_dtypes(
        include=["number"]
    ).columns:

        if column == target:
            continue

        unique_count = working_df[
            column
        ].nunique()

        if unique_count <= 20:

            discrete_numeric_columns.append(
                column
            )


    excluded_columns = sorted(
        set(
            id_columns
            +
            constant_columns
        )
    )


    schema_report = {

        "converted_numeric_columns":
            converted_numeric_columns,

        "id_columns":
            id_columns,

        "constant_columns":
            constant_columns,

        "high_cardinality_columns":
            high_cardinality_columns,

        "discrete_numeric_columns":
            discrete_numeric_columns,

        "excluded_columns":
            excluded_columns
    }


    return (
        working_df,
        schema_report
    )