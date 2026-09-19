# import pandas as pd


# def detect_problem_type(
#     df: pd.DataFrame,
#     target: str
# ) -> str:

#     if target not in df.columns:
#         raise ValueError(
#             f"Target '{target}' not found."
#         )

#     y = df[target]

#     unique_values = y.nunique()

#     # Object/category/bool targets
#     # usually indicate classification
#     if (
#         pd.api.types.is_object_dtype(y)
#         or isinstance(y.dtype, pd.CategoricalDtype)
#         or pd.api.types.is_bool_dtype(y)
#     ):
#         return "classification"

#     # Numeric target with few unique values
#     # is probably classification
#     if (
#         pd.api.types.is_numeric_dtype(y)
#         and unique_values <= 20
#     ):
#         return "classification"

#     # Otherwise assume regression
#     if pd.api.types.is_numeric_dtype(y):
#         return "regression"

#     raise ValueError(
#         "Could not determine problem type."
#     )

import pandas as pd


def detect_problem_type(
    df: pd.DataFrame,
    target: str
) -> str:
    """
    Automatically determine whether the target represents
    a classification or regression problem.
    """

    # -------------------------------------------------
    # 1. Validate target
    # -------------------------------------------------

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' was not found "
            f"in the dataset."
        )

    y = df[target].dropna()

    if y.empty:
        raise ValueError(
            f"Target column '{target}' contains no valid values."
        )

    unique_values = y.nunique()

    print("\nTarget Analysis")
    print("-------------------------")
    print("Target column :", target)
    print("Target dtype  :", y.dtype)
    print("Unique values :", unique_values)

    # -------------------------------------------------
    # 2. Non-numeric targets
    # -------------------------------------------------

    # Examples:
    # Yes / No
    # Churn / Stay
    # Fraud / Normal
    # Low / Medium / High

    if not pd.api.types.is_numeric_dtype(y):

        print(
            "Decision      : Non-numeric target detected"
        )

        return "classification"

    # -------------------------------------------------
    # 3. Boolean target
    # -------------------------------------------------

    if pd.api.types.is_bool_dtype(y):

        print(
            "Decision      : Boolean target detected"
        )

        return "classification"

    # -------------------------------------------------
    # 4. Numeric target with small number of classes
    # -------------------------------------------------

    # Example:
    # 0 / 1
    # 0 / 1 / 2
    # Ratings such as 1-5

    unique_ratio = (
        unique_values / len(y)
    )

    if (
        unique_values <= 20
        and unique_ratio < 0.05
    ):

        print(
            "Decision      : Discrete numeric target detected"
        )

        return "classification"

    # -------------------------------------------------
    # 5. Numeric continuous target
    # -------------------------------------------------

    # Examples:
    # House Price
    # Salary
    # Sales
    # Delivery Time

    if pd.api.types.is_numeric_dtype(y):

        print(
            "Decision      : Continuous numeric target detected"
        )

        return "regression"

    # -------------------------------------------------
    # Fallback
    # -------------------------------------------------

    raise ValueError(
        f"Could not determine problem type for target '{target}'. "
        f"Detected dtype: {y.dtype}"
    )