from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)


def get_models(problem_type: str):

    if problem_type == "classification":

        return {
            "Logistic Regression":
                LogisticRegression(
                    max_iter=2000
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                ),

            "Gradient Boosting":
                GradientBoostingClassifier(
                    random_state=42
                )
        }

    elif problem_type == "regression":

        return {
            "Linear Regression":
                LinearRegression(),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                ),

            "Gradient Boosting":
                GradientBoostingRegressor(
                    random_state=42
                )
        }

    else:
        raise ValueError(
            "Unknown problem type."
        )