from sklearn.pipeline import Pipeline


def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    models,
    problem_type,
    evaluator
):

    results = {}

    trained_models = {}

    for model_name, model in models.items():

        print(
            f"\nTraining: {model_name}"
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        metrics = evaluator(
            y_test,
            predictions
        )

        results[model_name] = metrics

        trained_models[model_name] = pipeline

    return results, trained_models