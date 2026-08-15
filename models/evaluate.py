from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import pandas as pd

from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error
)


def evaluate_multioutput(model, X_test, y_test):

    predictions = model.predict(X_test)

    results = {}

    horizons = [
        ("24", 0),
        ("48", 1),
        ("72", 2)
    ]

    for horizon, index in horizons:

        # Works for Pandas DataFrame and NumPy array
        if hasattr(y_test, "iloc"):
            y_true = y_test.iloc[:, index].values
        else:
            y_true = y_test[:, index]

        y_pred = predictions[:, index]

        results[f"r2_{horizon}"] = r2_score(
            y_true,
            y_pred
        )

        results[f"rmse_{horizon}"] = root_mean_squared_error(
            y_true,
            y_pred
        )

        results[f"mae_{horizon}"] = mean_absolute_error(
            y_true,
            y_pred
        )
    return results