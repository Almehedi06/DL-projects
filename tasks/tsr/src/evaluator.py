# tsr/src/evaluator.py

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(y_true, y_pred):
    """
    Evaluate model predictions with standard regression metrics.

    Works for all models as long as they output numeric predictions.
    """
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
