# tsr/src/trainer.py

from src.models.linear_regression import LinearRegressionModel
from src.evaluator import evaluate

def run_training_pipeline(X_train, y_train, X_test, y_test, model_type="linear_regression", config=None):
    """
    Universal training pipeline supporting multiple models and config-driven DL models.
    """

    # Model Selection
    if model_type == "linear_regression":
        model = LinearRegressionModel()
    elif model_type == "svr":
        from src.models.svr import SVRModel
        model = SVRModel()
    elif model_type == "rf":
        from src.models.rf import RFModel
        model = RFModel()
    elif model_type == "xgb":
        from src.models.xgb import XGBModel
        model = XGBModel()
    elif model_type == "lstm":
        from src.models.lstm import LSTMModel
        model = LSTMModel(config["model_params"]["lstm"])
    elif model_type == "bilstm":
        from src.models.bilstm import BiLSTMModel
        model = BiLSTMModel(config["model_params"]["bilstm"])
    elif model_type == "gru":
        from src.models.gru import GRUModel
        model = GRUModel(config["model_params"]["gru"])
    else:
        raise NotImplementedError(f"Model '{model_type}' is not recognized.")

    # Training
    model.train(X_train, y_train)

    # Prediction on Train and Test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluation
    train_metrics = evaluate(y_train, y_train_pred)
    test_metrics = evaluate(y_test, y_test_pred)

    # Display Results
    print(f"✅ {model_type} Training Results: {train_metrics}")
    print(f"✅ {model_type} Testing Results: {test_metrics}")

    return model, {"train": train_metrics, "test": test_metrics}
