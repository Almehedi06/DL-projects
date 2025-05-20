# tsr/run.py

import argparse
import yaml
from src.data_loader import load_stock_data, prepare_features_targets, split_time_series_data
from src.trainer import run_training_pipeline

def load_config(config_path="config/config.yaml"):
    """
    Load YAML configuration file.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main(model_type=None):
    # --- Load Config ---
    config = load_config()

    # --- Load Data ---
    data_path = config["paths"]["data_file"]
    df = load_stock_data(data_path)

    # --- Prepare Features and Target ---
    X, y = prepare_features_targets(df)

    # --- Split Data (Sequential) ---
    X_train, X_test, y_train, y_test = split_time_series_data(X, y, train_fraction=config["general"]["train_fraction"])

    # --- Define Models to Run ---
    models_to_run = [model_type] if model_type else config["models_to_run"]

    # --- Run Training Pipelines ---
    for m in models_to_run:
        print(f"\n=== Running {m} ===")
        model, metrics = run_training_pipeline(X_train, y_train, X_test, y_test, model_type=m, config=config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training pipeline.")
    parser.add_argument("--model", type=str, required=False,
                        help="Model to run: linear_regression, svr, rf, xgb, lstm, bilstm, gru. Run all if not specified.")
    args = parser.parse_args()

    main(args.model)
