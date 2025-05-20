# tsr/src/data_loader.py

import pandas as pd

def load_stock_data(file_path):
    """
    Load stock data CSV with 'Date' as datetime and returns a DataFrame.
    Assumes the CSV has a 'Date' column.
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    return df

def prepare_features_targets(df, target_column="Close"):
    """
    Prepare features and target.
    Features: ['Open', 'High', 'Low', 'Volume']
    Target: target_column (default 'Close')
    """
    features = df[['Open', 'High', 'Low', 'Volume']]
    target = df[target_column]
    return features, target

def split_time_series_data(features, target, train_fraction=0.8):
    """
    Sequential split based on time. No shuffling.
    Example: first 80% for training, remaining 20% for testing.
    """
    split_index = int(len(features) * train_fraction)
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]
    return X_train, X_test, y_train, y_test
