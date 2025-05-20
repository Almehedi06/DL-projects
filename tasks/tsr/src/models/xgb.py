# tsr/src/models/xgb.py

from xgboost import XGBRegressor

class XGBModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        """
        Initialize XGBoost Regressor with specified hyperparameters.
        """
        self.model = XGBRegressor(n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  random_state=random_state,
                                  objective='reg:squarederror')

    def train(self, X_train, y_train):
        """
        Train the XGBoost Regressor.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions using the trained XGBoost Regressor.
        """
        return self.model.predict(X)
