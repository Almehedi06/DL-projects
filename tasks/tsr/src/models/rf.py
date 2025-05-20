# tsr/src/models/rf.py

from sklearn.ensemble import RandomForestRegressor

class RFModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize Random Forest Regressor with specified hyperparameters.
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           random_state=random_state)

    def train(self, X_train, y_train):
        """
        Train the Random Forest Regressor.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions using the trained Random Forest Regressor.
        """
        return self.model.predict(X)
