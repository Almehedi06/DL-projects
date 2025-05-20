# tsr/src/models/svr.py

from sklearn.svm import SVR

class SVRModel:
    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1):
        """
        Initialize SVR with specified hyperparameters.
        """
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def train(self, X_train, y_train):
        """
        Train the SVR model.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions using the trained SVR model.
        """
        return self.model.predict(X)
