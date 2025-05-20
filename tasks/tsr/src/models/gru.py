# tsr/src/models/gru.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

class GRUModel:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(GRU(self.config['neurons'],
                      activation=self.config['activation'],
                      input_shape=(self.config['lookback'], self.config['num_features']),
                      return_sequences=False))
        model.add(Dropout(self.config['dropout']))
        model.add(Dense(1))
        model.compile(optimizer=self.config['optimizer'], loss=self.config['loss_function'])
        return model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train,
                       epochs=self.config['epochs'],
                       batch_size=self.config['batch_size'],
                       verbose=0)

    def predict(self, X):
        return self.model.predict(X).flatten()
