from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import Bidirectional, TimeDistributed, Conv1D, MaxPooling1D, Flatten, ConvLSTM2D


class VanillaLSTM:
    def __init__(self, n_steps, n_features):
        self.n_steps = n_steps
        self.n_features = n_features
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(self.n_steps, self.n_features)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, verbose=0):
        self.model.fit(X, y, verbose=verbose, epochs=200)

    def predict(self, X):
        return self.model.predict(X, verbose=0)


class StackedLSTM:
    def __init__(self, n_steps, n_features):
        self.n_steps = n_steps
        self.n_features = n_features
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, verbose=0):
        self.model.fit(X, y, verbose=verbose, epochs=200)

    def predict(self, X, verbose=0):
        return self.model.predict(X, verbose=verbose)


class BidirectionalLSTM:
    def __init__(self, n_steps, n_features):
        self.n_steps = n_steps
        self.n_features = n_features
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, verbose=0):
        self.model.fit(X, y, verbose=verbose, epochs=200)

    def predict(self, X, verbose=0):
        return self.model.predict(X, verbose=verbose)

class CNN_LSTM:
    def __init__(self, n_steps, n_features):
        self.model = Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                       input_shape=(None, n_steps, n_features)))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, verbose=0):
        self.model.fit(X, y, verbose=verbose, epochs=200)

    def predict(self, X, verbose=0):
        return self.model.predict(X, verbose=verbose)


class ConvLSTM:
    def __init__(self, n_seq, n_steps, n_features):
        self.model = Sequential()
        self.model.add(
            ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, verbose=0):
        self.model.fit(X, y, verbose=verbose, epochs=400)

    def predict(self, X, verbose=0):
        return self.model.predict(X, verbose=verbose)
