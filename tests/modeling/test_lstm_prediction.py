import unittest
from src.preparation.misc import *
from src.modeling.lstm_prediction import *


class LSTM_PredictionTest(unittest.TestCase):

    def test_lstm_prediction(self):
        """Test with synthetic data"""
        raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        n_steps = 3
        n_features = 1
        X, y = split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        vanilla_lstm = VanillaLSTM(n_steps, n_features)
        vanilla_lstm.fit(X, y)
        stacked_lstm = StackedLSTM(n_steps,n_features)
        stacked_lstm.fit(X,y)
        bidirectional_lstm = BidirectionalLSTM(n_steps,n_features)
        bidirectional_lstm.fit(X,y)
        x_test = array([70, 80, 90])
        x_test = x_test.reshape((1, n_steps, n_features))

        prediction = vanilla_lstm.predict(x_test)
        print("Vanilla LSTM "  + str(prediction))
        prediction = stacked_lstm.predict(x_test)
        print("Stacked LSTM " + str(prediction))
        prediction = bidirectional_lstm.predict(x_test)
        print("Bidirectional LSTM " + str(prediction))

        """CNN LSTM TEST"""
        n_steps = 4
        X, y = split_sequence(raw_seq, n_steps)
        n_features = 1
        n_seq = 2
        n_steps = 2
        X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
        cnn_lstm = CNN_LSTM(n_steps, n_features)
        cnn_lstm.fit(X, y)
        x_test = array([60, 70, 80, 90])
        x_test = x_test.reshape((1, n_seq, n_steps, n_features))
        prediction = cnn_lstm.predict(x_test)
        print("CNN LSTM " + str(prediction))

        """CONV LSTM TEST"""

        X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
        conv_lstm = ConvLSTM(n_seq, n_steps, n_features)
        conv_lstm.fit(X,y)
        x_test = x_test.reshape((1, n_seq, 1, n_steps, n_features))
        prediction = conv_lstm.predict(x_test)
        print("CONV LSTM " + str(prediction))


