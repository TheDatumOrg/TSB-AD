"""
This function is adapted from [chronos-forecasting] by [lostella et al.]
Original source: [https://github.com/amazon-science/chronos-forecasting]
"""

from autogluon.timeseries import TimeSeriesPredictor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tempfile

from .base import BaseDetector


class Chronos(BaseDetector):
    def __init__(self, 
                 win_size=100,
                 model_size = 'base',  # [tiny, small, base]
                 prediction_length=1, 
                 input_c=1, 
                 batch_size=128):

        self.model_name = 'Chronos'
        self.model_size = model_size
        self.win_size = win_size
        self.prediction_length = prediction_length
        self.input_c = input_c
        self.batch_size = batch_size
        self.score_list = []

    def fit(self, data):

        for channel in range(self.input_c):
            
            data_channel = data[:, channel].reshape(-1, 1)
            data_win, data_target = self.create_dataset(data_channel, slidingWindow=self.win_size, predict_time_steps=self.prediction_length)
            # print('data_win: ', data_win.shape)         # (2330, 100)
            # print('data_target: ', data_target.shape)   # (2330, 1)

            train_data = []
            count = 0
            for id in range(data_win.shape[0]):
                for tt in range(data_win.shape[1]):
                    train_data.append([id, count, data_win[id, tt]])
                    count += 1
            train_data = pd.DataFrame(train_data, columns=['item_id', 'timestamp', 'target'])

            with tempfile.TemporaryDirectory() as temp_dir:

                predictor = TimeSeriesPredictor(prediction_length=self.prediction_length, path=temp_dir).fit(
                        train_data, 
                        hyperparameters={
                        "Chronos": {
                        "model_path": self.model_size,   # base
                        "device": "cuda",
                        "batch_size": self.batch_size}},
                        skip_model_selection=True,
                        verbosity=0)

                predictions = predictor.predict(train_data)['mean'].to_numpy().reshape(-1, self.prediction_length)
                print('predictions: ', predictions.shape)

                ### using mse as the anomaly score
                scores = (data_target.squeeze() - predictions.squeeze()) ** 2
                self.score_list.append(scores)

        scores_merge = np.mean(np.array(self.score_list), axis=0)
        # print('scores_merge: ', scores_merge.shape)

        padded_decision_scores = np.zeros(len(data))
        padded_decision_scores[: self.win_size+self.prediction_length-1] = scores_merge[0]
        padded_decision_scores[self.win_size+self.prediction_length-1 : ]=scores_merge

        self.decision_scores_ = padded_decision_scores


    def decision_function(self, X):
        """
        Not used, present for API consistency by convention.
        """        
        pass

    def create_dataset(self, X, slidingWindow, predict_time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - slidingWindow - predict_time_steps+1):
            
            tmp = X[i : i + slidingWindow + predict_time_steps].ravel()
            # tmp= MinMaxScaler(feature_range=(0,1)).fit_transform(tmp.reshape(-1,1)).ravel()
            
            x = tmp[:slidingWindow]
            y = tmp[slidingWindow:]
            Xs.append(x)
            ys.append(y)
        return np.array(Xs), np.array(ys)