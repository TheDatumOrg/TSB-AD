"""
This function is adapted from [timesfm] by [siriuz42 et al.]
Original source: [https://github.com/google-research/timesfm]
"""

import timesfm
import numpy as np

class TimesFM():
    def __init__(self, 
                 win_size=96, 
                 prediction_length=1, 
                 input_c=1):

        self.model_name = 'TimesFM'
        self.win_size = win_size
        self.prediction_length = prediction_length
        self.input_c = input_c
        self.score_list = []

    def fit(self, data):

        for channel in range(self.input_c):
            
            data_channel = data[:, channel].reshape(-1, 1)
            data_win, data_target = self.create_dataset(data_channel, slidingWindow=self.win_size, predict_time_steps=self.prediction_length)
            # print('data_win: ', data_win.shape)         # (2330, 100)
            # print('data_target: ', data_target.shape)   # (2330, 1)

            # tfm = timesfm.TimesFm(
            #     context_len=self.win_size,
            #     horizon_len=self.prediction_length,
            #     input_patch_len=32,
            #     output_patch_len=128,
            #     num_layers=20,
            #     model_dims=1280,
            #     backend="gpu")
            # tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

            tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=32,
                    horizon_len=self.prediction_length,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
            )

            forecast_input = [data_win[i, :] for i in range(data_win.shape[0])]
            point_forecast, _ = tfm.forecast(forecast_input)

            print('predictions: ', point_forecast.shape)

            ### using mse as the anomaly score
            scores = (data_target.squeeze() - point_forecast.squeeze()) ** 2
            # scores = np.mean(scores, axis=1)
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