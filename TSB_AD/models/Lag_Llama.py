"""
This function is adapted from [lag-llama] by [ashok-arjun&kashif]
Original source: [https://github.com/time-series-foundation-models/lag-llama]
"""

from itertools import islice

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import torch
from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
import numpy as np
from ..utils.torch_utility import get_gpu

from lag_llama.gluon.estimator import LagLlamaEstimator

class Lag_Llama():
    def __init__(self, 
                 win_size=96, 
                 prediction_length=1, 
                 input_c=1,
                 use_rope_scaling=False,
                 batch_size=64,
                 num_samples=100,
                 ckpt_path='lag-llama.ckpt'):

        self.model_name = 'Lag_Llama'
        self.context_length = win_size
        self.prediction_length = prediction_length
        self.input_c = input_c
        self.ckpt_path = ckpt_path
        self.use_rope_scaling = use_rope_scaling
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.score_list = []

        self.cuda = True
        self.device = get_gpu(self.cuda)


    def fit(self, data):

        for channel in range(self.input_c):
            
            data_channel = data[:, channel].reshape(-1, 1)
            data_win, data_target = self.create_dataset(data_channel, slidingWindow=self.context_length, predict_time_steps=self.prediction_length)
            # print('data_win: ', data_win.shape)         # (2330, 100)
            # print('data_target: ', data_target.shape)   # (2330, 1)

            data_win = data_win.T

            date_rng = pd.date_range(start='2021-01-01', periods=data_win.shape[0], freq='H')   # Dummy timestep
            df_wide = pd.DataFrame(data_win, index=date_rng)
            # Convert numerical columns to float 32 format for lag-llama
            for col in df_wide.columns:
                # Check if column is not of string type
                if df_wide[col].dtype != 'object' and pd.api.types.is_string_dtype(df_wide[col]) == False:
                    df_wide[col] = df_wide[col].astype('float32')

            # Create a PandasDataset
            ds = PandasDataset(dict(df_wide))

            ckpt = torch.load(self.ckpt_path, map_location=self.device) # Uses GPU since in this Colab we use a GPU.
            estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

            rope_scaling_arguments = {
                "type": "linear",
                "factor": max(1.0, (self.context_length + self.prediction_length) / estimator_args["context_length"]),
            }

            estimator = LagLlamaEstimator(
                ckpt_path=self.ckpt_path,
                prediction_length=self.prediction_length,
                context_length=self.context_length, # Lag-Llama was trained with a context length of 32, but can work with any context length

                # estimator args
                input_size=estimator_args["input_size"],
                n_layer=estimator_args["n_layer"],
                n_embd_per_head=estimator_args["n_embd_per_head"],
                n_head=estimator_args["n_head"],
                scaling=estimator_args["scaling"],
                time_feat=estimator_args["time_feat"],
                rope_scaling=rope_scaling_arguments if self.use_rope_scaling else None,

                batch_size=self.batch_size,
                num_parallel_samples=100,
                device=self.device,
            )

            lightning_module = estimator.create_lightning_module()
            transformation = estimator.create_transformation()
            predictor = estimator.create_predictor(transformation, lightning_module)

            forecast_it, ts_it = make_evaluation_predictions(
                dataset=ds,
                predictor=predictor,
                num_samples=self.num_samples
            )
            forecasts = list(forecast_it)
            tss = list(ts_it)

            predictions = np.array([pred.mean for pred in forecasts])

            # print('predictions: ', predictions.shape)

            ### using mse as the anomaly score
            scores = (data_target.squeeze() - predictions.squeeze()) ** 2
            self.score_list.append(scores)

        scores_merge = np.mean(np.array(self.score_list), axis=0)

        padded_decision_scores = np.zeros(len(data))
        padded_decision_scores[: self.context_length+self.prediction_length-1] = scores_merge[0]
        padded_decision_scores[self.context_length+self.prediction_length-1 : ]=scores_merge

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