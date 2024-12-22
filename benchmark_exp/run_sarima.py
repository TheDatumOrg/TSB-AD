import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging


from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore
from sklearn.preprocessing import MinMaxScaler
from pmdarima.arima import ARIMA, AutoARIMA
from pmdarima import model_selection


class SARIMA(BaseDetector):
    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize

        self.train_window_size = HP.get('train_window_size', 500)
        self.forecast_window_size = HP.get('prediction_window_size', 10)
        self.max_lag = HP.get('max_lag', None)
        self.period = HP.get('period', 1)
        self.max_iter = HP.get('max_iter', 20)
        self.exhaustive_search = HP.get('exhaustive_search', False)
        self.n_jobs = HP.get('n_jobs', 1)
        self.fixed_orders = HP.get('fixed_orders', None)

    def fit(self, X, y=None):
        """Fit SARIMA model."""
        if self.normalize:
            X = zscore(X, axis=0, ddof=0)

        train, _ = model_selection.train_test_split(X, train_size=self.train_window_size)
        self._fit(train)
        return self

    def _fit(self, X: np.ndarray) -> ARIMA:
        """Fit ARIMA or AutoARIMA model based on fixed orders."""
        if self.fixed_orders is not None:
            seasonal = list(self.fixed_orders["seasonal_order"])
            seasonal.append(self.period)
            self.fixed_orders["seasonal_order"] = tuple(seasonal)
            arima = ARIMA(
                max_iter=self.max_iter,
                suppress_warnings=False,
                **self.fixed_orders
            )
        else:
            arima = AutoARIMA(
                start_p=1, max_p=3,
                d=None, max_d=2,
                start_q=1, max_q=3,
                seasonal=True, m=self.period,
                start_P=1, max_P=2,
                D=None, max_D=1,
                start_Q=1, max_Q=2,
                maxiter=self.max_iter,
                suppress_warnings=True, error_action="warn", trace=1,
                stepwise=not self.exhaustive_search, n_jobs=self.n_jobs,
            )
        arima.fit(X)
        self._arima = arima
        return arima

    def decision_function(self, X):
        """Compute anomaly scores using the fitted SARIMA model."""
        n_samples = X.shape[0]
        decision_scores_ = np.zeros(n_samples)

        self._predictions = self._predict(X)
        self._scores = np.abs(X - self._predictions)
        
        decision_scores_ = self._scores
        self.decision_scores_ = decision_scores_

        return decision_scores_

    def _predict(self, X: np.ndarray, with_conf_int: bool = False) -> np.ndarray:
        """Generate predictions for SARIMA model."""
        N = len(X)
        self._predictions = np.zeros(shape=N)
        if with_conf_int:
            self._conf_ints = np.zeros(shape=(N, 2))

        self.max_lag = self.max_lag if self.max_lag else N
        i = self.train_window_size
        forecast_window_size = self.forecast_window_size
        lag_points = i
        while i < N:
            start = i
            end = i + forecast_window_size
            if lag_points >= self.max_lag:
                lag_points = 0
                self._fit(X[i - self.train_window_size:i])

            if end > N:
                end = N
                forecast_window_size = N - i

            prediction = self._arima.predict(forecast_window_size, return_conf_int=with_conf_int)
            if with_conf_int:
                y_hat, y_hat_conf = prediction
                self._predictions[start:end] = y_hat
                self._conf_ints[start:end, :] = y_hat_conf
            else:
                self._predictions[start:end] = prediction

            self._arima.update(X[start:end])
            i += forecast_window_size
            lag_points += forecast_window_size

        return self._predictions



def run_SARIMA_Unsupervised(data, HP):
    """Run SARIMA detector in unsupervised mode."""
    clf = SARIMA(HP=HP)
    clf.fit(data)
    
    clf.decision_function(data)
    
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score



if __name__ == '__main__':
    Start_T = time.time()

    parser = argparse.ArgumentParser(description='Running SARIMA')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='SARIMA', help='Anomaly detection method')
    args = parser.parse_args()

    SARIMA_HP = {
        'train_window_size': 500,
        'prediction_window_size': 10,
        'max_lag': None,
        'period': 1,
        'max_iter': 20,
        'exhaustive_search': False,
        'n_jobs': 1,
        'fixed_orders': None
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)

    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data

    start_time = time.time()

    output = run_SARIMA_Unsupervised(data, SARIMA_HP)

    pred = output > (np.mean(output) + 3 * np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)

    end_time = time.time()
    run_time = end_time - start_time
    print(f"Execution Time: {run_time} seconds")
    print('Evaluation Result: ', evaluation_result)