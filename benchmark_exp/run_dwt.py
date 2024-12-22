import warnings
from typing import Callable, List
import numpy as np
import pywt as wt
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import MinMaxScaler
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

warnings.filterwarnings(action='ignore', category=UserWarning)

class dwt(BaseDetector):

    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_samples, n_features = X.shape
        if self.normalize: 
            X = zscore(X, axis=0, ddof=0)

        self.data = X.flatten()
        self.n = len(self.data)
        self.wavelet = self.HP.get('wavelet', "haar")
        self.mode = self.HP.get('mode', "periodic")
        self.start_level = self.HP.get('start_level', 1)
        self.quantile_boundary_type = self.HP.get('quantile_boundary_type', "percentile")
        self.quantile_epsilon = self.HP.get('quantile_epsilon', 0.05)

        self.padded_data = self._pad_series(self.data)
        self.m = len(self.padded_data)
        self.max_level = int(np.log2(self.m)) - 1
        self.window_sizes = np.array([max(2, self.max_level - l - self.start_level + 1) for l in range(self.max_level)])
        self.track_coefs = self.HP.get('track_coefs', False)

        self.decision_scores_ = np.zeros(self.n)
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        levels, approx_coefs, detail_coefs = self._multilevel_dwt(self.padded_data)
        coef_anomaly_counts = self._score_anomalies(detail_coefs, approx_coefs, levels)

        if self.track_coefs:
            self.coefs_levels_ = levels
            self.coefs_approx_ = approx_coefs
            self.coefs_detail_ = detail_coefs
            self.coefs_scores_ = coef_anomaly_counts

        point_anomaly_scores = self._push_anomaly_counts_down_to_points(coef_anomaly_counts)
        self.decision_scores_ = point_anomaly_scores
        return point_anomaly_scores

    def _pad_series(self, data: np.ndarray) -> np.ndarray:
        n = len(data)
        exp = np.ceil(np.log2(n))
        m = int(np.power(2, exp))
        return wt.pad(data, (0, m - n), "periodic")

    def _multilevel_dwt(self, data: np.ndarray):
        levels, approx_coefs, detail_coefs = [], [], []
        a = data
        for i in range(self.max_level):
            a, d = wt.dwt(a, self.wavelet, self.mode)
            if i + 1 >= self.start_level:
                levels.append(i + 1)
                approx_coefs.append(a)
                detail_coefs.append(d)
        return np.array(levels), approx_coefs, detail_coefs

    def _estimate_gaussian_likelihoods(self, x_view: np.ndarray) -> np.ndarray:
        e_cov_est = EmpiricalCovariance(assume_centered=False)
        e_cov_est.fit(x_view)
        p = np.empty(shape=len(x_view))
        for i, window in enumerate(x_view):
            p[i] = e_cov_est.score(window.reshape(1, -1))
        return p

    def _mark_anomalous_windows(self, p: np.ndarray) -> np.ndarray:
        if self.quantile_boundary_type == "percentile":
            z_eps = np.percentile(p, self.quantile_epsilon * 100)
        else:
            raise ValueError(f"The quantile boundary type '{self.quantile_boundary_type}' is not implemented yet!")
        return p < z_eps

    def _reverse_windowing(self, data: np.ndarray, window_length: int, full_length: int,
                           reduction: Callable = np.mean, fill_value: float = np.nan) -> np.ndarray:
        mapped = np.full(shape=(full_length, window_length), fill_value=fill_value)
        mapped[:len(data), 0] = data
        for w in range(1, window_length):
            mapped[:, w] = np.roll(mapped[:, 0], w)
        return reduction(mapped, axis=1)

    def _score_anomalies(self, detail_coefs: List[np.ndarray], approx_coefs: List[np.ndarray], levels: np.ndarray) -> List[np.ndarray]:
        anomaly_scores = []
        for x, level in zip(self._combine_alternating(detail_coefs, approx_coefs), levels.repeat(2, axis=0)):
            level_index = level - 1
            if level_index < 0 or level_index >= len(self.window_sizes):
                raise IndexError(f"Level index {level_index} is out of bounds for window_sizes of length {len(self.window_sizes)}")
            window_size = self.window_sizes[level_index]
            x_view = sliding_window_view(x, window_size)
            p = self._estimate_gaussian_likelihoods(x_view)
            a = self._mark_anomalous_windows(p)
            xa = self._reverse_windowing(a, window_length=window_size, full_length=len(x), reduction=np.sum, fill_value=0)
            anomaly_scores.append(xa)
        return anomaly_scores

    def _combine_alternating(self, xs, ys):
        for x, y in zip(xs, ys):
            yield x
            yield y

    def _push_anomaly_counts_down_to_points(self, coef_anomaly_counts: List[np.ndarray]) -> np.ndarray:
        anomaly_counts = coef_anomaly_counts[0::2] + coef_anomaly_counts[1::2]
        counter = np.zeros(self.m)
        for ac in anomaly_counts:
            counter += ac.repeat(self.m // len(ac), axis=0)
        counter[counter < 2] = 0
        return counter[:self.n]

def run_dwt_Unsupervised(data, HP):
    clf = dwt(HP=HP)
    clf.fit(data)
    score = clf.decision_function(data)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

if __name__ == '__main__':

    import time, argparse, os
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank

    Start_T = time.time()
    parser = argparse.ArgumentParser(description='Running dwt')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='dwt')
    args = parser.parse_args()

    dwt_HP = {
        'wavelet': 'haar',
        'mode': 'periodic',
        'start_level': 1,
        'quantile_boundary_type': 'percentile',
        'quantile_epsilon': 0.05,
        'track_coefs': False
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

    #output = run_dwt_Semisupervised(data_train, data, HP=dwt_HP) #ignore
    output = run_dwt_Unsupervised(data, HP=dwt_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output)+3*np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)