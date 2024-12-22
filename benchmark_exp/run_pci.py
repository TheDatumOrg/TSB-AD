import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from typing import Tuple

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

class PCI(BaseDetector):
    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize
        self.k = HP.get('window_size', 20) // 2
        self.p = HP.get('thresholding_p', 0.4)
        self.w = np.concatenate((np.arange(1, self.k + 1), np.arange(1, self.k + 1)[::-1]))
        self.decision_scores_ = None
        self.anomaly_labels_ = None

    def fit(self, X, y=None):
        """Fit the anomaly detector."""
        n_samples = X.shape[0]
        if self.normalize: 
            X = zscore(X, axis=0, ddof=0)
        #print(X)
        self.decision_scores_, self.anomaly_labels_ = self._detect(X)
        self.decision_scores_ = MinMaxScaler(feature_range=(0, 1)).fit_transform(self.decision_scores_.reshape(-1, 1)).ravel()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score."""
        n_samples = X.shape[0]
        decision_scores_ = np.zeros(n_samples)
        decision_scores_, _ = self._detect(X)
        decision_scores_ = MinMaxScaler(feature_range=(0, 1)).fit_transform(decision_scores_.reshape(-1, 1)).ravel()
        return decision_scores_

    def _pci(self, v: float, window_predictions: np.ndarray, eta: np.ndarray) -> Tuple[float, float]:
        t = stats.t.ppf(self.p, df=2 * self.k - 1)
        s = (eta - window_predictions).std()
        lower_bound = v - t * s * np.sqrt(1 + (1 / (2 * self.k)))
        upper_bound = v + t * s * np.sqrt(1 + (1 / (2 * self.k)))
        return lower_bound, upper_bound

    def _predict(self, eta: np.ndarray) -> float:
        eta_no_nan = eta[~np.isnan(eta)]
        w_no_nan = self.w[~np.isnan(eta)]
        if w_no_nan.size == 0:
            return np.nan
        v_hat = np.matmul(eta_no_nan, w_no_nan) / w_no_nan.sum()
        return v_hat

    def _generate_window(self, ts: np.ndarray, i: int) -> np.ndarray:
        result = np.zeros(2 * self.k)
        m = len(ts)
        left_start = max(i - self.k, 0)
        left_end = max(i - 1 + 1, 0)
        left_length = left_end - left_start
        right_start = min(i + 1, m)
        right_end = min(i + self.k + 1, m)
        right_length = right_end - right_start

        if left_length > 0:
            result[self.k - left_length:self.k] = ts[left_start:left_end].flatten()

        if right_length > 0:
            result[self.k:self.k + right_length] = ts[right_start:right_end].flatten()

        return result




    def _combine_left_right(self, left: list, right: list) -> np.ndarray:
        prediction_window = np.zeros(2 * self.k)
        if len(left) > 0:
            prediction_window[self.k - len(left):self.k] = left[-self.k:]
        if len(right) > 0:
            prediction_window[self.k:self.k + len(right)] = right[-self.k:]
        return prediction_window

    def _detect(self, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        anomaly_scores = np.zeros(len(ts))
        anomaly_labels = np.zeros(len(ts), dtype=int)
        m = len(ts)
        left_predictions = []
        right_predictions = []
        for i in range(m):
            v = ts[i]
            start = i + len(right_predictions)
            for j in range(start, min(start + self.k, m) + 1):
                eta_ = self._generate_window(ts, j)
                right_v = self._predict(eta_)
                right_predictions.append(right_v)
            v_hat = right_predictions.pop(0) if right_predictions else np.nan
            predictions.append(v_hat)
            anomaly_scores[i] = abs(v_hat - v)
            prediction_window = self._combine_left_right(left_predictions, right_predictions)
            eta = self._generate_window(ts, i)
            lower_bound, upper_bound = self._pci(v_hat, prediction_window, eta)
            anomaly_labels[i] = int(not lower_bound < v_hat < upper_bound)
            left_predictions.append(v_hat)
            if len(left_predictions) > self.k:
                del left_predictions[0]
        return anomaly_scores, anomaly_labels

def run_PCI_anomaly_detection(data, HP):
    clf = PCI(HP=HP)
    clf.fit(data)
    scores = clf.decision_scores_ 
    return scores

if __name__ == '__main__':
    import time
    Start_T = time.time()
    
    ## ArgumentParser
    import argparse
    parser = argparse.ArgumentParser(description='Running PCI')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    args = parser.parse_args()

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    Custom_AD_HP = {
        'window_size': 20,
        'thresholding_p': 0.4,
    }

    slidingWindow = find_length_rank(data, rank=1)
    output = run_PCI_anomaly_detection(data, HP=Custom_AD_HP) 
    #print('Anomaly Scores:', output)  
    
    pred = output > (np.mean(output) + 3 * np.std(output))  
    #print('Predicted Anomalies (Thresholded):', pred)
    
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)

    end_time = time.time()
    run_time = end_time - Start_T
    print('Run time: ', run_time)