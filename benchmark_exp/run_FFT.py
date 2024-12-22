import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore
import time
import argparse
from dataclasses import dataclass

class FFT(BaseDetector):

    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize

        self.ifft_parameters = HP.get('ifft_parameters', 5)
        self.local_neighbor_window = HP.get('local_neighbor_window', 21)
        self.local_outlier_threshold = HP.get('local_outlier_threshold', 0.6)
        self.max_region_size = HP.get('max_region_size', 50)
        self.max_sign_change_distance = HP.get('max_sign_change_distance', 10)
        self.decision_scores_ = None

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods."""
        n_samples, n_features = X.shape
        if self.normalize: 
            if n_features == 1:
                X = zscore(X, axis=0, ddof=0)
            else: 
                X = zscore(X, axis=1, ddof=1)
        self.data = X
        self.decision_scores_ = self.detect_anomalies()  
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector."""
        n_samples, n_features = X.shape
        decision_scores_ = np.zeros(n_samples)
        self.data = X
        local_outliers = self.calculate_local_outliers()
        if not local_outliers:
            print("No local outliers detected.")
            return np.zeros_like(self.data)

        regions = self.calculate_region_outliers(local_outliers)
        anomaly_scores = np.zeros_like(self.data)
        for region in regions:
            start_index = local_outliers[region.start_idx].index
            end_index = local_outliers[region.end_idx].index
            anomaly_scores[start_index:end_index + 1] = region.score

        decision_scores_ = anomaly_scores
        return decision_scores_

    @staticmethod
    def reduce_parameters(f: np.ndarray, k: int) -> np.ndarray:
        transformed = f.copy()
        transformed[k:] = 0
        return transformed

    def calculate_local_outliers(self):
        n = len(self.data)
        k = max(min(self.ifft_parameters, n), 1)
        y = self.reduce_parameters(np.fft.fft(self.data), k)
        f2 = np.real(np.fft.ifft(y))

        so = np.abs(f2 - self.data)
        mso = np.mean(so)
        neighbor_c = self.local_neighbor_window // 2

        scores = []
        score_idxs = []
        for i in range(n):
            if so[i] > mso:
                nav = np.mean(self.data[max(i - neighbor_c, 0):min(i + neighbor_c + 1, n)])
                scores.append(self.data[i] - nav)
                score_idxs.append(i)

        if not scores:
            return []

        ms = np.mean(scores)
        sds = np.std(scores) + 1e-6  
        z_scores = (np.array(scores) - ms) / sds

        return [self.LocalOutlier(index=score_idxs[i], z_score=z_scores[i])
                for i in range(len(scores)) if abs(z_scores[i]) > self.local_outlier_threshold]

    def calculate_region_outliers(self, local_outliers):
        def distance(a: int, b: int) -> int:
            return abs(local_outliers[b].index - local_outliers[a].index)

        regions = []
        i = 0
        n_l = len(local_outliers) - 1
        while i < n_l:
            start_idx = i
            while i < n_l and distance(i, i + 1) <= self.max_sign_change_distance:
                i += 1
            end_idx = i
            if end_idx > start_idx:
                score = np.mean([abs(local_outliers[j].z_score) for j in range(start_idx, end_idx + 1)])
                regions.append(self.RegionOutlier(start_idx=start_idx, end_idx=end_idx, score=score))
            i += 1

        return regions

    @dataclass
    class LocalOutlier:
        index: int
        z_score: float

        @property
        def sign(self) -> int:
            return np.sign(self.z_score)

    @dataclass
    class RegionOutlier:
        start_idx: int
        end_idx: int
        score: float

    def detect_anomalies(self):
        """Detect anomalies by combining local and regional outliers."""
        local_outliers = self.calculate_local_outliers()
        if not local_outliers:
            print("No local outliers detected.")
            return np.zeros_like(self.data)

        regions = self.calculate_region_outliers(local_outliers)
        anomaly_scores = np.zeros_like(self.data)
        for region in regions:
            start_index = local_outliers[region.start_idx].index
            end_index = local_outliers[region.end_idx].index
            anomaly_scores[start_index:end_index + 1] = region.score

        return anomaly_scores

def run_FFT_Unsupervised(data, HP):
    clf = FFT(HP=HP)
    clf.fit(data)  
    score = clf.decision_scores_ 

    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()

    #print(score)
    return score

if __name__ == '__main__':
    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running FFT')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='FFT')
    args = parser.parse_args()

    FFT_HP = {
        'ifft_parameters': 5,
        'local_neighbor_window': 21,
        'local_outlier_threshold': 0.6,
        'max_region_size': 50,
        'max_sign_change_distance': 10,
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

    output = run_FFT_Unsupervised(data, FFT_HP)

    #print("Anomaly Scores (Normalized):", output)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output) + 3 * np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)