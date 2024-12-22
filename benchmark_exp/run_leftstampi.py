import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import math
from stumpy import stumpi
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

class leftstampi(BaseDetector):

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

        warmup = self.HP['n_init_train']
        ws = self.HP['anomaly_window_size']

        if ws > warmup:
            logging.warning(f"WARN: anomaly_window_size is larger than n_init_train. Adjusting to n_init_train={warmup}.")
            ws = warmup
        if ws < 3:
            logging.warning("WARN: anomaly_window_size must be at least 3. Adjusting to 3.")
            ws = 3

        self.stream = stumpi(X[:warmup, 0], m=ws, egress=False)
        for point in X[warmup:, 0]:
            self.stream.update(point)
  
        self.decision_scores_ = self.stream.left_P_
        self.decision_scores_[:warmup] = 0  

        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        n_samples = X.shape[0]
        padded_scores = self.pad_anomaly_scores(self.decision_scores_, n_samples, self.HP['anomaly_window_size'])
        return padded_scores

    @staticmethod
    def pad_anomaly_scores(scores, n_samples, window_size):
        """
        Pads the anomaly scores to match the length of the input time series.
        Padding is symmetric, using the first and last values.
        """
        left_padding = [scores[0]] * math.ceil((window_size - 1) / 2)
        right_padding = [scores[-1]] * ((window_size - 1) // 2)
        padded_scores = np.array(left_padding + list(scores) + right_padding)

        return padded_scores[:n_samples]

def run_leftstampi_Unsupervised(data, HP):
    clf = leftstampi(HP=HP)
    clf.fit(data)
    score = clf.decision_function(data)
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score

def run_leftstampi_Unsupervised(data_train, data_test, HP):
    clf = leftstampi(HP=HP)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running leftstampi')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='leftstampi')
    parser.add_argument('--anomaly_window_size', type=int, default=50)
    parser.add_argument('--n_init_train', type=int, default=100)
    args = parser.parse_args()

    leftstampi_HP = {
        'anomaly_window_size': args.anomaly_window_size,
        'n_init_train': args.n_init_train,
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

    output = run_leftstampi_Unsupervised(data_train, data, HP=leftstampi_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output) + 3 * np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)