import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import stumpy

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

class STOMP(BaseDetector):

    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize
        self.window_size = HP.get("anomaly_window_size", 30)
        self.n_jobs = HP.get("n_jobs", 1)
        self.random_state = HP.get("random_state", 42)
        self.verbose = HP.get("verbose", 1)

        np.random.seed(self.random_state)

        if self.window_size < 4:
            if self.verbose > 0:
                print("WARN: window_size must be at least 4. Dynamically setting window_size to 4.")
            self.window_size = 4

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

        self.X = X.ravel() 
        if y is not None:
            self.y = y.ravel()
        else:
            self.y = None

        if self.n_jobs <= 1:
            stomp_mp = stumpy.stump(self.X, m=self.window_size)
        else:
            stomp_mp = stumpy.stump(self.X, m=self.window_size, n_threads=self.n_jobs)

        self.anomaly_scores = stomp_mp[:, 0]
        return self

    def decision_function(self, X=None):
        """Predict raw anomaly score of X using the fitted detector.

        Parameters
        ----------
        X : Ignored for this method as anomaly scores are precomputed.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        n_samples = self.X.shape[0] if X is None else X.shape[0]
        reduced_length = self.anomaly_scores.shape[0]

        if reduced_length < n_samples:
            padding_size = (n_samples - reduced_length) // 2

            
            self.anomaly_scores = np.pad(
                self.anomaly_scores,
                (padding_size, n_samples - reduced_length - padding_size),
                mode='edge'
            )

        
        self.anomaly_scores = self.anomaly_scores[:n_samples]

        assert self.anomaly_scores.shape[0] == n_samples, (
            f"Length mismatch after padding: {self.anomaly_scores.shape[0]} != {n_samples}"
        )

        return self.anomaly_scores


def run_STOMP_Unsupervised(data, HP):
    clf = STOMP(HP=HP)
    clf.fit(data)
    score = clf.decision_function()
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running STOMP')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='STOMP')
    args = parser.parse_args()

    STOMP_HP = {
        'anomaly_window_size': 30,
        'n_jobs': 1,
        'random_state': 42,
        'verbose': 1
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)
    train_index = int(args.filename.split('.')[0].split('_')[-3])
    data_train = data

    start_time = time.time()

    output = run_STOMP_Unsupervised(data, HP=STOMP_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output) + 3 * np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)
