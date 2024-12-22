import pandas as pd
import numpy as np
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore
import stumpy

class Valmod(BaseDetector):

    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize
        self.window_min = max(HP.get("min_anomaly_window_size", 30), 4)
        self.window_max = max(self.window_min + 1, HP.get("max_anomaly_window_size", 40))
        self.heap_size = HP.get("heap_size", 50)
        self.exclusion_zone = HP.get("exclusion_zone", 0.5)
        self.verbose = HP.get("verbose", 1)
        self.random_state = HP.get("random_state", 42)
        np.random.seed(self.random_state)

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
        if self.normalize:
            X = zscore(X, axis=0, ddof=0)

        values = X[:, 0]
        matrix_profile = stumpy.stump(values, m=self.window_min)
        self.decision_scores_ = matrix_profile[:, 0]
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
        reduced_length = self.decision_scores_.shape[0] 

        if reduced_length < n_samples:
            padding_size = (n_samples - reduced_length) // 2

            self.decision_scores_ = np.pad(
                self.decision_scores_,
                (padding_size, n_samples - reduced_length - padding_size),
                mode='edge'
            )

        self.decision_scores_ = self.decision_scores_[:n_samples]

        assert self.decision_scores_.shape[0] == n_samples, (
            f"Length mismatch after padding: {self.decision_scores_.shape[0]} != {n_samples}"
        )

        return self.decision_scores_


def run_Valmod_Unsupervised(data, HP):
    clf = Valmod(HP=HP)
    clf.fit(data)
    score = clf.decision_function(data)
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score

if __name__ == '__main__':

    Start_T = time.time()
    # ArgumentParser
    parser = argparse.ArgumentParser(description='Running Valmod')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='Valmod')
    args = parser.parse_args()

    Valmod_HP = {
        "min_anomaly_window_size": 30,
        "max_anomaly_window_size": 40,
        "heap_size": 50,
        "exclusion_zone": 0.5,
        "verbose": 1,
        "random_state": 42
    }

    df = pd.read_csv(os.path.join(args.data_direc, args.filename)).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)

    start_time = time.time()

    output = run_Valmod_Unsupervised(data, Valmod_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output) + 3 * np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)
