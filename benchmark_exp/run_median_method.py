import pandas as pd
import numpy as np
import time
import argparse
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore
import bottleneck as bn

class MedianMethod(BaseDetector):
    """
    Anomaly detection using a moving median and standard deviation window-based method.

    :param dict HP: hyperparameters including neighbourhood_size
    :param bool normalize: whether to normalize the data before fitting
    """
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
        if self.normalize:
            X = zscore(X, axis=0, ddof=0)  
        self._timeseries = X.flatten() 
        self.decision_scores_ = self._compute_anomaly_scores(X)
        return self

    def _compute_windows(self, window_type):
        """Compute rolling windows for median or standard deviation."""
        if window_type == "std":
            windows = np.convolve(self._timeseries, np.ones(self.HP['neighbourhood_size']*2 + 1) / (self.HP['neighbourhood_size']*2 + 1), mode='same')
        else:
            windows = bn.move_median(self._timeseries, window=self.HP['neighbourhood_size'] * 2 + 1)  
        return windows


    def _compute_anomaly_scores(self, data):
        """Calculate the anomaly scores."""
        median_windows = self._compute_windows("median")
        std_windows = self._compute_windows("std")
        dist_windows = np.abs(median_windows - data.flatten())  
        scores = dist_windows / std_windows
        return np.nan_to_num(scores)

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector."""
        n_samples = X.shape[0]
        scores = self._compute_anomaly_scores(X)
        return scores

def run_MedianMethod_Unsupervised(data, HP):
    """Run MedianMethod for unsupervised anomaly detection."""
    clf = MedianMethod(HP=HP)
    clf.fit(data)
    scores = clf.decision_scores_
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
    return scores

if __name__ == '__main__':
    Start_T = time.time()

    # ArgumentParser
    parser = argparse.ArgumentParser(description='Running MedianMethod for Anomaly Detection')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='MedianMethod')
    args = parser.parse_args()


    HP = {
        'neighbourhood_size': 5, 
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print(f'Data shape: {data.shape}')
    print(f'Label shape: {label.shape}')

    slidingWindow = find_length_rank(data, rank=1) 

    train_index = int(args.filename.split('.')[0].split('_')[-3])
    data_train = data

    start_time = time.time()

    output = run_MedianMethod_Unsupervised(data_train, HP)

    end_time = time.time()
    run_time = end_time - start_time
    print(f'Run Time: {run_time:.3f} seconds')

    #print("Anomaly Scores:", output)

    pred = output > (np.mean(output) + 3 * np.std(output))

    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print(f'Evaluation Result: {evaluation_result}')