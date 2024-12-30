import numpy as np
import logging
import math
from stumpy import stumpi
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

class Left_STAMPi(BaseDetector):

    def __init__(self, n_init_train=100, window_size=50, normalize=True):
        super().__init__()
        self.n_init_train = n_init_train
        self.window_size = window_size
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

        warmup = self.n_init_train
        ws = self.window_size

        if ws > warmup:
            logging.warning(f"WARN: window_size is larger than n_init_train. Adjusting to n_init_train={warmup}.")
            ws = warmup
        if ws < 3:
            logging.warning("WARN: window_size must be at least 3. Adjusting to 3.")
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
        padded_scores = self.pad_anomaly_scores(self.decision_scores_, n_samples, self.window_size)
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