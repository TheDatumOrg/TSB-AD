# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

"""
This code is adapted from [pythresh] by [KulikDM]
Original source: [https://github.com/KulikDM/pythresh]
"""

import pandas as pd
import numpy as np
import inspect
import argparse, time
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

from .thresholding_utils import check_scores, normalize


class IQR(BaseDetector):
    """
    IQR class for Inter-Quartile Region thresholder.

    This class uses the inter-quartile region (IQR) to provide a non-parametric method 
    for thresholding decision scores. Outliers are defined as any value beyond the 
    third quartile (Q3) plus 1.5 times the inter-quartile range.

    For more details, see the reference: bardet2015iqr.

    Parameters
    ----------
    random_state : int, optional (default=1234)
        Random seed for the random number generators of the thresholders.
        Can also be set to None.

    Attributes
    ----------
    thresh_ : float
        The threshold value that separates inliers from outliers.

    dscores_ : ndarray of shape (n_samples,)
        1D array of decomposed decision scores.

    Notes
    -----
    The inter-quartile region (IQR) is calculated as:

        IQR = |Q3 - Q1|

    where Q1 and Q3 represent the first and third quartiles, respectively. 
    The threshold for decision scores is defined as:

        t = Q3 + 1.5 * IQR
    """

    def __init__(self, random_state=1234, normalize=True):
        super().__init__()
        self.random_state = random_state
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

        X = check_scores(X, random_state=self.random_state)

        #NOTE If we want to use this verify
        #NOTE here that axis should be 0 instead of 1
        # if self.normalize: X = zscore(X, axis=0, ddof=1)

        #NOTE the following is the PyThresh implementation
        if self.normalize: X = normalize(X)

        arg_map = {'old': 'interpolation', 'new': 'method'}
        arg_name = (arg_map['new'] if 'method' in
                    inspect.signature(np.percentile).parameters
                    else arg_map['old'])

        # First quartile (Q1)
        P1 = np.percentile(X, 25, **{arg_name: 'midpoint'})

        # Third quartile (Q3)
        P3 = np.percentile(X, 75, **{arg_name: 'midpoint'})

        # Calculate IQR and generate limit
        iqr = abs(P3-P1)
        limit = P3 + 1.5*iqr

        self.threshold_ = limit

        self.decision_scores_ = np.zeros(n_samples) #TODO should we keep this?

        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
            #NOTE here anomaly_scores == labels_ because
            thresholding method returns 0s and 1s (not anomaly scores)
        """
        n_samples, n_features = X.shape
        decision_scores_ = np.zeros(n_samples)
        decision_scores_[X >= self.threshold_] = 1

        return decision_scores_


def run_Custom_AD_Unsupervised(data, random_state):
    clf = IQR(random_state=random_state)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

# def run_Custom_AD_Semisupervised(data_train, data_test):
#     clf = IQR()
#     clf.fit(data_train)
#     score = clf.decision_function(data_test)
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
#     return score

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running IQR')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    # parser.add_argument('--data_direc', type=str, default='../Datasets/TSB-AD-U/')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='IQR')
    args = parser.parse_args()

    # parser.add_argument('--filename', type=str, default='057_SMD_id_1_Facility_tr_4529_1st_4629.csv')
    # parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-M/')


    Custom_AD_HP = {
        'random_state': 1234,
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    start_time = time.time()

    # output = run_Custom_AD_Semisupervised(data_train, data, **Custom_AD_HP)
    output = run_Custom_AD_Unsupervised(data, **Custom_AD_HP)
    # output = run_Custom_AD_Unsupervised(data) #NOTE no parameters for IQR

    end_time = time.time()
    run_time = end_time - start_time

    pred = output   #NOTE output has already the predictions
    # pred = output > (np.mean(output)+3*np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)