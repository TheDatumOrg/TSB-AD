# -*- coding: utf-8 -*-
# Author: [Your Name]
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import argparse
import time
from sklearn.preprocessing import MinMaxScaler
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector

class DSpotDetector(BaseDetector):
    def __init__(self, HP):
        super().__init__()
        self.HP = HP

    def fit(self, X, y=None):
        """Fit the DSpot algorithm to the data."""
        self.decision_scores_ = self.dspot(X)
        return self

    def decision_function(self, X):
        """Return anomaly scores."""
        return self.decision_scores_

    @staticmethod
    def dspot(data, num_init=50, depth=200, risk=1e-4):
        """Streaming Peak over Threshold with Drift."""
        logs = {'t': [], 'a': []}
        scores = np.zeros(data.size)
        base_data = data[:depth]
        init_data = data[depth:depth + num_init]
        rest_data = data[depth + num_init:]

        for i in range(num_init):
            temp = init_data[i]
            init_data[i] -= base_data.mean()
            np.delete(base_data, 0)
            np.append(base_data, temp)

        z, t = DSpotDetector.pot(init_data)
        k = num_init
        peaks = init_data[init_data > t] - t
        logs['t'] = [z] * (depth + num_init)

        for index, x in enumerate(rest_data):
            temp = x
            x -= base_data.mean()
            if x > z:
                logs['a'].append(index + num_init + depth)
                scores[index + num_init + depth] = 1 
            elif x > t:
                peaks = np.append(peaks, x - t)
                gamma, sigma = DSpotDetector.grimshaw(peaks, threshold=t)
                k += 1
                r = k * risk / peaks.size
                z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
                np.delete(base_data, 0)
                np.append(base_data, temp)
            else:
                k += 1
                np.delete(base_data, 0)
                np.append(base_data, temp)

            logs['t'].append(z)
        return scores 


    @staticmethod
    def pot(data, risk=1e-4, init_level=0.98, num_candidates=10, epsilon=1e-8):
        """Peak-over-Threshold Algorithm."""
        t = np.sort(data)[int(init_level * data.size)]
        peaks = data[data > t] - t
        if peaks.size == 0:
            return t, t

        gamma, sigma = DSpotDetector.grimshaw(peaks, threshold=t, num_candidates=num_candidates, epsilon=epsilon)

        r = data.size * risk / peaks.size
        if gamma != 0:
            z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            z = t - sigma * np.log(r)
        return z, t

    import scipy.optimize as opt

    @staticmethod
    def grimshaw(peaks, threshold, num_candidates=10, epsilon=1e-8):
        """Grimshaw's Method to estimate GPD parameters."""
        if peaks.size == 0:
            return 0, 0

        def gpd_log_likelihood(params):
            gamma, sigma = params
            if sigma <= 0 or gamma <= -1:
                return np.inf
            term = (1 + gamma * peaks / sigma)
            if np.any(term <= 0):
                return np.inf
            return -np.sum(np.log(term)) + peaks.size * np.log(sigma)

        result = opt.minimize(
            gpd_log_likelihood, x0=[0.1, np.std(peaks)], 
            bounds=[(-1 + epsilon, None), (epsilon, None)]
        )

        if result.success:
            return result.x[0], result.x[1] 
        return 0, 0



def run_DSpot_Unsupervised(data, HP):
    clf = DSpotDetector(HP=HP)
    clf.fit(data)
    scores = clf.decision_scores_
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
    return scores


if __name__ == '__main__':

    Start_T = time.time()
    # ArgumentParser
    parser = argparse.ArgumentParser(description='Running DSpot')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='DSpot')
    args = parser.parse_args()

    DSpot_HP = {
        'risk': 1e-4,
        'num_init': 50,
        'depth': 200,
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)
    train_index = int(args.filename.split('.')[0].split('_')[-3])
    data_train = data[:train_index, :]

    start_time = time.time()

    anomaly_scores = run_DSpot_Unsupervised(data, HP=DSpot_HP)

    end_time = time.time()
    run_time = end_time - start_time

    threshold = np.mean(anomaly_scores) + 3 * np.std(anomaly_scores)
    pred = anomaly_scores > threshold

    evaluation_result = get_metrics(anomaly_scores, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)

    print('Anomaly Scores:')
    print(anomaly_scores)

    output_filename = args.filename.replace('.csv', '_anomaly_scores.csv')
    pd.DataFrame({'Anomaly_Scores': anomaly_scores}).to_csv(output_filename, index=False)
    print(f'Anomaly scores saved to {output_filename}')
