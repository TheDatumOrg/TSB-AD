import argparse
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
from collections import defaultdict
from math import ceil


class TSBitmapper(BaseDetector):

    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize

        self.feature_window_size = HP['feature_window_size']
        self.alphabet_size = HP['alphabet_size']
        self.level_size = HP['level_size']
        self.lag_window_size = HP['lag_window_size']
        self.lead_window_size = HP['lead_window_size']
        self.compression_ratio = HP['compression_ratio']

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods."""
        n_samples, n_features = X.shape
        if self.normalize:
            X = zscore(X, axis=0, ddof=0)

        self.decision_scores_ = self.fit_predict(X)
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector."""
        n_samples, n_features = X.shape
        decision_scores_ = self.fit_predict(X)
        return decision_scores_

    def fit_predict(self, ts):
        """Unsupervised training of TSBitMaps."""
        self._ref_ts = ts
        self._ts_length = len(ts)
        scores = self._slide_chunks(ts)
        return scores

    def _slide_chunks(self, ts):
        lag_bitmap = {}
        lead_bitmap = {}
        binned_ts = self.discretize_by_sax_window(ts)
        scores = np.zeros(len(binned_ts))
        ts_len = len(binned_ts)

        lagws = self.lag_window_size
        leadws = self.lead_window_size
        featws = self.level_size
        for i in range(lagws, ts_len - leadws + 1):
            lag_chunk = binned_ts[i - lagws: i]
            lead_chunk = binned_ts[i: i + leadws]

            if i == lagws:
                lag_bitmap = self.get_bitmap_with_feat_window(lag_chunk)
                lead_bitmap = self.get_bitmap_with_feat_window(lead_chunk)
                scores[i] = self.bitmap_distance(lag_bitmap, lead_bitmap)
            else:
                ingress_lag_feat = lag_chunk[-featws:]
                ingress_lead_feat = lead_chunk[-featws:]

                lag_bitmap[ingress_lag_feat] += 1
                lag_bitmap[lag_chunk[0: featws]] -= 1

                lead_bitmap[ingress_lead_feat] += 1
                lead_bitmap[lead_chunk[0: featws]] -= 1

                scores[i] = self.bitmap_distance(lag_bitmap, lead_bitmap)

        return scores

    def discretize_by_sax_window(self, ts, feature_window_size=None):
        if feature_window_size is None:
            feature_window_size = self.feature_window_size
        n = len(ts)
        windows = ()
        for i in range(0, n - n % feature_window_size, feature_window_size):
            binned_fw = self.discretize_sax(ts[i: i + feature_window_size])
            windows += binned_fw
        if n % feature_window_size > 0:
            last_binned_fw = self.discretize_sax(ts[- (n % feature_window_size):])
            windows += last_binned_fw
        return windows

    def discretize_sax(self, ts):
        znorm_ts = znorm(ts)
        ts_string = ts_to_string(znorm_ts, cuts=cuts_for_asize(self.alphabet_size))
        return tuple(ts_string)

    def bitmap_distance(self, lag_bitmap, lead_bitmap):
        """Computes the dissimilarity of two bitmaps."""
        dist = 0
        lag_feats = set(lag_bitmap.keys())
        lead_feats = set(lead_bitmap.keys())
        shared_feats = lag_feats & lead_feats

        for feat in shared_feats:
            dist += (lead_bitmap[feat] - lag_bitmap[feat]) ** 2

        for feat in lag_feats - shared_feats:
            dist += lag_bitmap[feat] ** 2

        for feat in lead_feats - shared_feats:
            dist += lead_bitmap[feat] ** 2

        return dist

    def get_bitmap_with_feat_window(self, chunk, level_size=None, step=None):
        """Get bitmap for a feature window."""
        if step is None:
            step = self.feature_window_size
        if level_size is None:
            level_size = self.level_size
        bitmap = defaultdict(int)
        n = len(chunk)

        for i in range(0, n - n % step, step):
            for j in range(step - level_size + 1):
                feat = chunk[i + j: i + j + level_size]
                bitmap[feat] += 1 

        max_freq = max(bitmap.values())
        for feat in bitmap.keys():
            bitmap[feat] = bitmap[feat] / max_freq
        return bitmap

    def post_ts_bitmap(self, scores: np.ndarray):
        """Decompress scores based on the compression ratio."""
        if self.compression_ratio == 1:
            return scores
        window_sizes = np.diff(np.arange(0, self._ts_length, self.feature_window_size))
        mini_window_sizes = self.get_window_sizes(0, self.feature_window_size, self.compression_ratio)
        length_encoding = np.tile(mini_window_sizes, len(window_sizes))
        length_encoding_t = np.transpose(length_encoding)
        
        last_window_start = np.arange(0, self._ts_length, self.feature_window_size)[-1]
        length_encoding_full = np.append(length_encoding_t,
                                        self.get_window_sizes(last_window_start, self._ts_length, self.compression_ratio))

        decompressed_scores = np.zeros(self._ts_length)
        index_decompressed = 0
        for i in range(len(scores)):
            tmp = np.repeat(scores[i], length_encoding_full[i])
            decompressed_scores[index_decompressed: index_decompressed + length_encoding_full[i]] = tmp
            index_decompressed += length_encoding_full[i]

        return decompressed_scores

    def get_window_sizes(self, start, stop, step):
        """Compute window sizes based on the compression ratio."""
        return np.expand_dims(np.diff(np.append(np.arange(start, stop, step), stop)), axis=1)


def run_TSBitmapper_Unsupervised(data, HP):
    clf = TSBitmapper(HP=HP)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score


if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running TSBitmapper')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='TSBitmapper')
    args = parser.parse_args()

    TSBitmapper_HP = {
        'feature_window_size': 100,
        'alphabet_size': 5,
        'level_size': 3,
        'lag_window_size': 300,
        'lead_window_size': 200,
        'compression_ratio': 2
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    output = run_TSBitmapper_Unsupervised(data, TSBitmapper_HP)

    pred = output > (np.mean(output) + 3 * np.std(output))
    evaluation_result = get_metrics(output, label, pred=pred)
    print('Evaluation Result: ', evaluation_result)

    end_time = time.time()
    run_time = end_time - Start_T
    print(f"Total Runtime: {run_time:.2f} seconds")
