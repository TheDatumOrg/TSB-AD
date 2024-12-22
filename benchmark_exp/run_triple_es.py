import numpy as np
import pandas as pd
import random, argparse, time, os, logging
import math
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from stumpy import stumpi
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

class Triple_ES(BaseDetector):

    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize
        self.anomaly_window_size = HP.get("anomaly_window_size", 50)
        self.n_init_train = HP.get("n_init_train", 100)
        self.random_state = HP.get("random_state", 42)
        self.use_column_index = HP.get("use_column_index", 0)
        self.stream = None

    def set_random_state(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def initialize_stream(self, data, warmup: int, ws: int):
        self.stream = stumpi(data[:warmup], m=ws, egress=False)

    def fit(self, X, y=None):
        self.set_random_state()
        self.data = X[:, self.use_column_index].astype(float)
        self.labels = y if y is not None else np.zeros(len(self.data))

        warmup = self.n_init_train
        ws = self.anomaly_window_size

        if ws > warmup:
            ws = warmup
        if ws < 3:
            ws = 3

        self.initialize_stream(self.data, warmup, ws)

        for point in self.data[warmup:]:
            self.stream.update(point)

        mp = self.stream.left_P_
        mp[:warmup] = 0  
        self.decision_scores_ = mp
        return self


    def decision_function(self, X):
        n_samples, n_features = X.shape
        reduced_length = self.decision_scores_.shape[0]

        if reduced_length < n_samples:
          
            padding_length = n_samples - reduced_length
            self.decision_scores_ = np.pad(
                self.decision_scores_,
                (padding_length, 0),  
                'constant',
                constant_values=(0,)
            )

        assert self.decision_scores_.shape[0] == n_samples, (
            f"Length mismatch after padding: {self.decision_scores_.shape[0]} != {n_samples}"
        )

        return self.decision_scores_


def run_Triple_ES_Unsupervised(data, HP):
    clf = Triple_ES(HP=HP)
    clf.fit(data)
    score = clf.decision_function(data)
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score

def run_Triple_ES_Semisupervised(data_train, data_test, HP):
    clf = Triple_ES(HP=HP)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score

if __name__ == "__main__":

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running Triple_ES')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='Triple_ES')
    args = parser.parse_args()

    Triple_ES_HP = {
        'anomaly_window_size': 50,
        'n_init_train': 100,
        'random_state': 42,
        'use_column_index': 0
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data
    #print(data_train.shape)
    start_time = time.time()

    #output = run_Triple_ES_Semisupervised(data_train, data, Triple_ES_HP) #ignore
    output = run_Triple_ES_Unsupervised(data, Triple_ES_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output) + 3 * np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)