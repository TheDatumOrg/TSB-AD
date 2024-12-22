import numpy as np
import pandas as pd
import time
import argparse
from sklearn.preprocessing import MinMaxScaler
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore
import stumpy
import math


class STAMP(BaseDetector):
    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize
        self.window_size = HP.get("anomaly_window_size", 30)
        self.verbose = HP.get("verbose", 1)
        self.n_jobs = HP.get("n_jobs", 1)
        self.random_state = HP.get("random_state", 42)
        np.random.seed(self.random_state)

        if self.window_size < 4:
            if self.verbose > 0:
                print("WARN: window_size must be at least 4. Dynamically setting window_size to 4.")
            self.window_size = 4

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods."""
        
        n_samples, n_features = X.shape
        if self.normalize: 
            if n_features == 1:
                X = zscore(X, axis=0, ddof=0)
            else: 
                X = zscore(X, axis=1, ddof=1)
        
        self.values = X[:, 0]
        #print("values", self.values)
        self.decision_scores_ = self.compute_stamp(self.values)
        #print(self.values.shape,self.decision_scores_.shape)
        return self

    def compute_stamp(self, values):
        #print(f"Input values stats: min={values.min()}, max={values.max()}, std={values.std()}")
        
        if self.n_jobs <= 1:
            stamp_mp = stumpy.stump(values, m=self.window_size)
        else:
            stamp_mp = stumpy.stump(values, m=self.window_size, n_threads=self.n_jobs)
        
        #print("Matrix Profile (stumpy output):", stamp_mp)

        if np.all(stamp_mp[:, 0] == 0):
            print("WARNING: Matrix profile contains only zeros.")
        
        return stamp_mp[:, 0]

    def decision_function(self, X):
        """Predict raw anomaly scores using the fitted detector."""
        n_samples, n_features = X.shape
        reduced_length = self.decision_scores_.shape[0]

        if reduced_length < n_samples:
            self.decision_scores_ = np.array(
                [self.decision_scores_[0]] * math.ceil((self.window_size - 1) / 2) +
                list(self.decision_scores_) +
                [self.decision_scores_[-1]] * ((self.window_size - 1) // 2)
            )
            #print(self.decision_scores_.shape)
            self.decision_scores_ = self.decision_scores_[:n_samples]

        assert self.decision_scores_.shape[0] == n_samples, (
            f"Length mismatch after padding: {self.decision_scores_.shape[0]} != {n_samples}"
        )

        self._process_decision_scores() 
        return self.decision_scores_



def run_STAMP_Unsupervised(data, HP):
    clf = STAMP(HP=HP)
    #print("LOGGER", data)
    clf.fit(data)
    scores = clf.decision_function(data)
    

    scores_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
    return scores_scaled



if __name__ == '__main__':
    Start_T = time.time()
    parser = argparse.ArgumentParser(description='Running STAMP')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='STAMP')
    args = parser.parse_args()

    HP = {
        "anomaly_window_size": 30,  
        "verbose": 1,
        "n_jobs": 1,
        "random_state": 42
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    #print(data)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    
    slidingWindow = find_length_rank(data, rank=1)

    data_train = data 
    label_train = label 

    #print("Unique labels in label_train:", np.unique(label_train))
    start_time = time.time()

    
    output = run_STAMP_Unsupervised(data_train, HP)

    end_time = time.time()
    run_time = end_time - start_time

    trimmed_label_train = label_train 

    #print(trimmed_label_train)

    pred = output > (np.mean(output) + 3 * np.std(output))

    #print("Anomaly Scores: ")
    #print(output)

    evaluation_result = get_metrics(output, trimmed_label_train, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)