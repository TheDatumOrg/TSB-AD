import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from pyod.models.pca import PCA
from dataclasses import dataclass

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

class PCC(BaseDetector):
    def __init__(self, HP):
        super().__init__()
        self.HP = HP
        self.model = None

    def fit(self, X):
        """Fit the PCA-based anomaly detector."""
        self.model = PCA(
            contamination=self.HP.get('contamination', 0.1),
            n_components=self.HP.get('n_components', None),
            n_selected_components=self.HP.get('n_selected_components', None),
            whiten=self.HP.get('whiten', False),
            svd_solver=self.HP.get('svd_solver', 'auto'),
            tol=self.HP.get('tol', 0.0),
            iterated_power=self.HP.get('max_iter', "auto"),
            random_state=self.HP.get('random_state', 42),
            copy=True,
            weighted=True,
            standardization=True,
        )

        X = zscore(X, axis=0, ddof=0)
        self.model.fit(X)
        self.decision_scores_ = self.model.decision_scores_
        return self

    def decision_function(self, X):
        """Calculate anomaly scores for input samples."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        scores = self.model.decision_function(X)
        return scores


def run_PCC(data, HP):
    """Run PCA-based anomaly detection and normalize the scores."""
    clf = PCC(HP=HP)
    clf.fit(data)
    scores = clf.decision_scores_
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
    return scores


def load_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path, header=None, names=['value', 'label'])
    df['value'].fillna(df['value'].mean(), inplace=True)
    df['time'] = np.arange(len(df))

    data = df['value'].values.reshape(-1, 1)
    label = df['label'].values.astype(int)
    contamination = df['label'].mean()
    contamination = np.nextafter(0, 1) if contamination == 0 else contamination
    return data, label, contamination


def evaluate_anomaly_detection(scores, labels):
    """Evaluate anomaly detection performance using AUC-ROC."""
    return roc_auc_score(labels, scores)


if __name__ == '__main__':

    Start_T = time.time()
    # ArgumentParser
    parser = argparse.ArgumentParser(description='Running PCC')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='PCC')
    args = parser.parse_args()

    PCC_HP = {
        'n_components': 1,
        'whiten': False,
        'svd_solver': 'auto',
        'random_state': 42
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

    output = run_PCC(data_train, PCC_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output)+3*np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)