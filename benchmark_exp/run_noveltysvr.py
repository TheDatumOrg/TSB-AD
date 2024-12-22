import argparse
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import roc_auc_score
from TSB_AD.models.base import BaseDetector
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.utils.utility import zscore
from pyonlinesvr import OnlineSVR 
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import binom
from typing import Optional
import joblib
import warnings


class NoveltySVR(BaseDetector):
    def __init__(
        self,
        HP,
        train_window_size: int = 16,
        anomaly_window_size: int = 6,
        lower_suprise_bound: Optional[int] = None,
        scaling: str = "standard",
        forgetting_time: Optional[int] = None,
        epsilon: float = 0.1,
        verbose: int = 0,
        C: float = 30.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Optional[float] = None,
        coef0: float = 0.0,
        tol: float = 1e-3,
        stabilized: bool = True,
        normalize=True
    ):
        super().__init__()
        self.HP = HP
        self.train_window_size = train_window_size
        self.anomaly_window_size = anomaly_window_size
        self.verbose = verbose
        self.forgetting_time = forgetting_time
        self.lower_suprise_bound = lower_suprise_bound if lower_suprise_bound else anomaly_window_size // 2
        self.scaling = scaling
        self.normalize = normalize

        if scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "robust":
            self.scaler = RobustScaler()
        elif scaling == "power":
            self.scaler = PowerTransformer()
        else:
            self.scaler = DummyScaler() 

        self.svr = OnlineSVR(
            epsilon=epsilon,
            verbose=max(0, verbose - 2),
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            stabilized=stabilized,
            save_kernel_matrix=True,
        )

    def fit(self, X: np.ndarray, y=None):
        """ Fit the model with the data """
        n_samples, n_features = X.shape
        if self.normalize: 
            X = zscore(X, axis=0, ddof=0) 

        print(X)
        self.decision_scores_ = np.zeros(n_samples)
        X = self.scaler.fit_transform(X)  
        return self

    def decision_function(self, X):
        """ Compute the decision function """
        n_samples, n_features = X.shape
        decision_scores_ = np.zeros(n_samples)
        X = self.scaler.transform(X)  
        return decision_scores_

    def detect(self, X: np.ndarray) -> np.ndarray:
        """ Detect anomalies based on the model """
        X = self.scaler.transform(X)
        y_hat = np.zeros_like(X)
        qs = np.zeros_like(X)
        for i, xt in enumerate(X):
            y_hat[i] = self.svr.predict([xt])
            qs[i] = self._calc_current_q() 
        return MinMaxScaler().fit_transform(y_hat.reshape(-1, 1)).reshape(-1)


def run_Custom_AD_Semisupervised(data_train, data_test, HP):
    """ Run the Custom AD Semisupervised approach with NoveltySVR """
    clf = NoveltySVR(HP=HP)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score


if __name__ == '__main__':
    Start_T = time.time()
    parser = argparse.ArgumentParser(description='Running NoveltySVR')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='NoveltySVR')
    args = parser.parse_args()

    Custom_AD_HP = {
        'HP': ['HP'],  
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data

    start_time = time.time()

    output = run_Custom_AD_Semisupervised(data_train, data, **Custom_AD_HP)

    end_time = time.time()
    run_time = end_time - start_time

    print(f"Anomaly Scores: {output}")

    pred = output > (np.mean(output) + 3 * np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)

    auc_roc = roc_auc_score(label, output)
    print(f"AUC ROC Score: {auc_roc:.4f}")
