import numpy as np
import pandas as pd
import argparse
import time
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

class PS_SVM(BaseDetector):

    def __init__(self, HP, normalize=True):
        """Initialize the anomaly detector."""
        super().__init__()
        self.HP = HP
        self.normalize = normalize
        self.decision_scores_ = None

    def unfold(self, ts: np.ndarray, dim: int) -> np.ndarray:
        """Unfolds the time series into an embedding matrix."""
        start = 0
        n = len(ts) - start - dim + 1
        index = start + np.expand_dims(np.arange(dim), 0) + np.expand_dims(np.arange(n), 0).T
        return ts[index]

    def project(self, q: np.ndarray, dim: int) -> np.ndarray:
        """Projects the embedding matrix onto a hyperspace."""
        ones = np.ones(dim)
        proj_matrix = np.identity(dim) - (1 / dim) * ones * ones.T
        return np.dot(q, proj_matrix)

    def svm(self, X: np.ndarray, **svm_kwargs: dict) -> np.ndarray:
        """Train One-Class SVM and compute anomaly scores."""
        clf = OneClassSVM(**svm_kwargs)
        clf.fit(X)
        scores = clf.decision_function(X)
        return scores * -1  # Invert scores: higher scores indicate anomalies

    def fit(self, X: np.ndarray, y=None):
        """Fit the detector to the data."""
        n_samples, n_features = X.shape

        if self.normalize:
            X = zscore(X, axis=0, ddof=0)

        
        embed_dims = self.HP['embed_dim_range']
        projected_ps = self.HP.get('project_phasespace', False)
        svm_kwargs = {
            'nu': self.HP.get('nu', 0.5),
            'gamma': self.HP.get('gamma', 'scale'),
            'kernel': self.HP.get('kernel', 'rbf'),
            'degree': self.HP.get('degree', 3),
            'coef0': self.HP.get('coef0', 0.0),
            'tol': self.HP.get('tol', 1e-3),
            'shrinking': self.HP.get('shrinking', True),
            'cache_size': self.HP.get('cache_size', 200),
            'max_iter': self.HP.get('max_iter', -1)
        }

        score_list = []
        for dim in embed_dims:
            Q = self.unfold(X.ravel(), dim)  
            if projected_ps:
                Q = self.project(Q, dim)

            scores = self.svm(Q, **svm_kwargs)
            aligned_scores = np.full(X.shape[0], np.nan)
            aligned_scores[:-dim + 1] = scores  
            score_list.append(aligned_scores)

        self.decision_scores_ = np.nansum(np.array(score_list), axis=0)
        return self

    def decision_function(self, X):
        """Predict raw anomaly scores for new data."""
        return self.decision_scores_


def run_PS_SVM(data, HP):
    clf = PS_SVM(HP=HP)
    clf.fit(data)
    scores = clf.decision_scores_
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
    return scores

if __name__ == '__main__':

    Start_T = time.time()

    # ArgumentParser
    parser = argparse.ArgumentParser(description='Running Custom_AD')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='Custom_AD')
    args = parser.parse_args()

    HP = {
        'embed_dim_range': [50, 100, 150],
        'project_phasespace': False,
        'nu': 0.5,
        'gamma': 'scale',
        'kernel': 'rbf',
        'degree': 3,
        'coef0': 0.0,
        'tol': 1e-3,
        'shrinking': True,
        'cache_size': 200,
        'max_iter': -1
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)

    train_index = int(args.filename.split('.')[0].split('_')[-3])
    data_train = data

    start_time = time.time()

    output = run_PS_SVM(data, HP)
    # output = run_Custom_AD_Unsupervised(data, **Custom_AD_HP) #ignore

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output) + 3 * np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)
    print('Runtime: {:.2f} seconds'.format(run_time))
