import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

class dbstream(BaseDetector):

    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize
        self.radius = HP.get('radius', 0.1)
        self.lambda_ = HP.get('lambda_', 0.001)
        self.metric = HP.get('metric', 'euclidean')
        self.subsequence_length = HP.get('subsequence_length', 20)
        self.n_clusters = HP.get('n_clusters', 3)
        self.alpha = HP.get('alpha', 0.3)
        self.min_weight = HP.get('min_weight', 0.0)
        self.random_state = HP.get('random_state', 42)
        
        np.random.seed(self.random_state)

    def fit(self, X, y=None):
        """Fit the detector. This will prepare the data for anomaly detection."""
        n_samples, n_features = X.shape
        if self.normalize: 
            X = zscore(X, axis=0, ddof=0)
        
        self.decision_scores_ = np.zeros(n_samples)
        self.X_train = X  
        return self

    def decision_function(self, X):
        """Compute raw anomaly scores for the input data X."""
        n_samples, n_features = X.shape
        subsequence_count = n_samples - self.subsequence_length + 1
        subsequences = self._preprocess(X, subsequence_count, n_features)
        
        micro_clusters = np.random.randint(0, self.n_clusters, size=subsequence_count)
        macro_clusters = {i: i % 2 for i in range(self.n_clusters)}
        cluster_centers = np.random.rand(self.n_clusters, subsequences.shape[1]) 
        weights = np.random.rand(self.n_clusters) 
        
        subsequence_anomaly_scores = self._compute_anomaly_scores_for_sequences(
            macro_clusters, cluster_centers, micro_clusters, subsequences, weights
        )

        anomaly_scores = self._compute_anomaly_scores_for_points(subsequence_anomaly_scores, subsequence_count)
        self.decision_scores_ = anomaly_scores
        return MinMaxScaler(feature_range=(0, 1)).fit_transform(anomaly_scores.reshape(-1, 1)).ravel()

    def _preprocess(self, values, subsequence_count, dimensionality):
        """Prepare subsequences."""
        subsequences = np.zeros((subsequence_count, self.subsequence_length, dimensionality))
        for row in range(subsequence_count):
            for col in range(self.subsequence_length):
                subsequences[row, col, 0] = values[row + col] 
        return subsequences.reshape(subsequence_count, self.subsequence_length * dimensionality)

    def _compute_anomaly_scores_for_sequences(self, macro_clusters, cluster_centers, micro_clusters, df, weights):
        """Compute anomaly scores for each subsequence."""
        anomaly_scores = np.full(df.shape[0], -1.0)
        for i in range(df.shape[0]):
            micro_cluster = micro_clusters[i]
            if macro_clusters[micro_cluster] is None:
                anomaly_scores[i] = -1
            else:
                macro_cluster_index = macro_clusters[micro_cluster]
                distance_score = pairwise_distances(df[i:i+1], cluster_centers[macro_cluster_index:macro_cluster_index+1], metric=self.metric)[0, 0]
                anomaly_scores[i] = -weights[macro_cluster_index] * distance_score
        max_score = np.max(anomaly_scores)
        anomaly_scores[anomaly_scores == -1] = max_score + 1
        return anomaly_scores

    def _compute_anomaly_scores_for_points(self, subsequence_anomaly_scores, subsequence_count):
        """Compute anomaly scores for each point."""
        anomaly_scores = np.zeros((subsequence_count + self.subsequence_length - 1, 2))
        for i in range(subsequence_count):
            for j in range(i, i + self.subsequence_length):
                anomaly_scores[j, 0] += subsequence_anomaly_scores[i]
                anomaly_scores[j, 1] += 1
        return anomaly_scores[:, 0] / anomaly_scores[:, 1]

def run_dbstream_Unsupervised(data, HP):
    clf = dbstream(HP=HP)
    clf.fit(data)
    score = clf.decision_scores_
    return score

def run_dbstream_unsupervised(data_train, data_test, HP):
    clf = dbstream(HP=HP)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score

if __name__ == '__main__':
    import argparse
    import time
    
    # Argument parsing
    parser = argparse.ArgumentParser(description='Running dbstream')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='dbstream')
    args = parser.parse_args()

    dbstream_HP = {
        'radius': 0.1,
        'lambda_': 0.001,
        'metric': 'euclidean',
        'subsequence_length': 20,
        'n_clusters': 3,
        'alpha': 0.3,
        'min_weight': 0.0,
        'random_state': 42
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    slidingWindow = find_length_rank(data, rank=1)
    train_index = int(args.filename.split('.')[0].split('_')[-3])  
    data_train = data

    start_time = time.time()
    output = run_dbstream_unsupervised(data_train, data, dbstream_HP)
    end_time = time.time()

    pred = output > (np.mean(output) + 3 * np.std(output)) 
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)

    run_time = end_time - start_time
    print(f"Run Time: {run_time:.2f} seconds")