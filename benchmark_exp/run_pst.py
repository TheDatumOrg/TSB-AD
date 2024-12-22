import pandas as pd
import numpy as np
import random, argparse, time
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

class PST(BaseDetector):
    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.window_size = HP['window_size']
        self.n_bins = HP['n_bins']
        self.sim = HP['sim']
        self.random_state = HP['random_state']
        self.max_depth = HP['max_depth']
        self.n_min = HP['n_min']
        self.normalize = normalize

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        if self.normalize:
            X = zscore(X, axis=0, ddof=0)
        self.values = X.flatten()
        subsequence_count = len(self.values) - self.window_size + 1
        subsequences = self.split_into_subsequences(self.values, subsequence_count, self.window_size)
        tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.n_min)
        tree.fit(subsequences, np.zeros(subsequence_count))
        similarity_scores = self.compute_similarity_scores(subsequences)
        self.anomaly_scores = self.compute_anomaly_scores_for_points(similarity_scores, self.values, self.window_size)
        self.anomaly_scores = self.set_most_frequent_score_to_zero(self.anomaly_scores)
        return self

    def decision_function(self, X):
        n_samples, n_features = X.shape
        X = X.flatten()
        subsequence_count = len(X) - self.window_size + 1
        subsequences = self.split_into_subsequences(X, subsequence_count, self.window_size)
        similarity_scores = self.compute_similarity_scores(subsequences)
        anomaly_scores = self.compute_anomaly_scores_for_points(similarity_scores, X, self.window_size)
        anomaly_scores = self.set_most_frequent_score_to_zero(anomaly_scores)
        return MinMaxScaler(feature_range=(0, 1)).fit_transform(anomaly_scores.reshape(-1, 1)).ravel()

    def split_into_subsequences(self, values, subsequence_count, subsequence_length):
        subsequences = np.zeros((subsequence_count, subsequence_length), dtype=int)
        for row in range(subsequence_count):
            for col in range(subsequence_length):
                subsequences[row, col] = values[row + col]
        return subsequences

    def compute_similarity_scores(self, subsequences):
        if self.sim == "simo":
            return pairwise_distances(subsequences, metric='cosine')
        elif self.sim == "simn":
            return pairwise_distances(subsequences, metric='euclidean')

    def compute_anomaly_scores_for_points(self, similarity_scores, values, subsequence_length):
        anomaly_scores = np.zeros((len(values), 3))
        for i in range(len(similarity_scores)):
            for j in range(i, min(i + subsequence_length, len(values))):
                anomaly_scores[j, 0] += 1 / (1 + similarity_scores[i, j - i])
                anomaly_scores[j, 1] += 1
        return anomaly_scores[:, 0] / anomaly_scores[:, 1]

    def set_most_frequent_score_to_zero(self, anomaly_scores):
        flattened_scores = anomaly_scores.flatten()
        score_counts = Counter(flattened_scores)
        most_frequent_score = score_counts.most_common(1)[0][0]
        anomaly_scores[anomaly_scores == most_frequent_score] = 0
        return anomaly_scores

def run_PST_Unsupervised(data_train, data_test, HP):
    clf = PST(HP=HP)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score

if __name__ == '__main__':
    Start_T = time.time()
    parser = argparse.ArgumentParser(description='Running PST')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    args = parser.parse_args()

    PST_HP = {
        'window_size': 5,
        'n_bins': 5,
        'sim': 'simn',
        'random_state': 42,
        'max_depth': 4,
        'n_min': 30
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    slidingWindow = find_length_rank(data, rank=1)
    train_index = int(args.filename.split('.')[0].split('_')[-3])
    data_train = data
    
    start_time = time.time()

    output = run_PST_Unsupervised(data_train, data, PST_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output) + 3 * np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)