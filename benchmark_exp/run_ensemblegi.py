import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore
from sksequitur import parse, Grammar
from saxpy.sax import ts_to_string
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.alphabet import cuts_for_asize
from joblib import Parallel, delayed

class EnsembleGI(BaseDetector):

    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize
        self.window_size = HP.get('anomaly_window_size', 50)
        self.ensemble_size = HP.get('n_estimators', 10)
        self.w_max = HP.get('max_paa_transform_size', 20)
        self.a_max = HP.get('max_alphabet_size', 10)
        self.selectivity = HP.get('selectivity', 0.8)
        self.random_state = HP.get('random_state', 42)
        self.n_jobs = HP.get('n_jobs', 1)
        np.random.seed(self.random_state)
        self.decision_scores_ = None

    def sax(self, X, w, a, method="orig"):
        saxed = []
        if method == "sliding":
            iter_range = range(len(X) - self.window_size + 1)
        elif method == "tumbling":
            iter_range = range(0, len(X), self.window_size)
        else:
            iter_range = range(len(X) // self.window_size)

        for i in iter_range:
            _paa = paa(znorm(X[i:i + self.window_size]), w)
            saxed.append(_paa)

        if method == "orig":
            rest = len(X) % self.window_size
            rest_segments = int(w * (rest / self.window_size)) or 1
            if rest > 0:
                _paa = paa(znorm(X[-rest:]), rest_segments)
                saxed.append(_paa)

        saxed = np.concatenate(saxed)
        return ts_to_string(saxed, cuts_for_asize(a))

    @staticmethod
    def density_curve(grammar):
        rules = grammar[0]
        density_curve = []
        i = 0
        depth = 0
        depths = []
        while i < len(rules):
            value_at_i = rules[i]
            rule = grammar.get(value_at_i, value_at_i)
            if isinstance(rule, str):
                i += 1
                density_curve.append(depth)
                if depths:
                    depths[-1] -= 1
                    while depths and depths[-1] < 1:
                        depths.pop()
                        depth -= 1
                        if depths:
                            depths[-1] -= 1
            else:
                rules[i] = rule
                depth += 1
                depths.append(len(rule))
            rules = EnsembleGI.flatten(rules)
        return np.array(density_curve)

    @staticmethod
    def flatten(x):
        _x = []
        for dd in x:
            if isinstance(dd, list):
                _x.extend(dd)
            else:
                _x.append(dd)
        return _x

    @staticmethod
    def stretch_density_curve(density_curve, w, l):
        return np.interp(np.arange(l), np.linspace(0, l, len(density_curve)), density_curve)

    def induce_grammar(self, X, w, a, window_method):
        saxed = self.sax(X, w, a, method=window_method)
        parsed = parse(saxed)
        density_curve = self.density_curve(parsed)
        return self.stretch_density_curve(density_curve, w, len(X))

    def _random_params(self):
        n_combinations = (self.w_max - 2) * (self.a_max - 2)
        comb_idx = np.random.choice(n_combinations, size=self.ensemble_size, replace=False)
        w = (comb_idx // (self.w_max - 2)) + 2
        a = (comb_idx % (self.a_max - 2)) + 2
        return zip(w, a)

    def fit(self, X, y=None, window_method="orig"):
        n_samples, n_features = X.shape
        if self.normalize: 
            if n_features == 1:
                X = zscore(X, axis=0, ddof=0)
            else: 
                X = zscore(X, axis=1, ddof=1)

        #print(X)
        def calculate_density_curve(i, w, a):
            return self.induce_grammar(X.flatten(), w, a, window_method)

        density_curves = Parallel(n_jobs=self.n_jobs)(delayed(calculate_density_curve)(i, w, a)
                                                       for i, (w, a) in enumerate(self._random_params()))

        density_curves = np.stack(density_curves)
        indices = np.argsort(density_curves.std(axis=1))[::-1]
        selected_curves = indices[:int(self.ensemble_size * self.selectivity)]
        density_curves = density_curves[selected_curves]

        density_curves = density_curves / np.max(density_curves, axis=0)

        density_overall = 1 - np.median(density_curves, axis=0)

        self.decision_scores_ = density_overall
        return self

    def decision_function(self, X):
        if self.decision_scores_ is None:
            raise RuntimeError("fit must be called before decision_function.")

        return self.decision_scores_


def run_EnsembleGI(data, HP):
    clf = EnsembleGI(HP=HP)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    #print("Anomaly scores:", score)

    

    return score

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running EnsembleGI')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='EnsembleGI')
    args = parser.parse_args()

    EnsembleGI_HP = {
        'anomaly_window_size': 50,
        'n_estimators': 10,
        'max_paa_transform_size': 20,
        'max_alphabet_size': 10,
        'selectivity': 0.8,
        'random_state': 42,
        'n_jobs': 1
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

    output = run_EnsembleGI(data_train, EnsembleGI_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output) + 3 * np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)