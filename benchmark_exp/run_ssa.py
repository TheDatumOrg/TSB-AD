import os 
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

class SSA(BaseDetector):
    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize
        self.ep = HP.get('ep', 3)
        self.method = HP.get('method', 'GARCH')
        self.rf_method = HP.get('rf_method', 'all')
        self.n = HP.get('n', 720)
        self.a = HP.get('a', 0.2)

    def reference_time(self, index, a, ground_truth=False):
        n = self.n
        X = self.X_train_
        if ground_truth:
            return X[index - 2 * n: index - n], X[index - n: index]
        else:
            if isinstance(a, float):
                return X[index - 2 * n: index - n] * (1 - a) + a * X[index - n: index], X[index - n: index]
            else:
                num = math.floor(index / n)
                A = a[:num]
                A = A / np.sum(A)
                rf = np.zeros(n)
                for i in range(len(A)):
                    rf += A[i] * X[index - (i + 1) * n: index - i * n]
                return rf, X[index - n: index]

    def Linearization(self, X2, e=1):
        i = 0
        fit = {'index': [], 'rep': []}
        while i < len(X2):
            fit['index'].append(i)
            fit['Y' + str(i)] = X2[i]
            fit['rep'].append(np.array([i, X2[i]]))
            if i + 1 >= len(X2):
                break
            k = X2[i + 1] - X2[i]
            b = -i * (X2[i + 1] - X2[i]) + X2[i]
            fit['reg' + str(i)] = np.array([k, b])
            i += 2
            if i >= len(X2):
                break
            d = abs(X2[i] - (k * i + b))
            while d < e:
                i += 1
                if i >= len(X2):
                    break
                d = abs(X2[i] - (k * i + b))
        return fit

    def SSA(self, X2, X3, e=1):
        fit = self.Linearization(X2, e=e)
        fit2 = self.Linearization(X3, e=e)
        Index = list(set(fit['index'] + fit2['index']))
        Y = 0
        for i in Index:
            if i in fit['index'] and i in fit2['index']:
                Y += abs(fit['Y' + str(i)] - fit2['Y' + str(i)])
            elif i in fit['index']:
                J = np.max(np.where(np.array(fit2['index']) < i))
                index = fit2['index'][J]
                k, b = fit2['reg' + str(index)]
                Y += abs(k * i + b - fit['Y' + str(i)])
            elif i in fit2['index']:
                J = np.max(np.where(np.array(fit['index']) < i))
                index = fit['index'][J]
                k, b = fit['reg' + str(index)]
                Y += abs(k * i + b - fit2['Y' + str(i)])
        score = Y / len(Index)
        return score

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        if self.normalize:
            X = zscore(X, axis=0, ddof=0)
        print(X)
        self.X_train_ = X
        self.n_train_ = n_samples
        self.decision_scores_ = np.zeros(n_samples)

        n = self.n
        a = self.a
        ep = self.ep

        if self.rf_method == 'all':
            rf = np.zeros(self.n) 
            num = math.floor(self.n_train_ / self.n)
            for i in range(num):

                rf += X[i * self.n: (i + 1) * self.n].mean(axis=1) / num
            for i in range(2 * n, n_samples):
                X1 = X[i - n:i].mean(axis=1) 
                score = self.SSA(X1, rf)
                self.decision_scores_[i] = min(score / (2 * ep), 1)


        elif self.rf_method == 'alpha':
            for i in range(2 * n, n_samples):
                X1, X2 = self.reference_time(index=i, a=a)
                score = self.SSA(X1, X2)
                self.decision_scores_[i] = min(score / (2 * ep), 1)
        else:
            raise ValueError(self.method + " is not a valid reference timeseries method")

        return self

    def decision_function(self, X):
        #print(self.decision_scores_)
        return self.decision_scores_


def run_SSA_Unsupervised(data, HP):
    clf = SSA(HP=HP)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    return score

if __name__ == '__main__':

    import argparse, time

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running Custom_AD')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='Custom_AD')
    args = parser.parse_args()


    Custom_AD_HP = {
        'ep': 3,
        'method': 'GARCH',
        'rf_method': 'all',
        'n': 720,
        'a': 0.2,
    }

    df = pd.read_csv(os.path.join(args.data_direc, args.filename)).dropna()
    data = df.iloc[:, :-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)

    train_index = int(args.filename.split('.')[0].split('_')[-3])
    data_train = data

    start_time = time.time()

    output = run_SSA_Unsupervised(data_train, Custom_AD_HP)


    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output) + 3 * np.std(output))

    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)

    End_T = time.time()
    print(f"Execution Time: {End_T - Start_T:.2f} seconds")

