import numpy as np
import math
from utils.slidingWindows import find_length_rank
from sklearn.preprocessing import MinMaxScaler

# from models.NormA import NORMA
from models.LOF import LOF
from models.IForest import IForest
from models.MCD import MCD
from models.POLY import POLY
from models.MatrixProfile import MatrixProfile
from models.PCA import PCA
from models.HBOS import HBOS
from models.OCSVM import OCSVM
from models.KNN import KNN
from models.KMeansAD import KMeansAD
from models.COPOD import COPOD
from models.CBLOF import CBLOF
from models.COF import COF
from models.EIF import EIF
from models.RobustPCA import RobustPCA
from models.AE import AutoEncoder
from models.CNN import CNN
from models.LSTMAD import LSTMAD
from models.TranAD import TranAD
from models.USAD import USAD
from models.OmniAnomaly import OmniAnomaly
from models.AnomalyTransformer import AnomalyTransformer
from models.TimesNet import TimesNet
from models.FITS import FITS
from models.Donut import Donut
from models.OFA import OFA
from models.Lag_Llama import Lag_Llama
# from models.Chronos import Chronos

Unsupervise_AD_Pool = ['NORMA', 'IForest', 'IForest1', 'LOF', 'POLY', 'MatrixProfile', 'PCA', 'HBOS', 'KNN', 'KMeansAD', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA', 'Lag_Llama', 'Chronos']
Semisupervise_AD_Pool = ['MCD', 'OCSVM', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly', 'AnomalyTransformer', 'TimesNet', 'FITS', 'Donut', 'OFA']

def run_Unsupervise_AD(model_name, data, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message

def run_Semisupervise_AD(model_name, data_train, data_test, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data_train, data_test, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message

def run_IForest(data, periodicity=1, n_estimators=100, max_features=1, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_IForest1(data, periodicity=1, n_estimators=100, max_features=1, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = IForest(slidingWindow=slidingWindow, sub=False, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_LOF(data, periodicity=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_POLY(data, periodicity=1, power=3, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = POLY(power=power, window = slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_MatrixProfile(data, periodicity=1, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = MatrixProfile(slidingWindow = slidingWindow, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_PCA(data, periodicity=1, n_components=None, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

# def run_NORMA(data, periodicity=1, clustering='hierarchical', n_jobs=1):
#     slidingWindow = find_length_rank(data, rank=periodicity)
#     clf = NORMA(pattern_length=slidingWindow, nm_size=3*slidingWindow, clustering=clustering)
#     clf.fit(data)
#     score = clf.decision_scores_
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
#     score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
#     if len(score) > len(data):
#         start = len(score) - len(data)
#         score = score[start:]
#     return score

def run_HBOS(data, periodicity=1, n_bins=10, tol=0.5, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_OCSVM(data_train, data_test, kernel='rbf', nu='0.5', periodicity=1, n_jobs=1):
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_MCD(data_train, data_test, support_fraction=None, periodicity=1, n_jobs=1):
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_KNN(data, n_neighbors=10, method='largest', n_jobs=1):
    clf = KNN(n_neighbors=n_neighbors, method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_KMeansAD(data, n_clusters=20, window_size=20, n_jobs=1):
    clf = KMeansAD(k=n_clusters, window_size=window_size, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_COPOD(data, n_jobs=1):
    clf = COPOD(n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_CBLOF(data, n_clusters=8, alpha=0.9, n_jobs=1):
    clf = CBLOF(n_clusters=n_clusters, alpha=alpha, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_COF(data, n_neighbors=30):
    clf = COF(n_neighbors=n_neighbors)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_EIF(data, n_trees=100):
    clf = EIF(n_trees=n_trees)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_RobustPCA(data, max_iter=1000):
    clf = RobustPCA(max_iter=max_iter)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_AutoEncoder(data_train, data_test, window_size=100, hidden_neurons=[64, 32], n_jobs=1):
    clf = AutoEncoder(slidingWindow=window_size, hidden_neurons=hidden_neurons, batch_size=128, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_CNN(data_train, data_test, window_size=100, num_channel=[32, 32, 40], n_jobs=1):
    clf = CNN(window_size=window_size, num_channel=num_channel, feats=data_test.shape[1], batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_LSTMAD(data_train, data_test, window_size=100, lr=0.0008):
    clf = LSTMAD(window_size=window_size, pred_len=1, lr=lr, feats=data_test.shape[1], batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_TranAD(data_train, data_test, win_size=10, lr=1e-3):
    clf = TranAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_AnomalyTransformer(data_train, data_test, win_size=100, lr=1e-4):
    clf = AnomalyTransformer(win_size=win_size, input_c=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_OmniAnomaly(data_train, data_test, win_size=100, lr=0.002):
    clf = OmniAnomaly(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_USAD(data_train, data_test, win_size=5, lr=1e-4):
    clf = USAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_Donut(data_train, data_test, win_size=120, lr=1e-4):
    clf = Donut(win_size=win_size, input_c=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_TimesNet(data_train, data_test, win_size=96, lr=1e-4):
    clf = TimesNet(win_size=win_size, enc_in=data_test.shape[1], lr=lr, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_FITS(data_train, data_test, win_size=100, lr=1e-3):
    clf = FITS(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_OFA(data_train, data_test, win_size=100):
    clf = OFA(win_size=win_size, enc_in=data_test.shape[1], epochs=10)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_Lag_Llama(data, win_size=96):
    clf = Lag_Llama(win_size=win_size, input_c=data.shape[1], batch_size=128)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

# def run_Chronos(data, win_size):
#     clf = Chronos(win_size=win_size, prediction_length=1, input_c=data.shape[1], model_size='base')
#     clf.fit(data)
#     score = clf.decision_scores_
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
#     return score