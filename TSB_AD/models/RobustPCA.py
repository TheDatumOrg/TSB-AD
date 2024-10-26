"""
This function is adapted from [TimeEval-algorithms] by [CodeLionX&wenig]
Original source: [https://github.com/TimeEval/TimeEval-algorithms]
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional

from .base import BaseDetector
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array
from scipy.spatial.distance import cdist

class Robust_PCA:
    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)                            #this line implements step 3
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)             #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk
    
class RobustPCA(BaseDetector):
    def __init__(self, max_iter: int = 1000, n_components = None, zero_pruning = True):
        self.pca: Optional[PCA] = None
        self.max_iter = max_iter
        self.n_components = n_components
        self.zero_pruning = zero_pruning

    def fit(self, X, y=None):

        if self.zero_pruning:
            non_zero_columns = np.any(X != 0, axis=0)
            X = X[:, non_zero_columns]
        
        rpca = Robust_PCA(X)
        L, S = rpca.fit(max_iter=self.max_iter)
        self.detector_ = PCA(n_components=L.shape[1])
        self.detector_.fit(L)
        self.decision_scores_ = self.decision_function(L)
        return self

    # def decision_function(self, X):
    #     check_is_fitted(self, ['detector_'])
    #     X_transformed = self.detector_.transform(X)  # Transform the data into the PCA space
    #     X_reconstructed = self.detector_.inverse_transform(X_transformed)  # Reconstruct the data from the PCA space
    #     anomaly_scores = np.linalg.norm(X - X_reconstructed, axis=1)  # Compute the Euclidean norm between original and reconstructed data
    #     return anomaly_scores

    def decision_function(self, X):
        assert self.detector_, "Please train PCA before running the detection!"

        L = self.detector_.transform(X)
        S = np.absolute(X - L)
        return S.sum(axis=1)
