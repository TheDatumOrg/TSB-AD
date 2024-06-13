import matrixprofile as mp
from .base import BaseDetector
import numpy as np
import math

class MatrixProfile(BaseDetector):
    def __init__(self, slidingWindow = 100, n_jobs=1):
        self.slidingWindow = slidingWindow
        self.n_jobs = n_jobs
        self.model_name = 'MatrixProfile'

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """

        X = X.squeeze()
        self.profile = mp.compute(X, windows=self.slidingWindow, n_jobs=self.n_jobs)
        self.decision_scores_ = self.profile['mp']

        # padded decision_scores_
        if self.decision_scores_.shape[0] < X.shape[0]:
            self.decision_scores_ = np.array([self.decision_scores_[0]]*math.ceil((self.slidingWindow-1)/2) + 
                        list(self.decision_scores_) + [self.decision_scores_[-1]]*((self.slidingWindow-1)//2))
    
    def decision_function(self, X):
        """
        Not used, present for API consistency by convention.
        """        
        pass

    def top_k_discords(self, k=5):
        discords = mp.discover.discords(self.profile, exclusion_zone=self.window//2, k=k)
        return discords['discords']