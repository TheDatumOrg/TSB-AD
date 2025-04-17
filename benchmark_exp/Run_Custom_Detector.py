# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

"""
This function is adapted from [pyod] by [yzhao062]
Original source: [https://github.com/yzhao062/pyod]
"""

from __future__ import division
from __future__ import print_function
import json
import os
import traceback
import warnings

import numpy as np
import math
import argparse
import time
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.models.base import BaseDetector
from TSB_AD.models.feature import Window
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.utils.utility import check_parameter, standardizer


class Custom_AD(BaseDetector):
    """Principal component analysis (PCA) can be used in detecting outliers.
    PCA is a linear dimensionality reduction using Singular Value Decomposition
    of the data to project it to a lower dimensional space.

    In this procedure, covariance matrix of the data can be decomposed to
    orthogonal vectors, called eigenvectors, associated with eigenvalues. The
    eigenvectors with high eigenvalues capture most of the variance in the
    data.

    Therefore, a low dimensional hyperplane constructed by k eigenvectors can
    capture most of the variance in the data. However, outliers are different
    from normal data points, which is more obvious on the hyperplane
    constructed by the eigenvectors with small eigenvalues.

    Therefore, outlier scores can be obtained as the sum of the projected
    distance of a sample on all eigenvectors.
    See :cite:`shyu2003novel,aggarwal2015outlier` for details.

    Score(X) = Sum of weighted euclidean distance between each sample to the
    hyperplane constructed by the selected eigenvectors

    Parameters
    ----------
    n_components : int, float, None or string
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        if n_components == 'mle' and svd_solver == 'full', Minka\'s MLE is used
        to guess the dimension
        if ``0 < n_components < 1`` and svd_solver == 'full', select the number
        of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components
        n_components cannot be equal to n_features for svd_solver == 'arpack'.

    n_selected_components : int, optional (default=None)
        Number of selected principal components
        for calculating the outlier scores. It is not necessarily equal to
        the total number of the principal components. If not set, use
        all principal components.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < X.shape[1]
        randomized :
            run randomized SVD by the method of Halko et al.

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.

    weighted : bool, optional (default=True)
        If True, the eigenvalues are used in score computation.
        The eigenvectors with small eigenvalues comes with more importance
        in outlier score calculation.

    standardization : bool, optional (default=True)
        If True, perform standardization first to convert
        data to zero mean and unit variance.
        See http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.

    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.

        Equal to n_components largest eigenvalues
        of the covariance matrix of X.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=0)`.

    n_components_ : int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or n_features if n_components is None.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        computed the estimated data covariance and score samples.

        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, slidingWindow=100, sub=True, n_components=None, n_selected_components=None,
                 contamination=0.1, copy=True, whiten=False, svd_solver='auto',
                 tol=0.0, iterated_power='auto', random_state=0,
                 weighted=True, standardization=True, zero_pruning=True):

        super(Custom_AD, self).__init__(contamination=contamination)
        self.slidingWindow = slidingWindow
        self.sub = sub
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.weighted = weighted
        self.standardization = standardization
        self.zero_pruning = zero_pruning

    # noinspection PyIncorrectDocstring
    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_samples, n_features = X.shape

        # Converting time series data into matrix format
        X = Window(window=self.slidingWindow).convert(X)

        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # PCA is recommended to use on the standardized data (zero mean and
        # unit variance).
        if self.standardization:
            X, self.scaler_ = standardizer(X, keep_scalar=True)

        if self.zero_pruning:
            non_zero_columns = np.any(X != 0, axis=0)
            X = X[:, non_zero_columns]

        self.detector_ = sklearn_PCA(n_components=self.n_components,
                                     copy=self.copy,
                                     whiten=self.whiten,
                                     svd_solver=self.svd_solver,
                                     tol=self.tol,
                                     iterated_power=self.iterated_power,
                                     random_state=self.random_state)
        self.detector_.fit(X=X, y=y)

        # copy the attributes from the sklearn PCA object
        self.n_components_ = self.detector_.n_components_
        self.components_ = self.detector_.components_

        # validate the number of components to be used for outlier detection
        if self.n_selected_components is None:
            self.n_selected_components_ = self.n_components_
        else:
            self.n_selected_components_ = self.n_selected_components
        check_parameter(self.n_selected_components_, 1, self.n_components_,
                        include_left=True, include_right=True,
                        param_name='n_selected_components_')

        # use eigenvalues as the weights of eigenvectors
        self.w_components_ = np.ones([self.n_components_, ])
        if self.weighted:
            self.w_components_ = self.detector_.explained_variance_ratio_

        # outlier scores is the sum of the weighted distances between each
        # sample to the eigenvectors. The eigenvectors with smaller
        # eigenvalues have more influence
        # Not all eigenvectors are used, only n_selected_components_ smallest
        # are used since they better reflect the variance change

        self.selected_components_ = self.components_[
            -1 * self.n_selected_components_:, :]
        self.selected_w_components_ = self.w_components_[
            -1 * self.n_selected_components_:]

        self.decision_scores_ = np.sum(
            cdist(X, self.selected_components_) / self.selected_w_components_,
            axis=1).ravel()

        # padded decision_scores_
        if self.decision_scores_.shape[0] < n_samples:
            self.decision_scores_ = np.array([self.decision_scores_[0]]*math.ceil((self.slidingWindow-1)/2) +
                                             list(self.decision_scores_) + [self.decision_scores_[-1]]*((self.slidingWindow-1)//2))

        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['components_', 'w_components_'])

        n_samples, n_features = X.shape

        # Converting time series data into matrix format
        X = Window(window=self.slidingWindow).convert(X)

        X = check_array(X)
        if self.standardization:
            X = self.scaler_.transform(X)

        decision_scores_ = np.sum(
            cdist(X, self.selected_components_) / self.selected_w_components_,
            axis=1).ravel()
        # padded decision_scores_
        if decision_scores_.shape[0] < n_samples:
            decision_scores_ = np.array([decision_scores_[0]]*math.ceil((self.slidingWindow-1)/2) +
                                        list(decision_scores_) + [decision_scores_[-1]]*((self.slidingWindow-1)//2))
        return decision_scores_

    @property
    def explained_variance_(self):
        """The amount of variance explained by each of the selected components.

        Equal to n_components largest eigenvalues
        of the covariance matrix of X.

        Decorator for scikit-learn PCA attributes.
        """
        return self.detector_.explained_variance_

    @property
    def explained_variance_ratio_(self):
        """Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0.

        Decorator for scikit-learn PCA attributes.
        """
        return self.detector_.explained_variance_ratio_

    @property
    def singular_values_(self):
        """The singular values corresponding to each of the selected
        components. The singular values are equal to the 2-norms of the
        ``n_components`` variables in the lower-dimensional space.

        Decorator for scikit-learn PCA attributes.
        """
        return self.detector_.singular_values_

    @property
    def mean_(self):
        """Per-feature empirical mean, estimated from the training set.

        Decorator for scikit-learn PCA attributes.
        """
        return self.detector_.mean_

    @property
    def noise_variance_(self):
        """The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        computed the estimated data covariance and score samples.

        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

        Decorator for scikit-learn PCA attributes.
        """
        return self.detector_.noise_variance_


def run_Custom_AD_Unsupervised(data, **HP):
    clf = Custom_AD(**HP)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        score.reshape(-1, 1)).ravel()
    return score


def run_Custom_AD_Semisupervised(data_train, data_test, **HP):
    clf = Custom_AD(**HP)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        score.reshape(-1, 1)).ravel()
    return score


if __name__ == '__main__':

    Start_T = time.time()
    # ArgumentParser
    parser = argparse.ArgumentParser(description='Running Custom_AD')
    parser.add_argument('--file_list', type=str, default='Datasets/File_List/TSB-AD-M-Eva.csv',
                        help='Path to the CSV file containing the list of filenames to process')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-M/',
                        help='Directory containing the dataset CSV files')
    parser.add_argument('--output_direc', type=str, default='benchmark_exp/benchmark_eval_results',
                        help='Directory to save the individual metric JSON files')
    args = parser.parse_args()

    Custom_AD_HP = {'n_components': 0.25, }

    # --- Create Output Directory ---
    try:
        os.makedirs(args.output_direc, exist_ok=True)
        print(f"Output directory: {args.output_direc}")
    except OSError as e:
        print(
            f"ERROR: Could not create output directory '{args.output_direc}': {e}")
        exit(1)

    # --- Determine files to process FROM THE FILE LIST ---
    files_to_process = []
    try:
        df_file_list = pd.read_csv(args.file_list)
        if df_file_list.empty or df_file_list.shape[1] == 0:
            print(
                f"ERROR: File list CSV '{args.file_list}' is empty or has no columns.")
            exit()
        files_to_process = df_file_list.iloc[:, 0].astype(
            str).str.strip().tolist()
        if not files_to_process:
            print(
                f"ERROR: No valid filenames extracted from the first column of '{args.file_list}'.")
            exit()
        print(
            f"Loaded {len(files_to_process)} filenames to process from {args.file_list}.")
    except FileNotFoundError:
        print(f"ERROR: File list CSV not found: {args.file_list}")
        exit()
    except Exception as e:
        print(
            f"ERROR: Failed to read or parse file list '{args.file_list}': {e}")
        exit()
    print("-" * 50)

    processed_files_count = 0
    failed_files = []

    # --- Loop through files ---
    for current_filename in tqdm(files_to_process, desc="Processing Files"):
        file_path = os.path.join(args.data_direc, current_filename)

        if not os.path.exists(file_path):
            tqdm.write(
                f"  WARNING: Skipping {current_filename} - File not found.")
            failed_files.append(current_filename + " (Not Found)")
            continue

        # Construct output path before try block
        base_name = os.path.splitext(current_filename)[0]
        output_json_path = os.path.join(
            args.output_direc, f"{base_name}_metrics.json")

        try:
            # --- Load and Prepare Data ---
            df = pd.read_csv(file_path).dropna()
            if df.empty or 'Label' not in df.columns or df.shape[1] < 2:
                failed_files.append(current_filename + " (Format)")
                continue
            data = df.iloc[:, 0:-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()

            # This recalculates window per file
            slidingWindow = find_length_rank(data, rank=1)

            # --- Run Unsupervised Custom_AD ---
            # Pass the defined hyperparameters
            output = run_Custom_AD_Unsupervised(data, **Custom_AD_HP)

            # --- Process output & metrics ---
            if isinstance(output, np.ndarray) and output.size > 0:
                if output.ndim > 1:
                    if output.shape[1] == 1:
                        output = output.flatten()
                    else:
                        failed_files.append(current_filename + " (Shape)")
                        continue

                # Handle potential NaN/Inf in scores before scaling
                if np.any(~np.isfinite(output)):
                    tqdm.write(
                        f"  WARNING: Skipping {current_filename} - Non-finite values found in output scores.")
                    failed_files.append(current_filename + " (Score NaN/Inf)")
                    continue

                # Scale output scores
                output_scaled = MinMaxScaler(feature_range=(
                    0, 1)).fit_transform(output.reshape(-1, 1)).ravel()

                # Align lengths
                min_len = min(len(output_scaled), len(label))
                output_final = output_scaled[:min_len]
                label_final = label[:min_len]

                # --- Calculate Metrics ---
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    # Use pred=None for VUS-PR etc. to work correctly
                    evaluation_result = get_metrics(
                        output_final, label_final, slidingWindow=slidingWindow, pred=None)

                # Log captured warnings using tqdm.write
                if caught_warnings:
                    for warning_msg in caught_warnings:
                        tqdm.write(
                            f"  WARNING in {current_filename} - {warning_msg.category.__name__}: {warning_msg.message}")

                # --- Save metrics to JSON ---
                try:
                    with open(output_json_path, 'w') as f:
                        json.dump(evaluation_result, f, indent=4)
                    processed_files_count += 1
                except Exception as save_e:
                    tqdm.write(
                        f"  ERROR saving metrics for {current_filename} to {output_json_path}: {save_e}")
                    failed_files.append(current_filename + " (Save Error)")

            else:
                # Handle cases where run function returns non-array (e.g., error string)
                tqdm.write(
                    f'  Skipping {current_filename}: Invalid output type from model ({type(output)}).')
                failed_files.append(current_filename + " (Output Type)")

        except Exception as e:
            tb_str = traceback.format_exc()
            tqdm.write(
                f"\n  ERROR processing {current_filename}: {e}\n{tb_str}")
            failed_files.append(current_filename + " (Exception)")
    # --- End Inner file loop ---

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print("Processing Finished.")
    print(
        f"Successfully processed and saved metrics for: {processed_files_count}/{len(files_to_process)} files.")
    if failed_files:
        print(
            f"Failed/Skipped files ({len(failed_files)}): {', '.join(failed_files)}")
    print(f"Metric files saved in: {args.output_direc}")
    print(f"Total time: {time.time() - Start_T:.2f} seconds")
    print("-" * 60)
