"""A set of utility functions to support outlier detection.
"""

from __future__ import division
from __future__ import print_function
from joblib.parallel import cpu_count
import numpy as np
from numpy import percentile
import numbers

import sklearn
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
import torch.nn as nn

MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT

def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)

    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd

    return np.nan_to_num(res)

def pairwise_distances_no_broadcast(X, Y):
    """Utility function to calculate row-wise euclidean distance of two matrix.
    Different from pair-wise calculation, this function would not broadcast.
    For instance, X and Y are both (4,3) matrices, the function would return
    a distance vector with shape (4,), instead of (4,4).
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples
    Y : array of shape (n_samples, n_features)
        Second input samples
    Returns
    -------
    distance : array of shape (n_samples,)
        Row-wise euclidean distance of X and Y
    """
    X = check_array(X)
    Y = check_array(Y)

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError("pairwise_distances_no_broadcast function receive"
                         "matrix with different shapes {0} and {1}".format(
            X.shape, Y.shape))
        
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

def getSplit(X):
    """
    Randomly selects a split value from set of scalar data 'X'.
    Returns the split value.
    
    Parameters
    ----------
    X : array 
        Array of scalar values
    Returns
    -------
    float
        split value
    """
    xmin = X.min()
    xmax = X.max()
    return np.random.uniform(xmin, xmax)

def similarityScore(S, node, alpha):
    """
    Given a set of instances S falling into node and a value alpha >=0,
    returns for all element x in S the weighted similarity score between x
    and the centroid M of S (node.M)
    
    Parameters
    ----------
    S : array  of instances
        Array  of instances that fall into a node
    node: a DiFF tree node
        S is the set of instances "falling" into the node
    alpha: float
        alpha is the distance scaling hyper-parameter
    Returns
    -------
    array
        the array of similarity values between the instances in S and the mean of training instances falling in node
    """
    d = np.shape(S)[1]
    if len(S) > 0:
        d = np.shape(S)[1]
        U = (S-node.M)/node.Mstd # normalize using the standard deviation vector to the mean
        U = (2)**(-alpha*(np.sum(U*U/d, axis=1)))
    else:
        U = 0

    return U


def EE(hist):
    """
    given a list of positive values as a histogram drawn from any information source,
    returns the empirical entropy of its discrete probability function.
    
    Parameters
    ----------
    hist: array 
        histogram
    Returns
    -------
    float
        empirical entropy estimated from the histogram
    """
    h = np.asarray(hist, dtype=np.float64)
    if h.sum() <= 0 or (h < 0).any():
        return 0
    h = h/h.sum()
    return -(h*np.ma.log2(h)).sum()


def weightFeature(s, nbins):
    '''
    Given a list of values corresponding to a feature dimension, returns a weight (in [0,1]) that is 
    one minus the normalized empirical entropy, a way to characterize the importance of the feature dimension. 
    
    Parameters
    ----------
    s: array 
        list of scalar values corresponding to a feature dimension
    nbins: int
        the number of bins used to discretize the feature dimension using an histogram.
    Returns
    -------
    float
        the importance weight for feature s.
    '''
    if s.min() == s.max():
        return 0
    hist = np.histogram(s, bins=nbins, density=True)
    ent = EE(hist[0])
    ent = ent/np.log2(nbins)
    return 1-ent


def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    """Check if an input is within the defined range.

    Parameters
    ----------
    param : int, float
        The input parameter to check.

    low : int, float
        The lower bound of the range.

    high : int, float
        The higher bound of the range.

    param_name : str, optional (default='')
        The name of the parameter.

    include_left : bool, optional (default=False)
        Whether includes the lower bound (lower bound <=).

    include_right : bool, optional (default=False)
        Whether includes the higher bound (<= higher bound).

    Returns
    -------
    within_range : bool or raise errors
        Whether the parameter is within the range of (low, high)

    """

    # param, low and high should all be numerical
    if not isinstance(param, (numbers.Integral, np.integer, float)):
        raise TypeError('{param_name} is set to {param} Not numerical'.format(
            param=param, param_name=param_name))

    if not isinstance(low, (numbers.Integral, np.integer, float)):
        raise TypeError('low is set to {low}. Not numerical'.format(low=low))

    if not isinstance(high, (numbers.Integral, np.integer, float)):
        raise TypeError('high is set to {high}. Not numerical'.format(
            high=high))

    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        raise ValueError('Neither low nor high bounds is undefined')

    # if wrong bound values are used
    if low > high:
        raise ValueError(
            'Lower bound > Higher bound')

    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (include_left and not include_right) and (
            param < low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and include_right) and (
            param <= low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and not include_right) and (
            param <= low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
    else:
        return True


def check_detector(detector):
    """Checks if fit and decision_function methods exist for given detector
    Parameters
    ----------
    detector : pyod.models
        Detector instance for which the check is performed.
    """

    if not hasattr(detector, 'fit') or not hasattr(detector,
                                                   'decision_function'):
        raise AttributeError("%s is not a detector instance." % (detector))


def standardizer(X, X_t=None, keep_scalar=False):
    """Conduct Z-normalization on data to turn input samples become zero-mean
    and unit variance.
    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The training samples
    X_t : numpy array of shape (n_samples_new, n_features), optional (default=None)
        The data to be converted
    keep_scalar : bool, optional (default=False)
        The flag to indicate whether to return the scalar
    Returns
    -------
    X_norm : numpy array of shape (n_samples, n_features)
        X after the Z-score normalization
    X_t_norm : numpy array of shape (n_samples, n_features)
        X_t after the Z-score normalization
    scalar : sklearn scalar object
        The scalar used in conversion
    """
    X = check_array(X)
    scaler = StandardScaler().fit(X)

    if X_t is None:
        if keep_scalar:
            return scaler.transform(X), scaler
        else:
            return scaler.transform(X)
    else:
        X_t = check_array(X_t)
        if X.shape[1] != X_t.shape[1]:
            raise ValueError(
                "The number of input data feature should be consistent"
                "X has {0} features and X_t has {1} features.".format(
                    X.shape[1], X_t.shape[1]))
        if keep_scalar:
            return scaler.transform(X), scaler.transform(X_t), scaler
        else:
            return scaler.transform(X), scaler.transform(X_t)


def score_to_label(pred_scores, outliers_fraction=0.1):
    """Turn raw outlier outlier scores to binary labels (0 or 1).
    Parameters
    ----------
    pred_scores : list or numpy array of shape (n_samples,)
        Raw outlier scores. Outliers are assumed have larger values.
    outliers_fraction : float in (0,1)
        Percentage of outliers.
    Returns
    -------
    outlier_labels : numpy array of shape (n_samples,)
        For each observation, tells whether or not
        it should be considered as an outlier according to the
        fitted model. Return the outlier probability, ranging
        in [0,1].
    """
    # check input values
    pred_scores = column_or_1d(pred_scores)
    check_parameter(outliers_fraction, 0, 1)

    threshold = percentile(pred_scores, 100 * (1 - outliers_fraction))
    pred_labels = (pred_scores > threshold).astype('int')
    return pred_labels


def precision_n_scores(y, y_pred, n=None):
    """Utility function to calculate precision @ rank n.
    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).
    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.
    n : int, optional (default=None)
        The number of outliers. if not defined, infer using ground truth.
    Returns
    -------
    precision_at_rank_n : float
        Precision at rank n score.
    """

    # turn raw prediction decision scores into binary labels
    y_pred = get_label_n(y, y_pred, n)

    # enforce formats of y and labels_
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    return precision_score(y, y_pred)


def get_label_n(y, y_pred, n=None):
    """Function to turn raw outlier scores into binary labels by assign 1
    to top n outlier scores.
    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).
    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.
    n : int, optional (default=None)
        The number of outliers. if not defined, infer using ground truth.
    Returns
    -------
    labels : numpy array of shape (n_samples,)
        binary labels 0: normal points and 1: outliers
    Examples
    --------
    >>> from pyod.utils.utility import get_label_n
    >>> y = [0, 1, 1, 0, 0]
    >>> y_pred = [0.1, 0.5, 0.3, 0.2, 0.7]
    >>> get_label_n(y, y_pred)
    array([0, 1, 0, 0, 1])
    """

    # enforce formats of inputs
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    check_consistent_length(y, y_pred)
    y_len = len(y)  # the length of targets

    # calculate the percentage of outliers
    if n is not None:
        outliers_fraction = n / y_len
    else:
        outliers_fraction = np.count_nonzero(y) / y_len

    threshold = percentile(y_pred, 100 * (1 - outliers_fraction))
    y_pred = (y_pred > threshold).astype('int')

    return y_pred

def get_intersection(lst1, lst2):
    """get the overlapping between two lists
    Parameters
    ----------
    li1 : list or numpy array
        Input list 1.
    li2 : list or numpy array
        Input list 2.
    Returns
    -------
    difference : list
        The overlapping between li1 and li2.
    """
    return list(set(lst1) & set(lst2))


def get_list_diff(li1, li2):
    """get the elements in li1 but not li2. li1-li2
    Parameters
    ----------
    li1 : list or numpy array
        Input list 1.
    li2 : list or numpy array
        Input list 2.
    Returns
    -------
    difference : list
        The difference between li1 and li2.
    """

    return (list(set(li1) - set(li2)))

def get_diff_elements(li1, li2):
    """get the elements in li1 but not li2, and vice versa
    Parameters
    ----------
    li1 : list or numpy array
        Input list 1.
    li2 : list or numpy array
        Input list 2.
    Returns
    -------
    difference : list
        The difference between li1 and li2.
    """
    
    return (list(set(li1) - set(li2)) + list(set(li2) - set(li1)))

def argmaxn(value_list, n, order='desc'):
    """Return the index of top n elements in the list
    if order is set to 'desc', otherwise return the index of n smallest ones.
    Parameters
    ----------
    value_list : list, array, numpy array of shape (n_samples,)
        A list containing all values.
    n : int
        The number of elements to select.
    order : str, optional (default='desc')
        The order to sort {'desc', 'asc'}:
        - 'desc': descending
        - 'asc': ascending
    Returns
    -------
    index_list : numpy array of shape (n,)
        The index of the top n elements.
    """

    value_list = column_or_1d(value_list)
    length = len(value_list)

    # validate the choice of n
    check_parameter(n, 1, length, include_left=True, include_right=True,
                    param_name='n')

    # for the smallest n, flip the value
    if order != 'desc':
        n = length - n

    value_sorted = np.partition(value_list, length - n)
    threshold = value_sorted[int(length - n)]

    if order == 'desc':
        return np.where(np.greater_equal(value_list, threshold))[0]
    else:  # return the index of n smallest elements
        return np.where(np.less(value_list, threshold))[0]


def invert_order(scores, method='multiplication'):
    """ Invert the order of a list of values. The smallest value becomes
    the largest in the inverted list. This is useful while combining
    multiple detectors since their score order could be different.
    Parameters
    ----------
    scores : list, array or numpy array with shape (n_samples,)
        The list of values to be inverted
    method : str, optional (default='multiplication')
        Methods used for order inversion. Valid methods are:
        - 'multiplication': multiply by -1
        - 'subtraction': max(scores) - scores
    Returns
    -------
    inverted_scores : numpy array of shape (n_samples,)
        The inverted list
    Examples
    --------
    >>> scores1 = [0.1, 0.3, 0.5, 0.7, 0.2, 0.1]
    >>> invert_order(scores1)
    array([-0.1, -0.3, -0.5, -0.7, -0.2, -0.1])
    >>> invert_order(scores1, method='subtraction')
    array([0.6, 0.4, 0.2, 0. , 0.5, 0.6])
    """

    scores = column_or_1d(scores)

    if method == 'multiplication':
        return scores.ravel() * -1

    if method == 'subtraction':
        return (scores.max() - scores).ravel()


def _get_sklearn_version():  # pragma: no cover
    """ Utility function to decide the version of sklearn.
    PyOD will result in different behaviors with different sklearn version
    Returns
    -------
    sk_learn version : int
    """

    sklearn_version = str(sklearn.__version__)
    if int(sklearn_version.split(".")[1]) < 19 or int(
            sklearn_version.split(".")[1]) > 23:
        raise ValueError("Sklearn version error")

    return int(sklearn_version.split(".")[1])


def _sklearn_version_21():  # pragma: no cover
    """ Utility function to decide the version of sklearn
    In sklearn 21.0, LOF is changed. Specifically, _decision_function
    is replaced by _score_samples
    Returns
    -------
    sklearn_21_flag : bool
        True if sklearn.__version__ is newer than 0.21.0
    """
    sklearn_version = str(sklearn.__version__)
    if int(sklearn_version.split(".")[1]) > 20:
        return True
    else:
        return False


def generate_bagging_indices(random_state, bootstrap_features, n_features,
                             min_features, max_features):
    """ Randomly draw feature indices. Internal use only.
    Modified from sklearn/ensemble/bagging.py
    Parameters
    ----------
    random_state : RandomState
        A random number generator instance to define the state of the random
        permutations generator.
    bootstrap_features : bool
        Specifies whether to bootstrap indice generation
    n_features : int
        Specifies the population size when generating indices
    min_features : int
        Lower limit for number of features to randomly sample
    max_features : int
        Upper limit for number of features to randomly sample
    Returns
    -------
    feature_indices : numpy array, shape (n_samples,)
        Indices for features to bag
    """

    # Get valid random state
    random_state = check_random_state(random_state)

    # decide number of features to draw
    random_n_features = random_state.randint(min_features, max_features)

    # Draw indices
    feature_indices = generate_indices(random_state, bootstrap_features,
                                       n_features, random_n_features)

    return feature_indices


def generate_indices(random_state, bootstrap, n_population, n_samples):
    """ Draw randomly sampled indices. Internal use only.
    See sklearn/ensemble/bagging.py
    Parameters
    ----------
    random_state : RandomState
        A random number generator instance to define the state of the random
        permutations generator.
    bootstrap :  bool
        Specifies whether to bootstrap indice generation
    n_population : int
        Specifies the population size when generating indices
    n_samples : int
        Specifies number of samples to draw
    Returns
    -------
    indices : numpy array, shape (n_samples,)
        randomly drawn indices
    """

    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(n_population, n_samples,
                                             random_state=random_state)

    return indices


def EuclideanDist(x,y):
    return np.sqrt(np.sum((x - y) ** 2))

def dist2set(x, X):
    l=len(X)
    ldist=[]
    for i in range(l):
        ldist.append(EuclideanDist(x,X[i]))
    return ldist

def c_factor(n) :
    if(n<2):
        n=2
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))


def all_branches(node, current=[], branches = None):
    current = current[:node.e]
    if branches is None: branches = []
    if node.ntype == 'inNode':
        current.append('L')
        all_branches(node.left, current=current, branches=branches)
        current = current[:-1]
        current.append('R')
        all_branches(node.right, current=current, branches=branches)
    else:
        branches.append(current)
    return branches


def branch2num(branch, init_root=0):
    num = [init_root]
    for b in branch:
        if b == 'L':
            num.append(num[-1] * 2 + 1)
        if b == 'R':
            num.append(num[-1] * 2 + 2)
    return num

def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.
    See sklearn/utils/__init__.py for more information.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.
    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.
    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.
    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs.
    See sklearn/ensemble/base.py for more information.
    """
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs, dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _pprint(params, offset=0, printer=repr):
    # noinspection PyPep8
    """Pretty print the dictionary 'params'

    See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    and sklearn/base.py for more information.

    :param params: The dictionary to pretty print
    :type params: dict

    :param offset: The offset in characters to add at the begin of each line.
    :type offset: int

    :param printer: The function to convert entries to strings, typically
        the builtin str or repr
    :type printer: callable

    :return: None
    """

    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines

def get_activation_by_name(name):
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leakyrelu':nn.LeakyReLU()
    }

    if name in activations.keys():
        return activations[name]

    else:
        raise ValueError(name, "is not a valid activation function")

def get_optimal_n_bins(X, upper_bound=None, epsilon=1):
    """ Determine optimal number of bins for a histogram using the Birge 
    Rozenblac method (see :cite:`birge2006many` for details.)
     
    See  https://doi.org/10.1051/ps:2006001 
     
    Parameters 
    ---------- 
    X : array-like of shape (n_samples, n_features) 
        The samples to determine the optimal number of bins for. 
         
    upper_bound :  int, default=None 
        The maximum value of n_bins to be considered. 
        If set to None, np.sqrt(X.shape[0]) will be used as upper bound. 
         
    epsilon : float, default = 1 
        A stabilizing term added to the logarithm to prevent division by zero. 
         
    Returns 
    ------- 
    optimal_n_bins : int 
        The optimal value of n_bins according to the Birge Rozenblac method 
    """
    if upper_bound is None:
        upper_bound = int(np.sqrt(X.shape[0]))

    n = X.shape[0]
    maximum_likelihood = np.zeros((upper_bound - 1, 1))

    for i, b in enumerate(range(1, upper_bound)):
        histogram, _ = np.histogram(X, bins=b)

        maximum_likelihood[i] = np.sum(
            histogram * np.log(b * histogram / n + epsilon) - (
                    b - 1 + np.power(np.log(b), 2.5)))

    return np.argmax(maximum_likelihood) + 1