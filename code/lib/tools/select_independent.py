""" Independent feature selection transformer. """

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from .sklearn_validation import check_array, check_is_fitted


class SelectIndependent(BaseEstimator, SelectorMixin):

    """Feature selector that removes linearly dependent features.

    cf. https://github.com/scikit-learn/scikit-learn/blob/bb39b493ef084a4f362d77163c2ca506790c38b6/sklearn/feature_selection/variance_threshold.py
    """

    def __init__(self, tol=1e-05):
        """ Constructor. """
        self.tol = tol

    def fit(self, X, y=None):
        """Learn dependent columns from X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Sample input vectors from which to linear dependence.
        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.
        Returns
        -------
        self
        """
        X = check_array(X, dtype=np.float64)
        Q, R = np.linalg.qr(X)

        self.diag_ = R.diagonal()

        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'diag_')

        return np.abs(self.diag_) > self.tol
