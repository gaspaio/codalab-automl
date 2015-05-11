""" Independent feature selection transformer. """

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from .sklearn_validation import check_array, check_is_fitted


class NearZeroVar(BaseEstimator, SelectorMixin):

    """Feature selector that removes near zero variance features.

    As defined in the caret R package :
    http://www.inside-r.org/packages/cran/caret/docs/nearZeroVar
    """

    def __init__(self, freqCut=95/5, uniqueCut=0.1):
        """ Constructor. """
        self.freqCut = freqCut
        self.uniqueCut = uniqueCut

    def fit(self, X, y=None):
        """Remove near zero var columns from X.

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
        self.mask_ = np.apply_along_axis(self._notZeroVar, axis=0, arr=X)

        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'mask_')

        return self.mask_

    def _notZeroVar(self, x):
        unique, counts = np.unique(x, return_counts=True)
        if (len(counts) == 1):
            return False
        sc = sorted(counts, reverse=True)
        return len(unique)/float(len(x)) > self.uniqueCut \
            or sc[0]/float(sc[1]) < self.freqCut
