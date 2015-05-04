""" XXX. """

import numpy as np
from libscores import bac_metric


def bac_metric_wrapper(y, y_pred):
    """ transform y from a vector to a 2D array. """
    if y.ndim == 1:
        y = np.reshape(y, (len(y), 1))
    if y_pred.ndim == 1:
        y_pred = np.reshape(y_pred, (len(y_pred), 1))

    return bac_metric(y, y_pred)
