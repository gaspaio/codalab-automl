""" Base class for estimators. """

import copy
from abc import abstractmethod
from tools import TimerRegistry


class Estimator(object):

    """ Base class for all estimators. """

    def __init__(self, dataset, n_jobs=-1, random_seed=1):
        """ Constructor. """
        self.data = copy.deepcopy(dataset.data)
        self.jobs = n_jobs
        self.seed = random_seed
        self.sparse = dataset.info['is_sparse']
        self.has_missing = dataset.info['has_missing']
        self.task = dataset.info['task']
        self.target_num = dataset.info['target_num']
        self.feat_num = dataset.info['feat_num']
        self.results = []
        self.timers = TimerRegistry()

    @abstractmethod
    def fit_predict(self, time_budget):
        """ Main function. """
