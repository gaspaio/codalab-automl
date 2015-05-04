""" Auto ML box. """

import time

# from models import FilterClassifier
from models import RfeRfClassifier

class AutoML:

    """ docstring. """

    def __init__(self, dataset, config):
        """ Constructor. """
        self.D = dataset
        self.config = config

        self._Y = []

        return

    def run_predict(self, time_budget):
        """ Run AutoML. """
        start = time.time()

        predictor = self.init_predictor()
        predictor.fit_predict(time_budget)
        self._Y = predictor.results

        print "Predict total time: %5.2f sec" % (time.time() - start)

    def init_predictor(self):

        predictor = None

        """ Choose right prediction class. """
        if self.D.info['task'] != "regression":
            predictor = RfeRfClassifier(
                self.D, n_jobs=self.config["n_jobs"],
                random_seed=self.config["random_seed"])

        return predictor
