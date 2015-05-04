""" Base pipelined model for classification. """

import copy
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier


class BaseClassifier(object):

    """ A simple ensemble classifier, that respects a time budget. """

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
        self.time = []

        if self.task != 'binary.classification':
            raise Exception("Unsupported prediction task")

        self.benchmark_predictors = 50

        return

    def fit_predict(self, time_budget):
        """ XX. """
        time_spent = 0
        ts = time.time()

        # Ceil the time_budget
        time_budget = np.min([time_budget, 200])

        print "Prediction time budget: {} sec".format(time_budget)

        #  CYCLE 0: benchmark
        clf = self.binary_clf(self.benchmark_predictors)

        clf.fit(self.data['X_train'], self.data['Y_train'])  # Train
        self.results.append({
            "Y_valid": clf.predict(self.data['X_valid']),
            "Y_test": clf.predict(self.data['X_test'])
            })

        time_spent = time.time() - ts  # Compute remaining time
        ts = time.time()
        self.time.append(time_spent)
        time_budget -= time_spent
        print "Cycle #0: {} estimators, {:.2f} sec".format(self.benchmark_predictors, time_spent)

        n_estimators = self.benchmark_predictors * int(np.floor(time_budget/time_spent))
        clf = self.binary_clf(n_estimators)
        clf.fit(self.data['X_train'], self.data['Y_train'])  # Train
        self.results.append({
            "Y_valid": clf.predict(self.data['X_valid']),
            "Y_test": clf.predict(self.data['X_test'])
            })
        self.time.append(time.time() - ts)
        print "Cycle #0: {} estimators, {:.2f} sec".format(n_estimators, time.time() - ts)
        return

    def binary_clf(self, n_estimators):
        """ placeholder. """
        if self.sparse:
            pass
            # estimator = BaggingClassifier(
            #             base_estimator=BernoulliNB(),
            #             n_estimators=n_estimators/10,
            #             random_state=self.seed,
            #             n_jobs=self.jobs
            #         )
        else:
            estimator = RandomForestClassifier(
                            n_estimators,
                            random_state=self.seed,
                            n_jobs=self.jobs
                        )

        return estimator
