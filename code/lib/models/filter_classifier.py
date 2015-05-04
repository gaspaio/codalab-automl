""" Base pipelined model for classification. """

import numpy as np
import time
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import f_classif

from sklearn.pipeline import Pipeline, make_pipeline

from base_classifier import BaseClassifier
from tools import SelectIndependent, bac_metric_wrapper


class FilterClassifier(BaseClassifier):

    """ A simple ensemble classifier, that respects a time budget. """

    def __init__(self, dataset, n_jobs=-1, random_seed=1):
        """ Constructor. """
        super(FilterClassifier, self).__init__(
            dataset, n_jobs=-1, random_seed=1)

        # folds in GridSearchCV
        self.cv = 3

        # Filter param step (nb of features)
        self.K = 20

    def fit_predict(self, time_budget):
        """ XX. """
        time_spent = 0
        ts = time.time()

        f = make_pipeline(VarianceThreshold(), SelectIndependent(), StandardScaler())
        X_train = f.fit_transform(self.data["X_train"])
        X_valid = f.transform(self.data['X_valid'])
        X_test = f.transform(self.data['X_test'])

        time_spent = time.time() - ts
        ts = time.time()
        time_budget -= time_spent

        # Benchmark cycle
        clf = Pipeline([
            ('filterF', GenericUnivariateSelect(f_classif, 'k_best', 'all')),
            ('clf', RandomForestClassifier(n_jobs=-1, n_estimators=self.benchmark_predictors))
        ])
        clf.fit(X_train, self.data["Y_train"])

        time_spent = time.time() - ts  # Compute remaining time
        ts = time.time()
        time_budget -= time_spent
        print "Cycle #0: {} estimators, {:.2f} sec".format(
            self.benchmark_predictors, time_spent)

        self.results.append({
            "Y_valid": clf.predict(f.transform(self.data['X_valid'])),
            "Y_test": clf.predict(f.transform(self.data['X_test']))
            })

        N = self.benchmark_predictors * \
            int(np.floor(time_budget / (self.K * self.cv * time_spent)))

        # Search the best filter param
        feat_num_clean = X_train.shape[1]
        k_range = np.unique(range(feat_num_clean, 1, -feat_num_clean/self.K))
        param_grid = [{'filterF__param': k_range}]

        clf = Pipeline([
            ('filterF', GenericUnivariateSelect(f_classif, 'k_best')),
            ('clf', RandomForestClassifier(n_jobs=-1, n_estimators=N))
        ])

        gs = GridSearchCV(
            clf, param_grid, cv=self.cv,
            scoring=make_scorer(bac_metric_wrapper))
        gs.fit(X_train, self.data["Y_train"])
        clf = gs.best_estimator_
        self.results.append({
            "Y_valid": clf.predict(X_valid),
            "Y_test": clf.predict(X_test)
            })
        print "Cycle #1: {} estimators, {:.2f} sec".format(N, time.time() - ts)
        print "Result: K={}, score={:.2f}".format(
            gs.best_params_['filterF__param'], gs.best_score_)

        return
