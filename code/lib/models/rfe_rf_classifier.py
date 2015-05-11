""" Random Forest with RFE feature selection. """

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from models import Estimator
from tools import SelectIndependent, bac_metric_wrapper


class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


class RfeRfClassifier(Estimator):

    """ RF classifier with RFE feature selection. """

    def __init__(self, dataset, n_jobs=-1, random_seed=1):
        """ Constructor. """
        super(RfeRfClassifier, self).__init__(
            dataset, n_jobs=-1, random_seed=1)

        self.benchmark_predictors = 50

        # folds for CV
        self.folds = 3

        # How many feature selection steps to perform : 20 steps
        self.remove_per = 0.05

        self.feature_filter_ = make_pipeline(
            VarianceThreshold(), SelectIndependent(), StandardScaler())

    def fit_predict(self, time_budget):
        """ XX. """
        self.timers.start('total')

        X_train, X_valid, X_test = self.filterX(
            self.data["X_train"], self.data['X_valid'], self.data['X_test'])

        print "# Features {} / {}".format(self.data["X_train"].shape[1], X_train.shape[1])

        # Benchmark cycle
        self.timers.start('benchmark')
        clf = RandomForestClassifier(n_jobs=-1, n_estimators=self.benchmark_predictors)
        clf.fit(X_train, self.data["Y_train"])
        self.results.append({
            "Y_valid": clf.predict(X_valid),
            "Y_test": clf.predict(X_test)
            })
        self.timers.stop('benchmark')
        print "Cycle #0: {} estimators, {:.2f} sec".format(
            self.benchmark_predictors, self.timers.get('benchmark'))

        # Fit/Predict cycle
        self.timers.start('fit_predict')
        N = self.get_estimator_nb(time_budget - self.timers.get('total'), self.timers.get('benchmark'))
        clf = RFECV(
            RandomForestClassifierWithCoef(n_jobs=-1, n_estimators=N),
            step=self.remove_per, cv=self.folds, scoring=make_scorer(bac_metric_wrapper), verbose=True)
        clf.fit(X_train, self.data['Y_train'])

        cv_score = np.max(clf.grid_scores_)
        self.results.append({
            "Y_valid": clf.predict(X_valid),
            "Y_test": clf.predict(X_test)
            })

        self.timers.stop('fit_predict', 'total')
        print "Cycle #1: {} estimators, {:.2f} sec".format(N, self.timers.get('total'))
        print "Result: K={}, score={:.2f}".format(clf.n_features_, cv_score)

        return


    def get_estimator_nb(self, time_left, time_bench):
        """Very pessimistic estimation of number of allowed trees in time budget."""
        # Compute number of trees we can generate in out time_budget
        frem = int(np.floor(self.remove_per * self.feat_num))  # Features removed in each step
        iters = (self.feat_num + frem - 2)/frem + 1            # RFE iterations per CV fold

        N = self.benchmark_predictors * \
            int(np.floor(time_left / (iters * self.folds * time_bench)))

        print N, iters, time_left
        return N


    def get_estimator_nb_exact(self, time_left, time_bench):
        """Attempt to compute the number of estimators closelly.

        Doesn't work. Probably due to computation overhead. Often goes over budget.
        """
        # Compute number of trees we can generate in out time_budget
        frem = int(np.floor(self.remove_per * self.feat_num))  # Features removed in each step
        iters = (self.feat_num + frem - 2)/frem + 1            # RFE iterations per CV fold

        K = 1/np.sqrt(iters) * sum([np.sqrt(i) for i in range(iters + 1)])
        N = self.benchmark_predictors * \
            int(np.floor(time_left / (K * self.folds * time_bench)))

        print N, iters, time_left
        return N


    def filterX(self, X_train, *args):
        X_train_filtered = self.feature_filter_.fit_transform(X_train)

        if not args:
            return X_train_filtered

        Xs = map(self.feature_filter_.transform, args)
        Xs.insert(0, X_train_filtered)

        return Xs
