import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
import scipy.stats as scs

class Estimators:

    def __init__(self, param_dist, n_splits=5, scoring='roc_auc'):

        self.param_dist = param_dist
        self.n_splits = n_splits
        self.scoring = scoring
        self.splits = TimeSeriesSplit(n_splits=self.n_splits)
        self.X_train = None
        self.y_train = None
        self.opt_params = {}
        self.models_fit = {}
        self.models_pred = {}
        self.models_auc = {}

        self.models = {
            'bagging': BaggingClassifier,
            'random_forest': RandomForestClassifier,
            'ada_boost': AdaBoostClassifier,
            'xg_boost': XGBClassifier,
            'light_gbm': LGBMClassifier
        }

        self.model_args = {
            'bagging':{
                'bootstrap': False,
                'random_state': 42
            },
            'random_forest': {
                'random_state': 42
            },
            'ada_boost': {
                'random_state': 42
            },
            'xg_boost': {
                'eval_metric': 'auc',
                'use_label_encoder': False,
                'seed': 42
            },
            'light_gbm': {
                'random_state': 42
            }
        }

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        for model in self.models.keys():
            rand_search_cv = RandomizedSearchCV(estimator=self.models[model](**self.model_args[model]),
                                                param_distributions=self.param_dist[model],
                                                scoring=self.scoring,
                                                cv=self.splits,
                                                n_jobs=8,
                                                random_state=42)
            rand_search_cv.fit(self.X_train, self.y_train.flatten())
            self.opt_params[model] = rand_search_cv.best_params_
            self.models_fit[model] = rand_search_cv

        return self

    def predict(self, X_test, y_test):

        for model in self.models.keys():
            self.models_pred[model] = self.models_fit[model].predict(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, self.models_pred[model])
            self.models_auc[model] = auc(fpr, tpr)

        return self.models_pred
