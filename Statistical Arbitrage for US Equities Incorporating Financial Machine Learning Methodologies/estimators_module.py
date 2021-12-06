import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (TimeSeriesSplit, RandomizedSearchCV)
from sklearn.metrics import (roc_curve, auc, accuracy_score,
                             log_loss, roc_auc_score, make_scorer)
import scipy.stats as scs

# import warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.simplefilter("ignore", category=ConvergenceWarning)

class Estimators:

    def __init__(self, param_dist=None, n_splits=4):

        self.param_dist = param_dist
        self.n_splits = n_splits
        self.splits = TimeSeriesSplit(n_splits=self.n_splits)
        self.X_train = None
        self.y_train = None
        self.opt_params = {}
        self.models_fit = {}
        self.models_pred = {}
        self.models_auc = {}

        self.models = {
            # 'bagging': BaggingClassifier,
            'random_forest': RandomForestClassifier,
            'ada_boost': AdaBoostClassifier,
            'xg_boost': XGBClassifier,
            'light_gbm': LGBMClassifier,
            'n_network': MLPClassifier
        }

        self.model_args = {
            # 'bagging':{
            #     'bootstrap': False,
            #     'random_state': 42
            # },
            'random_forest': {
                'random_state': 42,
                'bootstrap': False
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
            },
            'n_network':{
                'activation': 'logistic',
                'solver': 'adam',
                'shuffle': False,
                'max_iter': 1000,
                'random_state': 42
            }
        }

        self.scoring = {
            'neg_log_loss': make_scorer(log_loss,
                                        labels=[0, 1],
                                        greater_is_better=False),
            'accuracy': 'accuracy'
        }

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        if self.param_dist is None:
            self.param_dist = {

                # 'bagging': {
                #     'n_estimators': [250, 500, 750, 1000],
                #     'max_samples': scs.uniform(0.05, 0.99),
                #     'max_features': scs.randint(1, int(self.X_train.shape[1] / 2))
                # },
                'random_forest': {
                    'n_estimators': [250, 500, 750, 1000],
                    #'max_depth': [None, 3, 5, 9, 11, 13, 15],
                    'max_features': scs.uniform(0.10, 1),
                    'min_samples_leaf': [1, 5, 10, 15, 30],
                    'ccp_alpha': scs.uniform(0.001, 0.5)
                },
                'ada_boost': {
                    'n_estimators': [250, 500, 750, 1000],
                    'learning_rate': scs.uniform(0.01, 0.5)
                },
                'xg_boost': {
                    'max_depth': [3, 5, 9, 11, 13, 15],
                    'learning_rate': scs.uniform(0.01, 0.5),
                    'subsample': [0.10, 0.25, 0.50, 0.75, 1.0],
                    'min_child_weight': scs.uniform(0.01, 1),
                    'n_estimators': [250, 500, 750, 1000]
                },
                'light_gbm': {
                    'max_depth': [3, 5, 9, 11, 13, 15],
                    'num_leaves': [2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7],
                    'learning_rate': scs.uniform(0.01, 0.5),
                    'subsample': [0.10, 0.25, 0.50, 0.75, 1.0],
                    'min_child_weight': scs.uniform(0.01, 1.0),
                    'n_estimators': [250, 500, 750, 1000]
                },
                'n_network': {
                    'hidden_layer_sizes': [(25, 25, 25),
                                           (50, 50, 50),
                                           (75, 75, 75),
                                           (100, 100, 100)],
                    'alpha': scs.uniform(0.01, 0.5),
                    'learning_rate': ['constant', 'invscaling', 'adaptive']

                }
            }

        for model in self.models.keys():
            rand_search_cv = RandomizedSearchCV(estimator=self.models[model](**self.model_args[model]),
                                                param_distributions=self.param_dist[model],
                                                scoring=self.scoring,
                                                cv=self.splits,
                                                n_jobs=8,
                                                random_state=42,
                                                refit = 'neg_log_loss')
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
