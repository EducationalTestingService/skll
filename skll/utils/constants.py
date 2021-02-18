"""
Constants useful for SKLL learners.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
:organization: ETS
"""

import os

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LinearRegression,
    LogisticRegression,
    RANSACRegressor,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

KNOWN_DEFAULT_PARAM_GRIDS = {AdaBoostClassifier:
                             {'learning_rate': [0.01, 0.1, 1.0, 10.0, 100.0]},
                             AdaBoostRegressor:
                             {'learning_rate': [0.01, 0.1, 1.0, 10.0, 100.0]},
                             BayesianRidge:
                             {'alpha_1': [1e-6, 1e-4, 1e-2, 1, 10],
                              'alpha_2': [1e-6, 1e-4, 1e-2, 1, 10],
                              'lambda_1': [1e-6, 1e-4, 1e-2, 1, 10],
                              'lambda_2': [1e-6, 1e-4, 1e-2, 1, 10]},
                             DecisionTreeClassifier:
                             {'max_features': ["auto", None]},
                             DecisionTreeRegressor:
                             {'max_features': ["auto", None]},
                             DummyClassifier:
                             {},
                             DummyRegressor:
                             {},
                             ElasticNet:
                             {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
                             GradientBoostingClassifier:
                             {'max_depth': [1, 3, 5]},
                             GradientBoostingRegressor:
                             {'max_depth': [1, 3, 5]},
                             HuberRegressor:
                             {'epsilon': [1.05, 1.35, 1.5, 2.0, 2.5, 5.0],
                              'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]},
                             KNeighborsClassifier:
                             {'n_neighbors': [1, 5, 10, 100],
                              'weights': ['uniform', 'distance']},
                             KNeighborsRegressor:
                             {'n_neighbors': [1, 5, 10, 100],
                              'weights': ['uniform', 'distance']},
                             MLPClassifier:
                             {'activation': ['logistic', 'tanh', 'relu'],
                              'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1],
                              'learning_rate_init': [0.001, 0.01, 0.1]},
                             MLPRegressor:
                             {'activation': ['logistic', 'tanh', 'relu'],
                              'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1],
                              'learning_rate_init': [0.001, 0.01, 0.1]},
                             MultinomialNB:
                             {'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]},
                             Lars:
                             {},
                             Lasso:
                             {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
                             LinearRegression:
                             {},
                             LinearSVC:
                             {'C': [0.01, 0.1, 1.0, 10.0, 100.0]},
                             LogisticRegression:
                             {'C': [0.01, 0.1, 1.0, 10.0, 100.0]},
                             SVC:
                             {'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                              'gamma': ['auto', 'scale', 0.01, 0.1, 1.0, 10.0, 100.0]},
                             RandomForestClassifier:
                             {'max_depth': [1, 5, 10, None]},
                             RandomForestRegressor:
                             {'max_depth': [1, 5, 10, None]},
                             RANSACRegressor:
                             {},
                             Ridge:
                             {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
                             RidgeClassifier:
                             {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
                             SGDClassifier:
                             {'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01],
                              'penalty': ['l1', 'l2', 'elasticnet']},
                             SGDRegressor:
                             {'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01],
                              'penalty': ['l1', 'l2', 'elasticnet']},
                             LinearSVR:
                             {'C': [0.01, 0.1, 1.0, 10.0, 100.0]},
                             SVR:
                             {'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                              'gamma': ['auto', 'scale', 0.01, 0.1, 1.0, 10.0, 100.0]},
                             TheilSenRegressor:
                             {}
                             }

KNOWN_REQUIRES_DENSE = (BayesianRidge, Lars, TheilSenRegressor)

MAX_CONCURRENT_PROCESSES = int(os.getenv('SKLL_MAX_CONCURRENT_PROCESSES', '5'))

VALID_FEATURE_SCALING_OPTIONS = frozenset(['both',
                                           'none',
                                           'with_std',
                                           'with_mean'])

VALID_SAMPLERS = frozenset(['Nystroem',
                            'RBFSampler',
                            'SkewedChi2Sampler',
                            'AdditiveChi2Sampler',
                            ''])

VALID_TASKS = frozenset(['cross_validate',
                         'evaluate',
                         'learning_curve',
                         'predict',
                         'train'])

#: Set of evaluation metrics only used for classification tasks
CLASSIFICATION_ONLY_METRICS = {
    'accuracy',
    'average_precision',
    'balanced_accuracy',
    'f1',
    'f1_score_least_frequent',
    'f1_score_macro',
    'f1_score_micro',
    'f1_score_weighted',
    'f05',
    'f05_score_macro',
    'f05_score_micro',
    'f05_score_weighted',
    'jaccard',
    'jaccard_macro',
    'jaccard_micro',
    'jaccard_weighted',
    'neg_log_loss',
    'precision',
    'precision_macro',
    'precision_micro',
    'precision_weighted',
    'recall',
    'recall_macro',
    'recall_micro',
    'recall_weighted',
    'roc_auc'
}


#: Set of evaluation metrics based on correlation
CORRELATION_METRICS = {'kendall_tau', 'pearson', 'spearman'}

#: Set of evaluation metrics that can use prediction probabilities
PROBABILISTIC_METRICS = frozenset(['average_precision',
                                   'neg_log_loss',
                                   'roc_auc'])

#: Set of evaluation metrics only used for regression tasks
REGRESSION_ONLY_METRICS = {
    'explained_variance',
    'max_error',
    'neg_mean_squared_error',
    'neg_mean_absolute_error',
    'r2'
}

#: Set of unweighted kappa agreement metrics
UNWEIGHTED_KAPPA_METRICS = {'unweighted_kappa', 'uwk_off_by_one'}

#: Set of weighed kappa agreement metrics
WEIGHTED_KAPPA_METRICS = {
    'linear_weighted_kappa',
    'lwk_off_by_one',
    'quadratic_weighted_kappa',
    'qwk_off_by_one'
}
