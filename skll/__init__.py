# License: BSD 3 clause
"""
This package provides a number of utilities to make it simpler to run
common scikit-learn experiments with pre-generated features.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

from sklearn.metrics import f1_score, make_scorer, SCORERS
from .logutils import (close_and_remove_logger_handlers,
                       get_skll_logger, orig_showwarning)
from .data import FeatureSet, Reader, Writer
from .experiments import run_configuration
from .learner import Learner
from .metrics import correlation, f1_score_least_frequent, kappa

__all__ = ['FeatureSet', 'Learner', 'Reader', 'get_skll_logger',
           'orig_showwarning', 'close_and_remove_logger_handlers',
           'run_configuration', 'Writer']

# Add our scorers to the sklearn dictionary here so that they will always be
# available if you import anything from skll
_scorers = {'f1_score_micro': make_scorer(f1_score,
                                          average='micro'),
            'f1_score_macro': make_scorer(f1_score,
                                          average='macro'),
            'f1_score_weighted': make_scorer(f1_score,
                                             average='weighted'),
            'f1_score_least_frequent': make_scorer(f1_score_least_frequent),
            'pearson': make_scorer(correlation, corr_type='pearson'),
            'spearman': make_scorer(correlation, corr_type='spearman'),
            'kendall_tau': make_scorer(correlation, corr_type='kendall_tau'),
            'unweighted_kappa': make_scorer(kappa),
            'quadratic_weighted_kappa': make_scorer(kappa,
                                                    weights='quadratic'),
            'linear_weighted_kappa': make_scorer(kappa, weights='linear'),
            'qwk_off_by_one': make_scorer(kappa, weights='quadratic',
                                          allow_off_by_one=True),
            'lwk_off_by_one': make_scorer(kappa, weights='linear',
                                          allow_off_by_one=True),
            'uwk_off_by_one': make_scorer(kappa, allow_off_by_one=True)}

SCORERS.update(_scorers)
