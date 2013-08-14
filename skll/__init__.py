# Copyright (C) 2012-2013 Educational Testing Service

# This file is part of SciKit-Learn Laboratory.

# SciKit-Learn Laboratory is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

# SciKit-Learn Laboratory is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# SciKit-Learn Laboratory.  If not, see <http://www.gnu.org/licenses/>.

'''
This package provides a number of utilities to make it simpler to run
common scikit-learn experiments with pre-generated features.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
'''

from __future__ import absolute_import, print_function, unicode_literals

from .data import load_examples
from .experiments import run_ablation, run_configuration
from .learner import Learner
from .metrics import (quadratic_weighted_kappa, unweighted_kappa, kendall_tau,
                      spearman, pearson, f1_score_least_frequent,
                      f1_score_macro, f1_score_micro, accuracy)
from .version import __version__, VERSION


__all__ = ['Learner', 'load_examples', 'quadratic_weighted_kappa',
           'unweighted_kappa', 'kendall_tau', 'spearman', 'pearson',
           'f1_score_least_frequent', 'f1_score_macro', 'f1_score_micro',
           'accuracy', 'run_configuration', 'run_ablation']
