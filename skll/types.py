# License: BSD 3 clause
"""
Custom type aliases for readability.

:author: Nitin Madnani (nmadnani@ets.org)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from typing_extensions import TypeAlias

# conditional imports to avoid circular dependency problem
if TYPE_CHECKING:
    import skll.data
    import skll.learner

# a class map that maps new labels (string)
# to list of old labels (list of string)
ClassMap: TypeAlias = Dict[str, List[str]]

# confusion matrix
ConfusionMatrix: TypeAlias = List[List[int]]

# list of feature dictionaries
FeatureDict: TypeAlias = Dict[str, Any]
FeatureDictList: TypeAlias = List[FeatureDict]

# an iterator over two featuresets, usually test and train
FeaturesetIterator: TypeAlias = Iterator[Tuple["skll.data.FeatureSet", "skll.data.FeatureSet"]]

# a mapping from example ID to fold ID;
# the example ID may be a float or a str
FoldMapping: TypeAlias = Dict[Union[float, str], str]

# a float or a string; this is useful
# for SKLL IDs that can be both
IdType: TypeAlias = Union[float, str]

# generator over two index numpy arrays - usually train and test
IndexIterator: TypeAlias = Generator[Tuple[np.ndarray, np.ndarray], None, None]

# a float, int, or a string; this is useful
# for SKLL labels that can be both
LabelType: TypeAlias = Union[float, int, str]

# learning curve sizes can either be a numpy array or a list of floats or ints
LearningCurveSizes: TypeAlias = Union[
    List[Union[float, int]],
    np.ndarray,
]

# a generator that yields a three-tuple:
# - an example ID (float or str)
# - a label (int, float, or str)
# - a feature dictionary
FeatGenerator: TypeAlias = Generator[Tuple[IdType, Optional[LabelType], FeatureDict], None, None]

# a string path or Path object
PathOrStr: TypeAlias = Union[Path, str]

# a sparse matrix for features
SparseFeatureMatrix: TypeAlias = csr_matrix

# Learner evaluate task results 5-tuple:
# - confusion matrix (classifier) / None (regressor)
# - accuracy (classifier) / None (regressor)
# - dictionary of results
# - score for the grid objective, None if no grid search
# - dictionary of score for any additional metrics
ComputeEvalMetricsResults: TypeAlias = Tuple[
    Optional[ConfusionMatrix],
    Optional[float],
    Dict[LabelType, Any],
    Optional[float],
    Dict[str, Optional[float]],
]

# Learner evaluate task results 6-tuple:
# - confusion matrix (classifier) / None (regressor)
# - accuracy (classifier) / None (regressor)
# - dictionary of results
# - model parameters dictionary
# - score for the grid objective, None if no grid search
# - dictionary of score for any additional metrics
EvaluateTaskResults: TypeAlias = Tuple[
    Optional[ConfusionMatrix],
    Optional[float],
    Dict[LabelType, Any],
    Dict[str, Any],
    Optional[float],
    Dict[str, Optional[float]],
]

# Learner cross-validate task results 5-tuple:
# - the confusion matrix, overall accuracy, per-label PRFs, model parameters,
#   objective function score, and evaluation metrics (if any) for each fold.
# - the grid search scores for each fold.
# - list of dictionaries of grid search CV results, one per fold, with keys
#   such as "params", "mean_test_score", etc, that are mapped to lists of values
#   associated with each hyperparameter set combination.
# - dictionary containing the test-fold number for each, None if not saved
# - a list of learners, one for each fold, None if not saved
CrossValidateTaskResults: TypeAlias = Tuple[
    List[EvaluateTaskResults],
    List[float],
    List[Dict[str, Any]],
    Optional[FoldMapping],
    Optional[List["skll.learner.Learner"]],
]

# Voting Learner cross-validate task results 3-tuple:
# - the confusion matrix, overall accuracy, per-label PRFs, model parameters,
#   objective function score, and evaluation metrics (if any) for each fold.
# - the grid search scores for each fold.
# - dictionary containing the test-fold number for each, None if not saved
# - a list of learners, one for each fold, None if not saved
VotingCrossValidateTaskResults: TypeAlias = Tuple[
    List[EvaluateTaskResults],
    Optional[FoldMapping],
    Optional[List["skll.learner.voting.VotingLearner"]],
]
