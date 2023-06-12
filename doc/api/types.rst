:mod:`types` Module
---------------------

The ``skll.types`` module contains custom type aliases that are used throughout
the SKLL code in type hints and docstrings.

.. autoclass:: skll.types.ClassMap

Class map that maps new labels (string) to list of old labels (list of string).

.. autoclass:: skll.types.ConfusionMatrix

Confusion matrix represented by a list of list of integers.

.. autoclass:: skll.types.FeatureDict

Feature dictionary that maps a string to other dictionaries or other objects.

.. autoclass:: skll.types.FeatureDictList

List of feature dictionaries.

.. autoclass:: skll.types.FeaturesetIterator

An iterator over two FeatureSets, usually test and train.

.. autoclass:: skll.types.FoldMapping

Mapping from example ID to fold ID; the example ID may be a float or a string
but the fold ID is always a string.

.. autoclass:: skll.types.IdType

A float or a string; this is useful or SKLL IDs that can be both.

.. autoclass:: skll.types.IndexIterator

Generator over two ``numpy`` arrays containing indices - usually for train and test
data.

.. autoclass:: skll.types.LabelType

A float, integer, or a string; this is useful for SKLL labels that can be any
of them.

.. autoclass:: skll.types.LearningCurveSizes

Learning curve sizes can either be a ``numpy`` array (or a list) containing
floats or integers.

.. autoclass:: skll.types.FeatGenerator

Generator that yields a 3-tuple containing:

1. An example ID (float or string).
2. A label (integer, float, or string).
3. A feature dictionary.

.. autoclass:: skll.types.PathOrStr

A string path or Path object.

.. autoclass:: skll.types.SparseFeatureMatrix

A ``scipy`` sparse matrix to hold SKLL features in FeatureSets.

.. autoclass:: skll.types.ComputeEvalMetricsResults

Learner evaluate task results 5-tuple containing:

1. The confusion matrix for a classifier, ``None`` for a regressor.
2. Accuracy for a classifier, ``None`` for a regressor.
3. The dictionary of results.
4. Score for the grid objective, ``None`` if no grid search was performed.
5. The dictionary of scores for any additional metrics.

.. autoclass:: skll.types.EvaluateTaskResults

Learner evaluate task results 6-tuple containing:

1. The confusion matrix for a classifier, ``None`` for a regressor.
2. Accuracy for a classifier, ``None`` for a regressor.
3. The dictionary of results.
4. The dictionary containing the model parameters.
5. Score for the grid objective, None if no grid search
6. The dictionary of score for any additional metrics.

.. autoclass:: skll.types.CrossValidateTaskResults

Learner cross-validate task results 5-tuple containing:

1. The confusion matrix, overall accuracy, per-label precision/recall/F1, model
   parameters, objective function score, and evaluation metrics (if any) for
   each fold.
2. The grid search scores for each fold.
3. The list of dictionaries of grid search CV results, one per fold, with keys
   such as "params", "mean_test_score", etc, that are mapped to lists of values
   associated with each combination of hyper-parameters.
4. The dictionary containing the test-fold number for each, ``None`` if folds
   were not saved.
5. The list of learners, one for each fold, ``None`` if the models were not
   saved.

.. autoclass:: skll.types.VotingCrossValidateTaskResults

Voting Learner cross-validate task results 3-tuple containing:

1. The confusion matrix, overall accuracy, per-label precision/recall/F1, model
   parameters, objective function score, and evaluation metrics (if any) for
   each fold.
2. The dictionary containing the test-fold number for each, ``None`` if folds
   were not saved.
3. The list of voting learners, one for each fold, ``None`` if the models were
   not saved.
