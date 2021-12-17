# License: BSD 3 clause
"""
A meta-learner class that wraps scikit-learn's `VotingClassifier` and `VotingRegressor`.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""
import copy
import logging
from importlib import import_module
from itertools import zip_longest
from multiprocessing import cpu_count

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.utils import shuffle as sk_shuffle
from sklearn.utils.multiclass import type_of_target

from skll.data import FeatureSet
from skll.learner import Learner
from skll.utils.constants import MAX_CONCURRENT_PROCESSES

from .utils import (
    _load_learner_from_disk,
    _save_learner_to_disk,
    add_unseen_labels,
    compute_evaluation_metrics,
    get_acceptable_classification_metrics,
    get_acceptable_regression_metrics,
    get_predictions,
    setup_cv_fold_iterator,
    setup_cv_split_iterator,
    train_and_score,
    write_predictions,
)


class VotingLearner(object):
    """
    Wrap ``VotingClassifier`` and ``VotingRegressor`` from scikit-learn.

    Note that this class does not inherit from the ``Learner`` class but rather
    uses different ``Learner`` instances underlyingly.

    Parameters
    ----------
    learner_names : list of str
        List of the learner names that will participate in the voting process.
    voting : str, optional
        One of "hard" or "soft". If "hard", the predicted class labels
        are used for majority rule voting. If "soft", the predicted class
        label is based on the argmax of the sums of the predicted
        probabilities from each of the underlying learnrs. This parameter
        is only relevant for classification.
        Defaults to "hard".
    custom_learner_path : str, optional
        Path to a Python file containing the definitions of any custom
        learners. Any and all custom learners in ``estimator_names`` must
        be defined in this file. If the custom learner does not inherit
        from an already existing scikit-learn estimator, it must explicitly
        define an `_estimator_type` attribute indicating whether it's a
        "classifier" or a "regressor".
    feature_scaling : str, optional
        How to scale the features, if at all for each estimator. Options are
        -  "with_std": scale features using the standard deviation
        -  "with_mean": center features using the mean
        -  "both": do both scaling as well as centering
        -  "none": do neither scaling nor centering
        Defaults to 'none'.
    pos_label : str, optional
        A string denoting the label of the class to be
        treated as the positive class in a binary classification
        setting, for each estimator. If ``None``, the class represented
        by the label that appears second when sorted is chosen as the
        positive class. For example, if the two labels in data are "A"
        and "B" and ``pos_label`` is not specified, "B" will
        be chosen as the positive class.
        Defaults to ``None``.
    min_feature_count : int, optional
        The minimum number of examples a feature must have a nonzero
        value in to be included, for each estimator.
        Defaults to 1.
    model_kwargs_list : list of dicts, optional
        A list of dictionaries of keyword arguments to pass to the
        initializer for each of the estimators. There's a one-to-one
        correspondence between the order of this list and the order
        of the ``learner_names`` list.
        Defaults to ``None``.
    sampler_list : list of str, optional
        The samplers to use for kernel approximation, if desired, for each
        estimator. Valid values are:
        -  "AdditiveChi2Sampler"
        -  "Nystroem"
        -  "RBFSampler"
        -  "SkewedChi2Sampler"
        There's a one-to-one correspondence between the order of this list
        and the order of the ``learner_names`` list.
        Defaults to ``None``.
    sampler_kwargs_list : dict, optional
        A list of dictionaries of keyword arguments to pass to the
        initializer for the specified sampler, one per estimator.
        There's a one-to-one correspondence between the order of this
        list and the order of the ``learner_names`` list.
        Defaults to ``None``.
    logger : logging object, optional
        A logging object. If ``None`` is passed, get logger from ``__name__``.
        Defaults to ``None``.
    """
    def __init__(self,
                 learner_names,
                 voting="hard",
                 custom_learner_path=None,
                 feature_scaling="none",
                 pos_label=None,
                 min_feature_count=1,
                 model_kwargs_list=None,
                 sampler_list=None,
                 sampler_kwargs_list=None,
                 logger=None):
        """
        Initializes a ``VotingLearner`` object with the specified settings.
        """

        # initialize various attributes
        self._model = None
        self.voting = voting
        self.label_dict = None
        self.logger = logger if logger else logging.getLogger(__name__)
        self.model_kwargs_list = [] if model_kwargs_list is None else model_kwargs_list
        self.sampler_list = [] if sampler_list is None else sampler_list
        self.sampler_kwargs_list = [] if sampler_kwargs_list is None else sampler_kwargs_list

        # check that the arguments that are supposed to be lists are lists;
        # if they are `None`, set them to be empty lists
        for argument_name in ["model_kwargs_list",
                              "sampler_list",
                              "sampler_kwargs_list"]:
            argument_value = locals()[argument_name]
            if argument_value is None:
                setattr(self, argument_name, [])
            else:
                if not isinstance(argument_value, list):
                    raise ValueError(f"{argument_name} should be a list, you "
                                     f"specified {argument_value}")
                else:
                    setattr(self, argument_name, argument_value)

        # check that the list arguments, if not empty, have the right length
        for attribute_name in ["model_kwargs_list",
                               "sampler_list",
                               "sampler_kwargs_list"]:
            attribute_value = getattr(self, attribute_name)
            try:
                assert (len(attribute_value) == 0 or
                        len(attribute_value) == len(learner_names))
            except AssertionError:
                raise ValueError(f"'{attribute_name}' must have {len(learner_names)} "
                                 "entries, same as the number of learners") from None

        # instantiate each of the given estimators
        self._learners = []
        learner_types = set()
        self._learner_names = learner_names
        for (learner_name,
             model_kwargs,
             sampler,
             sampler_kwargs) in zip_longest(self._learner_names,
                                            self.model_kwargs_list,
                                            self.sampler_list,
                                            self.sampler_kwargs_list):
            learner = Learner(learner_name,
                              custom_learner_path=custom_learner_path,
                              feature_scaling=feature_scaling,
                              min_feature_count=min_feature_count,
                              model_kwargs=model_kwargs,
                              pipeline=True,
                              pos_label=pos_label,
                              probability=self.voting == "soft",
                              sampler=sampler,
                              sampler_kwargs=sampler_kwargs,
                              logger=logger)
            learner_types.add(learner.model_type._estimator_type)
            self._learners.append(learner)

        # infer what type of metalearner we have - a classifier or
        # a regressor; it can only be one or the other
        try:
            assert (len(learner_types) == 1 and
                    (learner_types == {"classifier"} or
                     learner_types == {"regressor"}))
        except AssertionError:
            raise ValueError("cannot mix classifiers and regressors for voting")
        else:
            self.learner_type = list(learner_types)[0]

        # unset the voting attribute for regressors for downstream simplicity
        if self.learner_type == "regressor":
            self.voting = None

    @property
    def learners(self):
        """
        Return the underlying list of learners
        """
        return self._learners

    @property
    def model(self):
        """
        The underlying scikit-learn meta-estimator model
        """
        return self._model

    @property
    def model_type(self):
        """
        The meta-estimator model type (i.e., the class)
        """
        return self._model_type

    def _setup_underlying_learners(self, examples):
        """
        Initial, pre-training set up for learners
        """
        for learner in self.learners:
            learner._create_label_dict(examples)
            learner._train_setup(examples)

    def __getstate__(self):
        """
        Return the attributes that should be dumped. We need this
        because we do not want to dump loggers.
        """
        attribute_dict = dict(self.__dict__)
        if 'logger' in attribute_dict:
            del attribute_dict['logger']
        return attribute_dict

    def save(self, learner_path):
        """
        Save the ``VotingLearner`` instance to a file.

        Parameters
        ----------
        learner_path : str
            The path to save the ``VotingLearner`` instance to.
        """
        _save_learner_to_disk(self, learner_path)

    @classmethod
    def from_file(cls, learner_path, logger=None):
        """
        Load a saved ``VotingLearner`` instance from a file.

        Parameters
        ----------
        learner_path : str
            The path to a saved ``VotingLearner`` instance file.
        logger : logging object, optional
            A logging object. If ``None`` is passed, get logger from ``__name__``.
            Defaults to ``None``.

        Returns
        -------
        learner : skll.learner.voting.VotingLearner
            The ``VotingLearner`` instance loaded from the file.
        """
        # use the logger that's passed in or if nothing was passed in,
        # then create a new logger
        logger = logger if logger else logging.getLogger(__name__)

        # call the learner loding utility function
        return _load_learner_from_disk(cls, learner_path, logger)

    def train(self,
              examples,
              param_grid_list=None,
              grid_search_folds=5,
              grid_search=True,
              grid_objective=None,
              grid_jobs=None,
              shuffle=False):
        """
        Train the voting meta-estimator.

        First, we train each of the underlying estimators (represented by
        a skll ``Learner``), possibly with grid search. Then, we instantiate
        a ``VotingClassifier`` or ``VotingRegressor`` as appropriate with the
        scikit-learn ``Pipeline`` stored in the ``pipeline`` attribute
        of each trained ``Learner`` instance as the estimator. Finally,
        we call ``fit()`` on the ``VotingClassifier`` or ``VotingRegressor``
        instance. We follow this process because it allows us to use grid
        search to find good hyperparameter values for our underlying learners
        before passing them to the meta-estimator AND because it allows us to
        use SKLL featuresets and do all of the same pre-processing when
        doing inference.

        The trained meta-estimator is saved in the ``_model`` attribute.
        Nothing is returned.

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to use for training.
        param_grid_list : list, optional
            The list of parameter grids to search through for grid
            search, one for each underlying learner. The order of
            the dictionaries should correspond to the order in
            which the underlying estimators were specified when the
            ``VotingLearner`` was instantiated. If ``None``, the default
            parameter grids will be used for the underlying estimators.
            Defaults to ``None``.
        grid_search_folds : int or dict, optional
            The number of folds to use when doing the grid search
            for each of the underlying learners, or a mapping from
            example IDs to folds.
            Defaults to 3.
        grid_search : bool, optional
            Should we use grid search when training each underlying learner?
            Defaults to ``True``.
        grid_objective : str, optional
            The name of the objective function to use when
            doing the grid search for each underlying learner.
            Must be specified if ``grid_search`` is ``True``.
            Defaults to ``None``.
        grid_jobs : int, optional
            The number of jobs to run in parallel when doing the
            grid search for each underlying learner. If ``None`` or 0,
            the number of grid search folds will be used.
            Defaults to ``None``.
        shuffle : bool, optional
            Shuffle examples (e.g., for grid search CV.)
            Defaults to ``False``.
        """
        if param_grid_list is None:
            self._param_grids = []
        else:
            if not isinstance(param_grid_list, list):
                raise ValueError(f"`param_grid_list` should be a list of dictionaries, you specified: {param_grid_list}")
            else:
                self._param_grids = param_grid_list

        # train each of the underlying estimators with grid search, if required;
        # basically, we are just running grid search to find good hyperparameter
        # values that we can then pass to scikit-learn below
        for (learner, param_grid) in zip_longest(self.learners,
                                                 self._param_grids):
            _ = learner.train(examples,
                              grid_search=grid_search,
                              grid_objective=grid_objective,
                              param_grid=param_grid,
                              grid_search_folds=grid_search_folds,
                              grid_jobs=grid_jobs,
                              shuffle=shuffle)

        # once we have our instantiated learners, we use their `pipeline`
        # attribute as the input estimators to the specific voting learner type
        estimators = list(zip(self._learner_names,
                              [learner.pipeline for learner in self.learners]))
        if self.learner_type == 'classifier':
            self._model_type = VotingClassifier
            model_kwargs = {"voting": self.voting}
        else:
            self._model_type = VotingRegressor
            model_kwargs = {}
        meta_learner = self.model_type(estimators, **model_kwargs)

        # get the training features in the right dictionary format
        X_train = examples.vectorizer.inverse_transform(examples.features)

        # since label dictionaries are identical for all underlying
        # learners, save it into a easier to access attribute
        self.label_dict = self.learners[0].label_dict

        # get the training labels in the right format too
        # NOTE: technically, we could also use a `LabelEncoder` here but
        # that may not account for passing `pos_label` above when
        # instantiating the learners so we stick with the label dict
        if self.learner_type == "classifier":
            y_train = [self.label_dict[label] for label in examples.labels]
        else:
            # for regressors, the labels are just the labels
            y_train = examples.labels

        # now we need to fit the actual meta learner which will also fit
        # clones of the underlying pipelines;
        # NOTE: this will *not* yield the same results as the case where we take
        # the predictions from the trained SKLL learners above and do the
        # voting ourselves. This is because SKLL learners do a lot of things
        # differently (e.g., shuffling before grid search) that can cause
        # differences in results and/or floating point precision.
        self._model = meta_learner.fit(X_train, y_train)

    def predict(self,
                examples,
                prediction_prefix=None,
                append=False,
                class_labels=True,
                individual_predictions=False):
        """
        Generate predictions from the meta-estimator and, optionally, the
        underlying estimators on given ``FeatureSet``. The predictions are
        also written to disk if ``prediction_prefix`` is not ``None``.

        For regressors, the returned and written-out predictions are identical.
        However, for classifiers:

        - if ``class_labels`` is ``True``, class labels are returned as well as
          written out.

        - if ``class_labels`` is ``False`` and the classifier is probabilistic
          (i.e., ``self.probability`` is ``True``), class probabilities are
          returned as well as written out.

        - if ``class_labels`` is ``False`` and the classifier is non-probabilistic
          (i.e., ``self..probability`` is ``False``), class indices are returned
          and class labels are written out. This option is generally only
          meant for SKLL-internal use.

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to predict labels for.
        prediction_prefix : str, optional
            If saving the predictions, this is the prefix that will be used for
            the filename. It will be followed by ``"_predictions.tsv"``
            Defaults to ``None``.
        append : bool, optional
            Should we append the current predictions to the file if it exists?
            Defaults to ``False``.
        class_labels : bool, optional
            For classifier, should we convert class indices to their (str) labels
            for the returned array? Note that class labels are always written out
            to disk.
            Defaults to ``True``.
        individual_predictions : bool, optional
            Return (and, optionally, write out) the predictions from each
            underlying learner.
            Defaults to ``False``.

        Returns
        -------
        res : a 2-tuple
            The first element is the array of predictions returned by the
            meta-estimator and the second is an optional dictionary with the
            name of each underlying learner as the key and the array of its
            predictions as the value. The second element is ``None`` if
            ``individual_predictions`` is set to ``False``.
        """
        example_ids = examples.ids

        # get the test set features in the right format
        xtest = examples.vectorizer.inverse_transform(examples.features)

        # get all possible kinds of predictions from the meta-learner
        prediction_dict = get_predictions(self, xtest)

        # decide what predictions to write and what predictions to return
        # by default, these are just what is output by the model
        to_write = prediction_dict["raw"]
        to_return = prediction_dict["raw"]

        # for classifiers ...
        if self.learner_type == "classifier":

            # return and write class labels if they were explicitly asked for
            if class_labels:
                to_return = to_write = prediction_dict["labels"]
            else:
                # return and write probabilities
                if self.voting == "soft":
                    to_return = to_write = prediction_dict["probabilities"]
                # return class indices and write labels
                else:
                    to_return = prediction_dict["raw"]
                    to_write = prediction_dict["labels"]

        # for regressors, it's really simple
        else:
            to_write = to_return = prediction_dict["raw"]

        # write out the meta-estimator predictions if we are asked to
        if prediction_prefix is not None:
            write_predictions(example_ids,
                              to_write,
                              prediction_prefix,
                              self.learner_type,
                              append=append,
                              label_list=self.learners[0].label_list)

        # get and write each underlying learner's predictions if asked for
        if individual_predictions:

            # create a dictionary to hold the individual predictions
            individual_predictions_dict = {}

            # iterate over each underlying learner along with names
            for (name, learner) in zip(self.model.named_estimators_, self.learners):

                # the learner's `predict()` method should handle everything
                learner_prediction_prefix = f"{prediction_prefix}_{name}" if prediction_prefix is not None else None
                learner_predictions = learner.predict(examples,
                                                      prediction_prefix=learner_prediction_prefix,
                                                      append=append,
                                                      class_labels=class_labels)

                # save this estimator's predictions in the dictionary
                individual_predictions_dict[name] = learner_predictions
        else:
            individual_predictions_dict = None

        # return the tuple of the meta-estimator predictions array
        # and the dictionary containing the individual predictions
        return (to_return, individual_predictions_dict)

    def evaluate(self,
                 examples,
                 prediction_prefix=None,
                 append=False,
                 grid_objective=None,
                 individual_predictions=False,
                 output_metrics=[]):
        """
        Evaluates the meta-estimator on a given ``FeatureSet``.

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to evaluate the performance of the model on.
        prediction_prefix : str, optional
            If saving the predictions, this is the
            prefix that will be used for the filename.
            It will be followed by ``"_predictions.tsv"``
            Defaults to ``None``.
        append : bool, optional
            Should we append the current predictions to the file if
            it exists?
            Defaults to ``False``.
        grid_objective : function, optional
            The objective function that was used when doing
            the grid search.
            Defaults to ``None``.
        individual_predictions : bool, optional
            Optionally, write out the predictions from each underlying learner.
            Defaults to ``False``.
        output_metrics : list of str, optional
            List of additional metric names to compute in
            addition to grid objective. Empty by default.
            Defaults to an empty list.

        Returns
        -------
        res : 6-tuple
            The confusion matrix, the overall accuracy, the per-label
            PRFs, the model parameters, the grid search objective
            function score, and the additional evaluation metrics, if any.
        """

        # make the prediction on the test data; note that these
        # are either class indices or class probabilities
        yhat, _ = self.predict(examples,
                               class_labels=False,
                               prediction_prefix=prediction_prefix,
                               append=append,
                               individual_predictions=individual_predictions)

        # for classifiers, convert class labels indices for consistency
        # but account for any unseen labels in the test set that may not
        # have occurred in the training data for the underlying learners
        # at all; then get acceptable metrics based on the type of labels we have
        if self.learner_type == 'classifier':
            sorted_unique_labels = np.unique(examples.labels)
            test_label_list = sorted_unique_labels.tolist()
            train_and_test_label_dict = add_unseen_labels(self.label_dict,
                                                          test_label_list)
            ytest = np.array([train_and_test_label_dict[label]
                              for label in examples.labels])
            acceptable_metrics = get_acceptable_classification_metrics(sorted_unique_labels)
        # for regressors we do not need to do anything special to the labels
        else:
            train_and_test_label_dict = None
            ytest = examples.labels
            acceptable_metrics = get_acceptable_regression_metrics()

        # check that all of the output metrics are acceptable
        unacceptable_metrics = set(output_metrics).difference(acceptable_metrics)
        if unacceptable_metrics:
            label_type = examples.labels.dtype.type
            raise ValueError(f"The following metrics are not valid "
                             f"for this learner({self.model_type.__name__}) "
                             f"with these labels of type {label_type.__name__}: "
                             f"{list(unacceptable_metrics)}")

        # get the values of the evaluation metrics
        (conf_matrix,
         accuracy,
         result_dict,
         objective_score,
         metric_scores) = compute_evaluation_metrics(output_metrics,
                                                     ytest,
                                                     yhat,
                                                     self.learner_type,
                                                     label_dict=train_and_test_label_dict,
                                                     grid_objective=grid_objective,
                                                     probability=self.voting == "soft",
                                                     logger=self.logger)

        # add in the model parameters, excluding the ones
        # for the underlying estimators, and return
        model_params = self.model.get_params(deep=False)
        res = (conf_matrix, accuracy, result_dict, model_params,
               objective_score, metric_scores)
        return res

    def cross_validate(self,
                       examples,
                       stratified=True,
                       cv_folds=10,
                       cv_seed=123456789,
                       grid_search=True,
                       grid_search_folds=5,
                       grid_jobs=None,
                       grid_objective=None,
                       output_metrics=[],
                       prediction_prefix=None,
                       param_grid_list=None,
                       shuffle=False,
                       save_cv_folds=True,
                       save_cv_models=False,
                       individual_predictions=False,
                       use_custom_folds_for_grid_search=True):
        """
        Cross-validate the meta-estimator on the given examples.

        We follow essentially the same methodology as in
        ``Learner.cross_validate()`` - split the examples into
        training and testing folds, and then call ``self.train()``
        on the training folds and then ``self.evaluate()`` on the
        test fold. Note that this means that underlying estimators
        with different hyperparameters may be used for each fold, as is
        the case with ``Learner.cross_validate()``.

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to cross-validate learner performance on.
        stratified : bool, optional
            Should we stratify the folds to ensure an even
            distribution of labels for each fold?
            Defaults to ``True``.
        cv_folds : int or dict, optional
            The number of folds to use for cross-validation, or
            a mapping from example IDs to folds.
            Defaults to 10.
        cv_seed: int, optional
            The value for seeding the random number generator
            used to create the random folds. Note that this
            seed is *only* used if either ``grid_search`` or
            ``shuffle`` are set to ``True``.
            Defaults to 123456789.
        grid_search : bool, optional
            Should we do grid search when training each fold?
            Note: This will make this take *much* longer.
            Defaults to ``False``.
        grid_search_folds : int or dict, optional
            The number of folds to use when doing the
            grid search, or a mapping from
            example IDs to folds.
            Defaults to 3.
        grid_jobs : int, optional
            The number of jobs to run in parallel when doing the
            grid search. If ``None`` or 0, the number of
            grid search folds will be used.
            Defaults to ``None``.
        grid_objective : str, optional
            The name of the objective function to use when
            doing the grid search. Must be specified if
            ``grid_search`` is ``True``.
            Defaults to ``None``.
        output_metrics : list of str, optional
            List of additional metric names to compute in
            addition to the metric used for grid search. Empty
            by default.
            Defaults to an empty list.
        prediction_prefix : str, optional
            If saving the predictions, this is the
            prefix that will be used for the filename.
            It will be followed by ``"_predictions.tsv"``
            Defaults to ``None``.
        param_grid_list : list, optional
            The list of parameters grid to search through for grid
            search, one for each underlying learner. The order of
            the dictionaries should correspond to the order If ``None``,
            the default parameter grids will be used for the underlying
            estimators.
            Defaults to ``None``.
        shuffle : bool, optional
            Shuffle examples before splitting into folds for CV.
            Defaults to ``False``.
        save_cv_folds : bool, optional
             Whether to save the cv fold ids or not?
             Defaults to ``True``.
        save_cv_models : bool, optional
            Whether to save the cv models or not?
            Defaults to ``False``.
        individual_predictions : bool, optional
            Write out the cross-validated predictions from each
            underlying learner as well.
            Defaults to ``False``.
        use_custom_folds_for_grid_search : bool, optional
            If ``cv_folds`` is a custom dictionary, but
            ``grid_search_folds`` is not, perhaps due to user
            oversight, should the same custom dictionary
            automatically be used for the inner grid-search
            cross-validation?
            Defaults to ``True``.

        Returns
        -------
        results : list of 6-tuples
            The confusion matrix, overall accuracy, per-label PRFs, model
            parameters, objective function score, and evaluation metrics (if any)
            for each fold.
        skll_fold_ids : dict
            A dictionary containing the test-fold number for each id
            if ``save_cv_folds`` is ``True``, otherwise ``None``.
        models : list of skll.learner.VotingLearner
            A list of skll.learner.VotingLearner instances, one for each fold
            if ``save_cv_models`` is ``True``, otherwise ``None``.

        Raises
        ------
        ValueError
            If classification labels are not properly encoded as strings.
        ValueError
            If ``grid_search`` is ``True`` but ``grid_objective`` is ``None``.
        """

        # Seed the random number generator so that randomized algorithms are
        # replicable.
        random_state = np.random.RandomState(cv_seed)

        # We need to check whether the labels in the featureset are labels
        # or continuous values. If it's the latter, we need to raise an
        # an exception since the stratified splitting in sklearn does not
        # work with continuous labels. Note that although using random folds
        # _will_ work, we want to raise an error in general since it's better
        # to encode the labels as strings anyway for classification problems.
        if (self.learner_type == 'classifier' and
                type_of_target(examples.labels) not in ['binary', 'multiclass']):
            raise ValueError("Floating point labels must be encoded as strings for cross-validation.")

        # check that we have an objective since grid search is on by default
        # Note that `train()` would raise this error anyway later but it's
        # better to raise this early on so rather than after a whole bunch of
        # stuff has happened
        if grid_search and not grid_objective:
            raise ValueError("Grid search is on by default. You must "
                             "either specify a grid objective or turn off "
                             "grid search.")

        # Shuffle so that the folds are random for the inner grid search CV.
        # If grid search is True but shuffle isn't, shuffle anyway.
        # You can't shuffle a scipy sparse matrix in place, so unfortunately
        # we make a copy of everything (and then get rid of the old version)
        if grid_search or shuffle:
            if grid_search and not shuffle:
                self.logger.warning('Training data will be shuffled to randomize '
                                    'grid search folds.  Shuffling may yield '
                                    'different results compared to scikit-learn.')
            ids, labels, features = sk_shuffle(examples.ids, examples.labels,
                                               examples.features,
                                               random_state=random_state)
            examples = FeatureSet(examples.name, ids, labels=labels,
                                  features=features,
                                  vectorizer=examples.vectorizer)

        # Call some setup code which will properly initialize the underlying
        # learners before they are eventually trained
        self._setup_underlying_learners(examples)

        # Set up the cross-validation iterator.
        kfold, cv_groups = setup_cv_fold_iterator(cv_folds,
                                                  examples,
                                                  self.learner_type,
                                                  stratified=stratified,
                                                  logger=self.logger)

        # When using custom CV folds (a dictionary), if we are planning to do
        # grid search, set the grid search folds to be the same as the custom
        # cv folds unless a flag is set that explicitly tells us not to.
        # Note that this should only happen when we are using the API; otherwise
        # the configparser should take care of this even before this method is called
        if isinstance(cv_folds, dict):
            if grid_search and use_custom_folds_for_grid_search and grid_search_folds != cv_folds:
                self.logger.warning("The specified custom folds will be used for "
                                    "the inner grid search.")
                grid_search_folds = cv_folds

        # handle each fold separately & accumulate the predictions and results
        results = []
        append_predictions = False
        models = [] if save_cv_models else None
        skll_fold_ids = {} if save_cv_folds else None
        for fold_num, (train_indices,
                       test_indices) in enumerate(kfold.split(examples.features,
                                                              examples.labels,
                                                              cv_groups)):
            # Train model
            self._model = None  # prevent feature vectorizer from being reset.
            train_set = FeatureSet(examples.name,
                                   examples.ids[train_indices],
                                   labels=examples.labels[train_indices],
                                   features=examples.features[train_indices],
                                   vectorizer=examples.vectorizer)

            self.train(train_set,
                       param_grid_list=param_grid_list,
                       grid_search_folds=grid_search_folds,
                       grid_search=grid_search,
                       grid_objective=grid_objective,
                       grid_jobs=grid_jobs,
                       shuffle=grid_search)

            if save_cv_models:
                models.append(copy.deepcopy(self))

            # evaluate the voting meta-estimator on the test fold
            test_tuple = FeatureSet(examples.name,
                                    examples.ids[test_indices],
                                    labels=examples.labels[test_indices],
                                    features=examples.features[test_indices],
                                    vectorizer=examples.vectorizer)

            # save the results
            results.append(self.evaluate(test_tuple,
                                         prediction_prefix=prediction_prefix,
                                         append=append_predictions,
                                         grid_objective=grid_objective,
                                         output_metrics=output_metrics,
                                         individual_predictions=individual_predictions))
            append_predictions = True

            # save the fold number for each test ID if we were asked to
            if save_cv_folds:
                for index in test_indices:
                    skll_fold_ids[examples.ids[index]] = str(fold_num)

        # return list of results/outputs for all folds
        return (results, skll_fold_ids, models)

    def learning_curve(self,
                       examples,
                       metric,
                       cv_folds=10,
                       train_sizes=np.linspace(0.1, 1.0, 5),
                       override_minimum=False):
        """
        Generates learning curves for the voting meta-estimator on the training
        examples via cross-validation. Adapted from the scikit-learn code for
        learning curve generation (cf.``sklearn.model_selection.learning_curve``).

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to generate the learning curve on.
        cv_folds : int or dict, optional
            The number of folds to use for cross-validation, or
            a mapping from example IDs to folds.
            Defaults to 10.
        metric : str
            The name of the metric function to use
            when computing the train and test scores
            for the learning curve.
        train_sizes : list of float or int, optional
            Relative or absolute numbers of training examples
            that will be used to generate the learning curve.
            If the type is float, it is regarded as a fraction
            of the maximum size of the training set (that is
            determined by the selected validation method),
            i.e. it has to be within (0, 1]. Otherwise it
            is interpreted as absolute sizes of the training
            sets. Note that for classification the number of
            samples usually have to be big enough to contain
            at least one sample from each class.
            Defaults to  ``np.linspace(0.1, 1.0, 5)``.
        override_minimum : bool, optional
            Learning curves can be unreliable for very small sizes
            esp. for > 2 labels. If this option is set to ``True``, the
            learning curve would be generated even if the number
            of example is less 500 along with a warning. If ``False``,
            the curve is not generated and an exception is raised instead.
            Defaults to ``False``.

        Returns
        -------
        train_scores : list of float
            The scores for the training set.
        test_scores : list of float
            The scores on the test set.
        num_examples : list of int
            The numbers of training examples used to generate
            the curve

        Raises
        ------
        ValueError
            If the number of examples is less than 500.
        """
        # check that the number of training examples is more than the minimum
        # needed for generating a reliable learning curve
        if len(examples) < 500:
            if not override_minimum:
                raise ValueError(
                    f"Number of training examples provided ({len(examples)}) "
                    "is less than the minimum needed (500) for the "
                    "learning curve to be reliable."
                )
            else:
                self.logger.warning(
                    "Learning curves can be unreliable for examples fewer than "
                    f"500. You provided {len(examples)}."
                )

        # raise a warning if we are using a probabilistic classifier
        # since that means we cannot use the predictions directly
        if self.voting == "soft":
            self.logger.warning("For soft-voting classifiers, the most likely "
                                "class will be computed via an argmax before "
                                "computing the curve.")

        # Call some setup code which will properly initialize the underlying
        # learners before they are eventually trained
        self._setup_underlying_learners(examples)

        # set up the CV split iterator over the train/test featuresets
        # which also returns the maximum number of training examples
        (featureset_iter,
         n_max_training_samples) = setup_cv_split_iterator(cv_folds, examples)

        # Get the `_translate_train_sizes()` function from scikit-learn
        # since we need it to get the right list of sizes after cross-validation
        _module = import_module('sklearn.model_selection._validation')
        _translate_train_sizes = getattr(_module, '_translate_train_sizes')
        train_sizes_abs = _translate_train_sizes(train_sizes,
                                                 n_max_training_samples)
        n_unique_ticks = train_sizes_abs.shape[0]

        # limit the number of parallel jobs for this to be no higher than
        # MAX_CONCURRENT_PROCESSES or the number of cores, whichever is lower
        n_jobs = min(cpu_count(), MAX_CONCURRENT_PROCESSES)

        # Run jobs in parallel that train the model on each subset
        # of the training data and compute train and test scores
        parallel = joblib.Parallel(n_jobs=n_jobs, pre_dispatch=n_jobs)
        out = parallel(joblib.delayed(train_and_score)(self,
                                                       train_fs[:n_train_samples],
                                                       test_fs,
                                                       metric)
                       for train_fs, test_fs in featureset_iter
                       for n_train_samples in train_sizes_abs)

        # Reshape the outputs
        out = np.array(out)
        n_cv_folds = out.shape[0] // n_unique_ticks
        out = out.reshape(n_cv_folds, n_unique_ticks, 2)
        out = np.asarray(out).transpose((2, 1, 0))

        return list(out[0]), list(out[1]), list(train_sizes_abs)
