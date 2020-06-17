# License: BSD 3 clause
"""
This module provides the `VotingLearner` meta-learner class which is a wrapper
around scikit-learn's `VotingClassifier` and `VotingRegressor`.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""
import copy
import logging

from collections import defaultdict
from itertools import combinations, zip_longest

import numpy as np

from nose.tools import assert_dict_equal
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.utils.multiclass import type_of_target
from skll import Learner
from skll.data import FeatureSet

from .utils import (add_unseen_labels,
                    compute_evaluation_metrics,
                    get_acceptable_classification_metrics,
                    get_acceptable_regression_metrics,
                    setup_cv_iterator,
                    write_predictions)


class VotingLearner(object):
    """
    A class that wraps the scikit-learn ``VotingClassifier`` and
    ``VotingRegressor`` meta-estimators. Note that this class does not
    inherit from the `Learner` class but rather uses different `Learner`
    instances underlyingly.

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
    pos_label_str : str, optional
        A string denoting the label of the class to be
        treated as the positive class in a binary classification
        setting, for each estimator. If ``None``, the class represented
        by the label that appears second when sorted is chosen as the
        positive class. For example, if the two labels in data are "A"
        and "B" and ``pos_label_str`` is not specified, "B" will
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
                 feature_scaling='none',
                 pos_label_str=None,
                 min_feature_count=1,
                 model_kwargs_list=None,
                 sampler_list=None,
                 sampler_kwargs_list=None,
                 logger=None):
        """
        Initializes a ``VotingLearner`` object with the specified settings.
        """

        self.voting = voting
        self._model = None
        self.logger = logger if logger else logging.getLogger(__name__)

        self.model_kwargs_list = [] if model_kwargs_list is None else model_kwargs_list
        self.sampler_list = [] if sampler_list is None else sampler_list
        self.sampler_kwargs_list = [] if sampler_kwargs_list is None else sampler_kwargs_list

        # check that the arguments that are supposed to be lists are lists
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
                              pos_label_str=pos_label_str,
                              probability=self.voting == "soft",
                              sampler=sampler,
                              sampler_kwargs=sampler_kwargs)
            learner_types.add(learner.model_type._estimator_type)
            self._learners.append(learner)

        # infer what type of metalearner we have - a classifier or
        # a regressor, it can only be one or the other
        try:
            assert (len(learner_types) == 1 and
                    (learner_types == {"classifier"} or
                     learner_types == {"regressor"}))
        except AssertionError:
            raise ValueError("cannot mix classifiers and regressors for voting")
        else:
            self.learner_type = list(learner_types)[0]

        # unset the voting attribute for regressors which makes things
        # much simpler later
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

    def _fit_meta_estimator(self, estimators):
        pass

    def train(self,
              examples,
              param_grid_list=None,
              grid_search_folds=3,
              grid_search=True,
              grid_objective=None,
              grid_jobs=None,
              shuffle=False):
        """
        Train the voting meta-estimator.

        First, we train each of the underlying estimators (represented by
        a skll ``Learner``), possibly with grid search. Then, we instantiate
        a `VotingClassifier` or ``VotingRegressor`` as appropriate with the
        scikit-learn ``Pipeline`` stored in the ``pipeline`` attribute
        of each trained `Learner` instance as the estimator. Finally,
        we call ``fit()`` on the ``VotingClassifier`` or ``VotingRegressor``
        instance. We do this because it allows us to use grid search to
        find good hyperparameter values for our underlying learners before
        passing them to the meta-estimator AND because it allows us to
        use SKLL featuresets and do all of the same pre-processing when
        doing inference.

        The trained meta-estimator is saved in the ``_model`` attribute.
        Nothing is returned.

        Parameters
        ----------
        examples : skll.FeatureSet
            The ``FeatureSet`` instance to use for training.
        param_grid_list : list, optional
            The list of parameters grid to search through for grid
            search, one for each underlying learner. The order of
            the dictionaries should correspond to the order If ``None``, the default
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
        for (learner, param_grid) in zip_longest(self._learners,
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
                              [learner.pipeline for learner in self._learners]))
        if self.learner_type == 'classifier':
            self._model_type = VotingClassifier
            model_kwargs = {"voting": self.voting}
        else:
            self._model_type = VotingRegressor
            model_kwargs = {}
        meta_learner = self._model_type(estimators, **model_kwargs)

        # get the training features in the right dictionary format
        X_train = examples.vectorizer.inverse_transform(examples.features)

        # get the training label sin the right format too
        # NOTE: for classifiers, we can use any of the training learners since
        # the label dict will be identical for all of them; technically, we
        # could also use a `LabelEncoder` here but that may not account for
        # passing `pos_label_str` above when instantiating the learners so we
        # stick with the properly inferred label dict
        if self.learner_type == "classifier":
            y_train = [self._learners[0].label_dict[label] for label in examples.labels]
        else:
            # for regressors, the labels are just the labels
            y_train = examples.labels

        # now we need to fit the actual meta learner which will also fit
        # clones of the underlying pipelines;
        # NOTE: this will *not* yield the same results as the case where we take
        # the predictions from the trained SKLL learners above and did the
        # voting ourselves. This is because SKLL learners do a lot of things
        # differently (e.g., shuffling before grid search) that can cause
        # differences in results and/or floating point precision.
        self._model = meta_learner.fit(X_train, y_train)

    def predict(self,
                examples,
                prediction_prefix=None,
                append=False,
                class_labels=False,
                individual_predictions=False):
        """
        Generate predictions from the meta-estimator and, optionally, the
        underlying estimators on given ``FeatureSet``.

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
            Defaults to ``False``.
        individual_predictions : bool, optional
            Return (and, optionally, write out) the predictions from each
            underlying learner.
            Defaults to ``False``.

        Returns
        -------
        res : (array-like, dict)
            A 2-tuple that contains (1) the predictions returned by the
            meta-estimator and (2) an optional dictionary with the name of each
            underlying learner as the key and the array of its predictions
            as the value. The second element is ``None`` if
            ``individual_predictions`` is set to ``False``.
        """
        example_ids = examples.ids

        # get the test set features in the right format
        X_test = examples.vectorizer.inverse_transform(examples.features)
        self.logger.warning("If there is any between the features used "
                            "to train the underlying learners and the "
                            "features in the test set, the test set features "
                            "will be transformed to the trained model space.")

        # get the predictions from the meta-learner
        yhat = self._model.predict(X_test)

        # decide what predictions to write and what predictions to return
        # by default, these are just what is output by the model
        predictions_to_write = yhat
        predictions_to_return = yhat

        # if it's a classifier
        if self.learner_type == "classifier":

            # get its predicted classes
            classes = np.array([self.learners[0].label_list[int(pred)] for pred in yhat])

            # we always want to write out classes as our predictions
            predictions_to_write = classes

            # if the user specified `class_labels`, then we want
            # to return classes as well
            if class_labels:
                predictions_to_return = classes

        # write out the meta-estimator predictions if we are asked to
        if prediction_prefix is not None:
            write_predictions(example_ids,
                              predictions_to_write,
                              prediction_prefix,
                              self.learner_type,
                              append=append,
                              label_list=self.learners[0].label_list)

        # get and write each underlying estimator's predictions if asked for
        if individual_predictions:

            # create a dictionary to hold the individual predictions
            individual_predictions_dict = {}

            # iterate over each underlying estimator
            for name, estimator in self.model.named_estimators_.items():

                # get the right predictions first
                yhat = estimator.predict_proba(X_test) if self.voting == "soft" else estimator.predict(X_test)
                predictions_to_write = yhat
                predictions_to_return = yhat

                # for classifiers if we did hard voting, we always want to
                # write out the classes but return classes only if asked for
                if self.learner_type == "classifier" and self.voting != "soft":

                    # get its predicted classes
                    classes = np.array([self.learners[0].label_list[int(pred)] for pred in yhat])

                    # we always want to write out classes as our predictions
                    predictions_to_write = classes

                    # if the user specified `class_labels`, then we want
                    # to return classes as well
                    if class_labels:
                        predictions_to_return = classes

                # save this estimator's predictions in the dictionary
                individual_predictions_dict[name] = predictions_to_return

                # also write them out if asked
                if prediction_prefix is not None:
                    write_predictions(example_ids,
                                      predictions_to_write,
                                      prediction_prefix + f"_{name}",
                                      self.learner_type,
                                      append=append,
                                      label_list=self.learners[0].label_list,
                                      probability=self.voting == "soft")
        else:
            individual_predictions_dict = None

        # return the tuple of the meta-estimator predictions array
        # and the dictionary containing the individual predictions
        return (predictions_to_return, individual_predictions_dict)

    def evaluate(self,
                 examples,
                 prediction_prefix=None,
                 append=False,
                 grid_objective=None,
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
                               prediction_prefix=prediction_prefix,
                               append=append)

        # for classifiers, convert class labels indices for consistency
        # but account for any unseen labels in the test set that may not
        # have occurred in the training data for the underlying learners
        # at all; then get acceptable metrics based on the type of labels we have
        if self.learner_type == 'classifier':
            sorted_unique_labels = np.unique(examples.labels)
            test_label_list = sorted_unique_labels.tolist()
            train_and_test_label_dict = add_unseen_labels(self.learners[0].label_dict,
                                                          test_label_list)
            ytest = np.array([train_and_test_label_dict[label]
                              for label in examples.labels])
            acceptable_metrics = get_acceptable_classification_metrics(sorted_unique_labels)
        # for regressors we do not need to do anything special to the labels
        else:
            ytest = examples.labels
            acceptable_metrics = get_acceptable_regression_metrics()

        # check that all of the output metrics are acceptable
        unacceptable_metrics = set(output_metrics).difference(acceptable_metrics)
        if unacceptable_metrics:
            label_type = examples.labels.dtype.type
            raise ValueError("The following metrics are not valid "
                             "for this learner({}) with these labels of "
                             "type {}: {}".format(self._model_type.__name__,
                                                  label_type.__name__,
                                                  list(unacceptable_metrics)))

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
                                                     logger=self.logger)

        # add in the model parameters and return
        model_params = self.model.get_params()
        res = (conf_matrix, accuracy, result_dict, model_params,
               objective_score, metric_scores)
        return res

    def cross_validate(self,
                       examples,
                       stratified=True,
                       cv_folds=10,
                       grid_search=True,
                       grid_search_folds=3,
                       grid_jobs=None,
                       grid_objective=None,
                       output_metrics=[],
                       prediction_prefix=None,
                       param_grid_list=None,
                       shuffle=False,
                       save_cv_folds=False,
                       save_cv_models=False,
                       use_custom_folds_for_grid_search=True):

        # We need to check whether the labels in the featureset are labels
        # or continuous values. If it's the latter, we need to raise an
        # an exception since the stratified splitting in sklearn does not
        # work with continuous labels. Note that although using random folds
        # _will_ work, we want to raise an error in general since it's better
        # to encode the labels as strings anyway for classification problems.
        if (self.learner_type == 'classifier' and
                type_of_target(examples.labels) not in ['binary', 'multiclass']):
            raise ValueError("Floating point labels must be encoded as strings for cross-validation.")

        # Set up the cross-validation iterator.
        kfold, cv_groups = setup_cv_iterator(cv_folds,
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

        # handle each fold separately &accumulate the predictions and results
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
                       grid_objective=grid_objective,
                       grid_search=grid_search,
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
                                         output_metrics=output_metrics))
            append_predictions = True

            # save the fold number for each test ID if we were asked to
            if save_cv_folds:
                for index in test_indices:
                    skll_fold_ids[examples.ids[index]] = str(fold_num)

        # return list of results/outputs for all folds
        return (results, skll_fold_ids, models)

    def learning_curve(self):
        pass