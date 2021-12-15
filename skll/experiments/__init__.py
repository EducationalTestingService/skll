# License: BSD 3 clause
"""
Functions for running and interacting with SKLL experiments.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Chee Wee Leong (cleong@ets.org)
"""

import datetime
import json
import logging
from itertools import combinations
from os.path import exists, getsize, join

import matplotlib.pyplot as plt
import numpy as np
from sklearn import __version__ as SCIKIT_VERSION
from sklearn.metrics import SCORERS

from skll.config import parse_config_file
from skll.config.utils import _munge_featureset_name
from skll.learner import MAX_CONCURRENT_PROCESSES, Learner, load_custom_learner
from skll.learner.voting import VotingLearner
from skll.metrics import _CUSTOM_METRICS, register_custom_metric
from skll.utils.logging import close_and_remove_logger_handlers, get_skll_logger
from skll.version import __version__

from .input import load_featureset
from .output import (
    _print_fancy_output,
    _write_learning_curve_file,
    _write_skll_folds,
    _write_summary_file,
    generate_learning_curve_plots,
)
from .utils import NumpyTypeEncoder, _check_job_results, _create_learner_result_dicts

# Check if gridmap is available
try:
    from gridmap import Job, process_jobs
except ImportError:
    _HAVE_GRIDMAP = False
else:
    _HAVE_GRIDMAP = True

# Turn off interactive plotting for matplotlib
plt.ioff()


__all__ = ['generate_learning_curve_plots',
           'load_featureset',
           'run_configuration']


def _classify_featureset(args):  # noqa: C901
    """
    Classification job to be submitted to grid.

    Parameters
    ----------
    args : dict
        A dictionary with arguments for classifying the
        ``FeatureSet`` instance.

    Returns
    -------
    res : list of dicts
        The results of the classification, in the format
        of a list of dictionaries.

    Raises
    ------
    ValueError
        If extra unknown arguments are passed to the function.
    """

    # Extract all the arguments.
    # (There doesn't seem to be a better way to do this since one can't specify
    # required keyword arguments.)

    experiment_name = args.pop("experiment_name")
    task = args.pop("task")
    sampler = args.pop("sampler")
    feature_hasher = args.pop("feature_hasher")
    hasher_features = args.pop("hasher_features")
    job_name = args.pop("job_name")
    featureset = args.pop("featureset")
    featureset_name = args.pop("featureset_name")
    learner_name = args.pop("learner_name")
    train_path = args.pop("train_path")
    test_path = args.pop("test_path")
    train_set_name = args.pop("train_set_name")
    test_set_name = args.pop("test_set_name")
    shuffle = args.pop('shuffle')
    model_path = args.pop("model_path")
    prediction_prefix = args.pop("prediction_prefix")
    grid_search = args.pop("grid_search")
    grid_objective = args.pop("grid_objective")
    output_metrics = args.pop("output_metrics")
    suffix = args.pop("suffix")
    job_log_file = args.pop("log_file")
    job_log_level = args.pop("log_level")
    probability = args.pop("probability")
    pipeline = args.pop("pipeline")
    results_path = args.pop("results_path")
    fixed_parameters = args.pop("fixed_parameters")
    sampler_parameters = args.pop("sampler_parameters")
    param_grid = args.pop("param_grid")
    pos_label = args.pop("pos_label")
    overwrite = args.pop("overwrite")
    feature_scaling = args.pop("feature_scaling")
    min_feature_count = args.pop("min_feature_count")
    folds_file = args.pop("folds_file")
    grid_search_jobs = args.pop("grid_search_jobs")
    grid_search_folds = args.pop("grid_search_folds")
    cv_folds = args.pop("cv_folds")
    cv_seed = args.pop("cv_seed")
    save_cv_folds = args.pop("save_cv_folds")
    save_cv_models = args.pop("save_cv_models")
    use_folds_file_for_grid_search = args.pop("use_folds_file_for_grid_search")
    stratified_folds = args.pop("do_stratified_folds")
    label_col = args.pop("label_col")
    id_col = args.pop("id_col")
    ids_to_floats = args.pop("ids_to_floats")
    class_map = args.pop("class_map")
    custom_learner_path = args.pop("custom_learner_path")
    custom_metric_path = args.pop("custom_metric_path")
    quiet = args.pop('quiet', False)
    learning_curve_cv_folds = args.pop("learning_curve_cv_folds")
    learning_curve_train_sizes = args.pop("learning_curve_train_sizes")
    save_votes = args.pop("save_votes")

    if args:
        raise ValueError("Extra arguments passed to _classify_featureset: "
                         f"{args.keys()}")
    start_timestamp = datetime.datetime.now()

    # create a new SKLL logger for this specific job and
    # use the given log level
    logger = get_skll_logger(job_name,
                             job_log_file,
                             log_level=job_log_level)

    try:

        # log messages
        logger.info(f"Task: {task}")

        # check if we have any possible custom metrics
        possible_custom_metric_names = []
        for metric_name in output_metrics + [grid_objective]:
            # metrics that are not in `SCORERS` or `None` are candidates
            # (the `None` is a by-product of how jobs with single tuning
            # objectives are created)
            if metric_name not in SCORERS and metric_name is not None:
                possible_custom_metric_names.append(metric_name)
            # if the metric is already in `SCORERS`, is it a custom one
            # that we already registered? if so, log that
            elif metric_name in _CUSTOM_METRICS:
                logger.info(f"custom metric '{metric_name}' is already registered")

        # initialize list that will hold any invalid metrics
        # that we could not register as custom metrics
        invalid_metric_names = []

        # if we have possible custom metrics
        if possible_custom_metric_names:

            # check that we have a file to load them from
            if not custom_metric_path:
                raise ValueError(f"invalid metrics specified: {possible_custom_metric_names}")
            else:
                # try to register each possible custom metric
                # raise an exception if we fail, if we don't then
                # add the custom metric function to `globals()` so
                # that it serializes properly for gridmap
                for custom_metric_name in possible_custom_metric_names:
                    try:
                        custom_metric_func = register_custom_metric(custom_metric_path, custom_metric_name)
                    except (AttributeError, NameError, ValueError):
                        invalid_metric_names.append(custom_metric_name)
                    else:
                        logger.info(f"registered '{custom_metric_name}' as a "
                                    f"custom metric")
                        globals()[custom_metric_name] = custom_metric_func

        # raise an error if we have any invalid metrics
        if invalid_metric_names:
            raise ValueError(f"invalid metrics specified: {invalid_metric_names}. "
                             f"If these are custom metrics, check the function "
                             f"names.")

        if task == 'cross_validate':
            if isinstance(cv_folds, int):
                num_folds = cv_folds
            else:  # folds_file was used, so count the unique fold ids.
                num_folds = len(set(cv_folds.values()))
            logger.info(f"Cross-validating ({num_folds} folds, seed={cv_seed}) on "
                        f"{train_set_name}, feature set {featureset} ...")
        elif task == 'evaluate':
            logger.info(f"Training on {train_set_name}, Test on "
                        f"{test_set_name}, feature set {featureset} ...")
        elif task == 'train':
            logger.info(f"Training on {train_set_name}, feature set "
                        f"{featureset} ...")
        elif task == 'learning_curve':
            logger.info("Generating learning curve "
                        f"({learning_curve_cv_folds} 80/20 folds, "
                        f"sizes={learning_curve_train_sizes}, "
                        f"objective={grid_objective}) on {train_set_name}, "
                        f"feature set {featureset} ...")
        else:  # predict
            logger.info(f"Training on {train_set_name}, Making predictions on"
                        f" {test_set_name}, feature set {featureset} ...")

        # check whether a trained model on the same data with the same
        # featureset already exists if so, load it and then use it on test data
        modelfile = join(model_path, f'{job_name}.model')
        if (task in ['cross_validate', 'learning_curve'] or
                not exists(modelfile) or
                overwrite):
            train_examples = load_featureset(train_path,
                                             featureset,
                                             suffix,
                                             label_col=label_col,
                                             id_col=id_col,
                                             ids_to_floats=ids_to_floats,
                                             quiet=quiet,
                                             class_map=class_map,
                                             feature_hasher=feature_hasher,
                                             num_features=hasher_features,
                                             logger=logger)

            train_set_size = len(train_examples.ids)
            if not train_examples.has_labels:
                raise ValueError('Training examples do not have labels')

            # set up some keyword arguments for instantiating the learner
            # object; note that these are shared by all types of learners
            # supported by SKLL (regular and voting)
            common_learner_kwargs = {"custom_learner_path": custom_learner_path,
                                     "feature_scaling": feature_scaling,
                                     "pos_label": pos_label,
                                     "min_feature_count": min_feature_count,
                                     "logger": logger}

            # instantiate the right type of learner object
            if learner_name in ["VotingClassifier", "VotingRegressor"]:
                # the fixed parameters dictionary must at least
                # contains the estimator names for the voting;
                # the rest can be set to default values
                try:
                    learner_names = fixed_parameters["estimator_names"]
                except KeyError:
                    raise ValueError("'estimator names' must be specified as "
                                     "fixed parameters for voting classifiers "
                                     "and/or regressors.") from None
                else:
                    voting_type = fixed_parameters.get("voting_type", "hard")
                    model_kwargs_list = fixed_parameters.get("estimator_fixed_parameters")
                    sampler_list = fixed_parameters.get("estimator_samplers")
                    sampler_kwargs_list = fixed_parameters.get("estimator_sampler_parameters")
                    param_grids_list = fixed_parameters.get("estimator_param_grids")
                    learner = VotingLearner(learner_names,
                                            voting=voting_type,
                                            model_kwargs_list=model_kwargs_list,
                                            sampler_list=sampler_list,
                                            sampler_kwargs_list=sampler_kwargs_list,
                                            **common_learner_kwargs)
            else:
                learner = Learner(learner_name,
                                  model_kwargs=fixed_parameters,
                                  pipeline=pipeline,
                                  sampler=sampler,
                                  sampler_kwargs=sampler_parameters,
                                  probability=probability,
                                  **common_learner_kwargs)

        # load the model if it already exists
        else:
            # import custom learner into global namespace if we are reusing
            # a saved model
            if custom_learner_path:
                globals()[learner_name] = load_custom_learner(custom_learner_path, learner_name)
            train_set_size = 'unknown'

            # load the non-custom learner from disk
            if exists(modelfile) and not overwrite:
                logger.info(f"Loading pre-existing {learner_name} model: "
                            f"{modelfile}")
            if learner_name in ["VotingClassifier", "VotingRegressor"]:
                learner = VotingLearner.from_file(modelfile, logger=logger)
            else:
                learner = Learner.from_file(modelfile, logger=logger)

        # Load test set if there is one
        if task == 'evaluate' or task == 'predict':
            test_examples = load_featureset(test_path,
                                            featureset,
                                            suffix,
                                            label_col=label_col,
                                            id_col=id_col,
                                            ids_to_floats=ids_to_floats,
                                            quiet=quiet,
                                            class_map=class_map,
                                            feature_hasher=feature_hasher,
                                            num_features=hasher_features)
            test_set_size = len(test_examples.ids)
        else:
            test_set_size = 'n/a'

        # compute information about xval and grid folds that can be put in results
        # in readable form
        if isinstance(cv_folds, dict):
            cv_folds_to_print = f'{len(set(cv_folds.values()))} via folds file'
        else:
            cv_folds_to_print = str(cv_folds)

        if isinstance(grid_search_folds, dict):
            grid_search_folds_to_print = \
                f'{len(set(grid_search_folds.values()))} via folds file'
        else:
            grid_search_folds_to_print = str(grid_search_folds)

        # create a list of dictionaries of the results information
        learner_result_dict_base = {'experiment_name': experiment_name,
                                    'train_set_name': train_set_name,
                                    'train_set_size': train_set_size,
                                    'test_set_name': test_set_name,
                                    'test_set_size': test_set_size,
                                    'featureset': json.dumps(featureset),
                                    'featureset_name': featureset_name,
                                    'shuffle': shuffle,
                                    'learner_name': learner_name,
                                    'task': task,
                                    'start_timestamp':
                                        start_timestamp.strftime('%d %b %Y %H:%M:'
                                                                 '%S.%f'),
                                    'version': __version__,
                                    'feature_scaling': feature_scaling,
                                    'folds_file': folds_file,
                                    'grid_search': grid_search,
                                    'grid_objective': grid_objective,
                                    'grid_search_folds': grid_search_folds_to_print,
                                    'min_feature_count': min_feature_count,
                                    'cv_folds': cv_folds_to_print,
                                    'using_folds_file':
                                        isinstance(cv_folds, dict) or isinstance(grid_search_folds, dict),
                                    'save_cv_folds': save_cv_folds,
                                    'save_cv_models': save_cv_models,
                                    'use_folds_file_for_grid_search': use_folds_file_for_grid_search,
                                    'stratified_folds': stratified_folds,
                                    'scikit_learn_version': SCIKIT_VERSION}

        # check if we're doing cross-validation, because we only load/save
        # models when we're not.
        task_results = None
        if task == 'cross_validate':
            logger.info('Cross-validating')

            # set up the keyword arguments for learner cross-validation;
            # note that most are shared by all types of learners
            # supported by SKLL (regular and voting)
            xval_kwargs = {"shuffle": shuffle,
                           "stratified": stratified_folds,
                           "prediction_prefix": prediction_prefix,
                           "grid_search": grid_search,
                           "grid_search_folds": grid_search_folds,
                           "cv_folds": cv_folds,
                           "cv_seed": cv_seed,
                           "grid_objective": grid_objective,
                           "output_metrics": output_metrics,
                           "grid_jobs": grid_search_jobs,
                           "save_cv_folds": save_cv_folds,
                           "save_cv_models": save_cv_models,
                           "use_custom_folds_for_grid_search": use_folds_file_for_grid_search}

            # voting learners require an optional list of parameter grids
            # passed as fixed parameters whereas regular learners only need
            # a single parameter grid
            if isinstance(learner, VotingLearner):
                xval_kwargs["param_grid_list"] = param_grids_list
                xval_kwargs["individual_predictions"] = save_votes
            else:
                xval_kwargs["param_grid"] = param_grid

            # cross-validate the learner
            results = learner.cross_validate(train_examples, **xval_kwargs)

            # voting learners return only a subset of the results
            # for cross-validation (no grid search results)
            if isinstance(learner, Learner):
                (task_results,
                 grid_scores,
                 grid_search_cv_results_dicts,
                 skll_fold_ids,
                 models) = results
            else:
                (task_results,
                 skll_fold_ids,
                 models) = results
                grid_scores = [None] * cv_folds
                grid_search_cv_results_dicts = [None] * cv_folds

            if models:
                for index, m in enumerate(models, start=1):
                    modelfile = join(model_path, f'{job_name}_fold{index}.model')
                    m.save(modelfile)

        elif task == 'learning_curve':
            logger.info("Generating learning curve(s)")
            (curve_train_scores,
             curve_test_scores,
             computed_curve_train_sizes) = learner.learning_curve(train_examples,
                                                                  grid_objective,
                                                                  cv_folds=learning_curve_cv_folds,
                                                                  train_sizes=learning_curve_train_sizes)
        else:
            # if we do not have a saved model, we need to train one
            grid_scores = [None]
            grid_search_cv_results_dicts = [None]
            if not exists(modelfile) or overwrite:
                logger.info(f"Featurizing and training new {learner_name} "
                            "model")

                # set up the keyword arguments for learner training;
                # note that most are shared by all types of learners
                # supported by SKLL (regular and voting)
                train_kwargs = {"grid_search": grid_search,
                                "grid_search_folds": grid_search_folds,
                                "grid_objective": grid_objective,
                                "grid_jobs": grid_search_jobs,
                                "shuffle": shuffle}

                # voting learners require an optional list of parameter grids
                # passed as fixed parameters whereas regular learners only need
                # a single parameter grid
                if isinstance(learner, VotingLearner):
                    train_kwargs["param_grid_list"] = param_grids_list
                else:
                    train_kwargs["param_grid"] = param_grid

                # train the model
                results = learner.train(train_examples, **train_kwargs)

                # regular learners return a grid score and results
                if isinstance(learner, Learner):
                    grid_scores = [results[0]]
                    grid_search_cv_results_dicts = [results[1]]
                    if grid_search:
                        logger.info(f"Best {grid_objective} grid search score: "
                                    f"{round(results[0], 3)}")

                # save model, if asked
                if model_path:
                    learner.save(modelfile)

            # print out the model parameters; note that for
            # voting learners, we exclude the parameters for
            # the underlying estimators
            params = learner.model.get_params(deep=isinstance(learner, Learner))
            param_out = (f'{param_name}: {param_value}' for
                         param_name, param_value in params.items())
            logger.info(f"Hyperparameters: {', '.join(param_out)}")

            # evaluate the model on the test set;
            if task == 'evaluate':
                logger.info("Evaluating predictions")
                # voting learners have an extra keyword argument indicating
                # whether we want to save the individual predictions (or votes)
                extra_kwargs = {}
                if isinstance(learner, VotingLearner):
                    extra_kwargs["individual_predictions"] = save_votes
                task_results = [learner.evaluate(test_examples,
                                                 prediction_prefix=prediction_prefix,
                                                 grid_objective=grid_objective,
                                                 output_metrics=output_metrics,
                                                 **extra_kwargs)]
            elif task == 'predict':
                logger.info("Writing predictions")
                # we set `class_labels` to `False` so that if the learner is
                # probabilistic, probabilities are written instead of labels;
                # voting learners have an extra keyword argument indicating
                # whether we want to save the individual predictions (or votes)
                extra_kwargs = {}
                if isinstance(learner, VotingLearner):
                    extra_kwargs["individual_predictions"] = save_votes
                learner.predict(test_examples,
                                prediction_prefix=prediction_prefix,
                                class_labels=False,
                                **extra_kwargs)

        end_timestamp = datetime.datetime.now()
        learner_result_dict_base['end_timestamp'] = end_timestamp.strftime(
            '%d %b %Y %H:%M:%S.%f')
        total_time = end_timestamp - start_timestamp
        learner_result_dict_base['total_time'] = str(total_time)

        if task == 'cross_validate' or task == 'evaluate':
            results_json_path = join(results_path,
                                     f'{job_name}.results.json')

            res = _create_learner_result_dicts(task_results,
                                               grid_scores,
                                               grid_search_cv_results_dicts,
                                               learner_result_dict_base)

            # write out the result dictionary to a json file
            with open(results_json_path, 'w') as json_file:
                json.dump(res, json_file, cls=NumpyTypeEncoder)

            with open(join(results_path, f'{job_name}.results'),
                      'w') as output_file:
                _print_fancy_output(res, output_file)

        elif task == 'learning_curve':
            results_json_path = join(results_path, f'{job_name}.results.json')

            res = {}
            res.update(learner_result_dict_base)
            res.update({'learning_curve_cv_folds': learning_curve_cv_folds,
                        'given_curve_train_sizes': learning_curve_train_sizes,
                        'learning_curve_train_scores_means': np.mean(curve_train_scores, axis=1),
                        'learning_curve_test_scores_means': np.mean(curve_test_scores, axis=1),
                        'learning_curve_train_scores_stds': np.std(curve_train_scores, axis=1, ddof=1),
                        'learning_curve_test_scores_stds': np.std(curve_test_scores, axis=1, ddof=1),
                        'computed_curve_train_sizes': computed_curve_train_sizes})

            # we need to return and write out a list of dictionaries
            res = [res]

            # write out the result dictionary to a json file
            with open(results_json_path, 'w') as json_file:
                json.dump(res, json_file, cls=NumpyTypeEncoder)

        # For all other tasks, i.e. train or predict
        else:
            if results_path:
                results_json_path = join(results_path,
                                         f'{job_name}.results.json')

                assert len(grid_scores) == 1
                assert len(grid_search_cv_results_dicts) == 1
                grid_search_cv_results_dict = {"grid_score": grid_scores[0]}
                grid_search_cv_results_dict["grid_search_cv_results"] = \
                    grid_search_cv_results_dicts[0]
                grid_search_cv_results_dict.update(learner_result_dict_base)
                # write out the result dictionary to a json file
                with open(results_json_path, 'w') as json_file:
                    json.dump(grid_search_cv_results_dict, json_file, cls=NumpyTypeEncoder)
            res = [learner_result_dict_base]

        # write out the cv folds if required
        if task == 'cross_validate' and save_cv_folds:
            skll_fold_ids_file = f'{experiment_name}_skll_fold_ids.csv'
            with open(join(results_path, skll_fold_ids_file),
                      'w') as output_file:
                _write_skll_folds(skll_fold_ids, output_file)

    finally:
        close_and_remove_logger_handlers(logger)

    return res


def run_configuration(config_file, local=False, overwrite=True, queue='all.q',  # noqa: C901
                      hosts=None, write_summary=True, quiet=False,
                      ablation=0, resume=False, log_level=logging.INFO):
    """
    Takes a configuration file and runs the specified jobs on the grid.

    Parameters
    ----------
    config_file : str
        Path to the configuration file we would like to use.
    local : bool, optional
        Should this be run locally instead of on the cluster?
        Defaults to ``False``.
    overwrite : bool, optional
        If the model files already exist, should we overwrite
        them instead of re-using them?
        Defaults to ``True``.
    queue : str, optional
        The DRMAA queue to use if we're running on the cluster.
        Defaults to ``'all.q'``.
    hosts : list of str, optional
        If running on the cluster, these are the machines we should use.
        Defaults to ``None``.
    write_summary : bool, optional
        Write a TSV file with a summary of the results.
        Defaults to ``True``.
    quiet : bool, optional
        Suppress printing of "Loading..." messages.
        Defaults to ``False``.
    ablation : int, optional
        Number of features to remove when doing an ablation
        experiment. If positive, we will perform repeated ablation
        runs for all combinations of features removing the
        specified number at a time. If ``None``, we will use all
        combinations of all lengths. If 0, the default, no
        ablation is performed. If negative, a ``ValueError`` is
        raised.
        Defaults to 0.
    resume : bool, optional
        If result files already exist for an experiment, do not
        overwrite them. This is very useful when doing a large
        ablation experiment and part of it crashes.
        Defaults to ``False``.
    log_level : str, optional
        The level for logging messages.
        Defaults to ``logging.INFO``.

    Returns
    -------
    result_json_paths : list of str
        A list of paths to .json results files for each variation in the
        experiment.

    Raises
    ------
    ValueError
        If value for ``"ablation"`` is not a positive int or ``None``.
    OSError
        If the lenth of the ``FeatureSet`` name > 210.
    """

    try:

        # Read configuration
        (experiment_name, task, sampler, fixed_sampler_parameters, feature_hasher,
         hasher_features, id_col, label_col, train_set_name, test_set_name, suffix,
         featuresets, do_shuffle, model_path, do_grid_search, grid_objectives,
         probability, pipeline, results_path, pos_label, feature_scaling,
         min_feature_count, folds_file, grid_search_jobs, grid_search_folds, cv_folds,
         cv_seed, save_cv_folds, save_cv_models, use_folds_file_for_grid_search,
         do_stratified_folds, fixed_parameter_list, param_grid_list, featureset_names,
         learners, prediction_dir, log_path, train_path, test_path, ids_to_floats,
         class_map, custom_learner_path, custom_metric_path,
         learning_curve_cv_folds_list, learning_curve_train_sizes,
         output_metrics, save_votes) = parse_config_file(config_file, log_level=log_level)

        # get the main experiment logger that will already have been
        # created by the configuration parser so we don't need anything
        # except the name `experiment`.
        logger = get_skll_logger('experiment')

        # Check if we have gridmap
        if not local and not _HAVE_GRIDMAP:
            local = True
            logger.warning('gridmap 0.10.1+ not available. Forcing local '
                           'mode.  To run things on a DRMAA-compatible '
                           'cluster, install gridmap>=0.10.1 via pip.')

        # No grid search or ablation for learning curve generation
        if task == 'learning_curve':
            if ablation is None or ablation > 0:
                ablation = 0
                logger.warning("Ablating features is not supported during "
                               "learning curve generation. Ignoring.")

        # if we just had a train file and a test file, there are no real featuresets
        # in which case there are no features to ablate
        if len(featuresets) == 1 and len(featuresets[0]) == 1:
            if ablation is None or ablation > 0:
                ablation = 0
                logger.warning("Not enough featuresets for ablation. Ignoring.")

        # if performing ablation, expand featuresets to include combinations of
        # features within those sets
        if ablation is None or ablation > 0:
            # Make new feature set lists so that we can iterate without issue
            expanded_fs = []
            expanded_fs_names = []
            for features, featureset_name in zip(featuresets, featureset_names):
                features = sorted(features)
                featureset = set(features)
                # Expand to all feature combinations if ablation is None
                if ablation is None:
                    for i in range(1, len(features)):
                        for excluded_features in combinations(features, i):
                            expanded_fs.append(sorted(featureset -
                                                      set(excluded_features)))
                            expanded_fs_names.append(
                                f'{featureset_name}_minus_'
                                f'{_munge_featureset_name(excluded_features)}'
                            )
                # Otherwise, just expand removing the specified number at a time
                else:
                    for excluded_features in combinations(features, ablation):
                        expanded_fs.append(sorted(featureset -
                                                  set(excluded_features)))
                        expanded_fs_names.append(
                            f'{featureset_name}_minus_'
                            f'{_munge_featureset_name(excluded_features)}'
                        )
                # Also add version with nothing removed as baseline
                expanded_fs.append(features)
                expanded_fs_names.append(f'{featureset_name}_all')

            # Replace original feature set lists
            featuresets = expanded_fs
            featureset_names = expanded_fs_names
        elif ablation < 0:
            raise ValueError('Value for "ablation" argument must be either '
                             'positive integer or None.')

        # the list of jobs submitted (if running on grid)
        if not local:
            jobs = []

        # the list to hold the paths to all the result json files
        result_json_paths = []

        # check if the length of the featureset_name exceeds the maximum length
        # allowed
        for featureset_name in featureset_names:
            if len(featureset_name) > 210:
                raise OSError(
                    f'System generated file length "{featureset_name}" '
                    'exceeds the maximum length supported.  Please specify '
                    'names of your datasets with "featureset_names".  If you '
                    'are running ablation experiment, please reduce the '
                    'length of the features in "featuresets" because the '
                    'auto-generated name would be longer than the file system'
                    ' can handle')

        # if the task is learning curve, and ``metrics`` was specified, then
        # assign the value of ``metrics`` to ``grid_objectives`` - this lets
        # us piggyback on the parallelization of the objectives that is already
        # set up for us to use
        if task == 'learning_curve' and len(output_metrics) > 0:
            grid_objectives = output_metrics

        # if there were no grid objectives provided, just set it to
        # a list containing a single None so as to allow the parallelization
        # to proceeed and to pass the correct default value of grid_objective
        # down to _classify_featureset().
        if not grid_objectives:
            grid_objectives = [None]

        # Run each featureset-learner-objective combination
        for featureset, featureset_name in zip(featuresets, featureset_names):
            for learner_num, learner_name in enumerate(learners):
                for grid_objective in grid_objectives:

                    # for the individual job name, we need to add the feature set name
                    # and the learner name
                    if grid_objective is None or len(grid_objectives) == 1:
                        job_name_components = [experiment_name, featureset_name,
                                               learner_name]
                    else:
                        job_name_components = [experiment_name, featureset_name,
                                               learner_name, grid_objective]

                    job_name = '_'.join(job_name_components)

                    # change the prediction prefix to include the feature set
                    prediction_prefix = join(prediction_dir, job_name)

                    # the log file that stores the actual output of this script (e.g.,
                    # the tuned parameters, what kind of experiment was run, etc.)
                    logfile = join(log_path, f'{job_name}.log')

                    # Figure out result json file path
                    result_json_path = join(results_path,
                                            f'{job_name}.results.json')

                    # save the path to the results json file that will be written
                    result_json_paths.append(result_json_path)

                    # If result file already exists and we're resuming, move on
                    if resume and (exists(result_json_path) and getsize(result_json_path)):
                        logger.info('Running in resume mode and '
                                    f'{result_json_path} exists, so skipping '
                                    'job.')
                        continue

                    # create job if we're doing things on the grid
                    job_args = {}
                    job_args["experiment_name"] = experiment_name
                    job_args["task"] = task
                    job_args["sampler"] = sampler
                    job_args["feature_hasher"] = feature_hasher
                    job_args["hasher_features"] = hasher_features
                    job_args["job_name"] = job_name
                    job_args["featureset"] = featureset
                    job_args["featureset_name"] = featureset_name
                    job_args["learner_name"] = learner_name
                    job_args["train_path"] = train_path
                    job_args["test_path"] = test_path
                    job_args["train_set_name"] = train_set_name
                    job_args["test_set_name"] = test_set_name
                    job_args["shuffle"] = do_shuffle
                    job_args["model_path"] = model_path
                    job_args["prediction_prefix"] = prediction_prefix
                    job_args["grid_search"] = do_grid_search
                    job_args["grid_objective"] = grid_objective
                    job_args['output_metrics'] = output_metrics
                    job_args["suffix"] = suffix
                    job_args["log_file"] = logfile
                    job_args["log_level"] = log_level
                    job_args["probability"] = probability
                    job_args["pipeline"] = pipeline
                    job_args["save_votes"] = save_votes
                    job_args["results_path"] = results_path
                    job_args["sampler_parameters"] = (fixed_sampler_parameters
                                                      if fixed_sampler_parameters
                                                      else dict())
                    job_args["fixed_parameters"] = (fixed_parameter_list[learner_num]
                                                    if fixed_parameter_list
                                                    else dict())
                    job_args["param_grid"] = (param_grid_list[learner_num]
                                              if param_grid_list else None)
                    job_args["pos_label"] = pos_label
                    job_args["overwrite"] = overwrite
                    job_args["feature_scaling"] = feature_scaling
                    job_args["min_feature_count"] = min_feature_count
                    job_args["grid_search_jobs"] = grid_search_jobs
                    job_args["grid_search_folds"] = grid_search_folds
                    job_args["folds_file"] = folds_file
                    job_args["cv_folds"] = cv_folds
                    job_args["cv_seed"] = cv_seed
                    job_args["save_cv_folds"] = save_cv_folds
                    job_args["save_cv_models"] = save_cv_models
                    job_args["use_folds_file_for_grid_search"] = use_folds_file_for_grid_search
                    job_args["do_stratified_folds"] = do_stratified_folds
                    job_args["label_col"] = label_col
                    job_args["id_col"] = id_col
                    job_args["ids_to_floats"] = ids_to_floats
                    job_args["quiet"] = quiet
                    job_args["class_map"] = class_map
                    job_args["custom_learner_path"] = custom_learner_path
                    job_args["custom_metric_path"] = custom_metric_path
                    job_args["learning_curve_cv_folds"] = learning_curve_cv_folds_list[learner_num]
                    job_args["learning_curve_train_sizes"] = learning_curve_train_sizes

                    if not local:
                        jobs.append(Job(_classify_featureset,
                                        [job_args],
                                        num_slots=(MAX_CONCURRENT_PROCESSES if
                                                   (do_grid_search or task == 'learning_curve') else 1),
                                        name=job_name,
                                        queue=queue))
                    else:
                        _classify_featureset(job_args)

        # Call get_skll_logger again after _classify_featureset
        # calls are finished so that any warnings that may
        # happen after this point get correctly logged to the
        # main logger
        logger = get_skll_logger('experiment')

        # submit the jobs (if running on grid)
        if not local and _HAVE_GRIDMAP:
            if log_path:
                job_results = process_jobs(jobs, white_list=hosts,
                                           temp_dir=log_path)
            else:
                job_results = process_jobs(jobs, white_list=hosts)
            _check_job_results(job_results)

        # write out the summary results file
        if (task == 'cross_validate' or task == 'evaluate') and write_summary:
            summary_file_name = f'{experiment_name}_summary.tsv'
            with open(join(results_path,
                           summary_file_name), 'w', newline='') as output_file:
                _write_summary_file(result_json_paths,
                                    output_file,
                                    ablation=ablation)
        elif task == 'learning_curve':
            output_file_name = f'{experiment_name}_summary.tsv'
            output_file_path = join(results_path, output_file_name)
            with open(output_file_path, 'w', newline='') as output_file:
                _write_learning_curve_file(result_json_paths, output_file)

            # generate the actual plot if we have the requirements installed
            generate_learning_curve_plots(experiment_name,
                                          results_path,
                                          output_file_path)

    finally:

        # Close/remove any logger handlers
        close_and_remove_logger_handlers(get_skll_logger('experiment'))

    return result_json_paths
