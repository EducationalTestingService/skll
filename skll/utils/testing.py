"""Utility functions to make SKLL testing simpler."""


import os
import re
from collections import OrderedDict
from math import floor, log10
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.random import RandomState
from sklearn.datasets import (
    fetch_california_housing,
    load_digits,
    make_classification,
    make_regression,
)
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.utils import shuffle as sk_shuffle

from skll.config import _setup_config_parser, fix_json
from skll.data import FeatureSet, NDJWriter
from skll.types import PathOrStr

if env_test_dir := os.getenv("TESTDIR"):
    tests_dir = Path(env_test_dir) / "tests"
else:
    tests_dir = Path(__file__).resolve().parent.parent.parent / "tests"
config_dir = tests_dir / "configs"
backward_compatibility_dir = tests_dir / "backward_compatibility"
examples_dir = tests_dir.parent / "examples"
output_dir = tests_dir / "output"
other_dir = tests_dir / "other"
train_dir = tests_dir / "train"
test_dir = tests_dir / "test"


class BoolDict(dict):
    """Dictionary that returns ``False`` instead of ``None`` as default."""

    def __getitem__(self, key):
        """Get item for given key."""
        return super().get(key, False)


def unlink(file_path: PathOrStr):
    """
    Remove a file path if it exists.

    Parameters
    ----------
    file_path : :class:`skll.types.PathOrStr`
        File path to remove.

    """
    file_path = Path(file_path)
    if file_path.exists():
        file_path.unlink()


def fill_in_config_paths(config_template_path: PathOrStr) -> Path:
    """
    Fill paths in given configuration file template.

    Add various paths in the given configuration template file.

    Parameters
    ----------
    config_template_path : :class:`skll.types.PathOrStr`
        The path to the template configuration file.

    Returns
    -------
    Path
        The path to the filled configuration file.

    """
    # convert path to Path object
    config_template_path = Path(config_template_path)

    config = _setup_config_parser(config_template_path, validate=False)

    task = config.get("General", "task")

    config.set("Input", "train_directory", str(train_dir))

    to_fill_in = ["logs"]

    if task != "learning_curve":
        to_fill_in.append("predictions")

    if task not in ["cross_validate", "learning_curve"]:
        to_fill_in.append("models")

    if task in ["cross_validate", "evaluate", "learning_curve", "train"]:
        to_fill_in.append("results")

    for d in to_fill_in:
        config.set("Output", d, str(output_dir))

    if task == "cross_validate":
        folds_file = config.get("Input", "folds_file")
        if folds_file:
            config.set("Input", "folds_file", str(train_dir / folds_file))

    if task == "predict" or task == "evaluate":
        config.set("Input", "test_directory", str(test_dir))

    # set up custom learner path, if relevant
    custom_learner_path = config.get("Input", "custom_learner_path")
    custom_learner_abs_path = tests_dir / custom_learner_path
    config.set("Input", "custom_learner_path", str(custom_learner_abs_path))

    config_prefix_match = re.search(r"^(.*)\.template\.cfg", str(config_template_path))
    assert config_prefix_match is not None
    config_prefix = config_prefix_match.groups()[0]
    new_config_path = Path(f"{config_prefix}.cfg")

    with open(new_config_path, "w") as new_config_file:
        config.write(new_config_file)

    return new_config_path


def fill_in_config_paths_for_single_file(
    config_template_path: PathOrStr,
    train_file: str,
    test_file: str,
    train_directory: str = "",
    test_directory: str = "",
) -> Path:
    """
    Fill in input file and directory paths in given configuration template.

    Add paths to train files, test files, and output directories in the
    given config template file.

    Parameters
    ----------
    config_template_path : :class:`skll.types.PathOrStr`
        Path to the template configuration file.
    train_file : str
        Name of the training data file.
    test_file : str
        Name of the test data file.
    train_directory : str, optional
        Path to the directory containing the training data.
        Defaults to ''
    test_directory : str, optional
        Path to the directory containing the test data.
        Defaults to ''

    Returns
    -------
    Path
        The path to the filled configuration file.

    """
    # convert path to Path object if it's a string
    config_template_path = Path(config_template_path)

    config = _setup_config_parser(config_template_path, validate=False)

    task = config.get("General", "task")

    config.set("Input", "train_file", str(train_dir / train_file))
    if task == "predict" or task == "evaluate":
        config.set("Input", "test_file", str(test_dir / test_file))

    if train_directory:
        config.set("Input", "train_directory", str(train_dir / train_directory))

    if test_directory:
        config.set("Input", "test_directory", str(test_dir / test_directory))

    to_fill_in = ["logs"]
    if task != "train":
        to_fill_in.append("predictions")

    if task == "cross_validate":
        if config.get("Output", "save_cv_models"):
            to_fill_in.append("models")
    else:
        to_fill_in.append("models")

    if task in ["cross_validate", "evaluate", "train"]:
        to_fill_in.append("results")

    for d in to_fill_in:
        config.set("Output", d, str(output_dir))

    if task == "cross_validate":
        folds_file = config.get("Input", "folds_file")
        if folds_file:
            config.set("Input", "folds_file", str(train_dir / folds_file))

    config_prefix_match = re.search(r"^(.*)\.template\.cfg", str(config_template_path))
    assert config_prefix_match is not None
    config_prefix = config_prefix_match.groups()[0]
    new_config_path = Path(f"{config_prefix}.cfg")

    with open(new_config_path, "w") as new_config_file:
        config.write(new_config_file)

    return new_config_path


def fill_in_config_options(
    config_template_path: PathOrStr,
    values_to_fill_dict: Dict[str, Any],
    sub_prefix: str,
    good_probability_option: bool = False,
) -> Path:
    """
    Fill in configuration options in the given template file.

    Parameters
    ----------
    config_template_path : :class:`skll.types.PathOrStr`
        Path to the template configuration file.
    values_to_fill_dict : Dict[str, Any]
        Dictionary containing the options to fille the keys and the
        corresponding values.
    sub_prefix : str
        The sub-prefix to add to the name when creating
        the filled configuration file on disk.
    good_probability_option : bool, default=False
        Whether to add the "probability" option in the correct
        section or an incorrect section.

    Returns
    -------
    Path
        The path to the filled configuration file.

    """
    # convert path to Path object if it's a string
    config_template_path = Path(config_template_path)

    config = _setup_config_parser(config_template_path, validate=False)

    # The dictionary that says which option to fill in where
    # Note: (a) `bad_option` and `duplicate_option` are needed
    # for test_config_parsing_invalid_option and
    # test_config_parsing_duplicate_option() respectively.
    # (b) `probability` is deliberately specified in the wrong
    # section for test_config_parsing_option_in_wrong_section().
    to_fill_in = {
        "General": ["experiment_name", "task"],
        "Input": [
            "train_directory",
            "train_file",
            "test_directory",
            "test_file",
            "featuresets",
            "featureset_names",
            "custom_learner_path",
            "feature_hasher",
            "hasher_features",
            "learners",
            "sampler",
            "shuffle",
            "feature_scaling",
            "learning_curve_cv_folds_list",
            "folds_file",
            "learning_curve_train_sizes",
            "fixed_parameters",
            "num_cv_folds",
            "cv_seed",
            "bad_option",
            "duplicate_option",
            "suffix",
        ],
        "Tuning": [
            "grid_search",
            "objective",
            "min_feature_count",
            "use_folds_file_for_grid_search",
            "grid_search_folds",
            "pos_label",
            "param_grids",
            "objectives",
            "duplicate_option",
        ],
        "Output": [
            "results",
            "log",
            "logs",
            "models",
            "metrics",
            "predictions",
            "pipeline",
            "save_cv_folds",
            "save_cv_models",
            "save_votes",
            "wandb_credentials",
        ],
    }

    if good_probability_option:
        to_fill_in["Output"].append("probability")
    else:
        to_fill_in["Tuning"].append("probability")

    for section in to_fill_in:
        for param_name in to_fill_in[section]:
            if param_name in values_to_fill_dict:
                value = str(values_to_fill_dict[param_name])
                config.set(section, param_name, value)

    config_prefix_match = re.search(r"^(.*)\.template\.cfg", str(config_template_path))
    assert config_prefix_match is not None
    config_prefix = config_prefix_match.groups()[0]
    new_config_path = Path(f"{config_prefix}_{sub_prefix}.cfg")

    with open(new_config_path, "w") as new_config_file:
        config.write(new_config_file)

    return new_config_path


def fill_in_config_paths_for_fancy_output(config_template_path: PathOrStr) -> Path:
    """
    Fill in the template for more comprehensive ("fancier") output.

    Add paths to train, test, and output directories in the given
    config template file.

    Parameters
    ----------
    config_template_path : :class:`skll.types.PathOrStr`
        Path to the template configuration file.

    Returns
    -------
    Path
        The path to the filled configuration file.

    """
    # convert template path to a Path object if string
    config_template_path = Path(config_template_path)

    config = _setup_config_parser(config_template_path, validate=False)

    config.set("Input", "train_file", str(train_dir / "fancy_train.jsonlines"))
    config.set("Input", "test_file", str(test_dir / "fancy_test.jsonlines"))
    config.set("Output", "results", str(output_dir))
    config.set("Output", "logs", str(output_dir))
    config.set("Output", "predictions", str(output_dir))

    config_prefix_match = re.search(r"^(.*)\.template\.cfg", str(config_template_path))
    assert config_prefix_match is not None
    config_prefix = config_prefix_match.groups()[0]
    new_config_path = Path(f"{config_prefix}.cfg")

    with open(new_config_path, "w") as new_config_file:
        config.write(new_config_file)

    return new_config_path


def fill_in_config_options_for_voting_learners(
    learner_type: str, task: str, options_dict: BoolDict
) -> Tuple:
    """
    Fill in values specific to voting learners in the given template.

    Parameters
    ----------
    learner_type : str
        One of "classifier" or "regressor".
    task : str
        The voting learner task for which we are filling the values.
    options_dict : BoolDict
        A dictionary containing various boolean options which will
        be used to fill the values based on whether they are set
        to ``True`` or ``False`.

    Returns
    -------
    res : a 13-tuple
        A tuple containing the following items as determined by the options
        specified in ``options_dict``:
        - the path to the filled-in configuration file
        - the names of the underlying estimators used for the voting learner
        - a job prefix constructed using the experiment name, the task, the
          featureset name, and the learner name.
        - the path to any custom learner file used, if any
        - the names of the objective functions used, if any
        - the names of any output metrics used, if any
        - the list of fixed parameters used for the underlying estimators, if any
        - the list of parameter grids used for the underlying estimators, if any
        - the list of samplers used for the underlying estimators, if any
        - the number of cross-validation folds used (6 or 10)
        - the custom seed value used for cross-validation, if any
        - the number of learning curve cross-validation folds (10 or 20)
        - the list of learning curve training sizes

    """
    # setup learner-type specific values based on configuration options
    custom_learner = ""
    objectives = None
    output_metrics = []
    model_kwargs_list: Optional[List[Dict[str, Any]]] = None
    param_grid_list: Optional[List[Dict[str, Any]]] = None
    sampler_list = None
    num_cv_folds = learning_curve_cv_folds = 10
    cv_seed = 123456789
    fixed_parameters: List[Dict[str, Any]]
    learning_curve_train_sizes = [0.1, 0.325, 0.55, 0.775, 1.0]
    if learner_type == "classifier":
        learner_name = "VotingClassifier"
        estimator_names = ["LogisticRegression", "SVC", "MultinomialNB"]
        fixed_parameters = [{"estimator_names": estimator_names}]

        if options_dict["with_model_kwargs_list"]:
            model_kwargs_list = [{}, {"gamma": "auto"}, {"alpha": 0.01}]
            fixed_parameters[0].update({"estimator_fixed_parameters": model_kwargs_list})

        if options_dict["with_param_grid_list"]:
            param_grid_list = [{"penalty": ["l1", "l2"]}, {}, {}]
            fixed_parameters[0].update({"estimator_param_grids": param_grid_list})

        if options_dict["with_sampler_list"]:
            sampler_list = ["SkewedChi2Sampler", "Nystroem", ""]
            fixed_parameters[0].update({"estimator_samplers": sampler_list})

        if options_dict["with_soft_voting"]:
            fixed_parameters[0].update({"voting_type": "soft"})

        if options_dict["with_grid_search"]:
            objectives = (
                ["accuracy", "f1_score_macro"]
                if options_dict["with_multiple_objectives"]
                else ["accuracy"]
            )

        if options_dict["with_output_metrics"]:
            output_metrics = ["f05", "f1_score_macro"]

    else:
        learner_name = "VotingRegressor"
        estimator_names = ["LinearRegression", "SVR", "RandomForestRegressor"]
        fixed_parameters = [{"estimator_names": estimator_names}]

        if options_dict["with_model_kwargs_list"]:
            model_kwargs_list = [{}, {"kernel": "poly"}, {"n_estimators": 200}]
            fixed_parameters[0].update({"estimator_fixed_parameters": model_kwargs_list})

        if options_dict["with_param_grid_list"]:
            param_grid_list = [{}, {"degree": [3, 4, 5]}, {}]
            fixed_parameters[0].update({"estimator_param_grids": param_grid_list})

        if options_dict["with_sampler_list"]:
            sampler_list = ["", "AdditiveChi2Sampler", ""]
            fixed_parameters[0].update({"estimator_samplers": sampler_list})

        if options_dict["with_grid_search"]:
            objectives = (
                ["pearson", "neg_mean_squared_error"]
                if options_dict["with_multiple_objectives"]
                else ["pearson"]
            )

        if options_dict["with_output_metrics"]:
            output_metrics = ["spearman", "kendall_tau"]

    # create the dictionary that we will use to fill our config template
    experiment_name = "test_voting_learner"
    sub_prefix = task
    featureset_name = "f0"
    train_path = train_dir / f"{featureset_name}.jsonlines"
    job_name = f"{experiment_name}_{sub_prefix}_{learner_name}"
    values_to_fill_dict = {
        "experiment_name": f"{experiment_name}_{sub_prefix}",
        "train_file": str(train_path),
        "task": task,
        "learners": str([learner_name]),
        "grid_search": fix_json(str(options_dict["with_grid_search"])),
        "featureset_names": str([featureset_name]),
        "fixed_parameters": str(fixed_parameters),
    }

    # insert additional values in this dictionary
    if task in ["evaluate", "predict"]:
        test_path = str(test_dir / f"{featureset_name}.jsonlines")
        values_to_fill_dict["test_file"] = str(test_path)

    if options_dict["with_grid_search"]:
        values_to_fill_dict["objectives"] = str(objectives)

    if options_dict["with_gs_folds"]:
        values_to_fill_dict["grid_search_folds"] = "4"

    if options_dict["with_shuffle"]:
        values_to_fill_dict["shuffle"] = "true"

    if options_dict["with_centering"]:
        values_to_fill_dict["feature_scaling"] = "with_mean"

    if options_dict["with_min_feature_count"]:
        values_to_fill_dict["min_feature_count"] = "2"

    if options_dict["with_custom_learner_path"]:
        custom_learner = str(other_dir / "custom_logistic_wrapper.py")
        values_to_fill_dict["custom_learner_path"] = custom_learner

    if options_dict["with_pos_label"]:
        values_to_fill_dict["pos_label"] = "dog"

    if options_dict["with_individual_predictions"]:
        values_to_fill_dict["save_votes"] = "true"

    if options_dict["with_output_metrics"]:
        values_to_fill_dict["metrics"] = str(output_metrics)

    if options_dict["with_prediction_prefix"]:
        values_to_fill_dict["predictions"] = str(output_dir)

    if options_dict["with_cv_folds"]:
        num_cv_folds = 6
        values_to_fill_dict["num_cv_folds"] = str(num_cv_folds)

    if options_dict["with_custom_cv_seed"]:
        cv_seed = 987
        values_to_fill_dict["cv_seed"] = str(cv_seed)

    if options_dict["without_save_cv_folds"]:
        values_to_fill_dict["save_cv_folds"] = "false"

    if options_dict["with_save_cv_models"]:
        values_to_fill_dict["save_cv_models"] = "true"

    if options_dict["with_learning_curve_cv_folds"]:
        learning_curve_cv_folds = 20
        values_to_fill_dict["learning_curve_cv_folds_list"] = str([learning_curve_cv_folds])

    if options_dict["with_learning_curve_train_sizes"]:
        learning_curve_train_sizes = [0.1, 0.5, 1.0]
        values_to_fill_dict["learning_curve_train_sizes"] = str(learning_curve_cv_folds)

    if task != "learning_curve":
        values_to_fill_dict["models"] = (
            str(other_dir) if options_dict["with_existing_model"] else str(output_dir)
        )

    # locate the voting learner config template and instantiate it
    config_template_path = config_dir / f"{experiment_name}.template.cfg"
    config_path = fill_in_config_options(config_template_path, values_to_fill_dict, sub_prefix)

    return (
        config_path,
        estimator_names,
        job_name,
        custom_learner,
        objectives,
        output_metrics,
        model_kwargs_list,
        param_grid_list,
        sampler_list,
        num_cv_folds,
        cv_seed,
        learning_curve_cv_folds,
        learning_curve_train_sizes,
    )


def create_jsonlines_feature_files(path: PathOrStr):
    """
    Create dummy jsonlines feature files and save them under ``path``.

    Parameters
    ----------
    path : :class:`skll.types.PathOrStr`
        Full path under which to save the created feature files.

    """
    # convert to Path object
    path = Path(path)

    # we only need to create the feature files if they
    # don't already exist under the given path
    feature_files_to_create = [path / f"f{i}.jsonlines" for i in range(6)]
    if all([ff.exists() for ff in feature_files_to_create]):
        return
    else:
        num_examples = 1000
        np.random.seed(1234567890)

        # Create lists we will write files from
        ids = []
        features = []
        labels = []
        for j in range(num_examples):
            y = "dog" if j % 2 == 0 else "cat"
            ex_id = f"{y}{j}"
            x = {f"f{feat_num}": np.random.randint(0, 4) for feat_num in range(5)}
            x = OrderedDict(sorted(x.items(), key=lambda t: t[0]))
            ids.append(ex_id)
            labels.append(y)
            features.append(x)

        for i in range(5):
            file_path = path / f"f{i}.jsonlines"
            sub_features = []
            for example_num in range(num_examples):
                feat_num = i
                x = {f"f{feat_num}": features[example_num][f"f{feat_num}"]}
                sub_features.append(x)
            fs = FeatureSet("ablation_cv", ids, features=sub_features, labels=labels)

            writer = NDJWriter(file_path, fs)
            writer.write()

        # now write out the last file which is basically
        # identical to the last featureset we wrote
        # except that it has two extra instances
        fs = FeatureSet(
            "extra",
            ids + [f"cat{num_examples}", f"dog{num_examples + 1}"],
            features=sub_features + [{}, {}],
            labels=labels + ["cat", "dog"],
        )
        file_path = path / "f5.jsonlines"
        writer = NDJWriter(file_path, fs)
        writer.write()


def remove_jsonlines_feature_files(path: PathOrStr):
    """
    Remove all files created by ``create_jsonlines_feature_files()``.

    Parameters
    ----------
    path : :class:`skll.types.PathOrStr`
        Path to directory in which jsonlines files reside.

    """
    for i in range(6):
        unlink(Path(path) / f"f{i}.jsonlines")


def make_classification_data(
    num_examples=100,
    train_test_ratio=0.5,
    num_features=10,
    use_feature_hashing=False,
    feature_bins=4,
    num_labels=2,
    empty_labels=False,
    string_label_list=None,
    feature_prefix="f",
    id_type="string",
    class_weights=None,
    non_negative=False,
    one_string_feature=False,
    num_string_values=4,
    random_state=1234567890,
):
    """
    Create dummy classification data for use in various tests.

    Parameters
    ----------
    num_examples : int, optional
        Number of examples in the generated data.
        Defaults to 100.
    train_test_ratio : float, optional
        Ratio of train to test data in the generated data.
        Defaults to 0.5.
    num_features : int, optional
        Number of features in each generated example.
        Defaults to 10.
    use_feature_hashing : bool, optional
        Whether to use feature hashing.
        Defaults to ``False``.
    feature_bins : int, optional
        How many hashed feature bins to use, if ``use_feature_hashing``
        is ``True``.
        Defaults to 4.
    num_labels : int, optional
        Number of classes to use.
    empty_labels : bool, optional
        Whether to set the training and test labels to ``None``
        for certain tests.
        Defaults to ``False``.
    string_label_list : List[str], optional
        List of pre-specified string labels.
    feature_prefix : str, optional
        The prefix to use for all the feature names, if any.
        Defaults to "f" (i.e., "f01" et cetera).
    id_type : str, optional
        The data types to use for the generated example IDs.
        One of ["float", "integer", "integer_string", "string"].
        Defaults to "string".
    class_weights : List[float], optional
        The proportion of generated samples for each of the classes.
        If ``None``, use the same number of samples for each class.
        Defaults to ``None``.
    non_negative : bool, optional
        Whether to generate only non-negative features.
        Defaults to ``False``.
    one_string_feature : bool, optional
        Whether to generate one feature with string values in
        the generated data.
        Defaults to ``False``.
    num_string_values : int, optional
        If ``one_string_feature`` is set to ``True``, how many
        possible values to use for that feature.
        Defaults to 4.
    random_state : int, optional
        Random state to use for generating the data.
        Defaults to 123456789.

    Returns
    -------
    Tuple
        Tuple containing the generated training featureset and
        the generated test featureset.

    """
    # use sklearn's make_classification to generate the data for us
    num_numeric_features = num_features - 1 if one_string_feature else num_features
    X, y = make_classification(
        n_samples=num_examples,
        n_features=num_numeric_features,
        n_informative=num_numeric_features,
        n_redundant=0,
        n_classes=num_labels,
        weights=class_weights,
        random_state=random_state,
    )

    if string_label_list:
        assert len(string_label_list) == num_labels
        label_to_string = np.vectorize(lambda n: string_label_list[n])
        y = label_to_string(y)

    # if we were told to only generate non-negative features, then
    # we can simply take the absolute values of the generated features
    if non_negative:
        X = abs(X)

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs; we create IDs that either can also
    # be numbers or pure strings
    if id_type == "string":
        ids = [f"EXAMPLE_{n}" for n in range(1, num_examples + 1)]
    elif id_type == "integer_string":
        ids = [f"{n}" for n in range(1, num_examples + 1)]
    elif id_type == "float":
        ids = [float(n) for n in range(1, num_examples + 1)]
    elif id_type == "integer":
        ids = list(range(1, num_examples + 1))

    # create a string feature that has four possible values
    # 'a', 'b', 'c' and 'd' and add it to X at the end
    if one_string_feature:
        prng = RandomState(random_state)
        random_indices = prng.random_integers(0, num_string_values - 1, num_examples)
        possible_values = [chr(x) for x in range(97, 97 + num_string_values)]
        string_feature_values = [possible_values[i] for i in random_indices]
        string_feature_column = np.array(string_feature_values, dtype=object).reshape(100, 1)
        X = np.append(X, string_feature_column, 1)

    # create a list of dictionaries as the features
    feature_names = [f"{feature_prefix}{n:02d}" for n in range(1, num_features + 1)]
    features = [dict(zip(feature_names, row)) for row in X]

    # split everything into training and testing portions
    num_train_examples = int(round(train_test_ratio * num_examples))
    train_features, test_features = (features[:num_train_examples], features[num_train_examples:])
    train_y, test_y = y[:num_train_examples], y[num_train_examples:]
    train_ids, test_ids = ids[:num_train_examples], ids[num_train_examples:]

    # are we told to generate empty labels
    train_labels = None if empty_labels else train_y
    test_labels = None if empty_labels else test_y

    # create a FeatureHasher if we are asked to use feature hashing
    # with the specified number of feature bins
    vectorizer = FeatureHasher(n_features=feature_bins) if use_feature_hashing else None
    train_fs = FeatureSet(
        "classification_train",
        train_ids,
        labels=train_labels,
        features=train_features,
        vectorizer=vectorizer,
    )
    if train_test_ratio < 1.0:
        test_fs = FeatureSet(
            "classification_test",
            test_ids,
            labels=test_labels,
            features=test_features,
            vectorizer=vectorizer,
        )
    else:
        test_fs = None

    return (train_fs, test_fs)


def make_regression_data(
    num_examples=100,
    train_test_ratio=0.5,
    num_features=2,
    sd_noise=1.0,
    use_feature_hashing=False,
    feature_bins=4,
    start_feature_num=1,
    random_state=1234567890,
):
    """
    Create dummy regression data for use with tests.

    Parameters
    ----------
    num_examples : int, optional
        Number of examples in the generated data.
        Defaults to 100.
    train_test_ratio : float, optional
        Ratio of train to test data in the generated data.
        Defaults to 0.5.
    num_features : int, optional
        Number of features in each generated example.
        Defaults to 2.
    sd_noise : float, optional
        Amount of Gaussian noise added to the output.
        Defaults to 1.0.
    use_feature_hashing : bool, optional
        Whether to use feature hashing.
        Defaults to ``False``.
    feature_bins : int, optional
        How many hashed feature bins to use, if ``use_feature_hashing``
        is ``True``.
        Defaults to 4.
    start_feature_num : int, optional
        The integer suffix for the first feature name (i.e., "f01").
        Defaults to 1.
    random_state : int, optional
        Random state to use for generating the data.
        Defaults to 123456789.

    Returns
    -------
    Tuple
        3-tuple containing the generated training featureset, the generated
        test featureset, and a dictionary containing the oracle feature
        weights

    """
    # if we are doing feature hashing and we have asked for more
    # feature bins than number of total features, we need to
    # handle that because `make_regression()` doesn't know
    # about hashing
    if use_feature_hashing and num_features < feature_bins:
        num_features = feature_bins

    # use sklearn's make_regression to generate the data for us
    X, y, weights = make_regression(
        n_samples=num_examples,
        n_features=num_features,
        noise=sd_noise,
        random_state=random_state,
        coef=True,
    )

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = [f"EXAMPLE_{n}" for n in range(1, num_examples + 1)]

    # create a list of dictionaries as the features
    index_width_for_feature_name = int(floor(log10(num_features))) + 1
    feature_names = []
    for n in range(start_feature_num, start_feature_num + num_features):
        index_str = str(n).zfill(index_width_for_feature_name)
        feature_name = f"f{index_str}"
        feature_names.append(feature_name)
    features = [dict(zip(feature_names, row)) for row in X]

    # At this point the labels are generated using unhashed features
    # even if we want to do feature hashing. `make_regression()` from
    # sklearn doesn't know anything about feature hashing, so we need
    # a hack here to compute the updated labels ourselves
    # using the same command that sklearn uses inside `make_regression()`
    # which is to generate the X and the weights and then compute the
    # y as the dot product of the two. This y will then be used as our
    # labels instead of the original y we got from `make_regression()`.
    # Note that we only want to use the number of weights that are
    # equal to the number of feature bins for the hashing
    if use_feature_hashing:
        feature_hasher = FeatureHasher(n_features=feature_bins)
        hashed_X = feature_hasher.fit_transform(features)
        y = hashed_X.dot(weights[:feature_bins])

    # convert the weights array into a dictionary for convenience
    # if we are using feature hashing, we need to use the names
    # that would be output by `model_params()` instead of the
    # original names since that's what we would get from SKLL
    if use_feature_hashing:
        index_width_for_feature_name = int(floor(log10(feature_bins))) + 1
        hashed_feature_names = []
        for i in range(feature_bins):
            index_str = str(i + 1).zfill(index_width_for_feature_name)
            feature_name = f"hashed_feature_{index_str}"
            hashed_feature_names.append(feature_name)
        weightdict = dict(zip(hashed_feature_names, weights[:feature_bins]))
    else:
        weightdict = dict(zip(feature_names, weights))

    # split everything into training and testing portions
    num_train_examples = int(round(train_test_ratio * num_examples))
    train_features, test_features = (features[:num_train_examples], features[num_train_examples:])
    train_y, test_y = y[:num_train_examples], y[num_train_examples:]
    train_ids, test_ids = ids[:num_train_examples], ids[num_train_examples:]

    # create a FeatureHasher if we are asked to use feature hashing
    # with the specified number of feature bins
    vectorizer = FeatureHasher(n_features=feature_bins) if use_feature_hashing else None
    train_fs = FeatureSet(
        "regression_train",
        train_ids,
        labels=train_y,
        features=train_features,
        vectorizer=vectorizer,
    )
    test_fs = FeatureSet(
        "regression_test", test_ids, labels=test_y, features=test_features, vectorizer=vectorizer
    )

    return (train_fs, test_fs, weightdict)


def make_sparse_data(use_feature_hashing=False):
    """
    Create sparse data for use in various tests.

    Function to create sparse data with two features always zero
    in the training set and a different one always zero in the
    test set.

    Parameters
    ----------
    use_feature_hashing : bool, optional
        Whether to use feature hashing.
        Defaults to ``False``.

    Returns
    -------
    Tuple
        Tuple containing the generated training featureset and
        the generated test featureset.

    """
    # Create training data
    X, y = make_classification(
        n_samples=500,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=1234567890,
    )

    # we need features to be non-negative since we will be
    # using naive bayes laster
    X = np.abs(X)

    # make sure that none of the features are zero
    X[np.where(X == 0)] += 1

    # since we want to use SKLL's FeatureSet class, we need to
    # create a list of IDs
    ids = [f"EXAMPLE_{n}" for n in range(1, 501)]

    # create a list of dictionaries as the features
    # with f1 and f5 always 0
    feature_names = [f"f{n}" for n in range(1, 6)]
    features = []
    for row in X:
        row = [0] + row.tolist() + [0]
        features.append(dict(zip(feature_names, row)))

    # use a FeatureHasher if we are asked to do feature hashing
    vectorizer = FeatureHasher(n_features=4) if use_feature_hashing else None
    train_fs = FeatureSet("train_sparse", ids, features=features, labels=y, vectorizer=vectorizer)

    # now create the test set with f4 always 0 but nothing else
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=1234567890,
    )
    X = np.abs(X)
    X[np.where(X == 0)] += 1
    ids = [f"EXAMPLE_{n}" for n in range(1, 101)]

    # create a list of dictionaries as the features
    # with f4 always 0
    feature_names = [f"f{n}" for n in range(1, 6)]
    features = []
    for row in X:
        row = row.tolist()
        row = row[:3] + [0] + row[3:]
        features.append(dict(zip(feature_names, row)))

    test_fs = FeatureSet("test_sparse", ids, features=features, labels=y, vectorizer=vectorizer)

    return train_fs, test_fs


def make_digits_data(num_examples=None, test_size=0.2, use_digit_names=False):
    """
    Create train/test featuresets from the digits dataset.

    Parameters
    ----------
    num_examples : int, optional
        Number of total examples to use. 80% of these examples
        will be in the training set and the remaining 20% in
        the test set. If ``None``, it will use all of the
        examples in the dataset.
        Defaults to ``None``.
    test_size : float, optional
        Fraction of ``num_examples`` to use for the test
        featureset. Should be between 0 and 1. If this is 0,
        the training featureset will contain all of the
        examples and the test featureset will be ``None``.
        Defaults to 0.2.
    use_digit_names : bool, optional
        If ``True``, use the names of the digits ("zero", "one", etc.) as the
        labels in the featuresets rather than the integer-valued targets
        (0, 1, etc.).
        Defaults to ``False``.

    Returns
    -------
    fs_tuple : tuple
        A 2-tuple containing the created train and test featuresets.
        The test featureset will be ``None`` if ``test_size`` is 0.

    Raises
    ------
    ValueError
        If ``num_examples`` is greater than the number of available
        examples.

    """
    # load the digits data
    digits = load_digits(as_frame=True)
    df_digits = digits.frame

    # use all examples if ``num_examples`` was ``None``
    num_examples = len(df_digits) if num_examples is None else num_examples

    # raise an exception if the number of desired examples is greater
    # than the number of available examples in the data
    if num_examples > len(df_digits):
        raise ValueError(
            f"invalid value {num_examples} for 'num_examples', "
            f"only {len(df_digits)} examples are available."
        )

    # select the desired number of examples
    row_indices = np.arange(len(df_digits))
    prng = np.random.default_rng(123456789)
    if num_examples < len(df_digits):
        chosen_row_indices = prng.choice(
            row_indices, size=num_examples, replace=False, shuffle=False
        )
    else:
        chosen_row_indices = row_indices

    # now split the chosen indices into train and test indices
    # assuming ``test_size`` > 0
    if test_size > 0:
        splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=123456789)
        train_indices, test_indices = list(splitter.split(chosen_row_indices))[0]
    else:
        train_indices = chosen_row_indices
        test_indices = np.array([])

    # by default, we will use the "target" column of the frame as our labels
    label_column = "target"

    # if we are asked to use digit names instead of integers
    if use_digit_names:
        # create a dictionary mapping integer labels to fake class names
        label_dict = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
        }

        # add a new class column that contains the class names
        df_digits["class"] = df_digits.apply(lambda row: label_dict[row["target"]], axis=1)

        # drop the target column and use "class" as the label column
        df_digits.drop(columns=["target"], axis=1, inplace=True)
        label_column = "class"

    # now subset the training and test rows from the data frame
    df_train = df_digits.iloc[train_indices].copy()
    df_test = df_digits.iloc[test_indices].copy()

    # now create the train and test feature sets from the data frames
    train_fs = FeatureSet.from_data_frame(df_train, "digits_train", labels_column=label_column)
    if len(df_test) > 0:
        test_fs = FeatureSet.from_data_frame(
            df_test, "digits_test", labels_column=label_column, vectorizer=train_fs.vectorizer
        )
    else:
        test_fs = None

    return train_fs, test_fs


def make_california_housing_data(num_examples=None, test_size=0.2):
    """
    Create train/test featuresets from the California housing dataset.

    All columns are standardized into z-scores before creating the
    two feature sets.

    Parameters
    ----------
    num_examples : int, optional
        Number of total examples to use. 80% of these examples
        will be in the training set and the remaining 20% in
        the test set. If ``None``, it will use all of the
        examples in the dataset.
        Defaults to ``None``.
    test_size : float, optional
        Fraction of ``num_examples`` to use for the test
        featureset. Should be between 0 and 1. If this is 0,
        the training featureset will contain all of the
        examples and the test featureset will be ``None``.
        Defaults to 0.2.

    Returns
    -------
    fs_tuple : tuple
        A 2-tuple containing the created train and test featuresets.
        The test featureset will be ``None`` if ``test_size`` is 0.

    Raises
    ------
    ValueError
        If ``num_examples`` is greater than the number of available
        examples.

    """
    # load the housing data
    housing = fetch_california_housing(
        data_home=str(other_dir), download_if_missing=False, as_frame=True
    )
    df_housing = housing.frame

    # standardize all of the values to get them on the same scale
    df_housing = (df_housing - df_housing.mean()) / df_housing.std()

    # use all examples if ``num_examples`` was ``None``
    num_examples = len(df_housing) if num_examples is None else num_examples

    # raise an exception if the number of desired examples is greater
    # than the number of available examples in the data
    if num_examples > len(df_housing):
        raise ValueError(
            f"invalid value {num_examples} for 'num_examples', "
            f"only {len(df_housing)} examples are available."
        )

    # select the desired number of examples
    row_indices = np.arange(len(df_housing))
    prng = np.random.default_rng(123456789)
    if num_examples < len(df_housing):
        chosen_row_indices = prng.choice(
            row_indices, size=num_examples, replace=False, shuffle=False
        )
    else:
        chosen_row_indices = row_indices

    # now split the chosen indices into train and test indices
    # assuming ``test_size`` > 0
    if test_size > 0:
        splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=123456789)
        train_indices, test_indices = list(splitter.split(chosen_row_indices))[0]
    else:
        train_indices = chosen_row_indices
        test_indices = np.array([])

    # now subset the training and test rows from the data frame
    df_train = df_housing.iloc[train_indices].copy()
    df_test = df_housing.iloc[test_indices].copy()

    # now create the train and test feature sets from the data frames
    train_fs = FeatureSet.from_data_frame(df_train, "housing_train", labels_column="MedHouseVal")
    if len(df_test) > 0:
        test_fs = FeatureSet.from_data_frame(
            df_test, "housing_test", labels_column="MedHouseVal", vectorizer=train_fs.vectorizer
        )
    else:
        test_fs = None

    return train_fs, test_fs


def compute_expected_folds_for_cv_testing(featureset, num_folds=10, stratified=True, seed=None):
    """
    Compute the fold IDs expected from SKLL's ``cross_validate()`` methods.

    This function is useful for cross-validation tests where
    we are trying to confirm that that SKLL is correctly computing
    the k-fold cross-validation folds for the given data for cases
    where k is a number. If ``seed`` is specified, the function first
    shuffles the data using that seed and then uses a
    ``StratifiedKFold()`` splitter to split the data into
    ``num_folds``.

    Parameters
    ----------
    featureset : skll.data.FeatureSet
        The given featureset for which to compute the folds.
    num_folds : int, optional
        The number of folds into which to split the given featureset.
        Defaults to 10.
    stratified : bool, optional
        Whether to use stratified k-fold splitting or regular splitting.
        Defaults to ``True`` (stratified).
    seed : int, optional
        The seed to use for the random number generator to
        shuffle the data before splitting. If ``None``, no
        shuffling takes place.

    Returns
    -------
    expected_fold_ids : dict
        A dictionary mapping each ID in the featureset to a fold ID.
        Fold IDs range from 0 to ``num_folds``-1.

    """
    # initialize the return dictionary
    expected_fold_ids = {}

    # initialize an RNG with the given seed, if available
    if seed:
        random_state = np.random.RandomState(seed)

        # shuffle and split the featureset ids, labels and features
        ids, labels, features = sk_shuffle(
            featureset.ids, featureset.labels, featureset.features, random_state=random_state
        )

        featureset = FeatureSet(
            featureset.name, ids, labels=labels, features=features, vectorizer=featureset.vectorizer
        )

    # split the featureset IDs into the given number of folds
    # and save the fold ID for each featureset ID
    kfold = StratifiedKFold(n_splits=num_folds) if stratified else KFold(n_splits=num_folds)
    for fold_num, (_, test_indices) in enumerate(
        kfold.split(featureset.features, featureset.labels)
    ):
        for index in test_indices:
            expected_fold_ids[featureset.ids[index]] = str(fold_num)

    return expected_fold_ids
