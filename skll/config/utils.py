# License: BSD 3 clause
"""
Utility classes and functions to parse SKLL configuration files.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
"""

import csv
import errno
import logging
from os.path import exists, isabs, join, normpath

import ruamel.yaml as yaml


def fix_json(json_string):
    """
    Fixes incorrectly formatted quotes and capitalized booleans in the given
    JSON string.

    Parameters
    ----------
    json_string : str
        A JSON-style string.

    Returns
    -------
    json_string : str
        The normalized JSON string.
    """
    json_string = json_string.replace('True', 'true')
    json_string = json_string.replace('False', 'false')
    json_string = json_string.replace("'", '"')
    return json_string


def load_cv_folds(folds_file, ids_to_floats=False):
    """
    Loads cross-validation folds from a CSV file with two columns for example
    ID and fold ID (and a header).

    Parameters
    ----------
    folds_file : str
        The path to a folds file to read.

    ids_to_floats : bool, default=False
        Whether to convert IDs to floats.

    Returns
    -------
    res : dict
        A dictionary with example IDs as the keys and fold IDs as the values.

    Raises
    ------
    ValueError
        If example IDs cannot be converted to floats and `ids_to_floats` is `True`.
    """
    with open(folds_file) as f:
        reader = csv.reader(f)
        next(reader)  # discard the header
        res = {}
        for row in reader:
            if ids_to_floats:
                try:
                    row[0] = float(row[0])
                except ValueError:
                    raise ValueError('You set ids_to_floats to true, but ID '
                                     f'{row[0]} could not be converted to '
                                     'float')
            res[row[0]] = row[1]

    return res


def locate_file(file_path, config_dir):
    """
    Locate a file, given a file path and configuration directory.

    Parameters
    ----------
    file_path : str
        The file to locate. Path may be absolute or relative.

    config_dir : str
        The path to the configuration file directory.

    Returns
    -------
    path_to_check : str
        The normalized absolute path, if it exists.

    Raises
    ------
    IOError
        If the file does not exist.
    """
    if not file_path:
        return ''
    path_to_check = file_path if isabs(file_path) else normpath(join(config_dir,
                                                                     file_path))
    ans = exists(path_to_check)
    if not ans:
        raise IOError(errno.ENOENT, "File does not exist", path_to_check)
    else:
        return path_to_check


def _munge_featureset_name(featureset):
    """
    Joins features in ``featureset`` with a '+' if ``featureset`` is not a string.
    Otherwise, returns ``featureset``.

    Parameters
    ----------
    featureset : skll.data.FeatureSet
        A SKLL ``FeatureSet`` object.

    Returns
    -------
    res : str
        ``featureset`` names joined with '+', if ``featureset`` is not a string.
    """
    if isinstance(featureset, str):
        return featureset

    res = '+'.join(sorted(featureset))
    return res


def _parse_and_validate_metrics(metrics, option_name, logger=None):
    """
    Given a string containing a list of metrics, this function
    parses that string into a list and validates some specific
    metric names.

    Parameters
    ----------
    metrics : str
        A string containing a list of metrics.

    option_name : str
        The name of the option with which the metrics are associated.

    logger : logging.Logger, default=None
        A logging object.

    Returns
    -------
    metrics : list of str
        A list of metrics for the given option.

    Raises
    ------
    TypeError
        If the given string cannot be converted to a list.

    ValueError
        If "mean_squared_error" is specified as a metric.
    """

    # create a logger if one was not passed in
    if not logger:
        logger = logging.getLogger(__name__)

    # make sure the given metrics data type is a list
    # and parse it correctly
    metrics = yaml.safe_load(fix_json(metrics))
    if not isinstance(metrics, list):
        raise TypeError(f"{option_name} should be a list, not a "
                        f"{type(metrics)}.")

    # `mean_squared_error` is no longer supported.
    # It has been replaced by `neg_mean_squared_error`
    if 'mean_squared_error' in metrics:
        raise ValueError("The metric \"mean_squared_error\" is no longer "
                         "supported. please use the metric "
                         "\"neg_mean_squared_error\" instead.")

    return metrics
