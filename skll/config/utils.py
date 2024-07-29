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
from pathlib import Path
from typing import Iterable, List, Union

from ruamel.yaml import YAML

from skll.types import FoldMapping, PathOrStr


def fix_json(json_string: str) -> str:
    """
    Fix incorrectly formatted quotes and capitalized booleans in JSON string.

    Parameters
    ----------
    json_string : str
        A JSON-style string.

    Returns
    -------
    str
        The normalized JSON string.

    """
    json_string = json_string.replace("True", "true")
    json_string = json_string.replace("False", "false")
    json_string = json_string.replace("'", '"')
    return json_string


def load_cv_folds(folds_file: PathOrStr, ids_to_floats=False) -> FoldMapping:
    """
    Load cross-validation folds from a CSV file.

    The CSV file must contain two columns: example ID and fold ID (and a header).

    Parameters
    ----------
    folds_file : :class:`skll.types.PathOrStr`
        The path to a folds file to read.

    ids_to_floats : bool, default=False
        Whether to convert IDs to floats.

    Returns
    -------
    :class:`skll.types.FoldMapping`
        Dictionary with example IDs as the keys and fold IDs as the values.
        If `ids_to_floats` is set to `True`, the example IDs are floats but
        otherwise they are strings.

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
            example_id: Union[float, str] = row[0]
            fold_id: str = row[1]
            if ids_to_floats:
                try:
                    example_id = float(example_id)
                except ValueError:
                    raise ValueError(
                        "You set ids_to_floats to true, but ID "
                        f"{row[0]} could not be converted to "
                        "float"
                    )
            res[example_id] = fold_id

    return res


def locate_file(file_path: PathOrStr, config_dir: PathOrStr) -> str:
    """
    Locate a file, given a file path and configuration directory.

    Parameters
    ----------
    file_path : :class:`skll.types.PathOrStr`
        The file to locate. Path may be absolute or relative.

    config_dir : :class:`skll.types.PathOrStr`
        The path to the configuration file directory.

    Returns
    -------
    path_to_check : str
        The normalized absolute path, if it exists.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    """
    if not file_path:
        return ""

    # convert to Path objects
    file_path = Path(file_path)
    config_dir = Path(config_dir)

    config_relative_path = config_dir / file_path
    path_to_check = file_path if file_path.is_absolute() else config_relative_path.resolve()
    if not path_to_check.exists():
        raise FileNotFoundError(
            errno.ENOENT, f"File {path_to_check} does not exist", str(path_to_check)
        )
    else:
        return str(path_to_check)


def _munge_featureset_name(name_or_list: Union[Iterable, str]) -> str:
    """
    Create a munged name for the featureset.

    Join features in ``featureset`` with a '+' if ``featureset`` is not a string.
    Otherwise, returns ``featureset``.

    Parameters
    ----------
    name_or_list : Union[Iterable, str]
        A featureset name or name components in a list.

    Returns
    -------
    res : str
        name components joined with '+' if input is a list or the name itself.

    """
    if isinstance(name_or_list, str):
        return name_or_list

    res = "+".join(sorted(name_or_list))
    return res


def _parse_and_validate_metrics(metrics: str, option_name: str, logger=None) -> List[str]:
    """
    Parse and validate string containing list of metrics.

    Given a string containing a list of metrics, this function
    parses that string into a list and validates some specific
    metric names.

    Parameters
    ----------
    metrics : str
        A string containing a list of metrics.

    option_name : str
        The name of the option with which the metrics are associated.

    logger : Optional[logging.Logger], default=None
        A logging object.

    Returns
    -------
    metrics : List[str]
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
    yaml = YAML(typ="safe", pure=True)
    metrics = yaml.load(fix_json(metrics))
    if not isinstance(metrics, list):
        raise TypeError(f"{option_name} should be a list, not a " f"{type(metrics)}.")

    # `mean_squared_error` is no longer supported.
    # It has been replaced by `neg_mean_squared_error`
    if "mean_squared_error" in metrics:
        raise ValueError(
            'The metric "mean_squared_error" is no longer '
            "supported. please use the metric "
            '"neg_mean_squared_error" instead.'
        )

    return metrics
