#!/usr/bin/env python
# License: BSD 3 clause
"""
Compute additional evaluation metrics from predictions.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from skll.data import Reader, safe_float
from skll.metrics import use_score_func
from skll.types import PathOrStr
from skll.version import __version__

# Make warnings from built-in warnings module get formatted more nicely
logging.captureWarnings(True)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - " "%(message)s")
logger = logging.getLogger(__name__)


def get_prediction_from_probabilities(
    classes: Union[List[int], List[str]],
    probabilities: List[float],
    prediction_method: str,
    random_state: int = 1234567890,
) -> Union[int, str]:
    """
    Convert a list of class-probabilities into a class prediction.

    If the prediction method is ``"expected_value"``, the class labels must be integers.

    Parameters
    ----------
    classes: Union[List[int], List[str]]
        List of string or integer class names.
    probabilities: List[float]
        List of probabilities for the respective classes.
    prediction_method: str
        Indicates how to get a single class prediction from the probabilities.
        Must be one of:
            1. ``"highest"``: Selects the class with the highest probability. If
               multiple classes have the same probability, a class is selected randomly.
            2. ``"expected_value"``: Calculates an expected value over integer classes and
               rounds to the nearest int.
    random_state: int, default=1234567890
        Seed for ``numpy.random.RandomState``, used for randomly selecting a class
        when necessary.

    Returns
    -------
    Union[int, str]
        The predicted class.

    Raises
    ------
    ValueError
        If ``classes`` does not contain integers and ``prediction_method`` is
        ``"expected_value"``.

    """
    prng = np.random.RandomState(random_state)
    if prediction_method == "highest":
        highest_p = max(probabilities)
        best_classes = [classes[i] for i, p in enumerate(probabilities) if p == highest_p]
        if len(best_classes) > 1:
            ans = prng.choice(best_classes)
        else:
            ans = best_classes[0]
    elif prediction_method == "expected_value":
        exp_val = 0.0
        for class_, probability in zip(classes, probabilities):
            exp_val += class_ * probability  # type: ignore
        ans = int(round(exp_val))

    return ans


def compute_eval_from_predictions(
    examples_file: PathOrStr,
    predictions_file: PathOrStr,
    metric_names: List[str],
    prediction_method: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics from prediction files after running an experiment.

    Parameters
    ----------
    examples_file: :class:`skll.types.PathOrStr`
        Path to a SKLL examples file (in .jsonlines or other format).
    predictions_file: :class:`skll.types.PathOrStr`
        Path to a SKLL predictions output TSV file with id and prediction column names.
    metric_names: List[str]
        A list of SKLL metric names (e.g., ``["pearson", "unweighted_kappa"]``).
    prediction_method: Optional[str], default=None
        Indicates how to get a single class prediction from the probabilities.
        Currently supported options are ``"highest"``, which selects the class
        with the highest probability, and ``"expected_value"``, which calculates
        an expected value over integer classes and rounds to the nearest int.
        If predictions file does not contain probabilities, this should be set
        to ``None``.

    Returns
    -------
    Dict[str, float]
        Mapping of metrics names to corresponding values.

    Raises
    ------
    ValueError
        If the requested prediction method is ``"expected_value"`` but
        the class names can't be converted to ints.

    """
    # convert the examples file and predictions file to a Path
    examples_file = Path(examples_file)
    predictions_file = Path(predictions_file)

    # read gold standard labels
    data = Reader.for_path(examples_file).read()
    if data.labels is not None:
        gold = dict(zip(data.ids, data.labels))

    # read predictions
    pred = {}
    with open(predictions_file) as pred_file:
        reader = csv.reader(pred_file, dialect=csv.excel_tab)
        header = next(reader)

        # If there are more than two columns, assume column 0 contains the ids, and
        # columns 1-n contain class probabilities. Convert them to a class prediction
        # using the specified `method`.
        classes: Union[List[int], List[str]]
        if len(header) > 2:
            classes = [c for c in header[1:] if c]
            if prediction_method is None:
                prediction_method = "highest"
                logger.info("No prediction method specified. Using 'highest'.")
            if prediction_method == "expected_value":
                try:
                    classes = [int(c) for c in classes]
                except ValueError as e:
                    raise e
            for row in reader:
                probabilities = [safe_float(p) for p in row[1:]]
                prediction = get_prediction_from_probabilities(
                    classes, probabilities, prediction_method  # type: ignore
                )
                pred[row[0]] = safe_float(prediction)
        else:
            if prediction_method is not None:
                logger.warning(
                    "A prediction method was provided, but the "
                    "predictions file doesn't contain "
                    "probabilities. Ignoring prediction method "
                    f"'{prediction_method}'."
                )

            for row in reader:
                pred[row[0]] = safe_float(row[1])

    # make a sorted list of example ids in order to match up
    # labels and predictions
    if set(gold.keys()) != set(pred.keys()):
        raise ValueError("The example and prediction IDs do not match.")
    example_ids = sorted(gold.keys())

    res = {}
    for metric_name in metric_names:
        score = use_score_func(
            metric_name,
            np.array([gold[ex_id] for ex_id in example_ids]),
            np.array([pred[ex_id] for ex_id in example_ids]),
        )
        res[metric_name] = score
    return res


def main(argv: Optional[List[str]] = None) -> None:
    """
    Handle command line arguments and get things started.

    Parameters
    ----------
    argv: Optional[List[str]], default=None
        List of arguments, as if specified on the command-line. If ``None``,
        then ``sys.argv[1:]`` is used instead.

    """
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Computes evaluation metrics from prediction files after "
        "you have run an experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("examples_file", help="SKLL input file with labeled examples")
    parser.add_argument("predictions_file", help="file with predictions from SKLL")
    parser.add_argument("metric_names", help="metrics to compute", nargs="+")
    parser.add_argument(
        "--method",
        "-m",
        help="How to generate a prediction from the class "
        "probabilities. Supported methods are 'highest' "
        "(default) and 'expected_value' (only works with"
        " integer classes).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    supported_prediction_methods = {"highest", "expected_value"}
    if (args.method is not None) and (args.method not in supported_prediction_methods):
        raise KeyError(
            f"Unrecognized prediction method '{args.method}'. "
            "Supported methods are 'highest' and "
            "'expected_value'."
        )

    scores = compute_eval_from_predictions(
        args.examples_file, args.predictions_file, args.metric_names, args.method
    )

    for metric_name in args.metric_names:
        print(f"{scores[metric_name]}\t{metric_name}\t{args.predictions_file}")


if __name__ == "__main__":
    main()
