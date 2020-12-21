#!/usr/bin/env python
# License: BSD 3 clause
"""
script for computing additional evaluation metrics

:author: Michael Heilman (mheilman@ets.org)
"""

import argparse
import csv
import logging

from numpy.random import RandomState

from skll.data import Reader, safe_float
from skll.metrics import use_score_func
from skll.version import __version__

# Make warnings from built-in warnings module get formatted more nicely
logging.captureWarnings(True)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - '
                           '%(message)s')
logger = logging.getLogger(__name__)


def get_prediction_from_probabilities(classes,
                                      probabilities,
                                      prediction_method,
                                      random_state=1234567890):
    """
    Convert a list of class-probabilities into a class prediction. This function assumes
    that, if the prediction method is 'expected_value', the class labels are integers.

    Parameters
    ----------
    classes: list of str or int
        List of string or integer class names.
    probabilities: list of float
        List of probabilities for the respective classes.
    prediction_method: str
        Indicates how to get a single class prediction from the probabilities.
        Must be one of:
            1. "highest": Selects the class with the highest probability. If
               multiple classes have the same probability, a class is selected randomly.
            2. "expected_value": Calculates an expected value over integer classes and
               rounds to the nearest int.

    random_state: int
        Seed for `RandomState`, used for randomly selecting a class when necessary.

    Returns
    -------
    predicted_class: str or int
        Predicted class.

    """
    prng = RandomState(random_state)
    if prediction_method == 'highest':
        highest_p = max(probabilities)
        best_classes = [classes[i] for i, p in enumerate(probabilities) if p == highest_p]
        if len(best_classes) > 1:
            return prng.choice(best_classes)
        else:
            return best_classes[0]

    elif prediction_method == 'expected_value':
        exp_val = sum([classes[i] * prob for i, prob in enumerate(probabilities)])
        return int(round(exp_val))


def compute_eval_from_predictions(examples_file,
                                  predictions_file,
                                  metric_names,
                                  prediction_method=None):
    """
    Compute evaluation metrics from prediction files after you have run an
    experiment.

    Parameters
    ----------
    examples_file: str
        Path to a SKLL examples file (in .jsonlines or other format).
    predictions_file: str
        Path to a SKLL predictions output TSV file with id and prediction column names.
    metric_names: list of str
        A list of SKLL metric names (e.g., [pearson, unweighted_kappa]).
    prediction_method: str or None
        Indicates how to get a single class prediction from the probabilities. Currently
        supported options are  "highest", which selects the class with the highest
        probability, and "expected_value", which calculates an expected value over
        integer classes and rounds to the nearest int. If predictions file does not
        contain probabilities, this should be set to None.

    Returns
    -------
    dict
        Maps metrics names to corresponding values.

    Raises
    ------
    ValueError
        If the requested prediction method is 'expected_value' but the class names can't
        be converted to ints.
    """

    # read gold standard labels
    data = Reader.for_path(examples_file).read()
    gold = dict(zip(data.ids, data.labels))

    # read predictions
    pred = {}
    with open(predictions_file) as pred_file:
        reader = csv.reader(pred_file, dialect=csv.excel_tab)
        header = next(reader)

        # If there are more than two columns, assume column 0 contains the ids, and
        # columns 1-n contain class probabilities. Convert them to a class prediction
        # using the specified `method`.
        if len(header) > 2:
            classes = [c for c in header[1:] if c]
            if prediction_method is None:
                prediction_method = "highest"
                logger.info("No prediction method specified. Using 'highest'.")
            if prediction_method == 'expected_value':
                try:
                    classes = [int(c) for c in classes]
                except ValueError as e:
                    raise e
            for row in reader:
                probabilities = [safe_float(p) for p in row[1:]]
                prediction = get_prediction_from_probabilities(classes,
                                                               probabilities,
                                                               prediction_method)
                pred[row[0]] = safe_float(prediction)
        else:
            if prediction_method is not None:
                logger.warning("A prediction method was provided, but the "
                               "predictions file doesn't contain "
                               "probabilities. Ignoring prediction method "
                               f"'{prediction_method}'.")

            for row in reader:
                pred[row[0]] = safe_float(row[1])

    # make a sorted list of example ids in order to match up
    # labels and predictions
    if set(gold.keys()) != set(pred.keys()):
        raise ValueError('The example and prediction IDs do not match.')
    example_ids = sorted(gold.keys())

    res = {}
    for metric_name in metric_names:
        score = use_score_func(metric_name,
                               [gold[ex_id] for ex_id in example_ids],
                               [pred[ex_id] for ex_id in example_ids])
        res[metric_name] = score
    return res


def main(argv=None):
    """
    Handles command line arguments and gets things started.

    Parameters
    ----------
    argv: list of str
        List of arguments, as if specified on the command-line. If None, `sys.argv[1:]`
        is used instead.
    """
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Computes evaluation metrics from prediction files after "
                    "you have run an experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('examples_file',
                        help='SKLL input file with labeled examples')
    parser.add_argument('predictions_file',
                        help='file with predictions from SKLL')
    parser.add_argument('metric_names',
                        help='metrics to compute',
                        nargs='+')
    parser.add_argument('--method', '-m',
                        help="How to generate a prediction from the class "
                             "probabilities. Supported methods are 'highest' "
                             "(default) and 'expected_value' (only works with"
                             " integer classes).")
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')
    args = parser.parse_args(argv)

    supported_prediction_methods = {"highest", "expected_value"}
    if (args.method is not None) and (args.method not in supported_prediction_methods):
        raise KeyError(f"Unrecognized prediction method '{args.method}'. "
                       "Supported methods are 'highest' and "
                       "'expected_value'.")

    scores = compute_eval_from_predictions(args.examples_file,
                                           args.predictions_file,
                                           args.metric_names,
                                           args.method)

    for metric_name in args.metric_names:
        print(f"{scores[metric_name]}\t{metric_name}\t{args.predictions_file}")


if __name__ == '__main__':
    main()
