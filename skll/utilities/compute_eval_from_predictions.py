#!/usr/bin/env python
# License: BSD 3 clause
"""
script for computing additional evaluation metrics

:author: Michael Heilman (mheilman@ets.org)
"""
from __future__ import print_function, unicode_literals

import argparse
import csv
import logging
import random

from skll.data import Reader, safe_float
from skll.metrics import use_score_func
from skll.version import __version__


def get_prediction_from_probabilities(classes, probs, prediction_method):
    """
    Convert a list of class-probabilities into a class prediction. This function assumes
    that, if the prediction method is 'expected_value', the class labels are integers.

    Parameters
    ----------
    classes: list
        List of str or int class names.

    probs: list of float
        Probabilities for respective classes.

    prediction_method: str
        Indicates how to get a single class prediction from the probabilities. Currently
        supported options are  "highest", which selects the class with the highest
        probability, and "expected_value", which calculates an expected value over
        integer classes and rounds to the nearest int.

    Returns
    -------
    str or int
        Predicted class.

    """
    if prediction_method == 'highest':
        if len(set(probs)) == 1:  # probabilities are all equal
            return random.choice(classes)
        else:
            return classes[probs.index(max(probs))]
    elif prediction_method == 'expected_value':
        exp_val = sum([classes[i] * prob for i, prob in enumerate(probs)])
        return int(round(exp_val))


def compute_eval_from_predictions(examples_file, predictions_file,
                                  metric_names, probabilities=False,
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

    probabilities: boolean
        Indicates whether the predictions file contains probabilities instead of predictions.

    prediction_method: str
        Indicates how to get a single class prediction from the probabilities. Currently
        supported options are  "highest", which selects the class with the highest
        probability, and "expected_value", which calculates an expected value over
        integer classes and rounds to the nearest int.

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

        # If prediction file contains probabilities instead of predictions, convert
        # class probabilities to a class prediction using the specified
        # `prediction_mathod`.
        if probabilities:
            classes = [c for c in header[1:] if c]
            if prediction_method == 'expected_value':
                if any(not c.isdigit() for c in classes):
                    raise ValueError("Cannot calculate expected value with non-integer "
                                     "classes.")
                else:
                    classes = [int(c) for c in classes]

            for row in reader:
                probs = [safe_float(p) for p in row[1:]]
                prediction = get_prediction_from_probabilities(classes, probs,
                                                               prediction_method)
                pred[row[0]] = safe_float(prediction)

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

    :param argv: List of arguments, as if specified on the command-line.
                 If None, ``sys.argv[1:]`` is used instead.
    :type argv: list of str
    """
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Computes evaluation metrics from prediction files after \
                     you have run an experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('examples_file',
                        help='SKLL input file with labeled examples')
    parser.add_argument('predictions_file',
                        help='file with predictions from SKLL')
    parser.add_argument('metric_names',
                        help='metrics to compute',
                        nargs='+')
    parser.add_argument('--probability', '-p',
                        help="Whether the predictions file contains probabilities.",
                        action='store_true')
    parser.add_argument('--prediction_method', '-P',
                        help="How to generate a prediction from the class probabilities "
                             "(only use with -p flag). Supported methods: 'choose_best' "
                             "(default) and 'expected_value' (only works with integer "
                             "classes", default='highest')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'))

    scores = compute_eval_from_predictions(args.examples_file,
                                           args.predictions_file,
                                           args.metric_names,
                                           args.probability,
                                           args.prediction_method)

    for metric_name in args.metric_names:
        print("{}\t{}\t{}".format(scores[metric_name],
                                  metric_name, args.predictions_file))


if __name__ == '__main__':
    main()
