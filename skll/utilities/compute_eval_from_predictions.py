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

from skll.data import Reader, safe_float
from skll.metrics import use_score_func
from skll.version import __version__


def compute_eval_from_predictions(examples_file, predictions_file,
                                  metric_names):
    """
    Compute evaluation metrics from prediction files after you have run an
    experiment.

    :param examples_file: a SKLL examples file (in .jsonlines or other format)
    :param predictions_file: a SKLL predictions output TSV file with id
                             and prediction column names
    :param metric_names: a list of SKLL metric names
                         (e.g., [pearson, unweighted_kappa])

    :returns: a dictionary from metrics names to values
    """

    # read gold standard labels
    data = Reader.for_path(examples_file).read()
    gold = dict(zip(data.ids, data.labels))

    # read predictions
    pred = {}
    with open(predictions_file) as pred_file:
        reader = csv.reader(pred_file, dialect=csv.excel_tab)
        next(reader)  # skip header
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
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'))

    scores = compute_eval_from_predictions(args.examples_file,
                                           args.predictions_file,
                                           args.metric_names)

    for metric_name in args.metric_names:
        print("{}\t{}\t{}".format(scores[metric_name],
                                  metric_name, args.predictions_file))


if __name__ == '__main__':
    main()
