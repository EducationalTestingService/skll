#!/usr/bin/env python
# License: BSD 3 clause
"""
Simple script for printing out model weights.

:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import argparse
import logging
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from skll.learner import Learner
from skll.version import __version__


def main(argv=None):
    """
    Handles command line arguments and gets things started.

    Parameters
    ----------
    argv : list of str
        List of arguments, as if specified on the command-line.
        If None, ``sys.argv[1:]`` is used instead.
    """

    parser = argparse.ArgumentParser(description="Prints out the weights of a"
                                                 " given model.",
                                     conflict_handler='resolve',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_file', help='model file to load')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--k',
                       help='number of top features to print (0 for all)',
                       type=int, default=50)
    group.add_argument("--sort_by_labels", '-s', action='store_true',
                       default=False, help="order the features by classes")
    parser.add_argument('--sign',
                        choices=['positive', 'negative', 'all'],
                        default='all',
                        help='show only positive, only negative or all weights')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - '
                               '%(message)s')

    k = args.k if args.k > 0 else None

    learner = Learner.from_file(args.model_file)
    (weights, intercept) = learner.model_params

    multiclass = False
    model = learner._model
    if (isinstance(model, LinearSVC)
        or (isinstance(model, LogisticRegression)
            and len(learner.label_list) > 2)
        or (isinstance(model, SVC)
            and model.kernel == 'linear')):
        multiclass = True
    weight_items = weights.items()
    if args.sign == 'positive':
        weight_items = (x for x in weight_items if x[1] > 0)
    elif args.sign == 'negative':
        weight_items = (x for x in weight_items if x[1] < 0)

    if intercept is not None:
        # subclass of LinearModel
        if '_intercept_' in intercept:
            # Some learners (e.g. LinearSVR) may return an array of intercepts but
            # sometimes that array is of length 1 so we don't need to print that
            # as an array/list. First, let's normalize these cases.
            model_intercepts = intercept['_intercept_']
            intercept_is_array = isinstance(model_intercepts, np.ndarray)
            num_intercepts = len(model_intercepts) if intercept_is_array else 1
            if intercept_is_array and num_intercepts == 1:
                model_intercepts = model_intercepts[0]
                intercept_is_array = False

            # now print out the intercepts
            print(f"intercept = {model_intercepts:.12f}")
        else:
            print("== intercept values ==")
            for (label, val) in intercept.items():
                print(f"{val:.12f}\t{label}")
        print()

    print("Number of nonzero features:", len(weights), file=sys.stderr)
    weight_by_class = defaultdict(dict)
    if multiclass and args.sort_by_labels:
        for label_feature, weight in weight_items:
            label, feature = label_feature.split()
            weight_by_class[label][feature] = weight
        for label in sorted(weight_by_class):
            for feat, val in sorted(weight_by_class[label].items(), key=lambda x: -abs(x[1])):
                print(f"{val:.12f}\t{label}\t{feat}")
    else:
        for feat, val in sorted(weight_items, key=lambda x: -abs(x[1]))[:k]:
            print(f"{val:.12f}\t{feat}")


if __name__ == '__main__':
    main()
