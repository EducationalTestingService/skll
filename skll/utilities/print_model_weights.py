#!/usr/bin/env python
# License: BSD 3 clause
"""
Simple script for printing out model weights.

:author: Michael Heilman (mheilman@ets.org)
:organization: ETS
"""

from __future__ import print_function, unicode_literals

import argparse
import logging
import sys

from six import iteritems
import numpy as np

from skll import Learner
from skll.version import __version__


def main(argv=None):
    """
    Handles command line arguments and gets things started.

    :param argv: List of arguments, as if specified on the command-line.
                 If None, ``sys.argv[1:]`` is used instead.
    :type argv: list of str
    """
    parser = argparse.ArgumentParser(description="Prints out the weights of a \
                                                  given model.",
                                     conflict_handler='resolve',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_file', help='model file to load')
    parser.add_argument('--k',
                        help='number of top features to print (0 for all)',
                        type=int, default=50)
    parser.add_argument('--sign',
                        choices=['positive', 'negative', 'all'],
                        default='all',
                        help='show only positive, only negative, ' +
                             'or all weights')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'))

    k = args.k if args.k > 0 else None

    learner = Learner.from_file(args.model_file)
    (weights, intercept) = learner.model_params

    weight_items = iteritems(weights)
    if args.sign == 'positive':
        weight_items = (x for x in weight_items if x[1] > 0)
    elif args.sign == 'negative':
        weight_items = (x for x in weight_items if x[1] < 0)

    if intercept is not None:
        # subclass of LinearModel
        if '_intercept_' in intercept:
            # Some learners (e.g. LinearSVR) may return a list of intercepts
            if isinstance(intercept['_intercept_'], np.ndarray):
                intercept_list = ["%.12f" % i for i in intercept['_intercept_']]
                print("intercept = {}".format(intercept_list))
            else:
                print("intercept = {:.12f}".format(intercept['_intercept_']))
        else:
            print("== intercept values ==")
            for (label, val) in intercept.items():
                print("{:.12f}\t{}".format(val, label))
        print()

    print("Number of nonzero features:", len(weights), file=sys.stderr)
    for feat, val in sorted(weight_items, key=lambda x: -abs(x[1]))[:k]:
        print("{:.12f}\t{}".format(val, feat))


if __name__ == '__main__':
    main()
