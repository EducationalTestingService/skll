#!/usr/bin/env python
# License: BSD 3 clause
'''
Simple script for printing out model weights.

:author: Michael Heilman (mheilman@ets.org)
:organization: ETS
'''

from __future__ import print_function, unicode_literals

import argparse
import logging
import sys

from six import iteritems

from skll import Learner
from skll.version import __version__


def main():
    '''
    Handles command line arguments and gets things started.
    '''
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
    args = parser.parse_args()

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'))

    k = args.k if args.k > 0 else None

    learner = Learner.from_file(args.model_file)
    weights = learner.model_params

    print("Number of nonzero features:", len(weights), file=sys.stderr)

    weight_items = iteritems(weights)
    if args.sign == 'positive':
        weight_items = (x for x in weight_items if x[1] > 0)
    elif args.sign == 'negative':
        weight_items = (x for x in weight_items if x[1] < 0)

    for feat, val in sorted(weight_items, key=lambda x: -abs(x[1]))[:k]:
        print("{:.12f}\t{}".format(val, feat))


if __name__ == '__main__':
    main()