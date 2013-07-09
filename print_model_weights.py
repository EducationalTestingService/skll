#!/usr/bin/env python
'''
Simple script for printing out model weights.
'''

from __future__ import print_function, unicode_literals

import argparse
import sys

from skll import Learner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prints out the weights of a \
                                                  given model.",
                                     conflict_handler='resolve',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_file', help='model file to load')
    parser.add_argument('--k', help='number of top features to print (0 for all)',
                        type=int, default=50)
    args = parser.parse_args()

    k = args.k if args.k > 0 else None

    learner = Learner()
    learner.load_model(args.model_file)
    weights = learner.model_params

    print("number of nonzero features:", len(weights), file=sys.stderr)

    for feat, val in sorted(weights.items(), key=lambda x: -abs(x[1]))[:k]:
        print("{:.12f}\t{}".format(val, feat))
