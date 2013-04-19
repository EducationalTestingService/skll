#!/usr/bin/env python
'''
Simple script for printing out model weights.
'''

from __future__ import print_function, unicode_literals

import argparse
import classifier
import sys



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prints out the weights of a \
                                                  given model.",
                                     conflict_handler='resolve',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_file', help='model file to load')
    parser.add_argument('--k', help='number of top features to print',
                        type=int, default=50)
    args = parser.parse_args()

    clf = classifier.Classifier()
    clf.load_model(args.model_file)
    weights = clf.get_model_params()
   
    print("number of nonzero features:", len(weights), file=sys.stderr)

    for feat, val in sorted(weights.items(), key=lambda x: -abs(x[1]))[:args.k]:
        print("{:.12f}\t{}".format(val, feat))
