#!/usr/bin/env python

# Copyright (C) 2012-2013 Educational Testing Service

# This file is part of SciKit-Learn Lab.

# SciKit-Learn Lab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SciKit-Learn Lab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with SciKit-Learn Lab.  If not, see <http://www.gnu.org/licenses/>.

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

    for feat, val in sorted(iteritems(weights), key=lambda x: -abs(x[1]))[:k]:
        print("{:.12f}\t{}".format(val, feat))


if __name__ == '__main__':
    main()