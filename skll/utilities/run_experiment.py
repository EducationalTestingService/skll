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
Runs a bunch of scikit-learn jobs in parallel on the cluster given a
config file.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
'''


from __future__ import print_function, unicode_literals

import argparse
import logging
import sys
from functools import partial

from skll.experiments import run_configuration
from skll.version import __version__


def main(argv=None):
    '''
    Handles command line arguments and gets things started.

    :param argv: List of arguments, as if specified on the command-line.
                 If None, ``sys.argv[1:]`` is used instead.
    :type argv: list of str
    '''
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Runs the scikit-learn experiments in a given config file.\
                     If Grid Map is installed, jobs will automatically be \
                     created and run on a DRMAA-compatible cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('config_file',
                        help='Configuration file describing the sklearn task\
                              to run.',
                        nargs='+')
    parser.add_argument('-a', '--ablation',
                        help='Runs an ablation study where repeated \
                              experiments are conducted where the specified \
                              number of features in each featureset in the \
                              configuration file are held out.',
                        type=int, default=0,
                        metavar='NUM_FEATURES')
    parser.add_argument('-A', '--ablation_all',
                        help='Runs an ablation study where repeated \
                              experiments are conducted with all combinations \
                              of features in each featureset in the \
                              configuration file. Overrides --ablation \
                              setting.',
                        action='store_true')
    parser.add_argument('-k', '--keep_models',
                        help='If trained models already exists, re-use them\
                              instead of overwriting them.',
                        action='store_true')
    parser.add_argument('-l', '--local',
                        help='Do not use the Grid Engine for running jobs and\
                              just run everything sequentially on the local \
                              machine. This is for debugging.',
                        action='store_true')
    parser.add_argument('-m', '--machines',
                        help="comma-separated list of machines to add to\
                              gridmap's whitelist (if not specified, all\
                              available machines are used). Note that full \
                              names must be specified, e.g., \
                              \"nlp.research.ets.org\"",
                        default=None)
    parser.add_argument('-q', '--queue',
                        help="Use this queue for gridmap.",
                        default='all.q')
    parser.add_argument('-r', '--resume',
                        help='If result files already exist for an experiment, \
                              do not overwrite them. This is very useful when \
                              doing a large ablation experiment and part of it \
                              crashes.',
                        action='store_true')
    parser.add_argument('-v', '--verbose',
                        help='Print more status information. For every ' +
                             'additional time this flag is specified, ' +
                             'output gets more verbose.',
                        default=0, action='count')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    # Logging levels are really integer multiples of 10, so convert verbose flag
    log_level = max(logging.WARNING - (args.verbose * 10), logging.DEBUG)
    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=log_level)

    machines = None
    if args.machines:
        machines = args.machines.split(',')

    # Create partial function to map onto list of config files
    ablation = args.ablation
    if args.ablation_all:
        ablation = None

    # Run each config file sequentially
    for config_file in args.config_file:
        run_configuration(config_file, local=args.local, overwrite=not
                          args.keep_models, queue=args.queue, hosts=machines,
                          ablation=ablation, resume=args.resume)


if __name__ == '__main__':
    main()
