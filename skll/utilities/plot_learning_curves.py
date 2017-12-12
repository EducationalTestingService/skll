#!/usr/bin/env python
# License: BSD 3 clause
"""
A Helper script to generate learning plots from the learning curve output TSV file.

This is necessary in scenarios where the plots were not generated as part of the original
learning curve experiment, e.g. the experiment was run (a) on a remote server where plots
may not have been generated either due to a crash or incorrect setting of the DISPLAY
environment variable, or (b) in an environment where seaborn and pandas were't installed.

In these cases, the summary file should always be generated and this script can then be used
to generate the plots later.

:author: Nitin Madnani
:organization: ETS
"""

from __future__ import print_function, unicode_literals

import argparse
import logging
import sys

from os import makedirs
from os.path import basename, exists

from skll.experiments import _HAVE_PANDAS, _HAVE_SEABORN
from skll.experiments import _generate_learning_curve_plots
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

    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Generates learning curve plots from the learning curve TSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('tsv_file',
                        help='Learning curve TSV output file.')
    parser.add_argument('output_dir',
                        help='Directory to store the learning curve plots.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'))

    # make sure that the input TSV file that's being passed exists
    if not exists(args.tsv_file):
        logging.error("Error: the given file {} does not exist.".format(args.tsv_file))
        sys.exit(1)

    # create the output directory if it doesn't already exist
    if not exists(args.output_dir):
        makedirs(args.output_dir)

    # check that we have pandas and seaborn available
    if not (_HAVE_PANDAS and _HAVE_SEABORN):
        logging.error("Error: need pandas and seaborn to generate learning curve plots.")
        sys.exit(1)

    # get the experiment name from the learning curve TSV file
    # output_file_name = experiment_name + '_summary.tsv'
    experiment_name = basename(args.tsv_file).rstrip('_summary.tsv')
    logging.info("Generating learning curve(s)")
    _generate_learning_curve_plots(experiment_name, args.output_dir, args.tsv_file)

if __name__ == '__main__':
    main()
