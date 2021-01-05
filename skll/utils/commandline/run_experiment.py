#!/usr/bin/env python
# License: BSD 3 clause
"""
Runs a bunch of scikit-learn jobs in parallel on the cluster given a
config file.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
"""


import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from skll.experiments import run_configuration
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
    parser = ArgumentParser(description='Runs the scikit-learn experiments '
                                        'in a given config file. If Grid Map '
                                        'is installed, jobs will automatically'
                                        ' be created and run on a '
                                        'DRMAA-compatible cluster.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('config_file',
                        help='Configuration file describing the task to run.',
                        nargs='+')
    parser.add_argument('-a',
                        '--ablation',
                        help='Runs an ablation study where repeated '
                             'experiments are conducted where the specified '
                             'number of features in each featureset in the '
                             'configuration file are held out.',
                        type=int,
                        default=0,
                        metavar='NUM_FEATURES')
    parser.add_argument('-A',
                        '--ablation_all',
                        help='Runs an ablation study where repeated '
                             'experiments are conducted with all combinations '
                             'of features in each featureset in the '
                             'configuration file. Overrides --ablation '
                             'setting.',
                        action='store_true')
    parser.add_argument('-k',
                        '--keep_models',
                        help='If trained models already exists, re-use them '
                             'instead of overwriting them.',
                        action='store_true')
    parser.add_argument('-l',
                        '--local',
                        help='Do not use the Grid Engine for running jobs and'
                             ' just run everything sequentially on the local '
                             'machine. ',
                        action='store_true')
    parser.add_argument('-m',
                        '--machines',
                        help='comma-separated list of machines to add to the '
                             'gridmap whitelist (if not specified, all '
                             'available machines are used). Note that full '
                             'names must be specified, e.g., '
                             '"nlp.research.ets.org"',
                        default=None)
    parser.add_argument('-q',
                        '--queue',
                        help="Use this queue for gridmap.",
                        default='all.q')
    parser.add_argument('-r',
                        '--resume',
                        help='If result files already exist for an experiment'
                             ' do not overwrite them. This is very useful '
                             'when doing a large ablation experiment and part'
                             ' of it crashes.',
                        action='store_true')
    parser.add_argument('-v',
                        '--verbose',
                        help='Include debug information in the logging output.',
                        default=False,
                        action='store_true')
    parser.add_argument('--version',
                        action='version',
                        version=f'%(prog)s {__version__}')
    args = parser.parse_args(argv)

    # Default logging level is INFO unless we are being verbose
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - '
                               '%(message)s',
                        level=log_level)

    machines = None
    if args.machines:
        machines = args.machines.split(',')

    ablation = args.ablation
    if args.ablation_all:
        ablation = None

    # Run each config file sequentially
    for config_file in args.config_file:
        run_configuration(config_file,
                          local=args.local,
                          overwrite=not args.keep_models,
                          queue=args.queue,
                          hosts=machines,
                          ablation=ablation,
                          resume=args.resume,
                          log_level=log_level)


if __name__ == '__main__':
    main()
