#!/usr/bin/env python
# License: BSD 3 clause
"""
Generate learning plots from the learning curve output TSV file.

This is necessary in scenarios where the plots were not generated as part of
the original learning curve experiment, e.g. the experiment was run on a remote
server where plots may not have been generated either due to a crash or
incorrect setting of the DISPLAY environment variable.

In these cases, the summary file should always be generated and this script can
then be used to generate the plots later.

:author: Nitin Madnani
:organization: ETS
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from skll.experiments import generate_learning_curve_plots
from skll.version import __version__


def main(argv: Optional[List[str]] = None) -> None:
    """
    Handle command line arguments and gets things started.

    Parameters
    ----------
    argv : Optional[List[str]], default=None
        List of arguments, as if specified on the command-line.
        If ``None``, ``sys.argv[1:]`` is used instead.

    """
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Generates learning curve plots from the learning curve " "TSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
    )
    parser.add_argument("tsv_file", help="Learning curve TSV output file.")
    parser.add_argument("output_dir", help="Directory to store the learning curve plots.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - " "%(message)s")

    # convert to Path objects
    tsv_file = Path(args.tsv_file)
    output_dir = Path(args.output_dir)

    # make sure that the input TSV file that's being passed exists
    if not tsv_file.exists():
        logging.error(f"Error: the given file {args.tsv_file} does not " "exist.")
        sys.exit(1)

    # create the output directory if it doesn't already exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # get the experiment name from the learning curve TSV file
    # output_file_name = experiment_name + '_summary.tsv'
    experiment_name = tsv_file.name.rstrip("_summary.tsv")
    logging.info("Generating learning curve(s)")
    generate_learning_curve_plots(experiment_name, args.output_dir, args.tsv_file)


if __name__ == "__main__":
    main()
