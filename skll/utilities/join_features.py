#!/usr/bin/env python
# License: BSD 3 clause
'''
Script that joins a bunch of feature files together to create one file.

:author: Dan Blanchard (dblanchard@ets.org)
:date: November 2014
'''

from __future__ import print_function, unicode_literals

import argparse
import logging
import os
import sys

from skll.data.readers import EXT_TO_READER
from skll.data.writers import ARFFWriter, DelimitedFileWriter, EXT_TO_WRITER
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
        description="Joins multiple input feature files together into one \
                     file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile',
                        help='input feature file (ends in .arff, .csv, \
                              .jsonlines, .megam, .ndj, or .tsv)',
                        nargs='+')
    parser.add_argument('outfile',
                        help='output feature file')
    parser.add_argument('-i', '--id_col',
                        help='Name of the column which contains the instance \
                              IDs in ARFF, CSV, or TSV files.',
                        default='id')
    parser.add_argument('-l', '--label_col',
                        help='Name of the column which contains the class \
                              labels in ARFF, CSV, or TSV files. For ARFF \
                              files, this must be the final column to count as\
                              the label.',
                        default='y')
    parser.add_argument('-q', '--quiet',
                        help='Suppress printing of "Loading..." messages.',
                        action='store_true')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - '
                                '%(message)s'))
    logger = logging.getLogger(__name__)

    # all extensions except .libsvm can be processed
    valid_extensions = {ext for ext in EXT_TO_READER if ext != '.libsvm'}

    # make sure the input file extensions are those we can process
    input_extension_set = {os.path.splitext(inf)[1].lower() for inf in
                           args.infile}
    output_extension = os.path.splitext(args.outfile)[1].lower()

    # make sure all the files are in the same format except libsvm files
    if len(input_extension_set) > 1:
        logger.error('All input files must be in the same format.')
        sys.exit(1)

    input_extension = list(input_extension_set)[0]

    if input_extension not in valid_extensions:
        logger.error(('Input files must be in either .arff, .csv, .jsonlines, '
                      '.megam, .ndj, or .tsv format. You specified: '
                      '{}').format(input_extension))
        sys.exit(1)

    if output_extension != input_extension:
        logger.error(('Output file must be in the same format as the input '
                      'file.  You specified: {}').format(output_extension))
        sys.exit(1)

    # Read and merge input files
    merged_set = None
    for infile in args.infile:
        reader = EXT_TO_READER[input_extension](infile, quiet=args.quiet,
                                                label_col=args.label_col,
                                                id_col=args.id_col)
        fs = reader.read()
        if merged_set is None:
            merged_set = fs
        else:
            merged_set += fs

    # write out the file in the requested output format
    writer_type = EXT_TO_WRITER[input_extension]
    writer_args = {'quiet': args.quiet}
    if writer_type is DelimitedFileWriter:
        writer_args['label_col'] = args.label_col
        writer_args['id_col'] = args.id_col
    elif writer_type is ARFFWriter:
        writer_args['label_col'] = args.label_col
        writer_args['id_col'] = args.id_col
        writer_args['regression'] = reader.regression
        writer_args['relation'] = reader.relation
    writer = writer_type(args.outfile, merged_set, **writer_args)
    writer.write()


if __name__ == '__main__':
    main()
