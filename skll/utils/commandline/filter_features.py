#!/usr/bin/env python
# License: BSD 3 clause
'''
Filter a given feature file to remove unwanted features, labels, or IDs.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
'''

import argparse
import logging
import os
import sys

from skll.data.readers import EXT_TO_READER, safe_float
from skll.data.writers import EXT_TO_WRITER, ARFFWriter, CSVWriter, TSVWriter
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
        description="Takes an input feature file and removes any instances or"
                    " features that do not match the specified patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input',
                        dest='infile',
                        required=True,
                        help='input feature file (ends in .arff, .csv, '
                             '.jsonlines, .ndj, or .tsv)')
    parser.add_argument('-o', '--output',
                        dest='outfile',
                        required=True,
                        help='output feature file (must have same extension '
                             'as input file)')
    parser.add_argument('-f', '--feature',
                        help='A feature in the feature file you would like to '
                             'keep.  If unspecified, no features are removed.',
                        nargs='*')
    parser.add_argument('-I', '--id',
                        help='An instance ID in the feature file you would '
                             'like to keep.  If unspecified, no instances are'
                             ' removed based on their IDs.',
                        nargs='*')
    parser.add_argument('--id_col',
                        help='Name of the column which contains the instance '
                             'IDs in ARFF, CSV, or TSV files.',
                        default='id')
    parser.add_argument('--inverse',
                        help='Instead of keeping features and/or examples in '
                             'lists, remove them.',
                        action='store_true')
    parser.add_argument('-L', '--label',
                        help='A label in the feature file you would like to '
                             'keep.  If unspecified, no instances are removed'
                             ' based on their labels.',
                        nargs='*')
    parser.add_argument('-l', '--label_col',
                        help='Name of the column which contains the class '
                             'labels in ARFF, CSV, or TSV files. For ARFF '
                             'files, this must be the final column to count '
                             'as the label.',
                        default='y')
    parser.add_argument('-rb', '--replace_blanks_with',
                        help='Specifies a new value with which to replace '
                             'blank values in all columns in a CSV/TSV file. To '
                             'replace blanks differently in each column, use '
                             'the SKLL Reader API directly.',
                        default=None)
    parser.add_argument('-db', '--drop_blanks',
                        action='store_true',
                        help='Drop all lines/rows in a CSV/TSV file that '
                             'contain any blank values. If each row of the file '
                             'contains a blank value, an exception will be raised.',
                        default=False)
    parser.add_argument('-q', '--quiet',
                        help='Suppress printing of "Loading..." messages.',
                        action='store_true')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - '
                               '%(message)s')
    logger = logging.getLogger(__name__)

    # all extensions except .libsvm can be processed
    valid_extensions = {ext for ext in EXT_TO_READER if ext != '.libsvm'}

    # make sure the input file extension is one we can process
    input_extension = os.path.splitext(args.infile)[1].lower()
    output_extension = os.path.splitext(args.outfile)[1].lower()

    if input_extension == '.libsvm':
        logger.error('Cannot filter LibSVM files.  Please use skll_convert to'
                     ' convert to a different datatype first.')
        sys.exit(1)

    if input_extension not in valid_extensions:
        logger.error('Input file must be in either .arff, .csv, .jsonlines, '
                     '.ndj, or .tsv format. You specified: '
                     f'{input_extension}')
        sys.exit(1)

    if output_extension != input_extension:
        logger.error('Output file must be in the same format as the input '
                     f'file.  You specified: {output_extension}')
        sys.exit(1)

    if input_extension == '.csv' or input_extension == '.tsv':
        replace_blanks_with = args.replace_blanks_with
        drop_blanks = args.drop_blanks
        if drop_blanks and replace_blanks_with is not None:
            raise ValueError("You cannot both drop blanks and replace them. "
                             "'replace_blanks_with' can only have a value "
                             "when 'drop_blanks' is `False`.")
        replace_blanks_with = (None if replace_blanks_with is None
                               else safe_float(replace_blanks_with))
        kwargs = {'replace_blanks_with': replace_blanks_with,
                  'drop_blanks': drop_blanks}
    else:
        kwargs = {}

    # Read input file
    reader = EXT_TO_READER[input_extension](args.infile,
                                            quiet=args.quiet,
                                            label_col=args.label_col,
                                            id_col=args.id_col,
                                            **kwargs)
    feature_set = reader.read()

    # convert the specified labels with `safe_float` before using
    # to match what the various featureset readers do
    converted_labels = None
    if args.label:
        converted_labels = [safe_float(label) for label in args.label]

    # Do the actual filtering
    feature_set.filter(ids=args.id,
                       labels=converted_labels,
                       features=args.feature,
                       inverse=args.inverse)

    # write out the file in the requested output format
    writer_type = EXT_TO_WRITER[input_extension]
    writer_args = {'quiet': args.quiet}
    if writer_type is CSVWriter or writer_type is TSVWriter:
        writer_args['label_col'] = args.label_col
        writer_args['id_col'] = args.id_col
    elif writer_type is ARFFWriter:
        writer_args['label_col'] = args.label_col
        writer_args['id_col'] = args.id_col
        writer_args['regression'] = reader.regression
        writer_args['relation'] = reader.relation
    writer = writer_type(args.outfile, feature_set, **writer_args)
    writer.write()


if __name__ == '__main__':
    main()
