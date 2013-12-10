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
Script that converts feature files from one format to another

:author: Nitin Madnani (nmadnani@ets.org)
:date: September 2013
'''

from __future__ import print_function, unicode_literals

import argparse
import logging
import os
import sys

from skll.data import (_CSVDictIter, _ARFFDictIter, _TSVDictIter, _JSONDictIter,
                       _MegaMDictIter, write_feature_file)
from skll.version import __version__


def main(argv=None):
    '''
    Handles command line arguments and gets things started.

    :param argv: List of arguments, as if specified on the command-line.
                 If None, ``sys.argv[1:]`` is used instead.
    :type argv: list of str
    '''
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Takes an input feature file \
                                                  and converts it to another \
                                                  format. Formats are \
                                                  determined automatically from\
                                                  file extensions.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile',
                        help='input feature file (ends in .jsonlines, .tsv, \
                              .csv, .arff, or .megam)')
    parser.add_argument('outfile',
                        help='output feature file (ends in .jsonlines, .tsv, \
                              .csv, .arff, or .megam)')
    parser.add_argument('-l', '--label_col',
                        help='Name of the column which contains the class \
                              labels in ARFF, CSV, or TSV files. For ARFF \
                              files, this must be the final column to count as\
                              the label.',
                        default='y')
    parser.add_argument('-q', '--quiet',
                        help='Suppress printing of "Loading..." messages.',
                        action='store_true')
    parser.add_argument('--arff_regression',
                        help='Create ARFF files for regression, not classification.',
                        action='store_true')
    parser.add_argument('--arff_relation',
                        help='Relation name to use for ARFF file.',
                        default='skll_relation')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'))
    logger = logging.getLogger(__name__)

    # make sure the input file extension is one we can process
    input_extension = os.path.splitext(args.infile)[1].lower()

    if input_extension == ".tsv":
        example_iter_type = _TSVDictIter
    elif input_extension == ".jsonlines" or input_extension == '.ndj':
        example_iter_type = _JSONDictIter
    elif input_extension == ".megam":
        example_iter_type = _MegaMDictIter
    elif input_extension == ".csv":
        example_iter_type = _CSVDictIter
    elif input_extension == ".arff":
        example_iter_type = _ARFFDictIter
    else:
        logger.error(('Input file must be in either .arff, .csv, .jsonlines, ' +
                      '.megam, .ndj, or .tsv format. You specified: ' +
                      '{}').format(input_extension))
        sys.exit(1)


    # Iterate through input file and collect the information we need
    ids = []
    classes = []
    feature_dicts = []
    example_iter = example_iter_type(args.infile, quiet=args.quiet,
                                     label_col=args.label_col)
    for example_id, class_name, feature_dict in example_iter:
        feature_dicts.append(feature_dict)
        classes.append(class_name)
        ids.append(example_id)

    # write out the file in the requested output format
    write_feature_file(args.outfile, ids, classes, feature_dicts,
                       arff_regression=args.arff_regression,
                       arff_relation=args.arff_relation)

if __name__ == '__main__':
    main()