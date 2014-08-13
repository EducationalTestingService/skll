#!/usr/bin/env python
# License: BSD 3 clause
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

from bs4 import UnicodeDammit
from sklearn.feature_extraction import DictVectorizer
from six import PY2

from skll.data import (_CSVDictIter, _ARFFDictIter, _TSVDictIter,
                       _JSONDictIter, _MegaMDictIter, _LibSVMDictIter,
                       write_feature_file)
from skll.version import __version__


def _pair_to_dict_tuple(pair):
    '''
    Little helper method for constructing mappings from feature/class names to
    numbers.
    '''
    number, name = pair.split('=')
    if PY2:
        name = name.encode('utf-8')
    number = int(number)
    return (name, number)


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
    parser.add_argument('--reuse_libsvm_map',
                        help='If you want to output multiple files that use \
                              the same mapping from classes and features to \
                              numbers when writing libsvm files, you can \
                              specify an existing .libsvm file to reuse the \
                              mapping from.',
                        type=argparse.FileType('rb'))
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - '
                                '%(message)s'))
    logger = logging.getLogger(__name__)

    # make sure the input file extension is one we can process
    input_extension = os.path.splitext(args.infile)[1].lower()
    output_extension = os.path.splitext(args.outfile)[1].lower()

    if input_extension == ".tsv":
        example_iter_type = _TSVDictIter
    elif input_extension == ".jsonlines" or input_extension == '.ndj':
        example_iter_type = _JSONDictIter
    elif input_extension == ".libsvm":
        example_iter_type = _LibSVMDictIter
    elif input_extension == ".megam":
        example_iter_type = _MegaMDictIter
    elif input_extension == ".csv":
        example_iter_type = _CSVDictIter
    elif input_extension == ".arff":
        example_iter_type = _ARFFDictIter
    else:
        logger.error(('Input file must be in either .arff, .csv, .jsonlines, '
                      '.libsvm, .megam, .ndj, or .tsv format. You specified: '
                      '{}').format(input_extension))
        sys.exit(1)

    # Build feature and label vectorizers from existing libsvm file if asked
    if args.reuse_libsvm_map and output_extension == '.libsvm':
        feat_map = {}
        label_map = {}
        for line in args.reuse_libsvm_map:
            line = UnicodeDammit(line, ['utf-8',
                                        'windows-1252']).unicode_markup
            if '#' not in line:
                logger.error('The LibSVM file you want to reuse the map from '
                             'was not created by SKLL and does not actually '
                             'contain the necessary mapping info.')
                sys.exit(1)
            comments = line.split('#')[1]
            _, label_map_str, feat_map_str = comments.split('|')
            feat_map.update(_pair_to_dict_tuple(pair) for pair in
                            feat_map_str.strip())
            label_map.update(_pair_to_dict_tuple(pair) for pair in
                             label_map_str
                             .strip())
        feat_vectorizer = DictVectorizer()
        feat_vectorizer.fit([{name: 1} for name in feat_map])
        feat_vectorizer.vocabulary_ = feat_map
    else:
        feat_vectorizer = None
        label_map = None

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
                       arff_relation=args.arff_relation,
                       feat_vectorizer=feat_vectorizer,
                       label_map=label_map)


if __name__ == '__main__':
    main()