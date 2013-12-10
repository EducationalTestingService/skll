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

"""
Join MegaM files

:author: Dan Blanchard (dblanchard@ets.org)
:date: July 2012
"""

from __future__ import print_function, unicode_literals

import argparse
import logging
import os
import re
import sys
from collections import OrderedDict

from six import iteritems

from skll.data import _MegaMDictIter
from skll.version import __version__


# Globals
warned_about = set()


def parse_num_list(num_string):
    '''
    Convert a string representing a range of numbers to a list of integers.
    '''
    range_list = []
    if (num_string != '') and (not re.match(r'^(\d+(-\d+)?,)*\d+(-\d+)?$',
                                            num_string)):
        raise argparse.ArgumentTypeError("'" + num_string + "' is not a range" +
                                         " of numbers. Expected forms are " +
                                         "'8-15', '4,8,15,16,23,42', or " +
                                         "'8-15,42'.")
    for rng in num_string.split(','):
        if rng.count('-'):
            split_range = [int(x) for x in rng.split('-')]
            split_range[1] += 1
            range_list.extend(range(*split_range))
        else:
            range_list.append(int(rng))
    return range_list


def get_unique_name(feature_name, prev_feature_set, filename):
    '''
    Get a name that doesn't overlap with the previous features.

    :param feature_name: The feature that we want a unique name for.
    :type feature_name: string
    :param prev_feature_set: The feature names encountered so far.
    :type prev_feature_set: set of strings
    :param filename: The current MegaM file we're processing.
    :type filename: string

    :returns: Either feature_name or feature_name with a unique suffix based
              on the current MegaM file if there was an overlap.
    '''
    global warned_about

    new_feature_name = feature_name
    suffix = os.path.splitext(os.path.basename(filename))[0].replace(' ', '_')
    # Add suffix multiple times if necessary
    while new_feature_name in prev_feature_set:
        new_feature_name += "_" + suffix

    logger = logging.getLogger(__name__)
    if new_feature_name != feature_name and feature_name not in warned_about:
        logger.warning(("Feature named {} already found in previous files. " +
                        "Renaming to {} to prevent " +
                        "duplicates.").format(feature_name,
                                              new_feature_name).encode('utf-8'))
        warned_about.add(feature_name)
    return new_feature_name


def main(argv=None):
    '''
    Handles command line arguments and gets things started.

    :param argv: List of arguments, as if specified on the command-line.
                 If None, ``sys.argv[1:]`` is used instead.
    :type argv: list of str
    '''
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Combine MegaM files that \
                                                  contain features for the same\
                                                  files.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('megam_file',
                        help='MegaM input file(s). Each feature line must be \
                              preceded by a comment with the filename/ID that \
                              the features should be joined on.',
                        nargs='+')
    parser.add_argument('-b', '--binary',
                        help='Converts all of the features in the specified \
                              range of files to presence/absence binary \
                              features. Files are numbered starting from 1, and\
                              if 0 is specified with this flag, all files are\
                              converted.',
                        type=parse_num_list)
    parser.add_argument('--doubleup',
                        help='Keep both the binary and numeric versions of any\
                              feature you convert to binary.',
                        action='store_true')
    parser.add_argument('-c', '--common',
                        help='Only output features for filenames that are \
                              common to all MegaM files.',
                        action='store_true')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'))
    logger = logging.getLogger(__name__)

    # Map from filenames to feature strings
    feature_dict = OrderedDict()
    class_dict = {}
    filename_set = set()

    # Set that will contain all of the features seen in previous files
    # (for duplicate detection)
    prev_feature_set = set()

    # Iterate through MegaM files
    for file_num, infile in enumerate(args.megam_file, start=1):
        # Initialize duplicate feature book-keeping variables
        curr_feature_set = set()

        # Initialize set for storing filenames mentioned in current MegaM file
        curr_filename_set = set()

        # Handle current MegaM file
        for curr_filename, class_name, feature_dict in _MegaMDictIter(infile):
            if curr_filename in class_dict:
                if class_dict[curr_filename] != class_name:
                    raise ValueError(("Inconsisten class label for instance " +
                                      "{} in {}.").format(curr_filename,
                                                          infile.name))
            else:
                class_dict[curr_filename] = class_name
            # If there are non-zero features, process them
            if feature_dict:
                for feat_name, feat_val in iteritems(feature_dict):
                    # Handle duplicate features
                    feat_name = get_unique_name(feat_name, prev_feature_set,
                                                infile.name)
                    # Ignore zero-valued features
                    try:
                        if feat_val != 'N/A' and float(feat_val) != 0:
                            # Convert feature to binary if necessary
                            if (args.binary and ((args.binary == [0]) or (file_num in args.binary))):
                                if args.doubleup:
                                    new_feat_pair = '{} {} '.format(feat_name, feat_val)
                                    feature_dict[curr_filename] = new_feat_pair if curr_filename not in feature_dict else feature_dict[curr_filename] + new_feat_pair
                                    curr_feature_set.add(feat_name)
                                    feat_name = get_unique_name(feat_name + "_binary", prev_feature_set, infile.name)
                                feat_val = 1

                            # Add feature pair to current string of features
                            new_feat_pair = '{} {} '.format(feat_name,
                                                            feat_val)
                            feature_dict[curr_filename] = new_feat_pair if curr_filename not in feature_dict else feature_dict[curr_filename] + new_feat_pair
                            curr_feature_set.add(feat_name)
                    except ValueError:
                        raise ValueError(("Invalid feature value in feature " +
                                          "pair '{} {}' for file {}").format(feat_name,
                                                                             feat_val,
                                                                             curr_filename).encode('utf-8'))

            # Otherwise warn about lack of features (although that really
            # just means all of them have zero values)
            else:
                if curr_filename not in feature_dict:
                    feature_dict[curr_filename] = ""
                logger.warning(("No features found for {} in {}. All are " +
                                "assumed to be zero.").format(curr_filename,
                                                              infile.name).encode('utf-8'))

        # Add current file's features to set of seen features
        prev_feature_set.update(curr_feature_set)

        # Either intersect or union current file's filenames with existing ones
        if args.common and filename_set:
            filename_set.intersection_update(curr_filename_set)
        else:
            filename_set.update(curr_filename_set)

    # Print new MegaM file
    for curr_filename in feature_dict.viewkeys():
        # Skip files that aren't common when args.common is true
        if curr_filename not in filename_set:
            continue
        print("# {}".format(curr_filename).encode('utf-8'))
        print("{}\t{}".format(class_dict[curr_filename],
                              feature_dict[curr_filename].strip()).encode('utf-8'))


if __name__ == '__main__':
    main()