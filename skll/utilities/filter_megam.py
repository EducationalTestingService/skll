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
Filter MegaM file to remove non-content word features

:author: Dan Blanchard (dblanchard@ets.org)
:date: Feb 2012
'''

from __future__ import print_function, unicode_literals

import argparse
import logging
import re
import sys

from six import iteritems

from skll.data import _MegaMDictIter
from skll.version import __version__


def main(argv=None):
    '''
    Handles command line arguments and gets things started.

    :param argv: List of arguments, as if specified on the command-line.
                 If None, ``sys.argv[1:]`` is used instead.
    :type argv: list of str
    '''
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Filter MegaM file to remove\
                                                  features with names in stop\
                                                  word list (or non alphabetic\
                                                  characters). Also has \
                                                  side-effect of removing TEST,\
                                                  TRAIN, and DEV lines if they\
                                                  are present.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', help='MegaM input file',
                        default='-', nargs='?')
    parser.add_argument('stopwordlist', help='Stop word file',
                        type=argparse.FileType('r'))
    parser.add_argument('-i', '--ignorecase',
                        help='Do case insensitive feature name matching.',
                        action='store_true')
    parser.add_argument('-k', '--keep',
                        help='Instead of removing features with names in the\
                              list, keep only those.',
                        action='store_true')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'))

    if args.infile.isatty():
        print("You are running this script interactively. Press CTRL-D at " +
              "the start of a blank line to signal the end of your input. " +
              "For help, run it with --help\n",
              file=sys.stderr)

    # Read stop word list
    if args.ignorecase:
        stopwords = {w.strip().lower() for w in args.stopwordlist}
    else:
        stopwords = {w.strip() for w in args.stopwordlist}

    # Iterate through MegaM file
    for example_id, class_name, feature_dict in _MegaMDictIter(args.infile):
        if example_id is not None:
            print("# {}".format(example_id))
        print(class_name, end="\t")
        first = True
        for feature, value in iteritems(feature_dict):
            feature = feature.strip()
            if (re.match(r'[\w-]*$', feature) and
                ((not args.keep and ((feature not in stopwords) or
                                     (args.ignorecase and
                                      (feature.lower() not in stopwords))))
                 or (args.keep and ((feature in stopwords) or
                                    (args.ignorecase and
                                     (feature.lower() in stopwords)))))):
                if first:
                    first = False
                else:
                    print(" ", end='')
                print('{} {}'.format(feature, value), end="")
        print()


if __name__ == '__main__':
    main()