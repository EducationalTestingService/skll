#!/usr/bin/env python
"""
Join MegaM files

@author: Dan Blanchard, dblanchard@ets.org
@date: July 2012
"""

from __future__ import print_function, unicode_literals

import argparse
import sys
from collections import defaultdict

from bs4 import UnicodeDammit


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Combine MegaM files that contain features for the same files.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('megam_file', help='MegaM input file(s). Each feature line must be preceded by a comment with the filename/ID ' +
                                           'that the features should be joined on.', type=argparse.FileType('r'), nargs='+')
    args = parser.parse_args()

    # Map from filenames to feature strings
    feature_dict = defaultdict(unicode)
    class_dict = dict()

    # Iterate through MegaM files
    for infile in args.megam_file:
        print("Loading {}...".format(infile.name), file=sys.stderr)
        sys.stderr.flush()
        for line in infile:
            stripped_line = UnicodeDammit(line.strip(), ['utf-8', 'windows-1252']).unicode_markup
            # Read current filename from comment
            if stripped_line.startswith('#'):
                curr_filename = stripped_line.lstrip('# ')
            # Ignore TEST and DEV lines and store features
            elif stripped_line not in ['TEST', 'DEV']:
                split_line = stripped_line.split('\t', 1)
                if len(split_line) == 2:
                    class_dict[curr_filename] = split_line[0]
                    feature_dict[curr_filename] += split_line[1] + ' '
                else:
                    print("Warning: No features found for {} in {}".format(curr_filename, infile.name), file=sys.stderr)

    # Print new MegaM file
    for curr_filename in feature_dict.viewkeys():
        print("# {}".format(curr_filename).encode('utf-8'))
        print("{}\t{}".format(class_dict[curr_filename], feature_dict[curr_filename].rstrip()).encode('utf-8'))
