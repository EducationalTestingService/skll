#!/usr/bin/env python
'''
Script that converts from CSV to MegaM format

@author: Dan Blanchard, dblanchard@ets.org
@date: June 2012

'''

from __future__ import unicode_literals, print_function

import argparse
import sys

from bs4 import UnicodeDammit


def sanitize_name(feature_name):
    '''
        Replaces bad characters in feature names.
    '''
    return feature_name.replace(" ", "_").replace("#", "HASH")


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Takes a delimited file with a header line and converts it to MegaM.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', help='MegaM input file', type=argparse.FileType('r'), default='-', nargs='?')
    parser.add_argument('-c', '--classfield', help='Index of class field in CSV file. Note: fields are numbered starting at 0.', default=-1, type=int)
    parser.add_argument('-d', '--delimiter', help='The column delimiter.', default=',')
    parser.add_argument('-i', '--idfield', help='Index of ID field in CSV file (if there is one). This will be included as a comment before each line.' +
                                                'Note: fields are numbered starting at 0.', type=int)
    args = parser.parse_args()

    if args.infile.isatty():
        print("You are running this script interactively. Press CTRL-D at the start of a blank line to signal the end of your input. For help, run it with --help\n",
              file=sys.stderr)

    # Initialize variables
    classes = set()
    instances = []
    fields = []

    # Iterate through input file
    first = True
    for line in args.infile:
        stripped_line = UnicodeDammit(line.strip(), ['utf-8', 'windows-1252']).unicode_markup
        split_line = stripped_line.split(args.delimiter)
        # Skip blank lines
        if split_line:
            # Process header
            if first:
                fields = split_line[1:] if split_line[0] == '#' else split_line  # Check for weird commented-out header
                fields = [sanitize_name(field) for field in fields]
                # Delete extra fields
                if args.idfield is not None:
                    # Have to sort descending so that we don't screw up the indices
                    for i in sorted((args.idfield, args.classfield), reverse=True):
                        del fields[i]
                else:
                    del fields[args.classfield]
                first = False
            else:
                # Delete extra fields
                if args.idfield is not None:
                    # Print id field
                    print("# {}".format(split_line[args.idfield]).encode('utf-8'))
                    # Print class
                    print('{}'.format(split_line[args.classfield]).encode('utf-8'), end='\t')

                    # Have to sort descending so that we don't screw up the indices
                    for i in sorted((args.idfield, args.classfield), reverse=True):
                        del split_line[i]
                else:
                    # Print class
                    print('{}'.format(split_line[args.classfield]).encode('utf-8'), end='\t')
                    del split_line[args.classfield]

                # Print features
                print(' '.join(['{} {}'.format(field, value) for field, value in zip(fields, split_line) if value not in ['.', '?'] and float(value) != 0]).encode('utf-8'))
