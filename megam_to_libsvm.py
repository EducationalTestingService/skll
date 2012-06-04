#!/usr/bin/env python

# Script that converts MegaM files to LibSVM format

# Author: Dan Blanchard, dblanchard@ets.org, May 2012

import argparse
import sys
from collections import defaultdict
from itertools import izip
from operator import itemgetter


# Globals
NUM_FIELDS = 0
NUM_CLASSES = 0


def get_next_field_num():
    global NUM_FIELDS
    NUM_FIELDS += 1
    return NUM_FIELDS


def get_next_class_num():
    global NUM_CLASSES
    NUM_CLASSES += 1
    return NUM_CLASSES


def convert_to_libsvm(lines):
    '''
        Converts a sequence of lines (e.g., a file or list of strings) in MegaM format to LibSVM format.
        @param lines: The sequence of lines to convert.
        @type lines: L{file} or L{list} of L{str}

        @return: A tuple of the newly formatted data, the mappings from class names to numbers, and the mappings from feature names to numbers.
        @rtype: 3-L{tuple} of (L{list} of L{str}, L{dict}, and L{dict})
    '''
    global NUM_FIELDS, NUM_CLASSES

    # Initialize variables
    NUM_FIELDS = 0
    NUM_CLASSES = 0
    field_num_dict = defaultdict(get_next_field_num)
    class_num_dict = defaultdict(get_next_class_num)

    result_list = []

    # Iterate through MegaM file
    for line in lines:
        result_string = ''
        split_line = line.strip().split()
        if len(split_line) > 1:
            result_string += str(class_num_dict[split_line[0]])
            del split_line[0]
            # Loop through all feature-value pairs printing out pairs separated by commas (and with feature names replaced with numbers)
            for field_num, value in sorted(izip([field_num_dict[field_name] for field_name in split_line[::2]], [float(value) for value in split_line[1::2]])):
                if float(value):
                    result_string += ' {}:{}'.format(field_num, value)
            result_list.append(result_string)

    return result_list, class_num_dict, field_num_dict


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Takes a MegaM-compatible file to be run with the '-fvals' switch and outputs a " +
                                                 "LibSVM/LibLinear-compatible file to STDOUT.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', help='MegaM input file', type=argparse.FileType('r'), default='-', nargs='?')
    parser.add_argument('-m', '--mappingfile', help='File to output mapping of feature/class indices to names to', type=argparse.FileType('w'), default='map.idx')
    args = parser.parse_args()

    if args.infile.isatty():
        print >> sys.stderr, "You are running this script interactively. Press CTRL-D at the start of a blank line to signal the end of your input. For help, run it with --help\n"

    line_list, class_num_dict, field_num_dict = convert_to_libsvm(args.infile)

    # Iterate through converted MegaM file
    for line in line_list:
        print line

    # Print out mappings to file
    print >> args.mappingfile, "CLASS NUM\tCLASS NAME"
    for class_name, class_num in sorted(class_num_dict.iteritems(), key=itemgetter(1)):
        print >> args.mappingfile, "{}\t{}".format(class_num, class_name)

    print >> args.mappingfile, "\n\nFEATURE NUM\tFEATURE NAME"
    for field_name, field_num in sorted(field_num_dict.iteritems(), key=itemgetter(1)):
        print >> args.mappingfile, "{}\t{}".format(field_num, field_name)
