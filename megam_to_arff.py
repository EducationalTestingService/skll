#!/usr/bin/env python

# Script that converts MegaM files to ARFF format

# Author: Dan Blanchard, dblanchard@ets.org, Oct 2011

import argparse
import sys


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Takes a MegaM-compatible file to be run with the '-fvals' switch and outputs a " +
                                                 "Weka-compatible ARFF file to STDOUT.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', help='MegaM input file', type=argparse.FileType('r'), default='-', nargs='?')
    parser.add_argument('-c', '--classname', help='Name of nominal class field for ARFF file', default='class')
    parser.add_argument('-r', '--relation', help='Name of relation for ARFF file', default='MegaM Relation')
    args = parser.parse_args()

    if args.infile.isatty():
        print >> sys.stderr, "You are running this script interactively. Press CTRL-D at the start of a blank line to signal the end of your input. For help, run it with --help\n"

    # Initialize variables
    classes = set()
    instances = []
    fields = []

    # Iterate through MegaM file
    first = True
    for line in args.infile:
        split_line = line.strip().split()
        if len(split_line) > 1:
            class_name = split_line[0]
            classes.add(class_name)
            field_pairs = split_line[1:]
            # Get names of all fields if this is the first row we've processed
            if first:
                fields = field_pairs[::2]
                first = False
            # Add all the field values (and the current class value) to the list of instances
            instances.append(field_pairs[1::2] + [class_name])

    # Add relation to header
    print "@relation '{}'\n".format(args.relation)

    # Loop through fields writing the header info for the ARFF file
    for field in fields:
        print "@attribute '{}' numeric".format(field.replace('\\', '\\\\').replace("'", "\\'"))
    print "@attribute {} ".format(args.classname) + "{" + ','.join([str(x) for x in classes]) + "}"

    print "\n@data"
    # Loop through the list of instances, writing the ARFF file
    for instance in instances:
        print ','.join([str(x) for x in instance])
