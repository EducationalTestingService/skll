#!/bin/env python

# Script that converts MegaM files to ARFF format

# Author: Dan Blanchard, dblanchard@ets.org, Oct 2011

import argparse
import sys


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Takes a MegaM-compatible file to be run with the '-fvals' switch and outputs a " +
                                                 "Weka-compatible ARFF file to STDOUT.")
    parser.add_argument('infile', help='MegaM input file (defaults to STDIN)', type=argparse.FileType('r'), default=sys.stdin, nargs='?')
    parser.add_argument('-c', '--classname', help='Name of nominal class field for ARFF file (defaults to "class")', default='class')
    parser.add_argument('-r', '--relation', help='Name of relation for ARFF file (defaults to "MegaM Relation")', default='MegaM Relation')
    args = parser.parse_args()

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
