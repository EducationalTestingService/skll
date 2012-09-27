#!/usr/bin/env python
'''
Script that converts MegaM files to ARFF format

@author: Dan Blanchard, dblanchard@ets.org
@date: Oct 2011
'''

from __future__ import print_function, unicode_literals

import argparse
import csv
import sys

from bs4 import UnicodeDammit


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
        print("You are running this script interactively. Press CTRL-D at the start of a blank line to signal the end of your input. For help, run it with --help\n",
              file=sys.stderr)

    # Initialize variables
    classes = set()
    instances = []
    fields = set()

    # Iterate through MegaM file
    first = True
    for line in args.infile:
        # Process encoding
        line = UnicodeDammit(line, ['utf-8', 'windows-1252']).unicode_markup

        # Ignore comments
        if not line.startswith('#'):
            split_line = line.strip().split()
            class_name = split_line[0]
            classes.add(class_name)
            if len(split_line) > 1:
                field_pairs = split_line[1:]
                # Add field names to set in case there are new ones
                field_names = field_pairs[::2]
                field_values = field_pairs[1::2]
                fields.update(field_names)
                # Add all the field values (and the current class value) to the list of instances
                instances.append(dict(zip(field_names, field_values) + [(args.classname, class_name)]))
            else:
                instances.append({args.classname: class_name})

    # Add relation to header
    print("@relation '{}'\n".format(args.relation))

    # Loop through fields writing the header info for the ARFF file
    sorted_fields = sorted(fields)
    for field in sorted_fields:
        print("@attribute '{}' numeric".format(field.replace('\\', '\\\\').replace("'", "\\'")))
    print("@attribute {} ".format(args.classname) + "{" + ','.join([unicode(x) for x in classes]) + "}")

    # Create CSV writer to handle missing values for lines in data section
    csv.excel.lineterminator = '\n'
    csv.unregister_dialect('excel')
    csv.register_dialect('excel', csv.excel)
    writer = csv.DictWriter(sys.stdout, sorted_fields + [args.classname], restval=0)

    print("\n@data")
    # Loop through the list of instances, writing the ARFF file
    for instance in instances:
        writer.writerow(instance)
