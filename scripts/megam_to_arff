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
from itertools import islice, izip

from bs4 import UnicodeDammit


def sanitize_line(line):
    ''' Return copy of line with all non-ASCII characters replaced with <U1234> sequences where 1234 is the value of ord() for the character. '''
    char_list = []
    for char in line:
        char_num = ord(char)
        char_list.append('<U{}>'.format(char_num) if char_num > 127 else char)
    return ''.join(char_list)

def megam_iter(instance_str_list, parsed_args):
    '''
    Generator for items in the instance_str_list as dictionaries.

    @param instance_str_list: List of instance lines in file
    @type instance_str_list: C{list} of C{unicode}
    @param parsed_args: The parsed command-line arguments returned by ArgumentParser
    '''

    for instance in instance_str_list:
        split_instance = instance.split()
        curr_info_dict = {parsed_args.class_name: split_instance[0]}
        if len(split_instance) > 1:
            # Get current instances feature-value pairs
            field_pairs = split_instance[1:]
            field_names = islice(field_pairs, 0, None, 2)
            field_values = islice(field_pairs, 1, None, 2)

            # Add the feature-value pairs to dictionary
            curr_info_dict.update(izip(field_names, field_values))
        yield curr_info_dict


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Takes a MegaM-compatible file to be run with the '-fvals' switch and outputs a " +
                                                 "Weka-compatible ARFF file to STDOUT.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', help='MegaM input file', type=argparse.FileType('r'), default='-', nargs='?')
    parser.add_argument('-c', '--class_name', help='Name of nominal class field for ARFF file', default='class')
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
    line_count = 0
    print("Loading {}...".format(args.infile.name).encode('utf-8'), end="", file=sys.stderr)
    sys.stderr.flush()
    for line in args.infile:
        # Process encoding
        line = sanitize_line(UnicodeDammit(line, ['utf-8', 'windows-1252']).unicode_markup).strip()

        # Ignore comments
        if not line.startswith('#'):
            split_line = line.split()
            classes.add(split_line[0])
            if len(split_line) > 1:
                # Add field names to set in case there are new ones
                fields.update(islice(islice(split_line, 1, None), 0, None, 2))
            instances.append(line)
        line_count += 1
        if line_count % 100 == 0:
            print(".", end="", file=sys.stderr)
    print("done", file=sys.stderr)

    # Add relation to header
    print("@relation '{}'\n".format(args.relation))

    # Loop through fields writing the header info for the ARFF file
    sorted_fields = sorted(fields)
    for field in sorted_fields:
        print("@attribute '{}' numeric".format(field.replace('\\', '\\\\').replace("'", "\\'")))
    print("@attribute {} ".format(args.class_name) + "{" + ','.join(sorted(classes)) + "}")

    # Create CSV writer to handle missing values for lines in data section
    csv.excel.lineterminator = '\n'
    csv.unregister_dialect('excel')
    csv.register_dialect('excel', csv.excel)
    writer = csv.DictWriter(sys.stdout, sorted_fields + [args.class_name], restval=0)

    print("\n@data")
    # Loop through the list of instances, writing the ARFF file
    for instance_dict in megam_iter(instances, args):
        writer.writerow(instance_dict)
