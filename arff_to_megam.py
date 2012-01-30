#!/bin/env python

# Script that converts ARFF files to MegaM format

# Author: Dan Blanchard, dblanchard@ets.org, Sep 2011

import argparse
import csv
import random
import sys
import re

from arff import arffread


def parse_num_list(num_string):
    '''
        Convert a string representing a range of numbers to a list of integers.
    '''
    range_list = []
    if (num_string != '') and (not re.match(r'^(\d+(-\d+)?,)*\d+(-\d+)?$', num_string)):
        raise argparse.ArgumentTypeError("'" + num_string + "' is not a range of numbers. Expected forms are '8-15', '4,8,15,16,23,42', or '3-16,42'.")
    for rng in num_string.split(','):
        if rng.count('-'):
            split_range = [int(x) for x in rng.split('-')]
            split_range[1] += 1
            range_list.extend(range(*split_range))
        else:
            range_list.append(int(rng))
    return range_list


def process_set(inst_set, class_dict, alist, args):
    '''
        Process an instance set and output MegaM-style instances
    '''
    for instance in inst_set:
        print (instance[-1] if args.namedclasses else str(class_dict[instance[-1]])) + "\t",
        # Loop through all attributes in instance set. We omit the last one, because it's class.
        for i in range(len(instance) - 1):
            # Check if this a feature we want to keep
            if ((not args.features) or ((i + 1) in args.features)) and alist[i][1] == 1:
                # Have to replace hash because it will cause MegaM to crash.
                if (args.binary and ((args.binary == [0]) or ((i + 1) in args.binary))):
                    print alist[i][0].replace(" ", "_").replace("#", "HASH"), int(bool(float(instance[i]))),
                    if args.doubleup:
                        print alist[i][0].replace(" ", "_").replace("#", "HASH"), instance[i],
                else:
                    print alist[i][0].replace(" ", "_").replace("#", "HASH"), instance[i],
            elif args.verbose:
                print >> sys.stderr, alist[i][0].replace(" ", "_").replace("#", "HASH"), instance[i],
        print


def read_arff(arff_file, quote_char):
    '''
        A simplified ARFF reader that is much more memory efficient than the arff module.
        @returns A tuple of the name of the relation, the list of attributes, and the list of instances.
    '''
    alist = []
    ilist = []
    relation = ''
    in_header = True
    for line in arff_file:
        if line.strip() != '':
            # Process header if we're still in it
            if in_header:
                # Split the line using CSV reader because it can handle quoted delimiters.
                split_header = csv.reader([line], delimiter=' ', quotechar=quote_char, escapechar='\\').next()
                row_type = split_header[0].lower()
                if row_type == '@attribute':
                    # Nominal
                    if split_header[2][0] == '{':
                        alist.append([split_header[1], 0, [x.strip() for x in split_header[2].strip('{}').split(',')]])
                    # Numeric or String
                    else:
                        alist.append([split_header[1], int(split_header[2] == 'numeric'), []])
                elif row_type == '@data':
                    in_header = False
                elif row_type == '@relation':
                    relation = split_header[1]
            # Process data lines
            else:
                ilist.append(csv.reader([line], quotechar=quote_char, escapechar='\\').next())
    return (relation, alist, ilist)


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Takes an ARFF file an outputs a MegaM-compatible file to be run with the '-fvals' switch." +
                                                 " Assumes last field is class. Ignores any relational or date fields. Automatically converts nominals" +
                                                 " to numerals.")
    parser.add_argument('infile', help='ARFF input file (defaults to STDIN)', type=argparse.FileType('r'), default=sys.stdin, nargs='?')
    parser.add_argument('-a', '--arffmodule', help='Use the Python ARFF module instead of my simple ARFF reader. Use this for complicated ARFF files that contain' +
                                                   ' datatypes other than string, numeric, and nominal.', action='store_true')
    parser.add_argument('-b', '--binary', help='Converts the specified range of numeric features to presence/absence binary features. Features are numbered ' +
                                               'starting from 1, and if 0 is specified with this flag, all numeric features are converted. Note: Any string ' +
                                               'features within the specified range are just ignored.', type=parse_num_list)
    parser.add_argument('--doubleup', help='Keep both the binary and numeric versions of any feature numeric feature you convert to binary.', action='store_true')
    parser.add_argument('-d', '--dev', help='Number of instances per class to reserve for development.', type=int, default=0)
    parser.add_argument('-f', '--features', help='Only keep the specified range of features in the MegaM output. Features are numbered starting from 1.',
                        type=parse_num_list)
    parser.add_argument('-m', '--max', help='Maximum number of instances to use for training for each class.', type=int, default=0)
    parser.add_argument('-n', '--namedclasses', help='Keep class names in MegaM file instead of converting the nomimal field to numeric.', action='store_true')
    parser.add_argument('-r', '--randomize', help='Process the instances in a random order to ensure that the dev and test sets are random.', action='store_true')
    parser.add_argument('-q', '--quotechar', help='Character to use for quoting strings in attribute names. (Default = \')', default="'")
    parser.add_argument('-t', '--test', help='Number of instances per class to reserve for testing.', type=int, default=0)
    parser.add_argument('-v', '--verbose', help='Print out fields that were not added output to MegaM file on STDERR.', action='store_true')
    args = parser.parse_args()

    # Read ARFF file
    if not args.arffmodule:
        (name, alist, ilist) = read_arff(args.infile, args.quotechar)
    else:
        (name, sparse, alist, ilist) = arffread(args.infile)

    # Randomize instances if asked
    if args.randomize:
        random.shuffle(ilist)

    # Convert class nominal to numeral
    class_dict = {}
    class_list = alist[-1][2]
    for i in range(len(class_list)):
        class_dict[class_list[i]] = i

    # Initialize dev, test, and train sets
    dev_sets = [set() for x in class_list]
    test_sets = [set() for x in class_list]
    train_sets = [set() for x in class_list]

    # Split instance list into dev, test, and training sets
    for instance in ilist:
        # Add to either dev or test sets, or just process normally
        if len(dev_sets[class_dict[instance[-1]]]) < args.dev:
            dev_sets[class_dict[instance[-1]]].add(tuple(instance))
        elif len(test_sets[class_dict[instance[-1]]]) < args.test:
            test_sets[class_dict[instance[-1]]].add(tuple(instance))
        elif (not args.max) or (len(train_sets[class_dict[instance[-1]]]) < args.max):
            train_sets[class_dict[instance[-1]]].add(tuple(instance))

    # Process each training set
    for inst_set in train_sets:
        process_set(inst_set, class_dict, alist, args)

    # Process each dev set
    if args.dev:
        print "DEV"
        for inst_set in dev_sets:
            process_set(inst_set, class_dict, alist, args)

    # Process each test set
    if args.test:
        print "TEST"
        for inst_set in test_sets:
            process_set(inst_set, class_dict, alist, args)
