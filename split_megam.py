#!/usr/bin/env python

# Script that splits MegaM files into training, test, and dev sections

# Author: Dan Blanchard, dblanchard@ets.org, Feb 2012

import argparse
import random
import sys


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Splits a MegaM-compatible into dev, training, and test sets. If -d and -t are ommitted, just strips existing " +
                                                 "DEV and TEST lines from file.")
    parser.add_argument('infile', help='MegaM input file (defaults to STDIN)', type=argparse.FileType('r'), default=sys.stdin, nargs='?')
    parser.add_argument('-d', '--dev', help='Number of instances per class to reserve for development.', type=int, default=0)
    parser.add_argument('-m', '--max', help='Maximum number of instances to use for training for each class.', type=int, default=0)
    parser.add_argument('-r', '--randomize', help='Randomly shuffle the instances before splitting into training, dev, and test sets.', action='store_true')
    parser.add_argument('-t', '--test', help='Number of instances per class to reserve for testing.', type=int, default=0)
    args = parser.parse_args()

    # Initialize variables
    classes = set()
    inst_str_list = []

    # Iterate through MegaM file
    first = True
    for line in args.infile:
        stripped_line = line.strip()
        split_line = stripped_line.split()
        if len(split_line) > 1:
            class_name = split_line[0]
            classes.add(class_name)
            # Add all the field values (and the current class value) to the list of instances
            inst_str_list.append(stripped_line)

    # Randomize if asked
    if args.randomize:
        random.shuffle(inst_str_list)

    # Initialize dev, test, and train sets
    dev_sets = dict([(x, set()) for x in classes])
    test_sets = dict([(x, set()) for x in classes])
    train_sets = dict([(x, set()) for x in classes])

    # Split instance list into dev, test, and training sets
    for i, inst_str in enumerate(inst_str_list):
        curr_class = inst_str.split()[0]
        if len(dev_sets[curr_class]) < args.dev:
            dev_sets[curr_class].add(i)
        elif len(test_sets[curr_class]) < args.test:
            test_sets[curr_class].add(i)
        elif (not args.max) or (len(train_sets[curr_class]) < args.max):
            train_sets[curr_class].add(i)

    # Process each training set
    for inst_set in train_sets.itervalues():
        for i in inst_set:
            print inst_str_list[i]

    # Process each dev set
    if args.dev:
        print "DEV"
        for inst_set in dev_sets.itervalues():
            for i in inst_set:
                print inst_str_list[i]

    # Process each test set
    if args.test:
        print "TEST"
        for inst_set in test_sets.itervalues():
            for i in inst_set:
                print inst_str_list[i]
