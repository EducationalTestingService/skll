#!/usr/bin/env python

# Filter MegaM file to remove non-content word features

# Author: Dan Blanchard, dblanchard@ets.org, Feb 2012

import argparse
import itertools
import re
import sys


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Filter MegaM file to remove features with names in stop word list (or non alphabetic characters).",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', help='MegaM input file', type=argparse.FileType('r'), default='-', nargs='?')
    parser.add_argument('-i', '--ignorecase', help='Do case insensitive feature name matching.', action='store_true')
    parser.add_argument('-k', '--keep', help='Instead of removing features with names in the list, keep only those.', action='store_true')
    parser.add_argument('-s', '--stopwordlist', help='Stop word file',
                        default=open('/home/nlp-text/static/corpora/nonets/pan-2010-plagiarism/scripts/big_stoplist'),
                        type=argparse.FileType('r'))
    args = parser.parse_args()

    # Read stop word list
    stopwords = set([w.strip().lower() for w in args.stopwordlist.readlines()]) if args.ignorecase else set([w.strip() for w in args.stopwordlist.readlines()])

    # Iterate through MegaM file
    first = True
    for line in args.infile:
        stripped_line = line.strip()
        if stripped_line in ['TEST', 'DEV']:
            print stripped_line
        else:
            split_line = stripped_line.split()
            feature_pairs = split_line[1:]
            sys.stdout.write(split_line[0] + "\t")
            for feature, value in itertools.izip(itertools.islice(feature_pairs, 0, None, 2), itertools.islice(feature_pairs, 1, None, 2)):
                if first:
                    first = False
                else:
                    sys.stdout.write(' ')
                if re.match(r'[\w-]*$', feature) and ((not args.keep and ((feature not in stopwords) or (args.ignorecase and (feature.lower() not in stopwords)))) or
                                                      (args.keep and ((feature in stopwords) or (args.ignorecase and (feature.lower() in stopwords))))):
                    sys.stdout.write('{} {}'.format(feature, value))
            print
