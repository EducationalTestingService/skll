#!/usr/bin/env python
'''
Filter MegaM file to remove non-content word features

@author: Dan Blanchard, dblanchard@ets.org
@date: Feb 2012

'''

from __future__ import print_function, unicode_literals

import argparse
import itertools
import re
import sys

from bs4 import UnicodeDammit


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Filter MegaM file to remove\
                                                  features with names in stop\
                                                  word list (or non alphabetic\
                                                  characters).",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', help='MegaM input file',
                        type=argparse.FileType('r'), default='-', nargs='?')
    parser.add_argument('-i', '--ignorecase',
                        help='Do case insensitive feature name matching.',
                        action='store_true')
    parser.add_argument('-k', '--keep',
                        help='Instead of removing features with names in the\
                              list, keep only those.',
                        action='store_true')
    parser.add_argument('-s', '--stopwordlist', help='Stop word file',
                        default='/home/nlp-text/static/corpora/nonets/pan-2010-plagiarism/scripts/big_stoplist',
                        type=argparse.FileType('r'))
    args = parser.parse_args()

    if args.infile.isatty():
        print("You are running this script interactively. Press CTRL-D at " +
              "the start of a blank line to signal the end of your input. " +
              "For help, run it with --help\n",
              file=sys.stderr)

    # Read stop word list
    if args.ignorecase:
        stopwords = {w.strip().lower() for w in args.stopwordlist}
    else:
        stopwords = {w.strip() for w in args.stopwordlist}

    # Iterate through MegaM file
    for line in args.infile:
        stripped_line = UnicodeDammit(line.strip(),
                                      ['utf-8', 'windows-1252']).unicode_markup
        # Print TEST, DEV, and comment lines
        if stripped_line in ['TEST', 'DEV'] or stripped_line.startswith('#'):
            print(stripped_line.encode('utf-8'))
        elif stripped_line:
            split_line = stripped_line.split()
            feature_pairs = split_line[1:]
            print(split_line[0], end="\t")
            first = True
            for feature, value in itertools.izip(itertools.islice(feature_pairs,
                                                                  0, None, 2),
                                                 itertools.islice(feature_pairs,
                                                                  1, None, 2)):
                feature = feature.strip()
                if (re.match(r'[\w-]*$', feature) and
                    ((not args.keep and ((feature not in stopwords) or
                                         (args.ignorecase and
                                          (feature.lower() not in stopwords))))
                     or (args.keep and ((feature in stopwords) or
                                        (args.ignorecase and
                                         (feature.lower() in stopwords)))))):
                    if first:
                        first = False
                    else:
                        print(" ", end='')
                    print('{} {}'.format(feature, value).format('utf-8'),
                          end="")
            print()
