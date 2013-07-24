# Copyright (C) 2012-2013 Educational Testing Service

# This file is part of SciKit-Learn Lab.

# SciKit-Learn Lab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SciKit-Learn Lab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with SciKit-Learn Lab.  If not, see <http://www.gnu.org/licenses/>.

'''
Handles loading data from various types of data files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
'''

from __future__ import print_function, unicode_literals

import json
import sys
from csv import DictReader
from itertools import islice

import numpy as np
from bs4 import UnicodeDammit
from six.moves import zip
from sklearn.feature_extraction import DictVectorizer


def _sanitize_line(line):
    '''
    :param line: The line to clean up.
    :type line: string

    :returns: Copy of line with all non-ASCII characters replaced with
    <U1234> sequences where 1234 is the value of ord() for the character.
    '''
    char_list = []
    for char in line:
        char_num = ord(char)
        char_list.append('<U{}>'.format(char_num) if char_num > 127 else char)
    return ''.join(char_list)


def _safe_float(text):
    '''
    Attempts to convert a string to a float, but if that's not possible, just
    returns the original value.
    '''
    try:
        return float(text)
    except ValueError:
        return text


def _megam_dict_iter(path, has_labels=True, quiet=False):
    '''
    Generator that yields tuples of classes and dictionaries mapping from
    features to values for each pair of lines in the MegaM -fvals file specified
    by path.

    :param path: Path to MegaM file (-fvals format)
    :type path: basestring
    :param has_labels: Whether or not the file has a class label separated by
                       a tab before the space delimited feature-value pairs.
    :type has_labels: bool
    :param quiet: Do not print "Loading..." status message to stderr.
    :type quiet: bool
    '''

    line_count = 0
    if not quiet:
        print("Loading {}...".format(path), end="", file=sys.stderr)
        sys.stderr.flush()
    with open(path, 'rb') as megam_file:
        curr_id = None
        for line in megam_file:
            # Process encoding
            line = UnicodeDammit(line, ['utf-8', 'windows-1252']).unicode_markup
            line = _sanitize_line(line.strip())
            # Handle instance lines
            if line.startswith('#'):
                curr_id = line[1:].strip()
            elif line and line not in ['TRAIN', 'TEST', 'DEV']:
                split_line = line.split()
                curr_info_dict = {}

                if has_labels:
                    class_name = split_line[0]
                    field_pairs = split_line[1:]
                else:
                    class_name = None
                    field_pairs = split_line

                if len(field_pairs) > 0:
                    # Get current instances feature-value pairs
                    field_names = islice(field_pairs, 0, None, 2)
                    # Convert values to floats, because otherwise features'll
                    # be categorical
                    field_values = (_safe_float(val) for val in 
                                    islice(field_pairs, 1, None, 2))

                    # TODO: Add some sort of check for duplicate feature names

                    # Add the feature-value pairs to dictionary
                    curr_info_dict.update(zip(field_names, field_values))

                yield curr_id, class_name, curr_info_dict
                curr_id = None
            line_count += 1
            if not quiet and line_count % 100 == 0:
                print(".", end="", file=sys.stderr)
        if not quiet:
            print("done", file=sys.stderr)


def load_examples(path, has_labels=True, sparse=True):
    '''
    Loads examples in the TSV, JSONLINES (a json dict per line), or MegaM
    formats.

    If you would like to include example/instance IDs in your files, they must
    be specified in the following ways:

    * MegaM: As a comment line directly preceding the line with feature values.
    * TSV: An "id" column.
    * JSONLINES: An "id" key in each JSON dictionary.

    :param path: The path to the file to load the examples from.
    :type path: basestring
    :param has_labels: Whether or not the file contains class labels.
    :type has_labels: bool
    :param sparse: Whether or not to store the features in a numpy CSR matrix.
    :type sparse: bool

    :return: 2-tuple of a (2 + n)-column numpy.array (where n is the number of 
             feature) of examples (with "SKLL_CLASS_LABEL" containing the class 
             labels, "SKLL_ID" containing the example IDs, and the remaining
             columns containing the features) and a DictVectorizer containing 
             the mapping between column/feature names and the column indices in 
             the example matrix.
    '''

    feat_vectorizer = DictVectorizer(sparse=sparse)

    # Build an appropriate generator for examples so we process the input file
    # through the feature vectorizer without using tons of memory
    if path.endswith(".tsv"):
        with open(path) as f:
            reader = DictReader(f, dialect=csv.excel_tab)
            example_generator = (_preprocess_tsv_row(row, reader.fieldnames, 
                                                     example_num, 
                                                     has_labels=has_labels) for 
                                 row in enumerate(reader))
    elif path.endswith(".jsonlines"):
        with open(path) as f:
            example_generator = (_preprocess_json_line(line, example_num) for 
                                 line, example_num in enumerate(f))
    elif path.endswith(".megam"):
        example_generator = (_preprocess_megam_example(curr_id, class_name, 
                                                       feature_dict) for 
                             curr_id, class_name, feature_dict in 
                             _megam_dict_iter(path))
    else:
        raise Exception('Example files must be in either TSV, MegaM, or ' +
                        '.jsonlines format. ' +
                        'You specified: {}'.format(path))

    return feat_vectorizer.fit_transform(example_generator), feat_vectorizer 


def _preprocess_tsv_row(row, header, example_num, has_labels=True):
    '''
    Convert current row dict from TSV file to a dictionary with the
    following fields: "SKLL_ID" (originally "id"), "SKLL_CLASS_LABEL"
    (originally "y"), and all of the feature-values in the sub-dictionary
    "x". Basically, we're flattening the structure, but renaming "y" and "id"
    to prevent possible conflicts with feature names in "x".

    :param row: The TSV row to convert.
    :type row: dict
    :param header: The header row from the TSV file.
    :type header: list
    :param example_num: The line number from the TSV file.
    :type example_num: int
    :param has_labels: Whether or not the TSV's first column is a class label.
    :type has_labels: bool
    '''
    example = row

    if has_labels:
        example_y = row[header[0]]
        del row[header[0]]
    else:
        example_y = None

    if "id" not in example:
        example_id = "EXAMPLE_{}".format(example_num)
    else:
        example_id = example["id"]
        del example["id"]
    example["SKLL_CLASS_LABEL"] = example_y
    example["SKLL_ID"] = example_id

    # Convert features to flaot
    for fname, fval in iteritems(example):
        fval_float = _safe_float(fval)
        # we don't need to explicitly store zeros
        if fval_float != 0.0:
            x["{}".format(fname)] = fval_float
    
    return example


def _preprocess_json_line(line, example_num):
    '''
    Convert current line in .jsonlines file to a dictionary with the
    following fields: "SKLL_ID" (originally "id"), "SKLL_CLASS_LABEL"
    (originally "y"), and all of the feature-values in the sub-dictionary
    "x". Basically, we're flattening the structure, but renaming "y" and "id"
    to prevent possible conflicts with feature names in "x".
    
    :param line: The JSON dictionary as a string.
    :type line: string
    :param example_num: The line number.
    :type example_num: int
    '''
    example = json.loads(line.strip())
    if "id" not in example:
        example_id = "EXAMPLE_{}".format(example_num)
    else:
        example_id = example["id"]
        del example["id"]
    example_y = example["y"]
    example = example["x"]
    example["SKLL_CLASS_LABEL"] = example_y
    example["SKLL_ID"] = example_id
    
    return example


def _preprocess_megam_example(curr_id, class_name, feature_dict):
    '''
    Takes the fields yielded by the _megam_dict_iter and converts them to the
    correct format for use as input to a DictVectorizer.

    :param curr_id: The ID of the current example
    :type curr_id: string
    :param class_name: The class label for the current example.
    :type class_name: string
    :param feature_dict: A dictionary of the feature values for the current
                         example.
    :type feature_dict: dictionary mapping from strings to floats
    
    :returns: a dictionary with the following fields: "SKLL_ID", 
              "SKLL_CLASS_LABEL", and all of the feature-values in feature_dict.
    '''
    example = feature_dict
    example["SKLL_CLASS_LABEL"] = class_name
    example["SKLL_ID"] = curr_id
    
    return example
