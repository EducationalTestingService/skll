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
Module to handle loading data from various types of data files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
'''

from __future__ import print_function, unicode_literals

import csv
import json
import sys
from itertools import islice

import numpy as np
from bs4 import UnicodeDammit
from six.moves import zip


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


def _megam_dict_iter(path, has_labels=True):
    '''
    Generator that yields tuples of classes and dictionaries mapping from
    features to values for each pair of lines in the MegaM -fvals file specified
    by path.

    :param path: Path to MegaM file (-fvals format)
    :type path: basestring
    :param has_labels: Whether or not the file has a class label separated by
                       a tab before the space delimited feature-value pairs.
    :type has_labels: bool
    '''

    line_count = 0
    print("Loading {}...".format(path).encode('utf-8'), end="", file=sys.stderr)
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
                    field_values = (float(val) for val in islice(field_pairs,
                                                                 1, None, 2))

                    # Add the feature-value pairs to dictionary
                    curr_info_dict.update(zip(field_names, field_values))
                yield curr_id, class_name, curr_info_dict
                curr_id = None
            line_count += 1
            if line_count % 100 == 0:
                print(".", end="", file=sys.stderr)
        print("done", file=sys.stderr)


def load_examples(path, has_labels=True):
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

    :return: 2-column numpy.array of examples with the "y" containing the
             class labels and "x" containing the features for each example.
    '''
    if path.endswith(".tsv"):
        out = []
        with open(path) as f:
            reader = csv.reader(f, dialect=csv.excel_tab)
            header = next(reader)
            out = [_preprocess_tsv_row(row, header, example_num,
                                       has_labels=has_labels)
                   for example_num, row in enumerate(reader)]
    elif path.endswith(".jsonlines"):
        out = []
        with open(path) as f:
            example_num = 0
            for line in f:
                example = json.loads(line.strip())
                if "id" not in example:
                    example["id"] = "EXAMPLE_{}".format(example_num)
                example_num += 1
                out.append(example)
    elif path.endswith(".megam"):
        out = [{"y": class_name,
                "x": feature_dict,
                "id": ("EXAMPLE_{}".format(example_num) if example_id is None
                       else example_id)}
               for example_num, (example_id, class_name, feature_dict)
               in enumerate(_megam_dict_iter(path, has_labels=has_labels))]
    else:
        raise Exception('Example files must be in either TSV, MegaM, or the' +
                        'preprocessed .jsonlines format. ' +
                        'You specified: {}'.format(path))

    return np.array(out)


def _preprocess_tsv_row(row, header, example_num, has_labels=True):
    '''
    Convert the current TSV row into a dictionary of the form {"y": class label,
    "x": dictionary of feature values, "id": instance id}

    :param row: The TSV row to convert.
    :type row: list
    :param header: The header row from the TSV file.
    :type header: list
    :param example_num: The line number from the TSV file.
    :type example_num: int
    :param has_labels: Whether or not the TSV's first column is a class label.
    :type has_labels: bool
    '''
    x = {}

    if has_labels:
        y = row[0]
        feature_start_col = 1
    else:
        y = None
        feature_start_col = 0

    example_id = "EXAMPLE_{}".format(example_num)
    for fname, fval in zip(islice(header, feature_start_col, None),
                           islice(row, feature_start_col, None)):
        if fname == "id":
            example_id = fval
        else:
            fval_float = float(fval)
            # we don't need to explicitly store zeros
            if fval_float != 0.0:
                x["{}".format(fname)] = fval_float

    return {"y": y, "x": x, "id": example_id}
