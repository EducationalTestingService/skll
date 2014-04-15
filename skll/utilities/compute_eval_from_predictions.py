#!/usr/bin/env python

# Copyright (C) 2012-2014 Educational Testing Service

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
script for computing additional evaluation metrics

:author: Michael Heilman (mheilman@ets.org)
'''

from skll.data import load_examples
import numpy as np
import argparse
from sklearn.metrics import SCORERS
import csv


def compute_score(y, y_pred, scorer_name):
    scorer = SCORERS[scorer_name]
    return scorer._score_func(y, y_pred, **scorer._kwargs)


def compute_eval_from_predictions(examples_file, predictions_file,
                                  metric_names):
    '''
    :param examples_file: a SKLL examples file (in .jsonlines or other format)
    :param predictions_file: a SKLL predictions output TSV file with id
                             and prediction column names
    :param metric_names: a list of SKLL metric names
                         (e.g., [pearson, unweighted_kappa])

    :returns: a dictionary from metrics names to values
    '''

    # read gold standard labels
    data = load_examples(examples_file)
    gold = dict(zip(data.ids, data.classes))

    # read predictions
    pred = {}
    with open(predictions_file) as pred_file:
        reader = csv.reader(pred_file, dialect=csv.excel_tab)
        reader.next()  # skip header
        for row in reader:
            pred[row[0]] = float(row[1])

    # make a sorted list of example ids in order to match up
    # labels and predictions
    if set(gold.keys()) != set(pred.keys()):
        raise ValueError('The example and prediction IDs do not match.')
    example_ids = sorted(gold.keys())

    res = {}
    for metric_name in metric_names:
        score = compute_score([gold[ex_id] for ex_id in example_ids],
                              [pred[ex_id] for ex_id in example_ids],
                              metric_name)
        res[metric_name] = score
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('examples_file',
                        help='SKLL input file with labeled examples')
    parser.add_argument('predictions_file',
                        help='file with predictions from SKLL')
    parser.add_argument('metric_names',
                        help='metrics to compute',
                        nargs='+')
    args = parser.parse_args()

    scores = compute_eval_from_predictions(args.examples_file,
                                           args.predictions_file,
                                           args.metric_names)

    for metric_name in args.metric_names:
        print("{}\t{}\t{}".format(scores[metric_name],
                                  metric_name, args.predictions_file))



if __name__ == '__main__':
    main()

