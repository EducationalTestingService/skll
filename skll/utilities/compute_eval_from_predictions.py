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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('examples_file', help='SKLL input file with labeled examples')
    parser.add_argument('predictions_file', help='file with predictions from SKLL')
    parser.add_argument('--metric_list', help='comma-separated list of metrics to compute',
                        type=str, default='quadratic_weighted_kappa,unweighted_kappa')
    args = parser.parse_args()
    metric_names = args.metric_list.split(',')

    # read gold standard labels
    data = load_examples(args.examples_file)
    gold = dict(zip(data.ids, data.classes))

    # read predictions
    pred = {}
    with open(args.predictions_file) as pred_file:
        reader = csv.reader(pred_file, dialect=csv.excel_tab)
        reader.next()  # skip header
        for row in reader:
            pred[row[0]] = float(row[1])

    # make a sorted list of example ids in order to match up labels and predictions
    if set(gold.keys()) != set(pred.keys()):
        raise ValueError('The IDs in the examples and predictions do not match.')
    example_ids = sorted(gold.keys())

    for metric_name in metric_names:
        print("{}\t{}\t{}".format(compute_score([gold[ex_id] for ex_id in example_ids],
                                                [pred[ex_id] for ex_id in example_ids],
                                                metric_name),
                                  metric_name, args.predictions_file))



if __name__ == '__main__':
    main()

