#!/usr/bin/env python

"""
This is a simple script to download and transform some example data from
sklearn.datasets.

:author: Michael Heilman (mheilman@ets.org)
:author: Aoife Cahill (acahill@ets.org)
:organization: ETS
"""

from __future__ import print_function, unicode_literals

import json
import os
import sys

import sklearn.datasets
from sklearn.cross_validation import train_test_split


def main():
    """
    Download some example data and split it into training and test data.
    The boston data set is meant for regression modeling.
    """
    print('Retrieving boston data from servers...', end='')
    boston_data = sklearn.datasets.load_boston()
    sys.stdout.flush()
    print('done')
    sys.stdout.flush()


    X = boston_data['data']
    Y = boston_data['target']

    examples = [{'id': 'EXAMPLE_{}'.format(i),
                 'y': y,
                 'x': {'f{}'.format(j): x_val for j, x_val in enumerate(x)}}
                for i, (x, y) in enumerate(zip(X, Y))]

    examples_train, examples_test = train_test_split(examples, test_size=0.33,
                                                     random_state=42)

    print('Writing training and testing files...', end='')
    for examples, suffix in [(examples_train, 'train'), (examples_test,
                                                         'test')]:
        boston_dir = os.path.join('boston', suffix)
        if not os.path.exists(boston_dir):
            os.makedirs(boston_dir)
        jsonlines_path = os.path.join(boston_dir,
                                      'example_boston_features.jsonlines')
        with open(jsonlines_path, 'w') as f:
            for ex in examples:
                f.write('{}\n'.format(json.dumps(ex)))
    print('done')


if __name__ == '__main__':
    main()
