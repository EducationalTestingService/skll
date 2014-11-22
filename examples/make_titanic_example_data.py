#!/usr/bin/env python
"""
This is a simple script to split the train.csv and test.csv files from the
Kaggle "Titanic: Machine Learning from Disaster" competition into the format
titanic.cfg expects.

:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
"""

from __future__ import division, print_function, unicode_literals

import logging
import os
import sys
from itertools import chain

from skll import Writer, Reader

def main():
    """
    Create directories and split CSV files into subsets.
    """
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=logging.INFO)
    logger = logging.getLogger(__name__)
    if not (os.path.exists('train.csv') and os.path.exists('test.csv')):
        logger.error('This script requires the train.csv and test.csv files ' +
                     'from http://www.kaggle.com/c/titanic-gettingStarted/' +
                     'data to be in the current directory in order to work. ' +
                     'Please download them and try again.')
        sys.exit(1)

    # Create dictionary of subsets to use for creating split feature files
    subset_dict = {'vitals': ['Sex', 'Age'],
                   'socioeconomic': ['Pclass', 'Fare'],
                   'family': ['SibSp', 'Parch'],
                   'misc': ['Embarked']}
    features_to_keep = list(chain(*subset_dict.values()))

    # Create directories to store files
    if not os.path.exists('titanic/train'):
        logger.info('Creating titanic/train directory')
        os.makedirs('titanic/train')
    if not os.path.exists('titanic/dev'):
        logger.info('Creating titanic/dev directory')
        os.makedirs('titanic/dev')
    if not os.path.exists('titanic/train+dev'):
        logger.info('Creating titanic/train+dev directory')
        os.makedirs('titanic/train+dev')
    if not os.path.exists('titanic/test'):
        logger.info('Creating titanic/test directory')
        os.makedirs('titanic/test')

    # Read and write training FeatureSet
    train_fs = Reader.for_path('train.csv', label_col='Survived',
                               id_col='PassengerId', quiet=False,
                               sparse=False).read()
    train_fs.filter(features=features_to_keep)
    num_train_dev = len(train_fs)
    num_train = int((num_train_dev / 5) * 4)
    writer = Writer.for_path('titanic/train/.csv',
                                       train_fs[:num_train],
                                       id_col='PassengerId',
                                       label_col='Survived',
                                       quiet=False,
                                       subsets=subset_dict)
    writer.write()

    # Write train+dev set for training model to use to generate predictions on
    # test
    writer = Writer.for_path('titanic/train+dev/.csv',
                                       train_fs,
                                       label_col='Survived',
                                       id_col='PassengerId',
                                       quiet=False,
                                       subsets=subset_dict)
    writer.write()

    # Write dev FeatureSet
    writer = Writer.for_path('titanic/dev/.csv',
                                       train_fs[num_train:],
                                       label_col='Survived',
                                       id_col='PassengerId',
                                       quiet=False,
                                       subsets=subset_dict)
    writer.write()

    # Read and write test FeatureSet
    test_fs = Reader.for_path('test.csv', label_col='Survived', quiet=False,
                              sparse=False).read()
    test_fs.filter(features=features_to_keep)
    num_test = len(test_fs)
    test_fs.ids = list(range(num_train_dev + 1, num_test + num_train_dev + 1))
    writer = Writer.for_path('titanic/test/.csv',
                                       test_fs,
                                       label_col='Survived',
                                       id_col='PassengerId',
                                       quiet=False,
                                       subsets=subset_dict)
    writer.write()


if __name__ == '__main__':
    main()
