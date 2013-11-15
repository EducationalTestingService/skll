#!/usr/bin/env python
'''
This is a simple script to split the train.csv and test.csv files from the
Kaggle "Titanic: Machine Learning from Disaster" competition into the format
titanic.cfg expects.

:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
'''

from __future__ import division, print_function, unicode_literals

import logging
import os
import sys

from skll import load_examples, write_feature_file

def main():
    '''
    Create directories and split CSV files into subsets.
    '''
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

    # Read and write training examples
    train_examples = load_examples('train.csv', label_col='Survived',
                                   quiet=False, sparse=False)
    num_train_dev = len(train_examples.classes)
    num_train = int((num_train_dev / 5) * 4)
    train_ids = list(range(1, num_train_dev + 1))
    write_feature_file('titanic/train/.csv',
                       train_ids[:num_train],
                       train_examples.classes[:num_train],
                       train_examples.features[:num_train, :],
                       feat_vectorizer=train_examples.feat_vectorizer,
                       subsets=subset_dict, label_col='Survived',
                       id_prefix='train_example')

    # Write train+dev set for training model to use to generate predictions on test
    write_feature_file('titanic/train+dev/.csv',
                       train_ids,
                       train_examples.classes,
                       train_examples.features,
                       feat_vectorizer=train_examples.feat_vectorizer,
                       subsets=subset_dict, label_col='Survived',
                       id_prefix='train_example')

    # Write dev examples
    write_feature_file('titanic/dev/.csv',
                       train_ids[num_train:],
                       train_examples.classes[num_train:],
                       train_examples.features[num_train:, :],
                       feat_vectorizer=train_examples.feat_vectorizer,
                       subsets=subset_dict, label_col='Survived',
                       id_prefix='dev_example')

    # Read and write test examples
    test_examples = load_examples('test.csv', label_col='Survived',
                                   quiet=False, sparse=False)
    num_test = len(test_examples.classes)
    test_ids = list(range(num_train_dev + 1, num_test + num_train_dev + 1))
    write_feature_file('titanic/test/.csv', test_ids,
                       test_examples.classes, test_examples.features,
                       feat_vectorizer=test_examples.feat_vectorizer,
                       subsets=subset_dict, label_col='Survived',
                       id_prefix='test_example')


if __name__ == '__main__':
    main()
