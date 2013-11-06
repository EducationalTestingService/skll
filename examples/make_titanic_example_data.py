#!/usr/bin/env python
'''
This is a simple script to split the train.csv and test.csv files from the
Kaggle "Titanic: Machine Learning from Disaster" competition into the format
titanic.cfg expects.

:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
'''

from __future__ import print_function, unicode_literals

import logging
import os

from skll import load_examples, write_feature_file

def main():
    '''
    Create directories and split CSV files into subsets.
    '''
    logger = logging.getLogger(__name__)
    if not (os.path.exists('train.csv') and os.path.exists('test.csv')):
        logger.error('This script requires the train.csv and test.csv files ' +
                     'from http://www.kaggle.com/c/titanic-gettingStarted/' +
                     'data to be in the current directory in order to work. ' +
                     'Please download them and try again.')
        sys.exit(1)

    # Create dictionary of subsets to use for creating split feature files
    subset_dict = {'vitals': ['Name', 'Sex', 'Age'],
                   'socioeconomic': ['Pclass', 'Fare', 'Cabin'],
                   'family': ['SibSp', 'Parch'],
                   'misc': ['Ticket', 'Embarked']}

    # Create directories to store files
    if not os.path.exists('titanic/train'):
        logger.info('Creating titanic/train directory')
        os.makedirs('titanic/train')
    if not os.path.exists('titanic/test'):
        logger.info('Creating titanic/test directory')
        os.makedirs('titanic/test')

    # Read and write training examples
    train_examples = load_examples('train.csv', label_col='Survived',
                                   quiet=False)
    write_feature_file('titanic/train/feats.csv', None,
                       train_examples.classes, train_examples.features,
                       feat_vectorizer=train_examples.feat_vectorizer,
                       subsets=subset_dict, label_col='Survived',
                       id_prefix='train_example')

    # Read and write test examples
    test_examples = load_examples('test.csv', label_col='Survived',
                                   quiet=False)
    write_feature_file('titanic/test/feats.csv', None,
                       test_examples.classes, test_examples.features,
                       feat_vectorizer=test_examples.feat_vectorizer,
                       subsets=subset_dict, label_col='Survived',
                       id_prefix='test_example')


if __name__ == '__main__':
    main()
