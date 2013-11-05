#!/usr/bin/env python
'''
This is a simple script to split the train.csv and test.csv files from the
Kaggle "Titanic: Machine Learning from Disaster" competition into the format
titanic.cfg expects.

:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
'''

from __future__ import print_function, unicode_literals

import os

from skll import load_examples, write_feature_file

def main():
    '''
    Create directories and split CSV files into subsets.
    '''
    # Create dictionary of subsets to use for creating split feature files
    subset_dict = {'vitals': ['Name', 'Sex', 'Age'],
                   'socioeconomic': ['Pclass', 'Fare', 'Cabin'],
                   'family': ['SibSp', 'Parch'],
                   'misc': ['Ticket', 'Embarked']}

    # Create directories to store files
    if not os.path.exists('titanic/train'):
        os.makedirs('titanic/train')
    if not os.path.exists('titanic/test'):
        os.makedirs('titanic/test')

    # Read and write training examples
    train_examples = load_examples('train.csv', label_col='Survived',
                                   quiet=False)
    write_feature_file('titanic/train/foo.csv', train_examples.ids,
                       train_examples.classes, train_examples.features,
                       feat_vectorizer=train_examples.feat_vectorizer,
                       subsets=subset_dict, label_col='Survived')

if __name__ == '__main__':
    main()
