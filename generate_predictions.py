#!/usr/bin/env python
'''
Loads a trained model and outputs predictions based on input feature files.

@author: Dan Blanchard
@contact: dblanchard@ets.org
@organization: ETS
@date: February 2013
'''

from __future__ import print_function, unicode_literals

import argparse
from itertools import islice, izip

from bs4 import UnicodeDammit
from classifier import Classifier, _sanitize_line


class Predictor(object):
    """ Little wrapper around a L{Classifier} to load models and get predictions for feature strings. """

    def __init__(self, model_prefix, threshold=None, positive_class=1):
        '''
        Initialize the predictor.

        @param model_prefix: Prefix to use when loading trained model (and its vocab).
        @type model_prefix: C{basestring}
        @param threshold: If the model we're using is generating probabilities of the positive class, return 1
                          if it meets/exceeds the given threshold and 0 otherwise.
        @type threshold: C{float}
        @param positive_class: If the model is only being used to predict the probability of a particular class, this specifies
                               the index of the class we're predicting. 1 = second class, which is default for binary classification.
        @type positive_class: C{int}
        '''
        self._classifier = Classifier()
        self._classifier.load_vocab('{}.vocab'.format(model_prefix))
        self._classifier.load_model('{}.model'.format(model_prefix))
        self._example_count = 0
        self._pos_index = positive_class
        self.threshold = threshold

    def predict(self, feature_string):
        ''' Return a prediction given a list of features '''
        self._example_count += 1

        # Process encoding
        feature_string = _sanitize_line(UnicodeDammit(feature_string, ['utf-8', 'windows-1252']).unicode_markup.strip())

        # Get current instances feature-value pairs
        field_pairs = feature_string.split()
        field_names = islice(field_pairs, 0, None, 2)
        field_values = islice(field_pairs, 1, None, 2)

        # Add the feature-value pairs to dictionary
        curr_info_dict = dict(izip(field_names, field_values))

        # Must make a list around a dictionary to fit format that Classifier.predict expects
        prediction_array = self._classifier.predict([{"y": None, "x": curr_info_dict, "id": "EXAMPLE_{}".format(self._example_count)}], None)

        if self._classifier.probability:
            return prediction_array[0, self._pos_index] if self.threshold is None else int(prediction_array[0, self._pos_index] >= self.threshold)
        else:
            return self._classifier.inverse_label_dict[int(prediction_array[0, 0])]


def main():
    ''' Main function that does all the work. '''
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Loads a trained model and outputs predictions based on input feature files.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler='resolve')
    parser.add_argument('model_prefix', help='Prefix to use when loading trained model (and its vocab).')
    parser.add_argument('feature_file', help='File containing list of space-delimited pairs of feature names and values, separated by spaces. For example: A 4.8 B 15 C 16.',
                        type=argparse.FileType('r'), nargs='+')
    parser.add_argument('-p', '--positive_class', help="If the model is only being used to predict the probability of a particular class, this specifies the index " +
                                                       "of the class we're predicting. 1 = second class, which is default for binary classification.", type=int)
    parser.add_argument('-t', '--threshold', help="If the model we're using is generating probabilities of the positive class, return 1 if it meets/exceeds " +
                                                  "the given threshold and 0 otherwise.", type=float)
    args = parser.parse_args()

    # Create the classifier and load the model
    predictor = Predictor(args.model_prefix, args.positive_class, args.threshold)

    for feature_file in args.feature_file:
        for line in feature_file:
            print(predictor.predict(line))


if __name__ == '__main__':
    main()
