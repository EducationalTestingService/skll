#!/usr/bin/env python
# License: BSD 3 clause
"""
Loads a trained model and outputs predictions based on input feature files.

:author: Dan Blanchard
:contact: dblanchard@ets.org
:organization: ETS
:date: February 2013
"""

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import logging
import os

from skll.data.readers import EXT_TO_READER
from skll.learner import Learner
from skll.version import __version__


class Predictor(object):
    """
    Little wrapper around a ``Learner`` to load models and get
    predictions for feature strings.
    """

    def __init__(self, model_path, threshold=None, positive_label=1):
        """
        Initialize the predictor.

        :param model_path: Path to use when loading trained model.
        :type model_path: str
        :param threshold: If the model we're using is generating probabilities
                          of the positive label, return 1 if it meets/exceeds
                          the given threshold and 0 otherwise.
        :type threshold: float
        :param positive_label: If the model is only being used to predict the
                               probability of a particular class, this
                               specifies the index of the class we're
                               predicting. 1 = second class, which is default
                               for binary classification.
        :type positive_label: int
        """
        self._learner = Learner.from_file(model_path)
        self._pos_index = positive_label
        self.threshold = threshold

    def predict(self, data):
        """
        Generate a list of predictions for the given examples.

        :param data: FeatureSet to get predictions for.
        :type data: FeatureSet

        :returns: A list of predictions the model generated for the given
                  examples.
        """
        # compute the predictions from the learner
        preds = self._learner.predict(data)
        preds = preds.tolist()

        if self._learner.probability:
            if self.threshold is None:
                return [pred[self._pos_index] for pred in preds]
            else:
                return [int(pred[self._pos_index] >= self.threshold)
                        for pred in preds]
        elif self._learner.model._estimator_type == 'regressor':
            return preds
        else:
            return [self._learner.label_list[pred if isinstance(pred, int) else
                                             int(pred[0])] for pred in preds]


def main(argv=None):
    """
    Handles command line arguments and gets things started.

    :param argv: List of arguments, as if specified on the command-line.
                 If None, ``sys.argv[1:]`` is used instead.
    :type argv: list of str
    """
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Loads a trained model and outputs predictions based \
                     on input feature files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('model_file',
                        help='Model file to load and use for generating \
                              predictions.')
    parser.add_argument('input_file',
                        help='A csv file, json file, or megam file \
                              (with or without the label column), \
                              with the appropriate suffix.',
                        nargs='+')
    parser.add_argument('-i', '--id_col',
                        help='Name of the column which contains the instance \
                              IDs in ARFF, CSV, or TSV files.',
                        default='id')
    parser.add_argument('-l', '--label_col',
                        help='Name of the column which contains the labels\
                              in ARFF, CSV, or TSV files. For ARFF files, this\
                              must be the final column to count as the label.',
                        default='y')
    parser.add_argument('-p', '--positive_label',
                        help="If the model is only being used to predict the \
                              probability of a particular label, this \
                              specifies the index of the label we're \
                              predicting. 1 = second label, which is default \
                              for binary classification. Keep in mind that \
                              labels are sorted lexicographically.",
                        default=1, type=int)
    parser.add_argument('-q', '--quiet',
                        help='Suppress printing of "Loading..." messages.',
                        action='store_true')
    parser.add_argument('-t', '--threshold',
                        help="If the model we're using is generating \
                              probabilities of the positive label, return 1 \
                              if it meets/exceeds the given threshold and 0 \
                              otherwise.",
                        type=float)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'))
    logger = logging.getLogger(__name__)

    # Create the classifier and load the model
    predictor = Predictor(args.model_file,
                          positive_label=args.positive_label,
                          threshold=args.threshold)

    # Iterate over all the specified input files
    for input_file in args.input_file:

        # make sure each file extension is one we can process
        input_extension = os.path.splitext(input_file)[1].lower()
        if input_extension not in EXT_TO_READER:
            logger.error(('Input file must be in either .arff, .csv, '
                          '.jsonlines, .libsvm, .megam, .ndj, or .tsv format. '
                          ' Skipping file {}').format(input_file))
            continue
        else:
            # Iterate through input file and collect the information we need
            reader = EXT_TO_READER[input_extension](input_file,
                                                    quiet=args.quiet,
                                                    label_col=args.label_col,
                                                    id_col=args.id_col)
            feature_set = reader.read()
            for pred in predictor.predict(feature_set):
                print(pred)


if __name__ == '__main__':
    main()
