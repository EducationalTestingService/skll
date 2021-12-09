#!/usr/bin/env python
# License: BSD 3 clause
"""
Loads a trained model and outputs predictions based on input feature files.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
:date: February 2013
"""

import argparse
import logging
import os
import sys

import numpy as np

from skll.data.readers import EXT_TO_READER
from skll.learner import Learner
from skll.version import __version__


def main(argv=None):
    """
    Handles command line arguments and gets things started.

    Parameters
    ----------
    argv : list of str
        List of arguments, as if specified on the command-line.
        If None, ``sys.argv[1:]`` is used instead.
    """

    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Loads a trained model and outputs predictions based "
                    "on input feature files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('model_file',
                        help='Model file to load and use for generating '
                             'predictions.')
    parser.add_argument('input_files',
                        help='A space-separated list of CSV, TSV, or '
                             'jsonlines files (with or without the label '
                             'column), with the appropriate suffix.',
                        nargs='+')
    parser.add_argument('-i', '--id_col',
                        help='Name of the column which contains the instance '
                             'IDs in ARFF, CSV, or TSV files.',
                        default='id')
    parser.add_argument('-l', '--label_col',
                        help='Name of the column which contains the labels '
                             'in ARFF, CSV, or TSV files. For ARFF files, '
                             'this must be the final column to count as the '
                             'label.',
                        default='y')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p', '--predict_labels',
                       help="If the model is doing probabilistic "
                            "classification, output the class label with the "
                            "highest probability instead of the class "
                            "probabilities.",
                       action='store_true',
                       default=False)
    group.add_argument('-t', '--threshold',
                       help="If the model we're using is doing probabilistic "
                            "binary classification, output the positive class"
                            " label if its probability meets/exceeds this "
                            "threshold and output the negative class label "
                            "otherwise.",
                       type=float)
    parser.add_argument('-q', '--quiet',
                        help='Suppress printing of "Loading..." messages.',
                        action='store_true')
    parser.add_argument('-o', '--output_file',
                        help="Path to output tsv file. If not specified, "
                             "predictions will be printed to stdout. For "
                             "probabilistic binary classification, the "
                             "probability of the positive class will always "
                             "be in the last column.")
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')

    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - '
                                '%(message)s'))
    logger = logging.getLogger(__name__)

    # load the model from disk
    learner = Learner.from_file(args.model_file)

    # is the model a regressor or a classifier?
    estimator_type = learner.model_type._estimator_type

    # if we are using a binary classification model, get the positive
    # class label string if from the `pos_label` attribute, and if
    # that is `None`, get it from the learner's label dictionary; also
    # get the string denoting the negative label which is just the other
    # label in the list
    if estimator_type == 'classifier':
        if len(learner.label_list) == 2:
            if learner.pos_label is not None:
                pos_label = learner.pos_label
            else:
                pos_label = [label for label in learner.label_dict if learner.label_dict[label] == 1][0]
            neg_label = [label for label in learner.label_list if label != pos_label][0]
            logger.info(f"{pos_label} is the label for the positive "
                        "class.")

    # if we want to choose labels by thresholding the probabilities,
    # make sure that the learner is probabilistic AND binary first
    is_probabilistic_classifier = hasattr(learner._model, 'predict_proba') and learner.probability
    if (args.threshold is not None and
        (not is_probabilistic_classifier or
         len(learner.label_list) != 2)):
        error_msg = ('Cannot threshold probabilities to predict positive '
                     f'class since given {learner._model_type.__name__} '
                     'learner is either multi-class, non-probabilistic, or '
                     'was not trained with probability=True.')
        logger.error(error_msg)
        raise ValueError(error_msg)

    # if we want to choose labels by predicting the most likely label,
    # make sure that the learner is probabilistic
    if args.predict_labels and not is_probabilistic_classifier:
        error_msg = ('Cannot predict most likely labels from probabilities '
                     f'since given {learner._model_type.__name__} learner is '
                     'either non-probabilistic or was not trained with '
                     'probability=True.')
        logger.error(error_msg)
        raise ValueError(error_msg)

    # iterate over all the specified input files
    for i, input_file in enumerate(args.input_files):

        # make sure each file extension is one we can process
        input_extension = os.path.splitext(input_file)[1].lower()
        if input_extension not in EXT_TO_READER:
            logger.error(f"Input file must be in either .arff, .csv, "
                         f".jsonlines, .libsvm, .ndj, or .tsv format. "
                         f" Skipping file {input_file}")
            continue
        else:
            # read in the file into a featureset
            reader = EXT_TO_READER[input_extension](input_file,
                                                    quiet=args.quiet,
                                                    label_col=args.label_col,
                                                    id_col=args.id_col)
            feature_set = reader.read()

            # for this featureset, get the predictions of either the
            # most likely class labels or the class label probabilities;
            # if the model is a regressor then `class_labels` will be
            # ignored entirely
            original_predictions = learner.predict(feature_set,
                                                   class_labels=not learner.probability or args.predict_labels)

            # get the appropriate header depending on the what we will
            # be outputting; if we are using a regressor or a non-probabilistic
            # learner, or thresholding probabilities, or predicting most likely
            # labels, we are outputting only two columns - the ID and the label,
            # otherwise we are outputting N + 1 columns where N = number of classes
            if (estimator_type == 'regressor' or
                    not learner.probability or
                    args.predict_labels or
                    args.threshold is not None):
                header = ["id", "prediction"]
            else:
                header = ["id"] + [str(x) for x in learner.label_list]

            # now let us start computing what we want to output based
            # on the predictions we have so far

            # Threshold the positive class label probability
            if args.threshold is not None:
                predictions = []
                for neg_label_prob, pos_label_prob in original_predictions:
                    chosen_label = pos_label if pos_label_prob >= args.threshold else neg_label
                    predictions.append(chosen_label)

            # For everything else, we can just use the original predictions
            else:
                predictions = original_predictions

            # now initialize the output file
            outputfh = None
            try:
                outputfh = open(args.output_file, 'a') if args.output_file else sys.stdout

                # write out the header first but only once
                if i == 0:
                    print("\t".join(header), file=outputfh)

                # and now write out the predictions
                for j, prediction in enumerate(predictions):
                    id_ = feature_set.ids[j]
                    if isinstance(prediction, (np.ndarray, list)):
                        prediction_str = "\t".join([str(p) for p in prediction])
                    else:
                        prediction_str = prediction
                    print(f"{id_}\t{prediction_str}", file=outputfh)

            finally:

                # close the file if we had opened one
                if args.output_file:
                    outputfh.close()


if __name__ == '__main__':
    main()
