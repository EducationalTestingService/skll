#!/usr/bin/env python
'''
Runs a bunch of sklearn jobs in parallel on the cluster given a config file.

@author: Nitin Madnani (nmadnani@ets.org)
@author: Dan Blanchard (dblanchard@ets.org)
'''

from __future__ import print_function, unicode_literals

import argparse
import ConfigParser
import re
import os
import sys
from collections import defaultdict, namedtuple

import classifier
from pythongrid import Job, process_jobs
from texttable import Texttable


# Named tuple for storing job results
ClassifierResultInfo = namedtuple('ClassifierResultInfo', ['train_set_name', 'test_set_name', 'featureset', 'given_classifier', 'task', 'task_results'])


def clean_path(path):
    ''' Replace all weird SAN paths with normal paths '''

    path = re.sub(r'/\.automount/\w+/SAN/NLP/(\w+)-(dynamic|static)', r'/home/nlp-\1/\2', path)
    path = re.sub(r'/\.automount/[^/]+/SAN/Research/HomeResearch', '/home/research', path)
    return path


def get_stat_string(class_result_dict, stat):
    ''' Little helper for getting output for precision, recall, and f-score columns in confusion matrix. '''
    if stat in class_result_dict and class_result_dict[stat]:
        return "{:.1f}%".format(class_result_dict[stat] * 100)
    else:
        return "N/A"


def print_fancy_output(result_tuples, output_file=sys.stdout):
    ''' Function to take all of the results from all of the folds and print nice tables with the resluts. '''
    num_folds = len(result_tuples)
    score_sum = 0.0
    prec_sum_dict = defaultdict(float)
    recall_sum_dict = defaultdict(float)
    f_sum_dict = defaultdict(float)
    classes = sorted(result_tuples[0][2].iterkeys())
    folds_with_class = defaultdict(int)
    for k, (conf_matrix, fold_score, result_dict) in enumerate(result_tuples, start=1):
        if num_folds > 1:
            print("\nFold: {}".format(k), file=output_file)
        result_table = Texttable(max_width=0)
        result_table.set_cols_align(["r"] * (len(classes) + 4))
        result_table.add_rows([[""] + classes + ["Precision", "Recall", "F-measure"]], header=True)
        for i, actual_class in enumerate(classes):
            conf_matrix[i][i] = "[{}]".format(conf_matrix[i][i])
            class_prec = get_stat_string(result_dict[actual_class], "Precision")
            class_recall = get_stat_string(result_dict[actual_class], "Recall")
            class_f = get_stat_string(result_dict[actual_class], "F-measure")
            if class_prec != 'N/A':
                folds_with_class[actual_class] += 1
                prec_sum_dict[actual_class] += float(class_prec[:-1])
                recall_sum_dict[actual_class] += float(class_recall[:-1])
                f_sum_dict[actual_class] += float(class_f[:-1])
                result_table.add_row([actual_class] + conf_matrix[i] + [class_prec, class_recall, class_f])
        print(result_table.draw(), file=output_file)
        print("(row = reference; column = predicted)", file=output_file)
        print("Accuracy = {:.1f}%\n".format(fold_score), file=output_file)
        score_sum += fold_score

    if num_folds > 1:
        print("\nAverage:", file=output_file)
        result_table = Texttable(max_width=0)
        result_table.set_cols_align(["l", "r", "r", "r"])
        result_table.add_rows([["Class", "Precision", "Recall", "F-measure"]], header=True)
        for actual_class in classes:
            if folds_with_class[actual_class]:
                result_table.add_row([actual_class] + ["{:.1f}%".format(prec_sum_dict[actual_class] / folds_with_class[actual_class]),
                                                       "{:.1f}%".format(recall_sum_dict[actual_class] / folds_with_class[actual_class]),
                                                       "{:.1f}%".format(f_sum_dict[actual_class] / folds_with_class[actual_class])])
        print(result_table.draw(), file=output_file)
        print("Accuracy = {:.1f}%".format(score_sum / num_folds), file=output_file)


def classify_featureset(jobname, featureset, given_classifier, train_path, test_path, train_set_name, test_set_name, modelpath, vocabpath, prediction_prefix, grid_search,
                        grid_objective, cross_validate, evaluate, suffix, log_path, probability, resultspath):
    ''' Classification job to be submitted to grid '''

    with open(log_path, 'w') as log_file:
        if cross_validate:
            print("Cross-validating on {}, feature set {} ...".format(train_set_name, featureset), file=log_file)
        else:
            print("Training on {}, Test on {}, feature set {} ...".format(train_set_name, test_set_name, featureset), file=log_file)

        # load the training and test examples
        train_examples = classifier.load_examples(os.path.join(train_path, featureset + suffix))
        if not cross_validate:
            test_examples = classifier.load_examples(os.path.join(test_path, featureset + suffix))

        # the name of the feature vocab file
        vocabfile = os.path.join(vocabpath, '{}.vocab'.format(featureset))

        # initialize a classifer object
        learner = classifier.Classifier(probability=probability, model_type=given_classifier)

        # check whether a trained model on the same data with the same featureset already exists
        # if so, load it (and the feature vocabulary) and then use it on the test data
        modelfile = os.path.join(modelpath, '{}.model'.format(jobname))

        # load the feature vocab if it already exists. We can do this since this is independent of the model type
        if os.path.exists(vocabfile):
            print('\tloading pre-existing feature vocab', file=log_file)
            learner.load_vocab(vocabfile)

        # check if we're doing cross-validation, because we only load/save models when we're not.
        if not cross_validate:

            # load the model if it already exists
            if os.path.exists(modelfile):
                print('\tloading pre-existing {} model'.format(given_classifier), file=log_file)
                learner.load_model(modelfile)

            # if we have do not have a saved model, we need to train one. However, we may be able to reuse a saved feature vocab file if that existed above.
            else:
                if learner.feat_vectorizer:
                    print('\ttraining new {} model'.format(given_classifier), file=log_file)
                    best_score = learner.train(train_examples, grid_search=grid_search, grid_objective=grid_objective)
                else:
                    print('\tfeaturizing and training new {} model'.format(given_classifier), file=log_file)
                    best_score = learner.train(train_examples, grid_search=grid_search, grid_objective=grid_objective)

                    # save vocab
                    learner.save_vocab(vocabfile)

                # save model
                learner.save_model(modelfile)

                if grid_search:
                    print('\tbest {} score: {}'.format(grid_objective.__name__, round(best_score, 3)), file=log_file)

            # print out the tuned parameters and best CV score
            param_out = ('{}: {}'.format(param_name, param_value) for param_name, param_value in learner.model.get_params().iteritems())
            print('\thyperparameters: {}'.format(', '.join(param_out)), file=log_file)

        # run on test set or cross-validate on training data, depending on what was asked for
        if cross_validate:
            print('\tcross-validating', file=log_file)
            results = learner.cross_validate(train_examples, prediction_prefix=prediction_prefix, grid_search=grid_search, grid_objective=grid_objective)
            task = 'cross-validate'
        elif evaluate:
            print('\tevaluating predictions', file=log_file)
            results = [learner.evaluate(test_examples, prediction_prefix=prediction_prefix)]
            task = 'evaluate'
        else:
            print('\twriting predictions', file=log_file)
            task = 'predict'
            learner.predict(test_examples, prediction_prefix)

        # write out results to file if we're not predicting
        result_info = ClassifierResultInfo(train_set_name, test_set_name, featureset, given_classifier, task, results)
        if task != 'predict':
            with open(os.path.join(resultspath, '{}.results'.format(jobname)), 'w') as output_file:
                print_fancy_output(result_info.task_results, output_file)

    return result_info


def run_configuration(config_file, local=False):
    ''' Takes a configuration file and runs the specified jobs on the grid. '''
    # initialize config parser
    configurator = ConfigParser.RawConfigParser({'test_location': '', 'log': '', 'results': '', 'predictions': '', "grid_search": False, 'objective': "f1_score_micro",
                                                 'probability': False})
    configurator.read(config_file)

    # extract sklearn parameters from the config file
    given_classifiers = eval(configurator.get('Input', 'classifiers'))
    given_featuresets = eval(configurator.get("Input", "featuresets"))

    # get all the input paths and directories
    train_path = configurator.get("Input", "train_location").rstrip('/')  # remove trailing / at the end of path name
    test_path = configurator.get("Input", "test_location").rstrip('/')
    suffix = configurator.get("Input", "suffix")

    # get all the output files and directories
    resultspath = configurator.get("Output", "results")
    logpath = configurator.get("Output", "log")
    modelpath = configurator.get("Output", "models")
    vocabpath = configurator.get("Output", "vocabs")
    probability = eval(configurator.get("Output", "probability"))

    # do we want to keep the predictions?
    prediction_dir = configurator.get("Output", "predictions")
    if prediction_dir:
        os.system("mkdir -p {}".format(prediction_dir))

    # make sure log path exists
    if logpath:
        os.system("mkdir -p {}".format(logpath))

    # make sure results path exists
    if resultspath:
        os.system("mkdir -p {}".format(resultspath))

    # make sure all the specified paths exist
    if not os.path.exists(train_path):
        print("Error: the training path specified in config file ({}) does not exist.".format(train_path), file=sys.stderr)
        sys.exit(2)
    if test_path and not os.path.exists(test_path):
        print("Error: the test path specified in config file ({}) does not exist.".format(test_path), file=sys.stderr)
        sys.exit(2)

    # do we need to run a grid search for the hyperparameters or are we just using the defaults
    do_grid_search = eval(configurator.get("Tuning", "grid_search"))

    # what is the objective function for the grid search?
    grid_objective_func = configurator.get("Tuning", "objective")
    if grid_objective_func not in {'f1_score_micro', 'f1_score_macro', 'accuracy', 'f1_score_least_frequent'}:
        print('Error: invalid grid objective function.', file=sys.stderr)
        sys.exit(2)
    else:
        grid_objective_func = 'classifier.' + grid_objective_func
    grid_objective = eval(grid_objective_func)

    # are we doing cross validation or actual testing or just generating predictions on a new test set?
    # If no test set was specified then assume that we are doing cross validation.
    # If the results field was not specified then assume that we are just generating predictions
    evaluate = False
    cross_validate = False
    predict = False
    if test_path and resultspath:
        evaluate = True
    elif not test_path:
        cross_validate = True
    else:
        predict = True

    if cross_validate:
        task = 'cross-validate'
    elif evaluate:
        task = 'evaluate'
    else:
        task = 'predict'

    # make sure that, if we are in prediction mode, we have a prediction_dir
    if predict and not prediction_dir:
        print('Error: you need to specify a prediction directory if you are using prediction mode (no "results" option in config file).', file=sys.stderr)
        sys.exit(2)

    # the list of jobs submitted (if running on grid)
    if not local:
        jobs = []

    # For each feature set
    for featureset in given_featuresets:

        # and for each classifier
        for given_classifier in given_classifiers:

            # store training/test set names for later use
            train_set_name = os.path.basename(train_path)
            test_set_name = os.path.basename(test_path) if test_path else "cv"

            # create a name for the job
            jobname = '{}_{}_{}_{}_{}_{}_{}'.format(train_set_name, test_set_name, featureset, given_classifier, "untuned" if not do_grid_search else "tuned",
                                                    grid_objective.__name__, task)

            # change the prediction prefix to include the feature set
            prediction_prefix = os.path.join(prediction_dir, jobname)

            # the log file that stores the actual output of this script (e.g., the tuned parameters, what kind of experiment was run, etc.)
            temp_logfile = os.path.join(logpath, '{}.log'.format(jobname))

            # create job if we're doing things on the grid
            if not local:
                job = Job(classify_featureset, [jobname, featureset, given_classifier, train_path, test_path, train_set_name, test_set_name,
                                                modelpath, vocabpath, prediction_prefix, do_grid_search, eval(grid_objective_func), cross_validate,
                                                evaluate, suffix, temp_logfile, probability, resultspath], num_slots=(5 if do_grid_search else 1), name=jobname)

                # Add job to list
                jobs.append(job)
            else:
                classify_featureset(jobname, featureset, given_classifier, train_path, test_path, train_set_name, test_set_name, modelpath, vocabpath, prediction_prefix,
                                    do_grid_search, eval(grid_objective_func), cross_validate, evaluate, suffix, temp_logfile, probability, resultspath)

    # submit the jobs (if running on grid)
    if not local:
        job_results = process_jobs(jobs)

        # Check for errors
        for result_info in job_results:
            if not hasattr(result_info, 'task'):
                print('There was an error running the experiment:\n{}'.format(result_info), file=sys.stderr)
                sys.exit(2)


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Runs a bunch of sklearn jobs in parallel on the cluster given a config file.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler='resolve')
    parser.add_argument('config_file', help='Configuration file describing the sklearn task to run.')
    parser.add_argument('-l', '--local', help='Do not use the Grid Engine for running jobs and just run everything sequential on the local machine. This is for debugging.',
                        action='store_true')
    args = parser.parse_args()

    run_configuration(args.config_file, args.local)
