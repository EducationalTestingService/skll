#!/usr/bin/env python
'''
Runs a bunch of sklearn jobs in parallel on the cluster given a config file.

@author: Nitin Madnani, nmadnani@ets.org
@author: Dan Blanchard, dblanchard@ets.org
'''

from __future__ import print_function, unicode_literals

import argparse
import cPickle as pickle
import ConfigParser
import re
import os
import sys
from collections import namedtuple

import classifier
from pythongrid import Job, process_jobs


# Named tuple for storing job results
ClassifierResultInfo = namedtuple('ClassifierResultInfo', ['train_set_name', 'test_set_name', 'featureset', 'given_classifier', 'task', 'task_results'])


def clean_path(path):
    ''' Replace all weird SAN paths with normal paths '''

    path = re.sub(r'/\.automount/\w+/SAN/NLP/(\w+)-(dynamic|static)', r'/home/nlp-\1/\2', path)
    path = re.sub(r'/\.automount/[^/]+/SAN/Research/HomeResearch', '/home/research', path)
    return path


def classify_featureset(featureset, given_classifiers, train_path, test_path, train_set_name, test_set_name, modelpath, vocabpath, prediction_prefix, grid_search,
                        grid_objective, cross_validate, evaluate, suffix, log_path):
    ''' Classification job to be submitted to grid '''
    result_list = []

    with open(log_path, 'w') as log_file:
        if cross_validate:
            print("Cross-validating on {}, feature set {} ...".format(train_set_name, featureset), file=log_file)
        else:
            print("Training on {}, Test on {}, feature set {} ...".format(train_set_name, test_set_name, featureset), file=log_file)

        # tunable parameters for each model type
        tunable_parameters = {'dtree': ['max_depth', 'max_features'], \
                              'svm_linear': ['C'], \
                              'svm_radial': ['C', 'gamma'], \
                              'logistic': ['C'], \
                              'naivebayes': ['alpha'], \
                              'rforest': ['max_depth', 'max_features']}

        # load the training and test examples
        train_examples = classifier.load_examples(os.path.join(train_path, featureset + suffix))
        if not cross_validate:
            test_examples = classifier.load_examples(os.path.join(test_path, featureset + suffix))

        # the name of the feature vocab file
        vocabfile = os.path.join(vocabpath, '{}.vocab'.format(featureset))

        # load the feature vocab if it already exists. We can do this since this is independent of the model type
        if os.path.exists(vocabfile):
            print('\tloading pre-existing feature vocab', file=log_file)
            with open(vocabfile) as f:
                feat_vectorizer, scaler, label_dict, inverse_label_dict = pickle.load(f)
        else:
            feat_vectorizer, scaler, label_dict, inverse_label_dict = [None] * 4

        # now go over each classifier
        for given_classifier in given_classifiers:

            # check whether a trained model on the same data with the same featureset already exists
            # if so, load it (and the feature vocabulary) and then use it on the test data
            modelfile = os.path.join(modelpath, given_classifier, '{}.model'.format(featureset))

            # load the model if it already exists
            if os.path.exists(modelfile):
                print('\tloading pre-existing {} model'.format(given_classifier), file=log_file)
                with open(modelfile) as f:
                    model = pickle.load(f)
            else:
                model = None

            # if we have do not have a saved model, we need to train one. However, we may be able to reuse a saved feature vocab file if that existed above.
            if not model:
                if feat_vectorizer:
                    print('\ttraining new {} model'.format(given_classifier), file=log_file)
                    model, best_score = classifier.train(train_examples, feat_vectorizer=feat_vectorizer, scaler=scaler, label_dict=label_dict, model_type=given_classifier,
                                                         modelfile=modelfile, grid_search=grid_search, grid_objective=grid_objective, inverse_label_dict=inverse_label_dict)[0:2]
                else:
                    print('\tfeaturizing and training new {} model'.format(given_classifier), file=log_file)
                    model, best_score, feat_vectorizer, scaler, label_dict, inverse_label_dict = classifier.train(train_examples, model_type=given_classifier, modelfile=modelfile,
                                                                                                                  vocabfile=vocabfile, grid_search=grid_search,
                                                                                                                  grid_objective=grid_objective)

                # print out the tuned parameters and best CV score
                if grid_search:
                    param_out = []
                    for param_name in tunable_parameters[given_classifier]:
                        param_out.append('{}: {}'.format(param_name, model.get_params()[param_name]))
                    print('\ttuned hyperparameters: {}'.format(', '.join(param_out)), file=log_file)
                    print('\tbest score: {}'.format(round(best_score, 3)), file=log_file)

            # run on test set or cross-validate on training data, depending on what was asked for
            if cross_validate:
                print('\tcross-validating', file=log_file)
                results = classifier.cross_validate(train_examples, model, feat_vectorizer, scaler, label_dict, inverse_label_dict, model_type=given_classifier,
                                                    prediction_prefix=prediction_prefix)
                task = 'cross-validate'
            elif evaluate:
                print('\tevaluating predictions', file=log_file)
                results = classifier.evaluate(test_examples, model, feat_vectorizer, scaler, label_dict, inverse_label_dict, model_type=given_classifier,
                                              prediction_prefix=prediction_prefix)
                task = 'evaluate'
            else:
                print('\twriting predictions', file=log_file)
                classifier.predict(test_examples, model, feat_vectorizer, scaler, label_dict, inverse_label_dict, prediction_prefix, model_type=given_classifier)
                continue

            # write out the tsv row to STDOUT
            result_list.append(ClassifierResultInfo(train_set_name, test_set_name, featureset, given_classifier, task, results))

    return result_list


def run_configuration(config_file):
    ''' Takes a configuration file and runs the specified jobs on the grid. '''
    # initialize config parser
    configurator = ConfigParser.RawConfigParser({'test_location': '', 'results': '', 'predictions': '', "grid_search": False, 'objective': "f1_score_micro"})
    configurator.read(config_file)

    # extract sklearn parameters from the config file
    given_classifiers = eval(configurator.get('Input', 'classifiers'))
    given_featuresets = eval(configurator.get("Input", "featuresets"))

    # get all the input paths and directories
    train_path = configurator.get("Input", "train_location").rstrip('/')  # remove trailing / at the end of path name
    test_path = configurator.get("Input", "test_location").rstrip('/')
    suffix = configurator.get("Input", "suffix")

    # get all the output files and directories
    resultsfile = configurator.get("Output", "results")
    resultspath = os.path.dirname(resultsfile) if resultsfile else ""
    logfile = configurator.get("Output", "log")
    logpath = os.path.dirname(logfile)
    modelpath = configurator.get("Output", "models")
    vocabpath = configurator.get("Output", "vocabs")

    # create the path of the resultsfile, logfile and the modelpath
    os.system("mkdir -p {} {}".format(resultspath, logpath))

    # do we want to keep the predictions?
    prediction_prefix = configurator.get("Output", "predictions")
    if prediction_prefix:
        predictdir = os.path.dirname(prediction_prefix)
        os.system("mkdir -p {}".format(predictdir))

    # make sure all the specified paths exist
    if not (os.path.exists(train_path) and (not test_path or os.path.exists(test_path))):
        print("Error: the training and/or test path(s) specified in config file does not exist.", file=sys.stderr)
        sys.exit(2)

    # make sure all the given classifiers are valid as well
    if set(given_classifiers).difference(set(['dtree', 'svm_linear', 'svm_radial', 'logistic', 'naivebayes', 'rforest'])):
        print("Error: unrecognized classifier in config file.", file=sys.stderr)
        sys.exit(2)

    # do we need to run a grid search for the hyperparameters or are we just using the defaults
    do_grid_search = eval(configurator.get("Tuning", "grid_search"))

    # what is the objective function for the grid search?
    grid_objective_func = configurator.get("Tuning", "objective")
    if grid_objective_func not in ['f1_score_micro', 'f1_score_macro', 'accuracy']:
        print('Error: invalid grid objective function.', file=sys.stderr)
        sys.exit(2)
    else:
        grid_objective_func = 'classifier.' + grid_objective_func

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

    # make sure that, if we are in prediction mode,we have a prediction_prefix
    if predict and not prediction_prefix:
        print('Error: you need to specify a prediction prefix if you are using prediction mode (no "results" option in config file).', file=sys.stderr)
        sys.exit(2)


    # the list of jobs submitted
    jobs = []

    # For each feature set
    for featureset in given_featuresets:
        # store training/test set names for later use
        train_set_name = os.path.basename(train_path)
        test_set_name = os.path.basename(test_path) if test_path else "cv"

        # create a name for the job
        jobname = 'run_{}_{}_{}'.format(train_set_name, test_set_name, featureset)

        # change the prediction prefix to include the feature set
        featset_prediction_prefix = prediction_prefix + '-' + featureset.replace('+', '-')

        # the log file
        temp_logfile = os.path.join(logpath, '{}.log'.format(jobname))

        # create job
        job = Job(classify_featureset, [featureset, given_classifiers, train_path, test_path, train_set_name, test_set_name,
                                        modelpath, vocabpath, featset_prediction_prefix, do_grid_search, eval(grid_objective_func), cross_validate,
                                        evaluate, suffix, temp_logfile], num_slots=(5 if do_grid_search else 1), name=jobname)

        # Add job to list
        jobs.append(job)

    # submit the jobs
    job_results = process_jobs(jobs)

    # Print out results
    with open(resultsfile, 'w') as output_file:
        for result_tuple_list in job_results:
            for result_tuple in result_tuple_list:
                print(result_tuple, file=output_file)


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Runs a bunch of sklearn jobs in parallel on the cluster given a config file.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler='resolve')
    parser.add_argument('config_file', help='Configuration file describing the sklearn task to run.')
    args = parser.parse_args()

    run_configuration(args.config_file)
