#!/usr/bin/env python
'''
Jinja template file to run a particular train/test/feature-set combination.
'''

from __future__ import print_function, unicode_literals

import cPickle as pickle
import csv
import os
import re
import sys

import classifier

print("Training on {}, Test on {}, feature set {} ...".format({{ train_set_name }}, {{ test_set_name }}, {{ featureset }}), file=sys.stderr)

# instantiate the classifier wrapper
clf = classifier.Classifier()

# create the csv writer object
w = csv.writer(sys.stdout, dialect=csv.excel_tab)

# tunable parameters for each model type
tunable_parameters = {'dtree': ['max_depth', 'max_features'], \
                      'svm_linear': ['C'], \
                      'svm_radial': ['C', 'gamma'], \
                      'logistic': ['C'], \
                      'naivebayes': ['alpha'], \
                      'rforest': ['max_depth', 'max_features']}

# load the training and test examples
train_examples = clf.load_examples(os.path.join({{ train_path }}, '{}.tsv'.format({{ featureset }})))
test_examples = clf.load_examples(os.path.join({{ test_path }}, '{}.tsv'.format({{ featureset }})))

# the path where model and vocab files are stored; create if necessary
modelpath = {{ modelpath }}

# the name of the feature vocab file
vocabfile = os.path.join({{ vocabpath }}, '{}.vocab'.format({{ featureset }}))

# load the feature vocab if it already exists. We can do this since this is independent of the model type
if os.path.exists(vocabfile):
    print('  loading pre-existing feature vocab', file=sys.stderr)
    with open(vocabfile) as f:
        feat_vectorizer, scaler, label_dict, inverse_label_dict = pickle.load(f)
else:
    feat_vectorizer, scaler, label_dict, inverse_label_dict = [None] * 4

# now go over each classifier
for given_classifier in {{ given_classifiers }}:

    # check whether a trained model on the same data with the same featureset already exists
    # if so, load it (and the feature vocabulary) and then use it on the test data
    modelfile = os.path.join(modelpath, given_classifier, '{}.model'.format({{ featureset }}))

    # load the model if it already exists
    if os.path.exists(modelfile):
        print('  loading pre-existing {} model'.format(given_classifier), file=sys.stderr)
        with open(modelfile) as f:
            model = pickle.load(f)
    else:
        model = None

    # if we have do not have a saved model, we need to train one. However, we may be able to reuse a saved feature vocab file if that existed above.
    if not model:
        if feat_vectorizer:
            print('  training new {} model'.format(given_classifier), file=sys.stderr)
            model, best_score = clf.train(train_examples, feat_vectorizer=feat_vectorizer, scaler=scaler, label_dict=label_dict, model_type=given_classifier, modelfile=modelfile, grid_search={{ grid_search }}, grid_objective={{ grid_objective }})[0:2]
        else:
            print('  featurizing and training new {} model'.format(given_classifier), file=sys.stderr)
            model, best_score, feat_vectorizer, scaler, label_dict, inverse_label_dict = clf.train(train_examples, model_type=given_classifier, modelfile=modelfile, vocabfile=vocabfile, grid_search={{ grid_search }}, grid_objective={{ grid_objective }})

        # print out the tuned parameters and best CV score
        if {{ grid_search }}:
            param_out = []
            for param_name in tunable_parameters[given_classifier]:
                param_out.append('{}: {}'.format(param_name, model.get_params()[param_name]))
            print('  tuned hyperparameters: {}'.format(', '.join(param_out)), file=sys.stderr)
            print('  best score: {}'.format(round(best_score, 3)), file=sys.stderr)

    # run on test set or cross-validate on training data, depending on what was asked for
    if {{ cross_validate }}:
        print('  cross-validating', file=sys.stderr)
        results = clf.cross_validate(train_examples, model, feat_vectorizer, scaler, label_dict, inverse_label_dict, model_type=given_classifier, prediction_prefix={{ prediction_prefix }})
    elif {{ evaluate }}:
        print('  making predictions', file=sys.stderr)
        results = clf.evaluate(test_examples, model, feat_vectorizer, scaler, label_dict, inverse_label_dict, model_type=given_classifier, prediction_prefix={{ prediction_prefix }})
    else:
        print('  writing predictions', file=sys.stderr)
        clf.predict(test_examples, model, feat_vectorizer, scaler, label_dict, inverse_label_dict, {{ prediction_prefix }}, model_type=given_classifier)
        continue

    # write out the tsv row to STDOUT
    row = [{{ train_set_name }}, {{ test_set_name }}, {{ featureset }}, given_classifier]
    row.extend(results)
    w.writerow(row)
