import cPickle as pickle
import csv
import os
import re
import sys

import classifier

sys.stderr.write("Lexicon {}, feature set {} ... \n".format({{ lexicon }}, {{ featureset }}))

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
train_examples = clf.load_examples(os.path.join({{ train_path }}, '{}-{}.tsv'.format({{ lexicon }}, {{ featureset }})))
test_examples = clf.load_examples(os.path.join({{ test_path }}, '{}-{}.tsv'.format({{ lexicon }}, {{ featureset }})))

# the path where model and vocab files are stored; create if necessary
modelpath = {{ modelpath }}

# the name of the feature vocab file
vocabfile = os.path.join({{ vocabpath }}, '{}-{}.vocab'.format({{ lexicon }}, {{ featureset }}))

# load the feature vocab if it already exists. We can do this since this is independent of the model type
if os.path.exists(vocabfile):
    sys.stderr.write('  loading pre-existing feature vocab\n')
    feat_vectorizer, scaler, label_dict, inverse_label_dict = pickle.loads(open(vocabfile).read())
else:
    feat_vectorizer, scaler, label_dict, inverse_label_dict = [None] * 4

# now go over each classifier
for given_classifier in {{ given_classifiers }}:

    # check whether a trained model on the same data with the same featureset already exists
    # if so, load it (and the feature vocabulary) and then use it on the test data
    modelfile = os.path.join(modelpath, given_classifier, '{}-{}.model'.format({{ lexicon }}, {{ featureset }}))

    # load the model if it already exists
    if os.path.exists(modelfile):
        sys.stderr.write('  loading pre-existing {} model\n'.format(given_classifier))
        model = pickle.loads(open(modelfile).read())
    else:
        model = None

    # if we have do not have a saved model, we need to train one. However, we may be able to reuse a saved feature vocab file if that existed above.
    if not model:
        if feat_vectorizer:
            sys.stderr.write('  training new {} model\n'.format(given_classifier))
            model, best_score = clf.train_without_featurization(train_examples, feat_vectorizer, scaler, label_dict, model_type=given_classifier, modelfile=modelfile, grid_search={{ grid_search }}, grid_objective={{ grid_objective }})
        else:
            sys.stderr.write('  featurizing and training new {} model\n'.format(given_classifier))
            model, best_score, feat_vectorizer, scaler, label_dict, inverse_label_dict = clf.train(train_examples, model_type=given_classifier, modelfile=modelfile, vocabfile=vocabfile, grid_search={{ grid_search }}, grid_objective={{ grid_objective }})

        # print out the tuned parameters and best CV score
        if {{ grid_search }}:
            param_out = []
            for param_name in tunable_parameters[given_classifier]:
                param_out.append('{}: {}'.format(param_name, model.get_params()[param_name]))
            sys.stderr.write('  tuned hyperparameters: {}\n'.format(', '.join(param_out)))
            sys.stderr.write('  best score: {}\n'.format(round(best_score, 3)))

    # run on test set or cross-validate on training data, depending on what was asked for
    if {{ cross_validate }}:
        sys.stderr.write('  cross-validating\n')
        results = clf.cross_validate(train_examples, model, feat_vectorizer, scaler, label_dict, inverse_label_dict, model_type=given_classifier, prediction_prefix={{ prediction_prefix }})
    elif {{ evaluate }}:
        sys.stderr.write('  making predictions\n')
        results = clf.evaluate(test_examples, model, feat_vectorizer, scaler, label_dict, inverse_label_dict, model_type=given_classifier, prediction_prefix={{ prediction_prefix }})
    else:
        sys.stderr.write('  writing predictions\n')
        clf.predict(test_examples, model, feat_vectorizer, scaler, label_dict, inverse_label_dict, {{ prediction_prefix }}, model_type=given_classifier)
        continue

    # write out the tsv row to STDOUT
    row = [{{ test_set_name }}, {{ lexicon }}, {{ featureset }}, {{ lexicon_info }}, given_classifier]
    row.extend(results)
    w.writerow(row)
