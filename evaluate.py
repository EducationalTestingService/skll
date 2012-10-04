#!/usr/bin/env python
'''
Runs a bunch of sklearn jobs in parallel on the cluster given a config file.

@author: Nitin Madnani, nmadnani@ets.org
@author: Dan Blanchard, dblanchard@ets.org
'''

from __future__ import print_function

import argparse
import ConfigParser
import os
import re
import subprocess
import sys

import classifier
from jinja2 import Environment, FileSystemLoader


def clean_path(path):
    ''' Replace all weird SAN paths with normal paths '''

    path = re.sub(r'/\.automount/\w+/SAN/NLP/(\w+)-(dynamic|static)', r'/home/nlp-\1/\2', path)
    path = re.sub(r'/\.automount/[^/]+/SAN/Research/HomeResearch', '/home/research', path)
    return path


# Get command line arguments
parser = argparse.ArgumentParser(description="Runs a bunch of sklearn jobs in parallel on the cluster given a config file.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 conflict_handler='resolve')
parser.add_argument('config_file', help='Configuration file describing the sklearn task to run.')
args = parser.parse_args()

# initialize config parser
configurator = ConfigParser.RawConfigParser({'test_location': '', 'results': '', 'predictions': '', "grid_search": False, 'objective': "f1_score_micro"})
configurator.read(args.config_file)

# initialize the jinja2 environment
thispath = os.path.dirname(clean_path(os.path.abspath(__file__)))
jinja_env = Environment(loader=FileSystemLoader(thispath))
scriptpath = thispath

# extract sklearn parameters from the config file
given_classifiers = eval(configurator.get('Input', 'classifiers'))
given_lexicons = eval(configurator.get("Input", "lexicons"))
given_featuresets = eval(configurator.get("Input", "featuresets"))

# get all the input paths and directories
train_path = configurator.get("Input", "train_location")
test_path = configurator.get("Input", "test_location")

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
do_grid_search = configurator.get("Tuning", "grid_search")

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


# instantiate the classifier wrapper
clf = classifier.Classifier()

# the list of jobids submitted
jobids = []

# For each feature set
for featureset in given_featuresets:

    # instantiate the jinja template and run with qsub
    template = jinja_env.get_template('featureset.template.py')

    # store training/test set names for later use
    train_set_name = os.path.basename(train_path)
    test_set_name = os.path.basename(test_path)

    # instantiate the python template into a python script
    jobname = 'run_{}_{}_{}'.format(train_set_name, test_set_name, featureset)
    with open(os.path.join(scriptpath, '{}.py'.format(jobname)), "w") as scriptfh:

        # change the prediction prefix to include the feature set
        featset_prediction_prefix = prediction_prefix + '-' + featureset.replace('+', '-')

        # write out the rendered template into this script
        scriptfh.write(template.render(featureset=repr(featureset), given_classifiers=repr(given_classifiers), train_path=repr(train_path), test_path=repr(test_path),
                                       train_set_name=repr(train_set_name), test_set_name=repr(test_set_name), modelpath=repr(modelpath), vocabpath=repr(vocabpath),
                                       prediction_prefix=repr(featset_prediction_prefix), grid_search=do_grid_search, grid_objective=grid_objective_func,
                                       cross_validate=cross_validate, evaluate=evaluate))

        # the log file
        temp_logfile = os.path.join(logpath, '{}.log'.format(jobname))

        # the output file
        temp_outfile = os.path.join(resultspath, '{}.tsv'.format(jobname))

        # request 5 slots for each job if we are doing a grid search to make things go even faster
        if do_grid_search:
            if evaluate or cross_validate:
                command_str = "/local/research/linux/sge6_2u6/bin/lx24-amd64/qsub -q nlp.q -b y -o /dev/null -j y -N {} -pe smp 5 'python {} > {} 2> {}'".format(jobname,
                                                                                                                                                                 scriptfh.name,
                                                                                                                                                                 temp_outfile,
                                                                                                                                                                 temp_logfile)
            else:
                command_str = "/local/research/linux/sge6_2u6/bin/lx24-amd64/qsub -q nlp.q -b y -o /dev/null -j y -N {} -pe smp 5 'python {} 2> {}'".format(jobname,
                                                                                                                                                            scriptfh.name,
                                                                                                                                                            temp_logfile)
        else:
            if evaluate or cross_validate:
                command_str = "/local/research/linux/sge6_2u6/bin/lx24-amd64/qsub -q nlp.q -b y -o /dev/null -j y -N {} 'python {} > {} 2> {}'".format(jobname,
                                                                                                                                                       scriptfh.name,
                                                                                                                                                       temp_outfile,
                                                                                                                                                       temp_logfile)
            else:
                command_str = "/local/research/linux/sge6_2u6/bin/lx24-amd64/qsub -q nlp.q -b y -o /dev/null -j y -N {} 'python {} 2> {}'".format(jobname, scriptfh.name,
                                                                                                                                                  temp_logfile)

    # submit the job
    try:
        qsub_proc = subprocess.Popen(command_str, stderr=subprocess.PIPE, stdin=None, stdout=subprocess.PIPE, shell=True)
        stdout_data, stderr_data = qsub_proc.communicate()
        jobid = re.findall(r'[0-9]+', stdout_data)[0]
        jobids.append(jobid)
    except subprocess.CalledProcessError, cpe:
        print('Error {}: {}'.format(cpe.returncode, cpe.output), file=sys.stderr)
        sys.stderr.flush()
        sys.exit(2)
    else:
        if stderr_data:
            print("[STDERR: {}".format(stderr_data), file=sys.stderr)
            sys.stderr.flush()
        print("Train on {}, Test on {}, feature set {}submitted".format(train_set_name, test_set_name, featureset))

# run a job that cleans up and merges the results on disk
if evaluate or cross_validate:
    cleanup_script = os.path.join(thispath, "cleanup_evaluate.sh")
    command_str = "/local/research/linux/sge6_2u6/bin/lx24-amd64/qsub -q nlp.q -b y -o /dev/null -j y -N cleanup -hold_jid {} 'bash {} {} {} {} {} {}'".format(",".join(jobids),
                                                                                                                                                               cleanup_script,
                                                                                                                                                               resultspath,
                                                                                                                                                               scriptpath,
                                                                                                                                                               logpath,
                                                                                                                                                               resultsfile,
                                                                                                                                                               logfile)
elif predict:
    cleanup_script = os.path.join(thispath, "cleanup_predict.sh")
    command_str = "/local/research/linux/sge6_2u6/bin/lx24-amd64/qsub -q nlp.q -b y -o /dev/null -j y -N cleanup -hold_jid {} 'bash {} {} {} {}'".format(",".join(jobids),
                                                                                                                                                         cleanup_script,
                                                                                                                                                         scriptpath,
                                                                                                                                                         logpath,
                                                                                                                                                         logfile)

try:
    qsub_proc = subprocess.Popen(command_str, stderr=subprocess.PIPE, stdin=None, stdout=subprocess.PIPE, shell=True)
    stdout_data, stderr_data = qsub_proc.communicate()
except subprocess.CalledProcessError, cpe:
    print('Error {}: {}'.format(cpe.returncode, cpe.output), file=sys.stderr)
    sys.stderr.flush()
    sys.exit(2)
else:
    if stderr_data:
        print("[STDERR: {}".format(stderr_data), file=sys.stderr)
        sys.stderr.flush()
    print('Cleanup job submitted')
