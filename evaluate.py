#!/usr/bin/env python

import ConfigParser
import os
import re
import subprocess
import sys

import classifier
from jinja2 import Environment, FileSystemLoader

args = sys.argv[1:]
if len(args) != 1:
    sys.stderr.write('Usage: evaluate.py <config_file>')
    sys.exit(2)
else:
    config_file = args[0]

configurator = ConfigParser.RawConfigParser()
configurator.read(config_file)

cleanpath = lambda path: re.sub(r'/\.automount/\w+/SAN/NLP/text-dynamic', '/home/nlp-text/dynamic', path)

# initialize the jinja2 environment
thispath = os.path.dirname(cleanpath(os.path.abspath(__file__)))
jinja_env = Environment(loader=FileSystemLoader(thispath))
scriptpath = thispath

# get the metadata about this run
run_name = config_file
run_desc = "" if not configurator.has_option("Info", "description") else configurator.get("Info", "description")
lexicon_info = "regular" if not configurator.has_option("Info", "lexicon_info") else configurator.get("Info", "lexicon_info")

# extract all the fields from the config file
given_classifiers = eval(configurator.get('Input', 'classifiers'))
given_lexicons = eval(configurator.get("Input", "lexicons"))
given_featuresets = eval(configurator.get("Input", "featuresets"))

# get all the input paths and directories
train_path = configurator.get("Input", "train_location")
test_path = "" if not configurator.has_option("Input", "test_location") else configurator.get("Input", "test_location")

# get all the output files and directories
resultsfile = "" if not configurator.has_option("Output", "results") else configurator.get("Output", "results")
resultspath = os.path.dirname(resultsfile) if resultsfile else ""
logfile = configurator.get("Output", "log")
logpath = os.path.dirname(logfile)
modelpath = configurator.get("Output", "models")
vocabpath = configurator.get("Output", "vocabs")

# create the path of the resultsfile, logfile and the modelpath
os.system("mkdir -p {} {}".format(resultspath, logpath))

# do we want to keep the predictions?
prediction_prefix = "" if not configurator.has_option("Output", "predictions") else configurator.get("Output", "predictions")
if prediction_prefix:
    predictdir = os.path.dirname(prediction_prefix)
    os.system("mkdir -p {}".format(predictdir))

# do we want to store the trees?
# keep_trees = False if not configurator.has_option("Output", "keep_trees") else eval(configurator.get("Output", "keep_trees"))
# treedir = "" if not configurator.has_option("Output", "treedir") else configurator.get("Output", "treedir")

# we need a treedir if we want to keep them
# if keep_trees and not bool(treedir):
#     sys.stderr.write("Error: you must specify keep_trees and treedir together.\n")
#     sys.exit(2)

# make sure all the specified paths exist
try:
    assert os.path.exists(train_path)
    if test_path:
        assert os.path.exists(test_path)
except AssertionError:
    sys.stderr.write("Error: a specified path in config file does not exist.\n")
    sys.exit(2)

# make sure all the given classifiers are valid as well
if set(given_classifiers).difference(set(['dtree', 'svm_linear', 'svm_radial', 'logistic', 'naivebayes', 'rforest'])):
    sys.stderr.write("Error: unrecognized classifier in config file.\n")
    sys.exit(2)

# do we need to run a grid search for the hyperparameters or are we just using the defaults
do_grid_search = False if not configurator.has_option("Tuning", "grid_search") else configurator.get("Tuning", "grid_search")

# what is the objective function for the grid search?
grid_objective_func = "f1_score_micro" if not configurator.has_option("Tuning", "objective") else configurator.get("Tuning", "objective")
if grid_objective_func not in ['f1_score_micro', 'f1_score_macro', 'accuracy']:
    sys.stderr.write('Error: invalid grid objective function.\n')
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
    sys.stderr.write('Error: you need to specify a prediction prefix if you are using prediction mode (no "results" option in config file).\n')
    sys.exit(2)


# instantiate the classifier wrapper
clf = classifier.Classifier()

# the list of jobids submitted
jobids = []

# For each lexicon
for lexicon in given_lexicons:

    # and then for each feature set
    for featureset in given_featuresets:

        # instantiate the jinja template and run with qsub
        template = jinja_env.get_template('featureset.template.py')

        # instantiate the python template into a python script
        jobname = 'run_{}_{}'.format(lexicon, featureset)
        scriptfh = file(os.path.join(scriptpath, '{}.py'.format(jobname)), "w")

        # change the prediction prefix to include the feature set
        featset_prediction_prefix = prediction_prefix + '-' + featureset.replace('+', '-')

        # write out the rendered template into this script
        scriptfh.write(template.render(lexicon=repr(lexicon), featureset=repr(featureset), lexicon_info=repr(lexicon_info), given_classifiers=repr(given_classifiers),
                                       train_path=repr(train_path), test_path=repr(test_path), test_set_name=repr(os.path.basename(test_path)), modelpath=repr(modelpath),
                                       vocabpath=repr(vocabpath), prediction_prefix=repr(featset_prediction_prefix), grid_search=do_grid_search,
                                       grid_objective=grid_objective_func, cross_validate=cross_validate, evaluate=evaluate))

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
                command_str = "/local/research/linux/sge6_2u6/bin/lx24-amd64/qsub -q nlp.q -b y -o /dev/null -j y -N {} 'python {} > {} 2> {}'".format(jobname, scriptfh.name,
                                                                                                                                                       temp_outfile, temp_logfile)
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
            sys.stderr.write('Error {}: {}\n'.format(cpe.returncode, cpe.output))
            sys.stderr.flush()
            sys.exit(2)
        else:
            if stderr_data:
                print >> sys.stderr, "[STDERR: {}".format(stderr_data)
                sys.stderr.flush()
            sys.stdout.write('Lexicon {}, Featureset {} submitted\n'.format(lexicon, featureset))

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
    sys.stderr.write('Error {}: {}\n'.format(cpe.returncode, cpe.output))
    sys.stderr.flush()
    sys.exit(2)
else:
    if stderr_data:
        print >> sys.stderr, "[STDERR: {}".format(stderr_data)
        sys.stderr.flush()
    sys.stdout.write('Cleanup job submitted\n')
