#!/usr/bin/env python
'''
Runs an ablation study, removing one feature file at a time.

@author: Michael Heilman (mheilman@ets.org)
'''


from six.moves import configparser
from six import string_types, iterkeys, iteritems, itervalues  # Python 2/3
import argparse
import json
import re
from run_experiment import fix_json, run_configuration
from multiprocessing import Pool


def run_experiment_without_feature(feature_type, given_features, config, args, machines):
    featureset = [[x for x in given_features if x != feature_type]]
    featureset_name = "{}_minus_{}".format(given_featureset_name, 
                                           feature_type)

    config.set("Input", "featuresets", json.dumps(featureset))
    config.set("Input", "featureset_names", "['{}']".format(featureset_name))

    m = re.search(r'^(.*)\.cfg$', args.config_file)
    if not m:
        raise ValueError("Configuration file should end in .cfg.")
    new_config_path = "{}_minus_{}.cfg".format(m.groups()[0], feature_type)

    with open(new_config_path, 'w') as new_config_file:
        config.write(new_config_file)

    with open(new_config_path, 'r') as new_config_file:
        run_configuration(new_config_file,
                          local=args.local,
                          queue=args.queue,
                          hosts=machines)


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Runs sklearn jobs for ablation study, given a \
                     config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('config_file',
                        help='Configuration file describing the sklearn task\
                              to run.',
                        type=argparse.FileType('r'))
    parser.add_argument('-l', '--local',
                        help='Do not use the Grid Engine for running jobs and\
                              just run everything sequential on the local \
                              machine. This is for debugging.',
                        action='store_true')
    parser.add_argument('-m', '--machines',
                        help="comma-separated list of machines to add to\
                              pythongrid's whitelist (if not specified, all\
                              available machines are used). Note that full \
                              names must be specified, e.g., \
                              \"nlp.research.ets.org\"",
                        type=str, default=None)
    parser.add_argument('-q', '--queue',
                        help="Use this queue for python grid.",
                        type=str, default='nlp.q')

    args = parser.parse_args()
    machines = None
    if args.machines:
        machines = args.machines.split(',')


    config = configparser.SafeConfigParser()
    config.readfp(args.config_file)

    given_featuresets = json.loads(fix_json(config.get("Input",
                                                       "featuresets")))
    given_featureset_names = json.loads(fix_json(config.get("Input", 
                                                 "featureset_names")))

    # make sure there is only one list of features
    if (isinstance(given_featuresets[0], list) and len(given_featuresets) > 0) \
       or (isinstance(given_featureset_names[0], list) 
       and len(given_featureset_names) > 0):
        raise ValueError("More than one feature set or list of names given.")

    # make a list of features rather than a list of lists
    given_features = given_featuresets[0]
    given_featureset_name = given_featureset_names[0]

    pool = Pool(processes=len(given_features))

    # for each feature file, make a copy of the config file
    # with all but that feature, and run the jobs.
    pool.apply_async(run_experiment_without_feature, 
                     [(feature_type, given_features, config, args, machines) for 
                      feature_type in given_features])
        
