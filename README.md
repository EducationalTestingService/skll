sklearn wrapper
---------------

This project consists of two utilities to make running common sklearn experiments on pre-generated features much simpler.

classifier.py contains a `Classifier` class and `load_examples` function that have a simple well-documented API for training, testing, cross-validation, and running grid search on a variety of sklearn models (for details see the documentation).

run_experiment.py is a command-line utility for running a series of classifiers on datasets specified in a configuration file. A sample configuration file is provided in the configs directory.