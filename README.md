## sklearn wrapper ##

This project consists of two utilities to make running common sklearn experiments on pre-generated features much simpler.

classifier.py contains a `Classifier` class and `load_examples` function that have a simple well-documented API for training, testing, cross-validation, and running grid search on a variety of sklearn models (for details see [the documentation](../blob/master/doc/index.html)).

run_experiment.py is a command-line utility for running a series of classifiers on datasets specified in a configuration file. For more information about the configuration file format, see the example configuration file `example.cfg`.

### Requirements ###

*  `pythongrid` must be in your `PYTHONPATH`
*  `export DRMAA_LIBRARY_PATH=/local/research/linux/sge6_2u6/lib/lx24-amd64/libdrmaa.so` must be present in your `.bashrc`