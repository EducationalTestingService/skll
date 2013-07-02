## SciKit-Learn Lab ##

This package provides a number of utilities to make it simpler to run common scikit-learn experiments with pre-generated features.

### Command-line Interface ###

`run_experiment.py` is a command-line utility for running a series of learners on datasets specified in a configuration file. For more information about the configuration file format, see the example configuration file [example.cfg](../blob/master/example.cfg).


### Python API ###

If you just want to avoid writing a lot of boilerplate learning code, you can use our simple well-documented Python API. The main way you'll want to use the API is through the `load_examples` function and the `Learner` class. For more details on how to simply train, test, cross-validate, and run grid search on a variety of scikit-learn models see [the documentation](../raw/master/doc/_build/html/index.html).


### Requirements ###

* Python 2.7+
* [scikit-learn](http://scikit-learn.org/stable/)
* [six](https://pypi.python.org/pypi/six)
* [PrettyTable](https://pypi.python.org/pypi/PrettyTable)
* To automatically run things in parallel on a [DRMAA](http://www.drmaa.org/)-compatible cluster, you must have the [ETS fork of Python Grid](https://github.com/EducationalTestingService/pythongrid) in your `PYTHONPATH`, and `DRMAA_LIBRARY_PATH` must be set to the path to `libdrmaa.so` for your cluster installation.