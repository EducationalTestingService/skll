SciKit-Learn Laboratory
-----------------------

This package provides a number of utilities to make it simpler to run
common scikit-learn experiments with pre-generated features.

Command-line Interface
~~~~~~~~~~~~~~~~~~~~~~

``run_experiment`` is a command-line utility for running a series of
learners on datasets specified in a configuration file. For more
information about using run_experiment (including a quick example),
go `here <https://scikit-learn-laboratory.readthedocs.org/en/latest/run_experiment.html>`__.

Python API
~~~~~~~~~~

If you just want to avoid writing a lot of boilerplate learning code,
you can use our simple well-documented Python API. The main way you'll
want to use the API is through the ``load_examples`` function and the
``Learner`` class. For more details on how to simply train, test,
cross-validate, and run grid search on a variety of scikit-learn models
see `the documentation <https://scikit-learn-laboratory.readthedocs.org/en/latest/index.html>`__.

Requirements
~~~~~~~~~~~~

-  Python 2.7+
-  `scikit-learn <http://scikit-learn.org/stable/>`__
-  `six <https://pypi.python.org/pypi/six>`__
-  `PrettyTable <http://pypi.python.org/pypi/PrettyTable>`__
-  `Grid Map <http://pypi.python.org/pypi/gridmap>`__ (only required if you plan
   to run things in parallel on a DRMAA-compatible cluster)

Changelog
~~~~~~~~~


-  v0.9.2

   +  Fixed example.cfg path issue. Updated some documentation.
   +  Made path in make_example_iris_data.py consistent with the updated one
      in example.cfg

-  v0.9.1

   +  Fixed bug where classification experiments would raise an error about class
      labels not being floats
   +  Updated documentation to include quick example for run_experiment.