SciKit-Learn Laboratory
-----------------------

.. image:: https://api.travis-ci.org/EducationalTestingService/skll.png
   :alt: Build status
   :target: https://travis-ci.org/EducationalTestingService/skll

.. image:: https://coveralls.io/repos/EducationalTestingService/skll/badge.png?branch=master
    :target: https://coveralls.io/r/EducationalTestingService/skll

.. image:: https://pypip.in/d/skll/badge.png
   :target: https://crate.io/packages/skll
   :alt: PyPI downloads

.. image:: https://pypip.in/v/skll/badge.png
   :target: https://crate.io/packages/skll
   :alt: Latest version on PyPI

.. image:: https://d2weczhvl823v0.cloudfront.net/EducationalTestingService/skll/trend.png
   :alt: Bitdeli badge
   :target: https://bitdeli.com/free

This Python package provides utilities to make it easier to run
machine learning experiments with scikit-learn.

Command-line Interface
~~~~~~~~~~~~~~~~~~~~~~

``run_experiment`` is a command-line utility for running a series of learners on
datasets specified in a configuration file. For more information about using
run_experiment (including a quick example), go
`here <https://skll2.readthedocs.org/en/latest/run_experiment.html>`__.

Python API
~~~~~~~~~~

If you just want to avoid writing a lot of boilerplate learning code, you can
use our simple Python API. The main way you'll want to use the API is through
the ``load_examples`` function and the ``Learner`` class. For more details on
how to simply train, test, cross-validate, and run grid search on a variety of
scikit-learn models see
`the documentation <https://skll2.readthedocs.org/en/latest/index.html>`__.

A Note on Pronunciation
~~~~~~~~~~~~~~~~~~~~~~~

SciKit-Learn Laboratory (SKLL) is pronounced "skull": that's where the learning
happens.

Requirements
~~~~~~~~~~~~

-  Python 2.7+
-  `scikit-learn <http://scikit-learn.org/stable/>`__
-  `six <https://pypi.python.org/pypi/six>`__
-  `PrettyTable <http://pypi.python.org/pypi/PrettyTable>`__
-  `BeautifulSoup 4 <http://www.crummy.com/software/BeautifulSoup/>`__
-  `Grid Map <http://pypi.python.org/pypi/gridmap>`__ (only required if you plan
   to run things in parallel on a DRMAA-compatible cluster)
-  `configparser <http://pypi.python.org/pypi/configparser>`__ (only required for
   Python 2.7)

Changelog
~~~~~~~~~

-  v0.13.2

   +  Now raise an exception when using ``ids_to_floats`` with non-numeric IDs.
   +  Fixed a number of inconsistencies with ``cv_folds-location`` and
      ``ids_to_floats`` (including GH issue #57).
   +  Fixed unit tests for ``cv_folds_location`` and ``ids_to_floats`` so that
      they actually test the right things now.

-  v0.13.1

   +  Fixed crash when using ``cv_folds_location`` with ``ids_to_floats``.

-  v0.13.0

   +  Will now skip IDs that are missing from ``cv_folds``/``grid_search_folds``
      dicts and print a warning instead of crashing.
   +  Added additional kappa unit tests to help detect/prevent future issues.
   +  **API change:** ``model_type`` is no longer a keyword argument to
      ``Learner`` constructor, and is now required. This was done to help
      prevent unexpected issues from defaulting to ``LogisticRegression``.
   +  No longer keep extra temporary config files around when running
      ``run_experiment`` in ablation mode.

-  v0.12.0

   +  Fixed crash with kappa when given two sets of ratings that are both
      missing an intermediate value (e.g., ``[1, 2, 4]``).
   +  Added ``summarize_results`` script for creating a nice summary TSV file
      from a list of JSON results files.
   +  Summary files for ablation studies now have an extra column that says
      which feature was removed.

-  v0.11.0

   +  Added initial version of ``skll_convert`` script for converting between
      .jsonlines, .megam, and .tsv data file formats.
   +  Fixed bug in ``_megam_dict_iter`` where labels for instances with all zero
      features were being incorrectly set to ``None``.
   +  Fixed bug in ``_tsv_dict_iter`` where features with zero values were being
      retained with values set as '0' instead of being removed completely. This
      caused `DictVectorizer` to create extra features, so **results may
      change** a little bit if you were using .tsv files.
   +  Fixed crash with predict and train_only modes when running on the grid.
   +  No longer use process pools to load files if
      ``SKLL_MAX_CONCURRENT_PROCESSES`` is 1.
   +  Added more informative error message when trying to load a file without
      any features.

-  v0.10.1

   +  Made processes non-daemonic to fix ``pool.map`` issue with running
      multiple configurations files at the same time with ``run_experiment``.

-  v0.10.0

   +  ``run_experiment`` can now take multiple configuration files.
   +  Fixed issue where model parameters and scores were missing in ``evaluate``
      mode

-  v0.9.17

   +  Added ``skll.data.convert_examples`` function to convert a list
      dictionaries to an ExamplesTuple.
   +  Added a new optional field to configuration file, ``ids_to_floats``, to
      help save memory if you have a massive number of instances with numeric
      IDs.
   +  Replaced ``use_dense_features`` and ``scale_features`` options with
      ``feature_scaling``. See the
      `run_experiment documentation <http://skll2.readthedocs.org/en/latest/run_experiment.html#creating-configuration-files>`__
      for details.

-  v0.9.16

   +  Fixed summary output for ablation experiments. Previously summary files
      would not include all results.
   +  Added ablation unit tests.
   +  Fixed issue with generating PDF documentation.

-  v0.9.15

   +  Added two new *required* fields to the configuration file format under the
      ``General`` heading: ``experiment_name`` and ``task``. See the
      `run_experiment documentation <http://skll2.readthedocs.org/en/latest/run_experiment.html#creating-configuration-files>`__
      for details.
   +  Fixed an issue where the "loading..." message was never being printed when
      loading data files.
   +  Fixed a bug where keyword arguments were being ignored for metrics when
      calculating final scores for a tuned model. This means that **previous**
      **reported results may be wrong for tuning metrics that use keywords**
      **arguments**: ``f1_score_micro``, ``f1_score_macro``,
      ``linear_weighted_kappa``, and ``quadratic_weighted_kappa``.
   +  Now try to convert IDs to floats if they look like them to save
      memory for very large files.
   +  ``kappa`` now supports negative ratings.
   +  Fixed a crash when specifing ``grid_search_jobs`` and pre-specified folds.

-  v0.9.14

   +  Hotfix to fix issue where ``grid_search_jobs`` setting was being overriden
      by ``grid_search_folds``.

-  v0.9.13

   +  Added ``skll.data.write_feature_file`` (also available as
      ``skll.write_feature_file``) to simplify outputting .jsonlines, .megam,
      and .tsv files.
   +  Added more unit tests for handling .megam and .tsv files.
   +  Fixed a bug that caused a crash when using gridmap.
   +  ``grid_search_jobs`` now sets both ``n_jobs`` and ``pre_dispatch`` for
      ``GridSearchCV`` under the hood. This prevents a potential memory issue
      when dealing with large datasets and learners that cannot handle sparse
      data.
   +  Changed logging format when using ``run_experiment`` to be a little more
      readable.

-  v0.9.12

   +  Fixed serious issue where merging feature sets was not working correctly.
      **All experiments conducted using feature set merging** (i.e., where you
      specified a list of feature files and had them merged into one set for
      training/testing) **should be considered invalid**. In general, your
      results should previously have been poor and now should be much better.
   +  Added more verbose regression output including descriptive statistics
      and Pearson correlation.

-  v0.9.11

   +  Fixed all known remaining compatibility issues with Python 3.
   +  Fixed bug in ``skll.metrics.kappa`` which would raise an exception if full
      range of ratings was not seen in both ``y_true`` and ``y_pred``. Also
      added a unit test to prevent future regressions.
   +  Added missing configuration file that would cause a unit test to fail.
   +  Slightly refactored ``skll.Learner._create_estimator`` to make it a lot
      simpler to add new learners/estimators in the future.
   +  Fixed a bug in handling of sparse matrices that would cause a crash if
      the number of features in the training and the test set were not the same.
      Also added a corresponding unit test to prevent future regressions.
   +  We now require the backported configparser module for Python 2.7 to make
      maintaining compatibility with both 2.x and 3.x a lot easier.

-  v0.9.10

   +  Fixed bug introduced in v0.9.9 that broke ``predict`` mode.

-  v0.9.9

   +  Automatically generate a result summary file with all results for
      experiment in one TSV.
   +  Fixed bug where printing predictions to file would cause a crash with some
      learners.
   +  Run unit tests for Python 3.3 as well as 2.7.
   +  More unit tests for increased coverage.

-  v0.9.8

   +  Fixed crash due to trying to print name of grid objective which is now a
      str and not a function.
   +  Added --version option to shell scripts.

-  v0.9.7

   +  Can now use any objective function scikit-learn supports for tuning (i.e.,
      any valid argument for scorer when instantiating GridSearchCV) in addition
      to those we define.
   +  Removed ml_metrics dependency and we now support custom weights for kappa
      (through the API only so far).
   +  Require's scikit-learn 0.14+.
   +  ``accuracy``, ``quadratic_weighted_kappa``, ``unweighted_kappa``,
      ``f1_score_micro``, and ``f1_score_macro`` functions are no longer
      available under ``skll.metrics``. The accuracy and f1 score ones are no
      longer needed because we just use the built-in ones. As for
      quadratic_weighted_kappa and unweighted_kappa, they've been superseded by
      the kappa function that takes a weights argument.
   +  Fixed issue where you couldn't write prediction files if you were
      classifying using numeric classes.

-  v0.9.6

   +  Fixes issue with setup.py importing from package when trying to install
      it (for real this time).

-  v0.9.5

   +  You can now include feature files that don't have class labels in your
      featuresets. At least one feature file has to have a label though,
      because we only support supervised learning so far.
   +  **Important:** If you're using TSV files in your experiments, you should
      either name the class label column 'y' or use the new ``tsv_label`` option
      in your configuration file to specify the name of the label column. This
      was necessary to support feature files without labels.
   +  Fixed an issue with how version number was being imported in setup.py that
      would prevent installation if you didn't already have the prereqs
      installed on your machine.
   +  Made random seeds smaller to fix crash on 32-bit machines. This means that
      experiments run with previous versions of skll will yield slightly
      different results if you re-run them with v0.9.5+.
   +  Added ``megam_to_csv`` for converting .megam files to CSV/TSV files.
   +  Fixed a potential rounding problem with ``csv_to_megam`` that could
      slightly change feature values in conversion process.
   +  Cleaned up test_skll.py a little bit.
   +  Updated documentation to include missing fields that can be specified in
      config files.

-  v0.9.4

   +  Documentation fixes
   +  Added requirements.txt to manifest to fix broken PyPI release tarball.

-  v0.9.3

   +  Fixed bug with merging feature sets that used to cause a crash.
   +  If you're running scikit-learn 0.14+, we use their StandardScaler, since
      the bug fix we include in FixedStandardScaler is in there.
   +  Unit tests all pass again
   +  Lots of little things related to using travis (which do not affect users)

-  v0.9.2

   +  Fixed example.cfg path issue. Updated some documentation.
   +  Made path in make_example_iris_data.py consistent with the updated one
      in example.cfg

-  v0.9.1

   +  Fixed bug where classification experiments would raise an error about class
      labels not being floats
   +  Updated documentation to include quick example for run_experiment.


