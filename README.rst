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
`here <https://skll.readthedocs.org/en/latest/run_experiment.html>`__.

Python API
~~~~~~~~~~

If you just want to avoid writing a lot of boilerplate learning code, you can
use our simple Python API. The main way you'll want to use the API is through
the ``load_examples`` function and the ``Learner`` class. For more details on
how to simply train, test, cross-validate, and run grid search on a variety of
scikit-learn models see
`the documentation <https://skll.readthedocs.org/en/latest/index.html>`__.

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

Changelog
~~~~~~~~~

-  v0.9.7

   +  Can now use any objective function scikit-learn supports for tuning (i.e.,
      any valid argument for scorer when instantiating GridSearchCV) in addition
      to those we define.
   +  Removed ml_metrics dependency and we now support custom weights for kappa
      (through the API only so far).
   +  Require's scikit-learn 0.14+.
   +  `accuracy`, `quadratic_weighted_kappa`, `unweighted_kappa`,
      `f1_score_micro`, and `f1_score_macro` functions are no longer available
      under `skll.metrics`. The accuracy and f1 score ones are no longer needed
      because we just use the built-in ones. As for quadratic_weighted_kappa and
      unweighted_kappa, they've been superseded by the kappa function that takes
      a weights argument.
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
      either name the class label column 'y' or use the new `tsv_label` option
      in your configuration file to specify the name of the label column. This
      was necessary to support feature files without labels.
   +  Fixed an issue with how version number was being imported in setup.py that
      would prevent installation if you didn't already have the prereqs
      installed on your machine.
   +  Made random seeds smaller to fix crash on 32-bit machines. This means that
      experiments run with previous versions of skll will yield slightly
      different results if you re-run them with v0.9.5+.
   +  Added `megam_to_csv` for converting .megam files to CSV/TSV files.
   +  Fixed a potential rounding problem with `csv_to_megam` that could slightly
      change feature values in conversion process.
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


