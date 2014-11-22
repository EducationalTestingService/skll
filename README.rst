SciKit-Learn Laboratory
-----------------------

.. image:: https://travis-ci.org/EducationalTestingService/skll.svg?branch=stable
   :alt: Build status
   :target: https://travis-ci.org/EducationalTestingService/skll

.. image:: http://img.shields.io/coveralls/EducationalTestingService/skll/stable.svg
    :target: https://coveralls.io/r/EducationalTestingService/skll

.. image:: http://img.shields.io/pypi/dm/skll.svg
   :target: https://warehouse.python.org/project/skll/
   :alt: PyPI downloads

.. image:: http://img.shields.io/pypi/v/skll.svg
   :target: https://warehouse.python.org/project/skll/
   :alt: Latest version on PyPI

.. image:: http://img.shields.io/pypi/l/skll.svg
   :alt: License

.. image:: http://img.shields.io/badge/DOI-10.5281%2Fzenodo.12521-blue.svg
   :target: http://dx.doi.org/10.5281/zenodo.12521
   :alt: DOI for citing SKLL 0.28.1

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
the ``Learner`` and ``Reader`` classes. For more details on how to simply
train, test, cross-validate, and run grid search on a variety of scikit-learn
models see
`the documentation <https://skll.readthedocs.org/en/latest/index.html>`__.

A Note on Pronunciation
~~~~~~~~~~~~~~~~~~~~~~~

SciKit-Learn Laboratory (SKLL) is pronounced "skull": that's where the learning
happens.

Requirements
~~~~~~~~~~~~

-  Python 2.7+
-  `scikit-learn <http://scikit-learn.org/stable/>`__
-  `six <https://warehouse.python.org/project/six>`__
-  `PrettyTable <https://warehouse.python.org/project/PrettyTable>`__
-  `BeautifulSoup 4 <http://www.crummy.com/software/BeautifulSoup/>`__
-  `Grid Map <https://warehouse.python.org/project/gridmap>`__ (only required if you plan
   to run things in parallel on a DRMAA-compatible cluster)
-  `joblib <https://warehouse.python.org/project/joblib>`__
-  `PyYAML <https://warehouse.python.org/project/PyYAML>`__
-  `configparser <https://warehouse.python.org/project/configparser>`__ (only required for
   Python 2.7)
-  `logutils <https://warehouse.python.org/project/logutils>`__ (only required for Python 2.7)
-  `mock <https://warehouse.python.org/project/mock>`__ (only required for Python 2.7)

Talks
~~~~~

-  *Simpler Machine Learning with SKLL*, Dan Blanchard, PyData NYC 2013 (`video <http://vimeo.com/79511496>`__ | `slides <http://www.slideshare.net/DanielBlanchard2/simple-machine-learning-with-skll>`__)

Books
~~~~~

SKLL is featured in `Data Science at the Command Line <http://datascienceatthecommandline.com>`__
by `Jeroen Janssens <http://jeroenjanssens.com>`__.

Changelog
~~~~~~~~~

See `GitHub releases <https://github.com/EducationalTestingService/skll/releases>`__.
