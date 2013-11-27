SciKit-Learn Laboratory
-----------------------

.. image:: https://travis-ci.org/EducationalTestingService/skll.png?branch=master
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
-  `configparser <http://pypi.python.org/pypi/configparser>`__ (only required for
   Python 2.7)
-  `futures <http://pypi.python.org/pypi/futures>`__ (only required for Python 2.7)
-  `logutils <http://pypi.python.org/pypi/logutils>`__ (only required for Python 2.7)

Talks
~~~~~

-  *Simpler Machine Learning with SKLL*, Dan Blanchard, PyData NYC 2013 (`video <http://vimeo.com/79511496>`__ | `slides <http://www.slideshare.net/DanielBlanchard2/simple-machine-learning-with-skll>`__)

Changelog
~~~~~~~~~

See `GitHub releases <https://github.com/EducationalTestingService/skll/releases>`__.
