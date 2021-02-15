.. _contributing:

Contributing
============

Thank you for your interest in contributing to SKLL! We welcome any and all contributions.

Guidelines
----------

The SKLL contribution guidelines can be found in our Github repository
`here <https://github.com/EducationalTestingService/skll/blob/main/CONTRIBUTING.md>`__. We strongly encourage all SKLL contributions to follow these guidelines.

SKLL Code Overview
------------------

This section will help you get oriented with the SKLL codebase by
describing how it is organized, the various SKLL entry points into the
code, and what the general code flow looks like for each entry point.

Organization
~~~~~~~~~~~~

The main Python code for the SKLL package lives inside the ``skll`` sub-directory of the repository. It contains the following files and sub-directories:

-  `__init__.py <https://github.com/EducationalTestingService/skll/blob/main/skll/__init__.py>`__ : Code used to initialize the ``skll`` Python package.

-  `config/ <https://github.com/EducationalTestingService/skll/tree/main/skll/config>`__ : Code to parse SKLL experiment configuration files.

-  `experiments/ <https://github.com/EducationalTestingService/skll/tree/main/skll/experiments>`__ : Code that is related to creating and running SKLL experiments. It also contains code that collects the various evaluation metrics and predictions for each SKLL experiment and writes them out to disk.

-  `learner/ <https://github.com/EducationalTestingService/skll/tree/main/skll/learner>`__ : Code for the `Learner <https://github.com/EducationalTestingService/skll/blob/main/skll/learner/__init__.py>`__ and `VotingLearner <https://github.com/EducationalTestingService/skll/blob/main/skll/learner/voting.py>`__ classes. The former is instantiated for all learner names specified in the experiment configuration file *except* ``VotingClassifier`` and ``VotingRegressor`` for which the latter is instantiated instead.

-  `metrics.py <https://github.com/EducationalTestingService/skll/blob/main/skll/metrics.py>`__ : Code for any custom metrics that are not in ``sklearn.metrics``, e.g., ``kappa``, ``kendall_tau``, ``spearman``, etc.

-  `data/ <https://github.com/EducationalTestingService/skll/tree/main/skll/data>`__

   -  `__init__.py <https://github.com/EducationalTestingService/skll/blob/main/skll/data/__init__.py>`__ : Code used to initialize the ``skll.data`` Python package.

   -  `featureset.py <https://github.com/EducationalTestingService/skll/blob/main/skll/data/featureset.py>`__ : Code for the ``FeatureSet`` class metadata for a given set of instances.

   -  `readers.py <https://github.com/EducationalTestingService/skll/blob/main/skll/data/readers.py>`__ : Code for classes that can read various file formats and create ``FeatureSet`` objects from them.

   -  `writers.py <https://github.com/EducationalTestingService/skll/blob/main/skll/data/writers.py>`__ : Code for classes that can write ``FeatureSet`` objects to files on disk in various formats.

   -  `dict_vectorizer.py <https://github.com/EducationalTestingService/skll/blob/main/skll/data/dict_vectorizer.py>`__ : Code for a ``DictVectorizer`` class that subclasses ``sklearn.feature_extraction.DictVectorizer`` to add an ``__eq__()`` method that we need for vectorizer equality.

-  `utils/ <https://github.com/EducationalTestingService/skll/tree/main/skll/utils>`__ : Code for different utility scripts, functions, and classes used throughout SKLL. The most important ones are the command line scripts in the ``utils.commandline`` submodule.

   - `compute_eval_from_predictions.py <https://github.com/EducationalTestingService/skll/blob/main/skll/utils/commandline/compute_eval_from_predictions.py>`__ : See `documentation <https://skll.readthedocs.io/en/latest/utilities.html#compute-eval-from-predictions>`__.

   -  `filter_features.py <https://github.com/EducationalTestingService/skll/blob/main/skll/utils/commandline/filter_features.py>`__ : See `documentation <https://skll.readthedocs.io/en/latest/utilities.html#filter-features>`__.

   -  `generate_predictions.py <https://github.com/EducationalTestingService/skll/blob/main/skll/utils/commandline/generate_predictions.py>`__ : See `documentation <https://skll.readthedocs.io/en/latest/utilities.html#generate-predictions>`__.

   -  `join_features.py <https://github.com/EducationalTestingService/skll/blob/main/skll/utils/commandline/join_features.py>`__ : See `documentation <https://skll.readthedocs.io/en/latest/utilities.html#join-features>`__.

   -  `plot_learning_curves.py <https://github.com/EducationalTestingService/skll/blob/main/skll/utils/commandline/plot_learning_curves.py>`__ : See `documentation <https://skll.readthedocs.io/en/latest/utilities.html#plot-learning-curves>`__.

   -  `print_model_weights.py <https://github.com/EducationalTestingService/skll/blob/main/skll/utils/commandline/print_model_weights.py>`__ : See `documentation <https://skll.readthedocs.io/en/latest/utilities.html#print-model-weights>`__.

   -  `run_experiment.py <https://github.com/EducationalTestingService/skll/blob/main/skll/utils/commandline/run_experiment.py>`__ : See `documentation <https://skll.readthedocs.io/en/latest/run_experiment.html#using-run-experiment>`__.

   -  `skll_convert.py <https://github.com/EducationalTestingService/skll/blob/main/skll/utils/commandline/skll_convert.py>`__ : See `documentation <https://skll.readthedocs.io/en/latest/utilities.html#skll-convert>`__.

   -  `summarize_results.py <https://github.com/EducationalTestingService/skll/blob/main/skll/utils/commandline/summarize_results.py>`__ : See `documentation <https://skll.readthedocs.io/en/latest/utilities.html#summarize-results>`__.

-  `version.py <https://github.com/EducationalTestingService/skll/blob/main/skll/version.py>`__ : Code to define the SKLL version. Only changed for new releases.

-  `tests/ <https://github.com/EducationalTestingService/skll/tree/main/tests>`__
   - ``test_*.py`` : These files contain the code for the unit tests and regression tests.

Entry Points & Workflow
~~~~~~~~~~~~~~~~~~~~~~~

There are three main entry points into the SKLL codebase:

1. **Experiment configuration files**. The primary way to interact with SKLL
   is by writing configuration files and then passing it to the
   `run_experiment <https://skll.readthedocs.io/en/latest/run_experiment.html#using-run-experiment>`__ script. When you run the command
   ``run_experiment <config_file>``, the following happens (at a high level):

   -  the configuration file is handed off to the `run_configuration() <https://github.com/EducationalTestingService/skll/blob/main/skll/experiments/__init__.py#L482>`__ function in ``experiments.py``.

   -  a `SKLLConfigParser <https://github.com/EducationalTestingService/skll/blob/main/skll/config/__init__.py#L41>`__ object is instantiated from ``config.py`` that parses all of the relevant fields out of the given configuration file.

   -  the configuration fields are then passed to the `_classify_featureset() <https://github.com/EducationalTestingService/skll/blob/main/skll/experiments/__init__.py#L56>`__ function in ``experiments.py`` which instantiates the learners (using code from ``learner.py``), the featuresets (using code from ``reader.py`` & ``featureset.py``), and runs the experiments, collects the results, and writes them out to disk.

2. **SKLL API**. Another way to interact with SKLL is via the SKLL API directly in your Python code rather than using configuration files. For example, you could use the `Learner.from_file() <https://github.com/EducationalTestingService/skll/blob/main/skll/learner/__init__.py#L324>`__ or `VotingLearner.from_file() <https://github.com/EducationalTestingService/skll/blob/main/skll/learner/voting.py#L254>`__ methods to load saved models of those types from disk and make predictions on new data. The documentation for the SKLL API can be found `here <https://skll.readthedocs.io/en/latest/api.html>`__.

3. **Utility scripts**. The scripts listed in the section above under ``utils`` are also entry points into the SKLL code. These scripts are convenient wrappers that use the SKLL API for commonly used tasks, e.g., generating predictions on new data from an already trained model.
