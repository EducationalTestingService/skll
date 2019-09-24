.. _contributing:

Contributing
============

Thank you for your interest in contributing to SKLL! We welcome any and
all contributions.

Guidelines
----------

The SKLL contribution guidelines can be found in our Github repository
`here <https://github.com/EducationalTestingService/skll/blob/master/CONTRIBUTING.md>`__. Please try to follow them as much as possible.

SKLL Code Overview
------------------

This section will help you get oriented with the SKLL codebase by
describing how it is organized, the various SKLL entry points into the
code, and what the general code flow looks like for each entry point.

Organization
~~~~~~~~~~~~

The main Python code for the SKLL package lives inside the ``skll`` sub-directory of the repository. It contains the following files and sub-directories:

-  `__init__.py <https://github.com/EducationalTestingService/skll/blob/master/skll/__init__.py>`__ : Code used to initialize the ``skll`` Python
   package.
-  `config.py <https://github.com/EducationalTestingService/skll/blob/master/skll/config.py>`__ : Code to parse SKLL experiment configuration files.
-  `experiments.py <https://github.com/EducationalTestingService/skll/blob/master/skll/experiments.py>`__ : Code that is related to creating and running
   SKLL experiments. It also contains code that collects the various
   evaluation metrics and predictions for each SKLL experiment and
   writes them out to disk.
-  `learner.py <https://github.com/EducationalTestingService/skll/blob/master/skll/learner.py>`__ : Code for the ``Learner`` class. This class is
   instantiated for every learner name specified in the experiment
   configuration file.
-  `metrics.py <https://github.com/EducationalTestingService/skll/blob/master/skll/metrics.py>`__ : Code for any custom metrics that are not in
   ``sklearn.metrics``, e.g., ``kappa``, ``kendall_tau``,
   ``spearman``, etc.
-  `logutils.py <https://github.com/EducationalTestingService/skll/blob/master/skll/logutils.py>`__ : Code for a custom logging solution that allows
   capturing any information logged to STDOUT/STDERR by SKLL and
   scikit-learn to also be captured into log files that are then
   saved on disk.
-  `version.py <https://github.com/EducationalTestingService/skll/blob/master/skll/version.py>`__ : Code to define the SKLL version. Only changed for
   new releases.
-  `data/ <https://github.com/EducationalTestingService/skll/tree/master/skll/data>`__

   -  `__init__.py <https://github.com/EducationalTestingService/skll/blob/master/skll/data/__init__.py>`__ : Code used to initialize the ``skll.data`` Python
      package.
   -  `featureset.py <https://github.com/EducationalTestingService/skll/blob/master/skll/data/featureset.py>`__ : Code for the ``FeatureSet`` class – the main
      class that encapsulates all of the features names, values, and
      metadata for a given set of instances.
   -  `readers.py <https://github.com/EducationalTestingService/skll/blob/master/skll/data/readers.py>`__ : Code for classes that can read various file
      formats and create ``FeatureSet`` objects from them.
   -  `writers.py <https://github.com/EducationalTestingService/skll/blob/master/skll/data/writers.py>`__ : Code for classes that can write ``FeatureSet``
      objects to files on disk in various formats.
   -  `dict_vectorizer.py <https://github.com/EducationalTestingService/skll/blob/master/skll/data/dict_vectorizer.py>`__ : Code for a ``DictVectorizer`` class that
      subclasses ``sklearn.feature_extraction.DictVectorizer`` to add an
      ``__eq__()`` method that we need for vectorizer equality.
-  `utilities/ <https://github.com/EducationalTestingService/skll/tree/master/skll/utilities>`__

   -  `compute_eval_from_predictions.py <https://github.com/EducationalTestingService/skll/blob/master/skll/utilities/compute_eval_from_predictions.py>`__ : See
      `documentation <https://skll.readthedocs.io/en/latest/utilities.html#compute-eval-from-predictions>`__.
   -  `filter_features.py <https://github.com/EducationalTestingService/skll/blob/master/skll/utilities/filter_features.py>`__ : See
      `documentation <https://skll.readthedocs.io/en/latest/utilities.html#filter-features>`__.
   -  `generate_predictions.py <https://github.com/EducationalTestingService/skll/blob/master/skll/utilities/generate_predictions.py>`__ : See
      `documentation <https://skll.readthedocs.io/en/latest/utilities.html#generate-predictions>`__.
   -  `join_features.py <https://github.com/EducationalTestingService/skll/blob/master/skll/utilities/join_features.py>`__ : See
      `documentation <https://skll.readthedocs.io/en/latest/utilities.html#join-features>`__.
   -  `plot_learning_curves.py <https://github.com/EducationalTestingService/skll/blob/master/skll/utilities/plot_learning_curves.py>`__ : See
      `documentation <https://skll.readthedocs.io/en/latest/utilities.html#plot-learning-curves>`__.
   -  `print_model_weights.py <https://github.com/EducationalTestingService/skll/blob/master/skll/utilities/print_model_weights.py>`__ : See
      `documentation <https://skll.readthedocs.io/en/latest/utilities.html#print-model-weights>`__.
   -  `run_experiment.py <https://github.com/EducationalTestingService/skll/blob/master/skll/utilities/run_experiment.py>`__ : See
      `documentation <https://skll.readthedocs.io/en/latest/run_experiment.html#using-run-experiment>`__.
   -  `skll_convert.py <https://github.com/EducationalTestingService/skll/blob/master/skll/utilities/skll_convert.py>`__ : See
      `documentation <https://skll.readthedocs.io/en/latest/utilities.html#skll-convert>`__.
   -  `summarize_results.py <https://github.com/EducationalTestingService/skll/blob/master/skll/utilities/summarize_results.py>`__ : See
      `documentation <https://skll.readthedocs.io/en/latest/utilities.html#summarize-results>`__.

-  `tests/ <https://github.com/EducationalTestingService/skll/tree/master/tests>`__ 

   - ``test_*.py`` : These files contain the code for the unit tests and regression tests.

Entry Points & Workflow
~~~~~~~~~~~~~~~~~~~~~~~

There are three main entry points into the SKLL codebase:

1. **Experiment configuration files**. The primary way to interact with SKLL
   is by writing configuration files and then passing it to the
   `run_experiment <https://skll.readthedocs.io/en/latest/run_experiment.html#using-run-experiment>`__ script. When you run the command
   ``run_experiment <config_file>``, the following happens (at a high level):

   -  the configuration file is handed off to the
      `run_configuration() <https://github.com/EducationalTestingService/skll/blob/master/skll/experiments.py#L981>`__ function in ``experiments.py``.
   -  a `SKLLConfigParser <https://github.com/EducationalTestingService/skll/blob/master/skll/config.py#L34>`__ object is instantiated from ``config.py``
      that parses all of the relevant fields out of the given
      configuration file.
   -  the configuration fields are then passed to the
      `_classify_featureset() <https://github.com/EducationalTestingService/skll/blob/master/skll/experiments.py#L449>`__ function in ``experiments.py`` which
      instantiates the learners (using code from ``learner.py``), the
      featuresets (using code from ``reader.py`` & ``featureset.py``),
      and runs the experiments, collects the results, and writes them
      out to disk.

2. **SKLL API**. Another way to interact with SKLL is via the SKLL API
   directly in your Python code rather than using configuration files.
   For example, you could use the `Learner.from_file() <https://github.com/EducationalTestingService/skll/blob/master/skll/learner.py#L967>`__ method to load a saved model from disk and make predictions on new data. The
   documentation for the SKLL API can be found
   `here <https://skll.readthedocs.io/en/latest/api.html>`__.

3. **Utility scripts**. The scripts listed in the section above under
   ``utilities`` are also entry points into the SKLL code. These scripts
   are convenient wrappers that use the SKLL API for commonly used
   tasks, e.g., generating predictions on new data from an already
   trained model.
