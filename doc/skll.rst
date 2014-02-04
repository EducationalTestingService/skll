Python API
==========
The complete Python API is documented below, but here is a quick run-down of
how you accomplish common tasks.


Load some examples::

    from skll import load_examples

    train_examples = load_examples('myexamples.megam')


Train a linear svm (assuming we have `train_examples`)::

    from skll import Learner

    learner = Learner('LinearSVC')
    learner.train(train_examples)


Evaluate a trained model::

    test_examples = load_examples('test.tsv')
    conf_matrix, accuracy, prf_dict, model_params, obj_score = learner.evaluate(test_examples)


Perform ten-fold cross-validation with a radial SVM::

    learner = Learner('SVC')
    fold_result_list, grid_search_scores = learner.cross-validate(train_examples)

``fold_result_list`` in this case is a list of the results returned by
``learner.evaluate`` for each fold, and ``grid_search_scores`` is the highest
objective function value achieved when tuning the model.


Generate predictions from a trained model::

    predictions = learner.predict(test_examples)


:mod:`skll` Package
-------------------

The most useful parts of our API are available at the package level in addition
to the module level. They are documented in both places for convenience.

From :py:mod:`~skll.data` Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: skll.convert_examples
.. autofunction:: skll.load_examples
.. autofunction:: skll.write_feature_file

From :py:mod:`~skll.experiments` Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: skll.run_ablation
.. autofunction:: skll.run_configuration

From :py:mod:`~skll.learner` Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: skll.Learner
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~skll.metrics` Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: skll.f1_score_least_frequent
.. autofunction:: skll.kappa
.. autofunction:: skll.kendall_tau
.. autofunction:: skll.spearman
.. autofunction:: skll.pearson


:mod:`data` Module
------------------

.. automodule:: skll.data
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`experiments` Module
-------------------------

.. automodule:: skll.experiments
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`learner` Module
---------------------

.. automodule:: skll.learner
    :members:
    :show-inheritance:

:mod:`metrics` Module
---------------------

.. automodule:: skll.metrics
    :members:
    :undoc-members:
    :show-inheritance:


