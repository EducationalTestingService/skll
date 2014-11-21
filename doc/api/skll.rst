:mod:`skll` Package
===================

The most useful parts of our API are available at the package level in addition
to the module level. They are documented in both places for convenience.

From :py:mod:`~skll.data` Package
---------------------------------
.. autoclass:: skll.FeatureSet
    :members:
    :show-inheritance:
.. autoclass:: skll.Reader
    :members:
    :show-inheritance:
.. autoclass:: skll.Writer
    :members:
    :show-inheritance:

From :py:mod:`~skll.experiments` Module
---------------------------------------
.. autofunction:: skll.run_configuration

From :py:mod:`~skll.learner` Module
-----------------------------------
.. autoclass:: skll.Learner
    :members:
    :show-inheritance:

From :py:mod:`~skll.metrics` Module
-----------------------------------
.. autofunction:: skll.f1_score_least_frequent
.. autofunction:: skll.kappa
.. autofunction:: skll.kendall_tau
.. autofunction:: skll.spearman
.. autofunction:: skll.pearson
