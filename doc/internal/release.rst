Release Process
===============

This document is only meant for the project administrators, not users and developers.

1. Create a release branch on GitHub.

2. In the release branch:

   a. update the version numbers in ``version.py``.

   b. update the conda recipe.

   c. update the documentation with any new features or details about changes.

   d. run ``make linkcheck`` on the documentation and fix any redirected/broken links.

   e. update the README.

3. Build the new conda package using the following command (*Note*: we should just be able to use ``conda-forge`` but due to a weirdness with the ``cycler`` package on that channel, we also have to include ``defaults``)::

    conda build -c defaults -c conda-forge --numpy=1.17 skll

4. Upload the package to anaconda.org using ``anaconda upload --user ets <package tarball>``.

5. Build the PyPI source distribution using ``python setup.py sdist build``.

6. Upload the source distribution to TestPyPI  using ``twine upload --repository testpypi dist/*``. You will need to have the ``twine`` package installed and set up your ``$HOME/.pypirc`` correctly. See details `here <https://packaging.python.org/guides/using-testpypi/>`__.

7. Test the conda package by creating a new environment on different platforms with this package installed and then running SKLL examples or tests from a SKLL working copy. If the package works, then move on to the next step. If it doesn't, figure out why and rebuild and re-upload the package.

8. Test the TestPyPI package by installing it as follows: ``pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple skll`` and then running SKLL examples or tests from a SKLL working copy. If the package works, then move on to the next step. If it doesn't, figure out why and rebuild and re-upload the package.

9. Upload source package to main PyPI using ``python setup.py sdist upload``.

10. Draft a release on GitHub.

11. Make a pull request with the release branch to be merged into ``master`` and request code review.

12. Once the Travis (Linux) and Appveyor (Windows) builds for the PR pass and the reviewers approve, merge the release branch into ``master``.

13. Make sure that the ReadTheDocs build for ``master`` passes.

14. Tag the latest commit in ``master`` with the appropriate release tag and publish the release on GitHub.

15. Send an email around at ETS announcing the release and the changes.

16. Post release announcement on Twitter/LinkedIn.
