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

3. Build the new conda package locally on your mac using the following command  (*Note*: you may have to replace the contents of the ``requirements()`` function in ``setup.py`` with a ``pass`` statement to get ``conda build`` to work)::

    conda build -c defaults -c conda-forge --python=3.6 --numpy=1.14 skll

4. Convert the package for both linux and windows::

    conda convert -p win-64 -p linux-64 <mac package tarball>

5. Upload each of the packages to anaconda.org using ``anaconda upload <package tarball>``.

6. Upload source and wheel packages to PyPI using ``python setup.py sdist upload`` and ``python setup.py bdist_wheel upload``.

7. Draft a release on GitHub.

8. Make a pull request with the release branch to be merged into ``master`` and request code review.

9. Once the build for the PR passes and the reviewers approve, merge the release branch into ``master``.

10. Make sure that the RTFD build for ``master`` passes.

11. Tag the latest commit in ``master`` with the appropriate release tag and publish the release on GitHub.

12. Send an email around at ETS announcing the release and the changes.
