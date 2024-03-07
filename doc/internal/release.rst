Release Process
===============

This document is only meant for the project administrators, not users and developers.

1. Create a release branch ``release/XX`` on GitHub.

2. In the release branch:

   a. Update the version numbers in ``version.py``.

   b. Make sure that `requirements.txt` only has the actual dependencies that
      are needed to run SKLL. Any dependencies needed only for
      development/testing (e.g., `sphinx`, `nose2` etc.) should be moved to
      `requirements.dev`. This means that `requirements.txt` *must* be a strict
      subset of `requirements.dev`.

   c. Make sure the versions in `doc/requirements.txt` are up to date with
      `requirements.txt` and only contains the dependencies needed to build the
      documentation.

   d. Make sure `.readthedocs.yml` is still accurate.

   e. Update the conda recipe.

   f. Update the documentation with any new features or details about changes.

   g. Run ``make linkcheck`` on the documentation and fix any redirected/broken links.

   h. Update the README and this release documentation, if necessary.

3. Build and upload the conda packages by following instructions in ``conda-recipe/README.md``.

4. Build the PyPI source distribution using ``python setup.py sdist build``.

5. Upload the source distribution to TestPyPI  using ``twine upload --repository testpypi dist/*``. You will need to have the ``twine`` package installed and set up your ``$HOME/.pypirc`` correctly. See details `here <https://packaging.python.org/en/latest/guides/using-testpypi/>`__.

6. Test the conda package by creating a new environment on different platforms with this package installed and then running SKLL examples or tests from a SKLL working copy. If the package works, then move on to the next step. If it doesn't, figure out why and rebuild and re-upload the package.

7. Test the TestPyPI package by installing it as follows::

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple skll

8. Then run some SKLL examples or tests from a SKLL working copy. If the TestPyPI package works, then move on to the next step. If it doesn't, figure out why and rebuild and re-upload the package.

9. Create pull requests on the `skll-conda-tester <https://github.com/EducationalTestingService/skll-conda-tester/>`_ and `skll-pip-tester <https://github.com/EducationalTestingService/skll-pip-tester/>`_ repositories to test the conda and TestPyPI packages on Linux and Windows.

10. Draft a release on GitHub while the Linux and Windows package tester builds are running.

11. Once both builds have passed, make a pull request with the release branch to be merged into ``main`` and request code review.

12. Once the build for the PR passes and the reviewers approve, merge the release branch into ``main``.

13. Upload source and wheel packages to PyPI using ``python setup.py sdist upload`` and ``python setup.py bdist_wheel upload``

14. Make sure that the ReadTheDocs build for ``main`` passes.

15. Tag the latest commit in ``main`` with the appropriate release tag and publish the release on GitHub.

16. Send an email around at ETS announcing the release and the changes.

17. Post release announcement on Twitter/LinkedIn.
