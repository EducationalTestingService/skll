
Contributing code
=================

*Note: This document is based on the contribution guidelines for scikit-learn.*

How to contribute
-----------------

0. Read the [part of the documentation](https://skll.readthedocs.io/en/latest/contributing.html) that provides an overview of the SKLL codebase, run the tutorial and examples, and get familiar with the SKLL outputs.

1. Fork the [project repository](http://github.com/EducationalTestingService/skll/): click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server. (**NOTE**: If you are officially on the SKLL Developers team, you should not fork; just clone the SKLL repository directly.)

2. Clone the fork (or the main repo) to your local disk:

          $ git clone git@github.com:YourLogin/skll.git
          $ cd skll

3. Create an isolated environment for SKLL development. We recommend using the [conda](https://conda.io/en/latest/) package manager. To create a `conda` environment, run the following command in the root of the working directory:

         $ conda create -n sklldev -c conda-forge --file conda_requirements.txt

4. Activate the conda environment

         $ conda activate sklldev

5. Run `pip install -e .` to install skll into the environment in editable mode,
   which is what we need for development.

6. Install [`pre-commit`](https://pre-commit.com/) for automatically running git commit hooks:

         $ pre-commit install

   [`pre-commit`](https://pre-commit.com/) is used to run pre-commit
   hooks, such as [`isort`](https://pycqa.github.io/isort/) and
   [`flake8`](https://flake8.pycqa.org/en/latest/). (Check
   [here](./.pre-commit-config.yaml) to see a full list of pre-commit
   hooks.) If you attempt to make a commit and it fails, you will be
   able to see which hooks passed/failed and you will have an
   opportunity to commit suggested changes and/or address problems.

   If you want to run all checks or specific checks before attempting a
   commit, it is possible to do. It is also possible to skip checks
   altogether (though this should be done only when well-motivated).

   To run all checks on all files (not just those that have changed):

         $ pre-commit run --all-files

   To run all hooks on changed files:

         $ pre-commit run

   To run the `isort` hook alone on changed files:

         $ pre-commit run isort

   To run the `isort` hook alone on a given file:

         $ pre-commit run isort <file-path>

   Finally, the `SKIP` environment variable can be used to indicate to
   `pre-commit` that certain checks should be skipped. It can be
   assigned a comma-separated list of check names:

         $ SKIP=check-added-large-files git commit -m "Adding a large file that we definitely need"

7. Create a feature branch to hold your changes:

          $ git checkout -b feature/my-new-addition

   and start making changes. **Never work in the ``main`` branch!**

8. During development, you can stage and commit your changes in git as follows:

          $ git add modified_files
          $ git commit

   Make sure to read step 6 above concerning `pre-commit` hooks.

9. Once you are done with your changes (including any new tests), run the tests
   locally:

         $ nosetests

10. After making sure all tests pass, you are ready to push your branch/fork to GitHub with:

          $ git push -u origin feature/my-new-addition

Finally, go to the web page of (your fork of) the SKLL repo,
and click 'Pull request' to send your changes to the maintainers for
review.

(If any of the above seems like magic to you, then look up the
[Git documentation](http://git-scm.com/documentation) on the web.)

We recommended that you check that your contribution complies with the
following rules before submitting a pull request:

-  All methods and functions should have informative docstrings.

-  All existing tests should pass when everything is rebuilt from scratch. You
   should be able to see this by running ``nosetests`` locally, or looking at the Gitlab CI build status after you create your pull request.

-  All new functionality must be covered by unit tests.

-  Every pull request description should contain a link to the issue that it is
   trying to address. This is easily done by just typing `#` and then picking the issue from the dropdown. If the issue is not visible in the first set of results, type a few characters from the issue title and the dropdown should update.

-  Address any PEP8 issues pointed out by the `pep8speaks` bot that comments on
   your PR after you submit it. The *same* comment will update after you make make any further commits so refer to it after every commit. You may want to install a linter in your development environment so that you can fix any PEP8 issues while you write your code. We generally ignore E501 messages about lines longer than 100 characters.

- You may need to add new tests if the code coverage after merging your branch
  will be lower than the current `main`. This will be reported by the `codecov` bot once you submit your PR.

After submitting a pull request, it is recommended to add at least 2-3 reviewers to
review it. See [Requesting a pull request review](https://help.github.com/en/articles/requesting-a-pull-request-review) for more details.


Easy Issues
-----------

A great way to start contributing to SKLL is to pick an item
from the list of issues labelled with the [`good first issue`](https://github.com/EducationalTestingService/skll/labels/good%20first%20issue)
tag. Resolving these issues allow you to start contributing to the project
without much prior knowledge. Your assistance in this area will be greatly
appreciated by the more experienced developers as it helps free up their time
to concentrate on other issues.

Large Issues
------------

If you are willing, there are often issues that are not incredibly
complex, but still take more time than the main developers have had
time to address them.  Any help with these issues would be *greatly*
appreciated.  They are labelled with the [help wanted](https://github.com/EducationalTestingService/skll/labels/help%20wanted)
tag on the issue list.


Documentation
-------------

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc.
reStructuredText documents live in the source code repository under the
doc/ directory.

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the doc/ directory.
Alternatively, ``make`` can be used to quickly generate the
documentation without the example gallery. The resulting HTML files will
be placed in _build/html/ and are viewable in a web browser. See the
README file in the doc/ directory for more information.

For building the documentation, you will need [sphinx](http://sphinx.pocoo.org/) as well as the readthedocs sphinx theme. To install both, just run:

      $ conda install sphinx sphinx_rtd_theme

in your existing conda environment.
