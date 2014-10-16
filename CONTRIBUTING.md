
Contributing code
=================

*Note: This document is based on the contribution guidelines for scikit-learn.*

How to contribute
-----------------

The preferred way to contribute to SKLL is to fork the
[main repository](http://github.com/EducationalTestingService/skll/) on
GitHub:

1. Fork the [project repository](http://github.com/EducationalTestingService/skll/):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

          $ git clone git@github.com:YourLogin/skll.git
          $ cd skll

3. Checkout the develop branch:

          $ git checkout develop

   You must do this to make sure that your branches are based off the
   ``develop`` branch, and not the ``master`` branch!

4. Create a feature branch to hold your changes:

          $ git checkout -b feature/my-new-addition

   and start making changes. Never work in the ``develop`` or ``master``
   branches!

5. Work on this copy on your computer using Git to do the version
   control. When you're done editing, do:

          $ git add modified_files
          $ git commit

   to record your changes in Git, then push them to GitHub with:

          $ git push -u feature/my-new-addition

Finally, go to the web page of the your fork of the SKLL repo,
and click 'Pull request' to send your changes to the maintainers for
review. **Please make sure that the "base" branch for your pull request is
set to ``develop`` and that the "base fork" is set to
``EducationalTestingService/skll``.

(If any of the above seems like magic to you, then look up the
[Git documentation](http://git-scm.com/documentation) on the web.)

It is recommended to check that your contribution complies with the
following rules before submitting a pull request:

-  All public methods should have informative docstrings.

-  All existing tests pass when everything is rebuilt from scratch. You should
   be able to see this by running ``nosetests`` locally, or looking at the
   Travis build status after you create your pull request.

-  All new functionality must be covered by unit tests.


You can also check for common programming errors with the following
tools:

-  Code with good unittest coverage (at least 80%), check with:

          $ pip install nose coverage
          $ nosetests --with-coverage path/to/tests_for_package

-  No pyflakes warnings, check with:

           $ pip install pyflakes
           $ pyflakes path/to/module.py

-  No PEP8 warnings, check with:

           $ pip install pep8
           $ pep8 path/to/module.py

-  AutoPEP8 can help you fix some of the easy redundant errors:

           $ pip install autopep8
           $ autopep8 path/to/pep8.py


Easy Issues
-----------

A great way to start contributing to SKLL is to pick an item
from the list of issues labelled with the [help wanted](https://github.com/EducationalTestingService/skll/labels/help%20wanted)
tag. Resolving these issues allow you to start contributing to the project
without much prior knowledge. Your assistance in this area will be greatly
appreciated by the more experienced developers as it helps free up their time
to concentrate on other issues.

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

For building the documentation, you will need [sphinx](http://sphinx.pocoo.org/).

