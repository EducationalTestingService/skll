### How to create and test conda package

1. To create the SKLL conda package run: `conda build -c conda-forge .`

2. This will create a single noarch Python package.

3. Upload the package file to anaconda.org using `anaconda upload --user ets <path_to_file>`.

4. Test the package: `conda create -n foobar -c ets -c conda-forge python=3.11 skll`. This should _always_ install the latest package from the ``ets`` conda channel.
   Note that we are specifying the ``ets`` channel first since SKLL is now also in conda-forge but runs a version behind until we do the actual release on GitHub.
