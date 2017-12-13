How to create and test conda package.

1. To create the SKLL conda package run:
   `conda build -c defaults -c conda-forge --python=3.6 --numpy=1.13 skll`
2. Upload the package to anaconda.org using `anaconda upload <path>`.
3. Test the package:
   `conda create -n foobar -c defaults -c conda-forge -c desilinguist python=3.6 skll=1.5`
