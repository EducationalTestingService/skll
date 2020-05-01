How to create and test conda package.

1. To create the SKLL conda package run:
   `conda build -c defaults -c conda-forge --numpy=1.17 .`
2. Upload the package to anaconda.org using `anaconda upload --user ets <path>`.
3. Test the package:
   `conda create -n foobar -c conda-forge -c ets skll=2.1`
