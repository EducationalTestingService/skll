### How to create and test conda package

1. To create the SKLL conda package run: `conda build -c conda-forge .`

2. This will create python 3.7, 3.8, and 3.9 packages for your native platform, e.g., `osx-64`.

3. Convert these built packages for the other two platforms. For example, if you ran the above command on macOS, run `conda convert -p linux-64 -p win-64 <packages files>`, where `<packages_files>` are the package files that were created in step 2.

4. Upload all 9 package files (3 Python versions x 3 platforms) to anaconda.org using `anaconda upload --user ets <path_to_files>`.

5. Test the package: `conda create -n foobar -c conda-forge -c ets python=3.9 skll`. This should _always_ install the latest package.
