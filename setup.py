#!/usr/bin/env python
# Copyright (C) 2012-2013 Educational Testing Service

# This file is part of SciKit-Learn Laboratory.

# SciKit-Learn Laboratory is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

# SciKit-Learn Laboratory is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# SciKit-Learn Laboratory.  If not, see <http://www.gnu.org/licenses/>.

import sys
from setuptools import setup

# To get around the fact that you can't import stuff from packages in setup.py
exec(compile(open('skll/version.py').read(), 'skll/version.py', 'exec'))
# (we use the above instead of execfile for Python 3.x compatibility)


def readme():
    with open('README.rst') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        reqs = f.read().splitlines()
    if sys.version_info < (3, 0):
        reqs.append('configparser')


setup(name='skll',
      version=__version__,
      description=('SciKit-Learn Laboratory makes it easier to run machine' +
                   'learning experiments with scikit-learn.'),
      long_description=readme(),
      keywords='learning scikit-learn',
      url='http://github.com/EducationalTestingService/skll',
      author='Daniel Blanchard',
      author_email='dblanchard@ets.org',
      license='GPL',
      packages=['skll'],
      scripts=['scripts/arff_to_megam', 'scripts/csv_to_megam',
               'scripts/filter_megam', 'scripts/generate_predictions',
               'scripts/join_megam', 'scripts/megam_to_arff',
               'scripts/megam_to_csv', 'scripts/megam_to_libsvm',
               'scripts/print_model_weights', 'scripts/run_experiment'],
      install_requires=requirements(),
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Programming Language :: Python',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   ],
      zip_safe=False)
