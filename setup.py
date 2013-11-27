#!/usr/bin/env python
# License: BSD 3 clause
import sys
from setuptools import setup

# To get around the fact that you can't import stuff from packages in setup.py
exec(compile(open('skll/version.py').read(), 'skll/version.py', 'exec'))
# (we use the above instead of execfile for Python 3.x compatibility)


def readme():
    with open('README.rst') as f:
        return f.read()


def requirements():
    # Use backported requirements for 2.7
    if sys.version_info < (3, 0):
        req_path = 'requirements_rtd.txt'
    # Use 3.x requirements
    else:
        req_path = 'requirements.txt'
    with open(req_path) as f:
        reqs = f.read().splitlines()
    return reqs


setup(name='skll',
      version=__version__,
      description=('SciKit-Learn Laboratory makes it easier to run machine' +
                   'learning experiments with scikit-learn.'),
      long_description=readme(),
      keywords='learning scikit-learn',
      url='http://github.com/EducationalTestingService/skll',
      author='Daniel Blanchard',
      author_email='dblanchard@ets.org',
      license='BSD 3 clause',
      packages=['skll'],
      scripts=['scripts/filter_megam', 'scripts/generate_predictions',
               'scripts/join_megam', 'scripts/megam_to_libsvm',
               'scripts/print_model_weights', 'scripts/run_experiment',
               'scripts/skll_convert', 'scripts/summarize_results'],
      install_requires=requirements(),
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: BSD License',
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
