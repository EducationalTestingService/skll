#!/usr/bin/env python
# License: BSD 3 clause
from setuptools import find_packages, setup

# Get version without importing, which avoids dependency issues
exec(compile(open('skll/version.py').read(), 'skll/version.py', 'exec'))
# (we use the above instead of execfile for Python 3.x compatibility)


def readme():
    with open('README.rst') as f:
        return f.read()


def requirements():
    req_path = 'requirements.txt'
    with open(req_path) as f:
        reqs = f.read().splitlines()
    return reqs


setup(name='skll',
      version=__version__,  # noqa: F821
      description=('SciKit-Learn Laboratory makes it easier to run machine '
                   'learning experiments with scikit-learn.'),
      long_description=readme(),
      keywords='learning scikit-learn',
      url='http://github.com/EducationalTestingService/skll',
      author='Nitin Madnani',
      author_email='nmadnani@ets.org',
      license='BSD 3 clause',
      packages=find_packages(exclude=['tests', 'examples']),
      entry_points={'console_scripts':
                    ['filter_features = skll.utils.commandline.filter_features:main',
                     'generate_predictions = skll.utils.commandline.generate_predictions:main',
                     'join_features = skll.utils.commandline.join_features:main',
                     'print_model_weights = skll.utils.commandline.print_model_weights:main',
                     'run_experiment = skll.utils.commandline.run_experiment:main',
                     'skll_convert = skll.utils.commandline.skll_convert:main',
                     'summarize_results = skll.utils.commandline.summarize_results:main',
                     'compute_eval_from_predictions = skll.utils.commandline.compute_eval_from_predictions:main',
                     'plot_learning_curves = skll.utils.commandline.plot_learning_curves:main']},
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
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10',
                   ],
      zip_safe=False)
