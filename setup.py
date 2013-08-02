from setuptools import setup

from skll import __version__


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='skll',
      version=__version__,
      description=('SciKit-Learn Laboratory provides a number of utilities to make ' +
                   'it simpler to run common scikit-learn experiments with ' +
                   'pre-generated features.'),
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
               'scripts/megam_to_libsvm', 'scripts/print_model_weights',
               'scripts/run_experiment'],
      install_requires=['scikit-learn', 'six', 'PrettyTable', 'beautifulsoup4',
                        'numpy', 'ml_metrics'],
      zip_safe=False)
