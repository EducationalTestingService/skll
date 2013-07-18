from setuptools import setup

from skll import __version__


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='skll',
      version='0.9',
      description='SciKit-Learn Lab provides a number of utilities to make it simpler to run common scikit-learn experiments with pre-generated features.',
      long_description=readme(),
      keywords='learning scikit-learn',
      url='http://github.com/EducationalTestingService/skll',
      author='Daniel Blanchard',
      author_email='dblanchard@ets.org',
      license='GPL',
      packages=['skll'],
      scripts=['scripts/generate_predictions.py', 'scripts/print_model_weights.py', 'scripts/run_ablation.py', 'scripts/run_experiment.py']
      install_requires=['scikit-learn', 'six', 'PrettyTable'],
      classifiers=
      zip_safe=False)
