from setuptools import setup


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
      install_requires=['scikit-learn', 'six', 'PrettyTable'],
      zip_safe=False)
