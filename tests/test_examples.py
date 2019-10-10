# License: BSD 3 clause
"""
Run the examples, just to make sure they are all still working

:author: Jeremy Biggs (jbiggs@@ets.org)
"""

from glob import glob
from os import chdir, getcwd, listdir, makedirs
from os.path import abspath, basename, dirname, join
from shutil import copyfile, rmtree

from examples import make_titanic_example_data as mtd
from examples import make_boston_example_data as mbd
from examples import make_iris_example_data as mid

from skll.experiments import run_configuration


_my_cwd = getcwd()
_my_dir = abspath(dirname(__file__))
_examples_dir = join(_my_dir, '..', 'examples')

_old_titanic_dir = join(_examples_dir, 'titanic')
_old_boston_dir = join(_examples_dir, 'boston')
_old_iris_dir = join(_examples_dir, 'iris')

_new_titanic_dir = join(_my_dir, 'other', 'titanic')
_new_boston_dir = join(_my_dir, 'other', 'boston')
_new_iris_dir = join(_my_dir, 'other', 'iris')


def run_function_in_custom_dir(func, old_dir, new_dir):
    """
    A helper function to create the data in new directory
    and then switch back to our original directory.
    """
    chdir(new_dir)
    func()
    chdir(old_dir)


def setup():
    """
    Create necessary directories for testing,
    and copy files to new locations.
    """

    # Create all of the directories we need
    makedirs(_new_titanic_dir, exist_ok=True)
    makedirs(_new_boston_dir, exist_ok=True)
    makedirs(_new_iris_dir, exist_ok=True)

    # Move the titanic data to our new directories
    for file in listdir(join(_old_titanic_dir, 'titanic')):
        copyfile(join(_old_titanic_dir, 'titanic', file),
                 join(_new_titanic_dir, file))

    # Create all of the data sets we need
    run_function_in_custom_dir(mtd.main, _my_cwd, _new_titanic_dir)
    run_function_in_custom_dir(mbd.main, _my_cwd, _new_boston_dir)
    run_function_in_custom_dir(mid.main, _my_cwd, _new_iris_dir)

    # Move all the configuration files to our new directories
    for file in glob(join(_old_titanic_dir, '**.cfg')):
        copyfile(file, join(_new_titanic_dir, 'titanic', basename(file)))

    for file in glob(join(_old_boston_dir, '**.cfg')):
        copyfile(file, join(_new_boston_dir, 'boston', basename(file)))

    for file in glob(join(_old_iris_dir, '**.cfg')):
        copyfile(file, join(_new_iris_dir, 'iris', basename(file)))


def tearDown():
    """
    Clean up after tests, remove all directories we created.
    """
    rmtree(_new_titanic_dir)
    rmtree(_new_boston_dir)
    rmtree(_new_iris_dir)


def test_titanic_configs():
    """
    Run all of the configuration files for the titanic example
    """
    for config_path in glob(join(_new_titanic_dir, 'titanic', '*.cfg')):
        run_configuration(config_path, quiet=True)


def test_boston_configs():
    """
    Run all of the configuration files for the boston example
    """
    for config_path in glob(join(_new_boston_dir, 'boston', '*.cfg')):
        run_configuration(config_path, quiet=True)


def test_iris_configs():
    """
    Run all of the configuration files for the iris example
    """
    for config_path in glob(join(_new_iris_dir, 'iris', '*.cfg')):
        run_configuration(config_path, quiet=True)
