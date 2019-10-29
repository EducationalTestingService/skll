# License: BSD 3 clause
"""
Run the examples, just to make sure they are all still working

:author: Jeremy Biggs (jbiggs@@ets.org)
"""

import json
import subprocess

from glob import glob
from os import getcwd, makedirs
from os.path import abspath, basename, dirname, exists, join
from shutil import copytree, copyfile, rmtree

from nose.tools import eq_, assert_almost_equal

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


def run_configuration_and_check_outputs(config_path):
    """
    Run the configuration, and then check the JSON results
    against expected JSON files
    """

    # run this experiment, get the `results_json_path`
    results_json_path = run_configuration(config_path, local=True, quiet=True)[0]

    # if the results path exists, check the output
    if exists(results_json_path):

        results_json_exp_path = join(_my_dir, 'other', 'expected', basename(results_json_path))
        results_obj = json.load(open(results_json_path, 'r'))[0]
        results_exp_obj = json.load(open(results_json_exp_path, 'r'))[0]

        # we check a subset of the values, just to make sure
        # that nothing weird is going on with our output
        for key in ["train_set_size", "test_set_size",
                    "learner_name", "cv_folds", "feature_scaling",
                    "grid_score", "grid_objective",
                    "accuracy", "score", "pearson"]:

            # we obviously want to skip any keys that we aren't expecting
            if key in results_exp_obj:

                actual = results_obj[key]
                expected = results_exp_obj[key]

                # if this is a float, then we check with less precision (4 decimals);
                # otherwise, we check to make sure things are matching exactly
                if isinstance(expected, float):
                    assert_almost_equal(actual, expected, places=4)
                else:
                    eq_(actual, expected)


def setup():
    """
    Create necessary directories for testing,
    and copy files to new locations.
    """

    # Create the directories we need for boston and iris;
    # if these directories already exist, it's fine
    makedirs(_new_boston_dir, exist_ok=True)
    makedirs(_new_iris_dir, exist_ok=True)

    # We get rid of the new titanic directory, if it already exists,
    # because `copytree()` will raise an error if it already exists.
    # Note :: In Python 3.8, `copytree()` has a new argument,
    # `dirs_exist_ok`, which would render this step unnecessary.
    if exists(_new_titanic_dir):
        rmtree(_new_titanic_dir)

    # Copy the titanic data to our new directories
    copytree(_old_titanic_dir, _new_titanic_dir)

    # Create all of the data sets we need
    subprocess.run(['python', join(_examples_dir, 'make_titanic_example_data.py')],
                   cwd=dirname(_new_titanic_dir))
    subprocess.run(['python', join(_examples_dir, 'make_boston_example_data.py')],
                   cwd=dirname(_new_boston_dir))
    subprocess.run(['python', join(_examples_dir, 'make_iris_example_data.py')],
                   cwd=dirname(_new_iris_dir))

    # Move all the configuration files to our new directories
    for cfg_file in glob(join(_old_titanic_dir, '**.cfg')):
        copyfile(cfg_file, join(_new_titanic_dir, basename(cfg_file)))

    for cfg_file in glob(join(_old_boston_dir, '**.cfg')):
        copyfile(cfg_file, join(_new_boston_dir, basename(cfg_file)))

    for cfg_file in glob(join(_old_iris_dir, '**.cfg')):
        copyfile(cfg_file, join(_new_iris_dir, basename(cfg_file)))


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
    for config_path in glob(join(_new_titanic_dir, '*.cfg')):
        run_configuration_and_check_outputs(config_path)


def test_boston_configs():
    """
    Run all of the configuration files for the boston example
    """
    for config_path in glob(join(_new_boston_dir, '*.cfg')):
        run_configuration_and_check_outputs(config_path)


def test_iris_configs():
    """
    Run all of the configuration files for the iris example
    """
    for config_path in glob(join(_new_iris_dir, '*.cfg')):
        run_configuration_and_check_outputs(config_path)
