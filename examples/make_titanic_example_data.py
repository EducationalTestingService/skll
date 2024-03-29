#!/usr/bin/env python
"""
Create data for Titanic classification example.

A script to split the train.csv and test.csv files from the
Kaggle "Titanic: Machine Learning from Disaster" competition
into the format `titanic.cfg` expects.

:author: Dan Blanchard (dblanchard@ets.org)
:organization: ETS
"""

import logging
from itertools import chain
from pathlib import Path

from skll.data import Reader, Writer


def main():
    """Create directories and split CSV files into subsets."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - " "%(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # Create dictionary of subsets to use for creating split feature files
    subset_dict = {
        "vitals": ["Sex", "Age"],
        "socioeconomic": ["Pclass", "Fare"],
        "family": ["SibSp", "Parch"],
        "misc": ["Embarked"],
    }
    features_to_keep = list(chain(*subset_dict.values()))

    # Create directories to store files
    root_path = Path("titanic")
    train_path = root_path / "train"
    dev_path = root_path / "dev"
    train_dev_path = root_path / "train+dev"
    test_path = root_path / "test"

    for path in [train_path, dev_path, train_dev_path, test_path]:
        if not path.exists():
            logger.info(f"Creating {path} directory")
            path.mkdir(parents=True)

    usecols_train = features_to_keep + ["PassengerId", "Survived"]
    usecols_test = features_to_keep + ["PassengerId"]

    # Read and write training FeatureSet
    train_fs = Reader.for_path(
        "titanic/train.csv",
        label_col="Survived",
        id_col="PassengerId",
        drop_blanks=True,
        pandas_kwargs={"usecols": usecols_train},
        quiet=False,
        sparse=False,
    ).read()

    train_fs.filter(features=features_to_keep)
    num_train_dev = len(train_fs)
    num_train = int((num_train_dev / 5) * 4)
    writer = Writer.for_path(
        "titanic/train/.csv",
        train_fs[:num_train],
        id_col="PassengerId",
        label_col="Survived",
        quiet=False,
        subsets=subset_dict,
    )
    writer.write()

    # Write train+dev set for training model to use to generate predictions on
    # test
    writer = Writer.for_path(
        "titanic/train+dev/.csv",
        train_fs,
        label_col="Survived",
        id_col="PassengerId",
        quiet=False,
        subsets=subset_dict,
    )
    writer.write()

    # Write dev FeatureSet
    writer = Writer.for_path(
        "titanic/dev/.csv",
        train_fs[num_train:],
        label_col="Survived",
        id_col="PassengerId",
        quiet=False,
        subsets=subset_dict,
    )
    writer.write()

    # Read and write test FeatureSet
    test_fs = Reader.for_path(
        "titanic/test.csv",
        label_col="Survived",
        drop_blanks=True,
        pandas_kwargs={"usecols": usecols_test},
        quiet=False,
        sparse=False,
    ).read()

    test_fs.filter(features=features_to_keep)
    num_test = len(test_fs)
    test_fs.ids = list(range(num_train_dev + 1, num_test + num_train_dev + 1))
    writer = Writer.for_path(
        "titanic/test/.csv", test_fs, id_col="PassengerId", quiet=False, subsets=subset_dict
    )
    writer.write()


if __name__ == "__main__":
    main()
