#!/usr/bin/env python

"""
This is a simple script to download and transform some example data from
sklearn.datasets.

:author: Michael Heilman (mheilman@ets.org)
:author: Aoife Cahill (acahill@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import json
import os

import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split


def main():
    """
    Download some example data and split it into training and test data.
    The california data set is meant for regression modeling.
    """
    print("Retrieving california data from servers...", end="")
    california = sklearn.datasets.fetch_california_housing()
    print("done")

    # this dataset contains 20,640 samples which is too many
    # let's just sample this dataset to get 500 samples
    rng = np.random.default_rng(42)
    chosen_indices = rng.integers(0, california.target.shape[0], size=500)
    X = california.data[chosen_indices, :]
    Y = california.target[chosen_indices]

    # crate example jsonlines dictionaries
    examples = [
        {"id": f"EXAMPLE_{i}", "y": y, "x": {f"f{j}": x_val for j, x_val in enumerate(x)}}
        for i, (x, y) in enumerate(zip(X, Y))
    ]

    (examples_train, examples_test) = train_test_split(examples, test_size=0.33, random_state=42)

    print("Writing training and testing files...", end="")
    for examples, suffix in [(examples_train, "train"), (examples_test, "test")]:
        california_dir = os.path.join("california", suffix)
        if not os.path.exists(california_dir):
            os.makedirs(california_dir)
        jsonlines_path = os.path.join(california_dir, "example_california_features.jsonlines")
        with open(jsonlines_path, "w") as f:
            for ex in examples:
                f.write(f"{json.dumps(ex)}\n")
    print("done")


if __name__ == "__main__":
    main()
