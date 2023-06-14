#!/usr/bin/env python
# License: BSD 3 clause
"""
Script that converts feature files from one format to another.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import UnicodeDammit

from skll.data.dict_vectorizer import DictVectorizer
from skll.data.readers import EXT_TO_READER
from skll.data.writers import (
    EXT_TO_WRITER,
    ARFFWriter,
    CSVWriter,
    LibSVMWriter,
    TSVWriter,
)
from skll.version import __version__


def _pair_to_dict_tuple(pair: str) -> Tuple[str, int]:
    """Construct mappings from feature/class names to numbers."""
    number_str, name = pair.split("=")
    number = int(number_str)
    return (name, number)


def main(argv: Optional[List[str]] = None) -> None:
    """
    Handle command line arguments and gets things started.

    Parameters
    ----------
    argv : Optional[List[str]], default=None
        List of arguments, as if specified on the command-line.
        If ``None``, ``sys.argv[1:]`` is used instead.
    """
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Takes an input feature file and converts it to another "
        "format. Formats are determined automatically from file "
        "extensions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "infile",
        help="input feature file (ends in .arff, .csv, " ".jsonlines, .libsvm, .ndj, or .tsv)",
    )
    parser.add_argument(
        "outfile",
        help="output feature file (ends in .arff, .csv, " ".jsonlines, .libsvm, .ndj, or .tsv)",
    )
    parser.add_argument(
        "-i",
        "--id_col",
        help="Name of the column which contains the instance " "IDs in ARFF, CSV, or TSV files.",
        default="id",
    )
    label_group = parser.add_mutually_exclusive_group(required=False)
    label_group.add_argument(
        "-l",
        "--label_col",
        help="Name of the column which contains the "
        "class labels in ARFF, CSV, or TSV files. "
        "For ARFF files, this must be the final "
        "column to count as the label.",
        default="y",
    )
    label_group.add_argument(
        "--no_labels",
        action="store_true",
        default=False,
        help="Used to indicate that the input data has " "no labels.",
    )
    parser.add_argument(
        "-q", "--quiet", help='Suppress printing of "Loading..." messages.', action="store_true"
    )
    parser.add_argument(
        "--arff_regression",
        help="Create ARFF files for regression, not " "classification.",
        action="store_true",
    )
    parser.add_argument(
        "--arff_relation", help="Relation name to use for ARFF file.", default="skll_relation"
    )
    parser.add_argument(
        "--reuse_libsvm_map",
        help="If you want to output multiple files that use "
        "the same mapping from labels and features to "
        "numbers when writing libsvm files, you can "
        "specify an existing .libsvm file to reuse the "
        "mapping from.",
        type=argparse.FileType("rb"),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - " "%(message)s")
    logger = logging.getLogger(__name__)

    # make sure the input file extension is one we can process
    input_extension = Path(args.infile).suffix.lower()
    output_extension = Path(args.outfile).suffix.lower()

    if input_extension not in EXT_TO_READER:
        logger.error(
            "Input file must be in either .arff, .csv, .jsonlines, "
            ".libsvm, .ndj, or .tsv format. You specified: "
            f"{input_extension}"
        )
        sys.exit(1)

    # Build feature and label vectorizers from existing libsvm file if asked
    feat_map: Dict[str, int]
    label_map: Optional[Dict[str, int]]
    if args.reuse_libsvm_map and output_extension == ".libsvm":
        feat_map = {}
        label_map = {}
        for line in args.reuse_libsvm_map:
            line = UnicodeDammit(line, ["utf-8", "windows-1252"]).unicode_markup
            if "#" not in line:
                logger.error(
                    "The LibSVM file you want to reuse the map from "
                    "was not created by SKLL and does not actually "
                    "contain the necessary mapping info."
                )
                sys.exit(1)
            comments = line.split("#")[1]
            _, label_map_str, feat_map_str = comments.split("|")
            feat_map.update(_pair_to_dict_tuple(pair) for pair in feat_map_str.strip().split())
            label_map.update(_pair_to_dict_tuple(pair) for pair in label_map_str.strip().split())
        feat_vectorizer = DictVectorizer()
        feat_vectorizer.fit([{name: 1} for name in feat_map])
        feat_vectorizer.vocabulary_ = feat_map
    else:
        feat_vectorizer = None
        label_map = None

    label_col = None if args.no_labels else args.label_col

    # Iterate through input file and collect the information we need
    reader = EXT_TO_READER[input_extension](
        args.infile, quiet=args.quiet, label_col=label_col, id_col=args.id_col
    )
    feature_set = reader.read()
    # write out the file in the requested output format
    writer_type = EXT_TO_WRITER[output_extension]
    writer_args = {"quiet": args.quiet}
    if writer_type is CSVWriter or writer_type is TSVWriter:
        writer_args["label_col"] = label_col
        writer_args["id_col"] = args.id_col
    elif writer_type is ARFFWriter:
        writer_args["label_col"] = label_col
        writer_args["id_col"] = args.id_col
        writer_args["regression"] = args.arff_regression
        writer_args["relation"] = args.arff_relation
    elif writer_type is LibSVMWriter:
        writer_args["label_map"] = label_map
    writer = writer_type(args.outfile, feature_set, **writer_args)
    writer.write()


if __name__ == "__main__":
    main()
