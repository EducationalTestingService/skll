.. sectionauthor:: Dan Blanchard <dblanchard@ets.org>

Running Experiments
===================
The simplest way to use SKLL is to create configuration files that describe
experiments you would like to run on pre-generated features. This document
describes the supported feature file formats, how to create configuration files
(and layout your directories), and how use ``run_experiment`` to get things
going.

Feature file formats
--------------------
The following feature file formats are supported:

	**megam**
		An expanded form of the input format for the
		`MegaM classification package <http://www.umiacs.umd.edu/~hal/megam/>`_
		with the ``-fvals`` switch.

		The basic format is::

			# Instance1
			CLASS1	F0 2.5 F1 3 FEATURE_2 -152000
			# Instance2
			CLASS2	F1 7.524

		where the comments before each instance are optional IDs for the following
		line, class names are separated from feature-value pairs with a tab, and
		feature-value pairs are separated by spaces. Any omitted features for a
		given instance are assumed to be zero, so this format is handy when dealing
		with sparse data. We also include several utility scripts for converting
		to/from this MegaM format and for adding/removing features from the files.

	**tsv**
		A simple tab-delimited format with the following restrictions:

		*	The first column contains the class label for each instance.
		*	If there is a column called "id" present, this will be treated as the
			ID for each row.
		*	All other columns contain feature values, and every feature value must
			be specified (making this a poor choice for sparse data).

	**jsonlines**
		A twist on the `JSON <http://www.json.org/>`_ format where every line is a
		JSON dictionary (the entire contents of a normal JSON file). Each dictionary
		is expected to contain the following keys:

		*	y: The class label.
		*	x: A dictionary of feature values.
		*	id: An optional instance ID.


Creating configuration files
----------------------------




Using run_experiment
--------------------