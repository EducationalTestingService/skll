"""Configure sphinx settings for SKLL documentation."""

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))

import skll  # noqa: E402,F401
from skll.version import __version__  # noqa: E402

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "SciKit-Learn Laboratory"
copyright = "2012-2023, Educational Testing Service"

# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = version

# intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# enable type hints
autodoc_typehints = "description"


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Output file base name for HTML help builder.
htmlhelp_basename = "SKLLdoc"

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "12pt"
    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    (
        "index",
        "SKLL.tex",
        "SciKit-Learn Laboratory Documentation",
        "Educational Testing Service",
        "manual",
    ),
]

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ("index", "SKLL", "SciKit-Learn Laboratory Documentation", ["Educational Testing Service"], 1)
]

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "SKLL",
        "SciKit-Learn Laboratory Documentation",
        "Educational Testing Service",
        "SciKit-LearnLab",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# Bibliographic Dublin Core info.
epub_title = "SciKit-Learn Laboratory"
epub_author = "Educational Testing Service"
epub_publisher = "Educational Testing Service"
epub_copyright = "2012-2023, Educational Testing Service"
