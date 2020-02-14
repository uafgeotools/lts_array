# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'lts_array'
copyright = '2019, Jordan W. Bishop, David Fee, and Curt Szuberla'
author = 'Jordan W. Bishop, David Fee, and Curt Szuberla'

release = '1.0.0'

# -- General configuration ---------------------------------------------------

language = 'python'

extensions = ['sphinxcontrib.apidoc',
              'sphinx.ext.autodoc',
              'spinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx_rtd_theme',
              'spinx.ext.viewcode']

autodoc_mock_imports = ['matplotlib',
                        'numpy',
                        'obspy',
                        'scipy',
                        ]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['build', '.DS_Store']

# List of source suffixes - the files that Sphinx will read and their
# markdown type.
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

# Parsers - rst is implicit. Add markdown parser.
source_parsers = {
   '.md': 'recommonmark.parser.CommonMarkParser',
}

html_theme = 'sphinx_rtd_theme'

# -- Options for docstrings -------------------------------------------------
# Docstring Parsing with napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- URL handling -----------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'obspy': ('https://docs.obspy.org/', None),
    'matplotlib': ('https://matplotlib.org/', None)
}
