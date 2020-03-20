# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath('../../lts_array/'))

# -- Project information -----------------------------------------------------
project = 'lts_array'
copyright = 'Jordan W. Bishop, David Fee, and Curt Szuberla'

# -- General configuration ---------------------------------------------------

language = 'python'
master_doc = 'index'

extensions = ['sphinxcontrib.apidoc',
              'sphinx.ext.autodoc',
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx_rtd_theme',
              'sphinx.ext.viewcode']

autodoc_mock_imports = ['numpy', 'scipy', 'obspy', 'matplotlib']
apidoc_module_dir = '../lts_array'
apidoc_output_dir = 'api'
apidoc_separate_modules = True
apidoc_toc_file = False

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
