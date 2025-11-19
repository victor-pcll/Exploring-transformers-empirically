import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'TPIV'
copyright = '2025, Peucelle Victor'
author = 'Peucelle Victor'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary'
]

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []
autodoc_mock_imports = ["torch", "numpy"]

# -- Options for HTML output -------------------------------------------------
# Use a theme compatible with GitHub Pages and relative URLs
html_theme = "sphinx_rtd_theme"
html_theme_options = {}
html_static_path = ['_static']
html_css_files = []
html_js_files = []
html_show_sourcelink = False
html_baseurl = "./"
