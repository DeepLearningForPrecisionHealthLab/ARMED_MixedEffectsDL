# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
sys.path.append('../')

project = 'ARMED'
copyright = '2023, Kevin P Nguyen, Alex Treacher, Albert Montillo'
author = 'Kevin P Nguyen, Alex Treacher, Albert Montillo'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Mock imports for autosummary
autodoc_mock_imports = ['sklearn', 'tensorflow', 'tensorflow_probability',
                        'tensorflow_addons', 'cv2', 'pandas', 'statsmodels',
                        'matplotlib', 'seaborn', 'scipy']

# Left align math blocks
mathjax3_config = {'chtml': {'displayAlign': 'left'}}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
