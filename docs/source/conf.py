import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

project = 'DeFi-ABM'
copyright = '2025, Auralshin'
author = 'Auralshin'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_logo = "_static/logo.png"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#007bff",
        "color-brand-content": "#0056b3",
    },
    "dark_css_variables": {
        "color-brand-primary": "#61dafb",
        "color-brand-content": "#21a1f1",
    },
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
