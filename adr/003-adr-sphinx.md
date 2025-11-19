# ADR 003 — Documentation System Using Sphinx

## Status
Accepted

## Context
The project requires a clear, maintainable, and automatically generated documentation system for Python modules.  
The documentation must be easy to build locally, compatible with ReadTheDocs/GitHub Pages, and support autodocumentation of Python code (docstrings).

Sphinx is the industry‑standard tool for Python documentation. It supports:
- Automatic documentation generation from docstrings (`autodoc`)
- Google/NumPy docstring styles (`napoleon`)
- HTML, PDF, and LaTeX outputs
- Modular documentation using `.rst` or `.md`
- API reference generation with `sphinx-apidoc`

## Decision
We adopt **Sphinx** as the documentation engine.

## Installation Procedure

### 1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Sphinx inside the virtual environment
```bash
pip install sphinx
```

(Optional recommended extensions:)
```bash
pip install sphinx-autobuild sphinx-rtd-theme
```

### 3. Initialize Sphinx in the project
From the project root (`tpiv-simulations/`):

```bash
sphinx-quickstart
```

This command creates:
```
source/conf.py
source/index.rst
Makefile
```

### 4. Enable autodoc & napoleon in `conf.py`
Add or modify:
```python
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]
autosummary_generate = True
```

Add project path:
```python
import os
import sys
sys.path.insert(0, os.path.abspath(".."))
```

## Usage

### Generating API documentation automatically
```bash
sphinx-apidoc -o source/ .
```

This generates `.rst` files for each Python module.

### Building the documentation
```bash
make html
```

Outputs will be located in:
```
build/html/index.html
```

You can open this file in your browser to view the documentation.

### Live reload (optional)
If `sphinx-autobuild` is installed:
```bash
sphinx-autobuild source build/html
```

### Github deployment

#### Manual deployment
```bash
git checkout --orphan gh-pages
cp -r build/html/* .
git add .
git commit -m "Deploy Sphinx doc"
git push origin gh-pages
```

#### Automated deployment via GitHub Actions

To automate building and deploying the documentation to the `gh-pages` branch on each push to `main`, add the following GitHub Actions workflow file `.github/workflows/deploy-docs.yml`:

```yaml
name: Deploy Sphinx Documentation

on:
  push:
    branches:
      - main

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.x'

    - name: Install dependencies
      run: |
        python -m venv .venv
        source .venv/bin/activate
        pip install sphinx sphinx-rtd-theme

    - name: Generate API docs
      run: |
        source .venv/bin/activate
        sphinx-apidoc -o source/ .

    - name: Build HTML docs
      run: |
        source .venv/bin/activate
        make html

    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./build/html
        publish_branch: gh-pages
```

This workflow will:
- Check out the repository
- Set up Python and install dependencies
- Generate API documentation
- Build the HTML documentation
- Deploy the built docs to the `gh-pages` branch automatically

Make sure the `gh-pages` branch is configured as the GitHub Pages source in your repository settings.

## Consequences
- Developers must maintain docstrings in NumPy or Google format.
- Documentation becomes reproducible and centralised.
- Works with CI systems for automated builds.

## Alternatives Considered
- MkDocs: simpler but weaker autodoc integration.
- pdoc: lightweight but insufficient for large structured documentation.
- Handwritten Markdown: unmaintainable for long-term evolution.

Sphinx provides the best long-term structure and flexibility.