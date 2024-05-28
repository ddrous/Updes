## Documentation website

## Getting started
Done with QuartoDoc. Three steps to build and deploy the documentation website:
- Install Quarto CLI, and the QuartoDoc Python package 
- Move into the `docs` folder: `cd docs`
- Automatically build the documentation from docstrings: `quartodoc build`
- Deploy the documentation website to GitHub Pages: `quarto publish`


## Interactive example with Shiny for Python
- Interactive cells: https://shiny.posit.co/py/
- Code at: https://github.com/posit-dev/py-shiny/tree/main/docs



## Building and deploying to PiPI
General steps for creating a new release: https://carpentries-incubator.github.io/python_packaging/instructor/05-publishing.html


- pip install build
- python3 -m build  (make sure the `dist` folder is empty)
- pip install twine
- twine check dist/*
- Add PyPI token to ~/.pypirc
- twine upload --repository testpypi dist/*
- twine upload dist/*
