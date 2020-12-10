# Nireact
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4313980.svg)](https://doi.org/10.5281/zenodo.4313980)

Using modeling to relate neuroimaging measures to reaction time data.

This project contains code used to run analyses and create plots and statistics for the following paper:

Morton NW, Schlichting ML, Preston AR. 2020. 
Representations of common event structure in medial temporal lobe and frontoparietal cortex support efficient inference. Proceedings of the National Academy of Sciences. 
117(47): 29338-29345. 
[doi:10.1073/pnas.1912338117](https://doi.org/10.1073/pnas.1912338117).

Nireact uses [PsiReact](https://github.com/mortonne/psireact), which provides general tools for modeling response time data.

## Installation

First, install swig. You can do this using Miniconda:

```bash
conda create -n nireact
conda activate nireact
conda install python=3.7 swig
```

Then you can install nireact into your conda environment using pip:

```bash
pip install git+git://github.com/mortonne/nireact 
```

## Reproducing analysis

Data are available on [OSF](https://osf.io/6eqbf/).
After downloading the data, you can run analyses for the paper using the Jupyter notebooks in the `jupyter` directory.
You will have to edit the first cell of each notebook to change the path to the data on your computer.
