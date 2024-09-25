# learning-metacontrol

This is code for the publication, ``Understanding human meta-control and its pathologies using deep neural networks" (Kai Sandbrink, Laurence Hunt, Christopher Summerfield).

The main supplementary figures are generated from the "fig_\*.ipynb" notebooks in the "nns" folder. To run these, the "humans" folder first needs to be installed as a Python package by calling

``` 
pip install -e . 
```

from the "humans" directory (which should only take a few seconds on a normal computer).

To train the neural networks, run the "run_\*.py" files in "nns". Then, run the needed "sample_\*.py" and "save_\*.py" files to generate the synthetic behavioral and neural data, updating the timestamps as needed.

To generate the POMPD solutions, run the "ovb_pepe_testcases_j_takes.jl" script followed by "ana_multi_efficacy_pomdp.py". 

A few individual supplementary panels are generated directly from the individual "ana_\*.py" scripts.

The human behavioral data is in the data folder. The scripts used for data collection are in the website folder.

### Versions

The neural network and analysis code uses Python v3.10.4 and associated numerous public libraries, as specified in the environment.yml file. The POMDP solver uses Julia v1.8.1 and associated libraries, as specified in the Manifest.toml. Code was operated on Ubuntu 22.04.3 LTS.

The website uses JSPsych v7.1.2.