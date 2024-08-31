# learning-metacontrol

This is code for the publication, ``Understanding human meta-control and its pathologies using deep neural networks" (Kai Sandbrink, Laurence Hunt, Christopher Summerfield).

The main supplementary figures are generated from the "fig_*.ipynb" notebooks in the "nns" folder. To run these, the "humans" folder first needs to be installed as a Python package by calling

``` 
pip install -e . 
```

from the "humans" directory. Please note the human behavioral data needs to be downloaded externally.

To train the neural networks, run the "run_*.py" files in "nns". Then, run the needed "sample_*.py" and "save_*.py" files to generate the synthetic behavioral and neural data, updating the timestamps as needed.

To generate the POMPD solutions, run the "ovb_pepe_testcases_j_takes.jl" script followed by "ana_multi_efficacy_pomdp.py". 

A few individual supplementary panels are generated directly from the individual "ana_*.py" scripts.