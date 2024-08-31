# Kai Sandbrink
# 2023-02-01
# This script computes the learning curves for specified test models

# %% LIBRARY IMPORTS

import torch
import pandas as pd
import numpy as np
from utils import Config, get_timestamp
from utils_project import load_config_files
from test_case import test
import os, copy, pickle
import multiprocessing as mp

from eval_pepe_learningcurves import test_model, test_model_taus

# %% SPECIFY TEST CASES

### NO HOLDOUT BIAS 0.5, VOL 0.1, 250k ENTROPY ANNEALING
## 12/11

ape_models = [
    20231017233748,
    20231017233746,
    20231017233744,
    20231017233742,
    20231017233741,
    20231017233738,
    20231017233737,
    20231017233735,
    20231017233734,
    20231017233732
]

control_models = [
    20231023003506,
    20231023003504,
    20231023003503,
    20231023003501,
    20231023003500, 
    20231023003457,
    20231023003456,
    20231023003454,
    20231023003453,
    20231023003451
]

# %% PARAMETERS

n_checkpoints = 100
n_repeats_case = 100
test_taus = np.arange(0,1.01,0.125)
model_folder = 'models'

#save_results_folder = os.path.join('results', 'levc', '%s_eval_learning_curves_%srepeats' %(get_timestamp(), n_repeats_case))
save_results_base = os.path.join('data', 'eval', 'levc', )

# %% TEST

if __name__ == '__main__':

    
    ## TORCH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = 'cpu'
    print(f"Using {device} device")

    
    ## OTHER
    #timestamp = get_timestamp()
    timestamp = '20230324153020'

    mp.set_start_method('spawn')
    processes = []

    n_simultaneous = 5
    counter = 0

    ### EVALUATE LEARNING CURVES
    for ape_model in ape_models:
        ape_model = str(ape_model)
        p = mp.Process(target=test_model, args=(ape_model, os.path.join(save_results_base, ape_model), n_repeats_case, timestamp, True, device))
        p.start()
        processes.append(p)

        counter += 1

        if counter % n_simultaneous == 0:

            for p in processes:
                p.join()

    for control_model in control_models:
        control_model = str(control_model)
        p = mp.Process(target=test_model, args=(control_model, os.path.join(save_results_base, control_model), n_repeats_case, timestamp, False, device))
        p.start()
        processes.append(p)
    
        counter += 1

        if counter % n_simultaneous == 0:

            for p in processes:
                p.join()
