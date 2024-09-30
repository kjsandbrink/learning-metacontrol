# Kai Sandbrink
# 2023-02-01
# This script samples trajectories across test tau values for the specified models

# %% LIBRARY IMPORTS

import torch
import pandas as pd
import numpy as np
from utils import Config, get_timestamp
from utils_project import load_config_files
from test_case import test
import os, copy, pickle

from nns.sample_task1_trajectories import sample_model_trajectory

import multiprocessing as mp

# %% SPECIFY TEST CASES

#### 5/23 WITH BIAS 0.5, VOL 0.1

pepe_models =  [
    20230427201627,
    20230427201629,
    20230427201630,
    20230427201632,
    20230427201633,
    
    20230427201644,
    20230427201646,
    20230427201647,
    20230427201648,
    20230427201649
]
    
levc_models = [
    20230519234556,
    20230519234555,
    20230519234553,
    20230519234552,
    20230519234550,

    20230519234540,
    20230519234538,
    20230519234536,
    20230519234534,
    20230519234533,
]

# %% PARAMETERS

checkpoint = ''
n_repeats_case = 100
test_taus = np.arange(0,1.01,0.250)
mistrained_base_model_folder = os.path.join('data','mistrained_models')
mistraining_timestamp = '20230523204912'
timestamp = get_timestamp()

#save_results_folder = os.path.join('results', 'pepe', '%s_eval_learning_curves_%srepeats' %(get_timestamp(), n_repeats_case))
save_results_base = os.path.join('data', 'mistrained_trajectories', )

# %% INITIALIZATIONS

## TORCH
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(f"Using {device} device")

## OTHER
timestamp = get_timestamp()

# %% TEST

if __name__ == '__main__':
    mp.set_start_method('spawn')
    processes = []

    ### EVALUATE LEARNING CURVES
    for ape_model in pepe_models:
        for tau in test_taus:
            ape_model = str(ape_model)
            p = mp.Process(target=sample_model_trajectory, args=(ape_model, test_taus, os.path.join(save_results_base, ape_model, 'mistrained_tau%d' %(tau*100)), True, False, n_repeats_case, '', timestamp, device, os.path.join(mistrained_base_model_folder, str(ape_model), mistraining_timestamp + '_mistrained_tau%d' %(tau*100))))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    for control_model in levc_models:
        for tau in test_taus:
            control_model = str(control_model)
            p = mp.Process(target=sample_model_trajectory, args=(control_model, test_taus, os.path.join(save_results_base, control_model, 'mistrained_tau%d' %(tau*100)), True, True, n_repeats_case, '', timestamp, device, os.path.join(mistrained_base_model_folder, str(control_model), mistraining_timestamp + '_mistrained_tau%d' %(tau*100))))
            p.start()
            processes.append(p)
    
        for p in processes:
            p.join()