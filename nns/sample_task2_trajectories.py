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

import multiprocessing as mp

from nns.sample_task1_trajectories import sample_model_trajectory

from nns.settings_ana import levc_human_ape_models as ape_models
from nns.settings_ana import levc_human_control_models as control_models

# %% PARAMETERS

checkpoint = ''
n_repeats_case = 100
test_taus = np.arange(0,1.01,0.125)
model_folder = 'models'

save_results_base = os.path.join('data', 'eval', 'levc', )

# %% INITIALIZATIONS

## TORCH
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

## OTHER
timestamp = get_timestamp()

# %% TEST

if __name__ == '__main__':
    mp.set_start_method('spawn')
    processes = []

    counter = 0
    n_simultaneous = 5
    
    for control_model in control_models:
        print(control_model)
        control_model = str(control_model)
        p = mp.Process(target=sample_model_trajectory, args=(control_model, test_taus, os.path.join(save_results_base, control_model), False, True, n_repeats_case, '', timestamp, device, os.path.join(model_folder, control_model)))
        p.start()
        processes.append(p)
    
        counter += 1

        if counter % n_simultaneous == 0:
            for p in processes:
                p.join()

# %%
