# Kai Sandbrink
# 2023-02-01
# This script samples trajectories across test tau values for the specified models

# %% LIBRARY IMPORTS

import torch
import pandas as pd
import numpy as np
from utils import Config, get_timestamp
from utils_project import load_config_files
from test_case import test, test_helplessness, perturbed_test
import os, copy, pickle

import multiprocessing as mp

# %% SPECIFY TEST CASES

### 5/23 BIAS 0.5 VOL 0.1

ape_models = [
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

# %% PARAMETERS

#checkpoint = ''
n_repeats_case = 100
test_taus = np.arange(0,1.01,0.125)
model_folder = 'models'

#save_results_folder = os.path.join('results', 'pepe', '%s_eval_learning_curves_%srepeats' %(get_timestamp(), n_repeats_case))
save_results_base = os.path.join('data', 'ablated', 'pepe', )

# %% INITIALIZATIONS

## TORCH
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

## OTHER
timestamp = get_timestamp()

# %% CREATE TEST FUNCTIONS

test_taus = np.arange(0,1.01,0.125)
target_taus = np.arange(-3, 4.01, 1).astype(float)

def sample_ablated_model_trajectory(modelname, test_taus, model_save_folder=None, ape=True, sleep=False, n_repeats_case = None, checkpoint='', timestamp = None, device='cpu', model_folder = None, target_tau = 0):

    config, task_options, ape_nn_options = load_config_files(modelname)

    if n_repeats_case is not None:
        config.n_repeats_case = n_repeats_case

    rews_taus_ape = []
    counters_peeks_taus_ape = []
    control_errs_taus_ape = []
    counters_sleeps_taus_ape = []

    for test_tau in test_taus:

        tau_task_options = Config(copy.deepcopy(task_options.__dict__))
        tau_task_options.starting_taus = {'peek': 0, 'take': test_tau}

        _, (rews_ape, _, counter_peeks_ape, counter_sleeps_taus_ape, _, _, control_errs_ape) = test_helplessness(config, ape_nn_options, tau_task_options, n_repeats_case = n_repeats_case, device=device, model_folder=model_folder, taus_clamp=target_tau)
        rews_taus_ape.append(rews_ape)
        counters_peeks_taus_ape.append(counter_peeks_ape)
        control_errs_taus_ape.append(control_errs_ape)
        counters_sleeps_taus_ape.append(counter_sleeps_taus_ape)

    if model_save_folder is not None:
        #model_save_folder = os.path.join(save_base_data_folder, str(modelname))

        if timestamp is None:
            timestamp = get_timestamp()

        model_save_folder = os.path.join(model_save_folder, 'ablated_tau%d' %target_tau)
            
        os.makedirs(model_save_folder, exist_ok=True)

        pickle.dump(rews_taus_ape, open(os.path.join(model_save_folder, '%s_rewss_taus.pkl' %timestamp), 'wb'))
        pickle.dump(counters_peeks_taus_ape, open(os.path.join(model_save_folder, '%s_ablated_counters_peeks_taus.pkl' %timestamp), 'wb'))

        if ape:
            pickle.dump(control_errs_taus_ape, open(os.path.join(model_save_folder, '%s_ablated_control_errs_taus_ape.pkl' %timestamp), 'wb'))    

        if sleep:
            pickle.dump(counters_sleeps_taus_ape, open(os.path.join(model_save_folder, '%s_ablated_sleep_errs_taus_ape.pkl' %timestamp), 'wb'))    

    return rews_taus_ape, counters_peeks_taus_ape, control_errs_taus_ape, counters_sleeps_taus_ape

# %% TEST

if __name__ == '__main__':
    mp.set_start_method('spawn')
    processes = []

    ### EVALUATE LEARNING CURVES
    for ape_model in ape_models:
        ape_model = str(ape_model)

        for target_tau in target_taus:
            p = mp.Process(target=sample_ablated_model_trajectory, args=(ape_model, test_taus, os.path.join(save_results_base, ape_model), True, False, n_repeats_case, '', timestamp, device, os.path.join(model_folder, ape_model), target_tau))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
