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

from nns.settings_ana import pepe_human_ape_models as ape_models

# %% PARAMETERS

#checkpoint = ''
n_repeats_case = 100
model_folder = 'models'

save_results_base = os.path.join('data', 'perturbed', 'pepe', )

# %% INITIALIZATIONS

## TORCH
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

## OTHER
timestamp = get_timestamp()

# %% CREATE TEST FUNCTIONS

test_taus = np.arange(0,1.01,0.125)
#target_taus = np.arange(-0.5, 1.51, 0.5).astype(float)
target_taus = None
perturbation_taus = np.arange(-1.5, 1.6, 0.25).astype(float)

def sample_perturbed_model_trajectory(modelname, test_taus, model_save_folder=None, ape=True, sleep=False, n_repeats_case = None, checkpoint='', timestamp = None, device='cpu', model_folder = None, target_tau = None, perturbation = None):

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

        if target_tau is not None:
            target = target_tau
        else:
            target = test_tau
        if perturbation is not None:
            target += perturbation

        _, (rews_ape, _, counter_peeks_ape, counter_sleeps_taus_ape, _, _, control_errs_ape) = perturbed_test(config, ape_nn_options, tau_task_options, n_repeats_case = n_repeats_case, device=device, model_folder=model_folder, target_tau=target)
        rews_taus_ape.append(rews_ape)
        counters_peeks_taus_ape.append(counter_peeks_ape)
        control_errs_taus_ape.append(control_errs_ape)
        counters_sleeps_taus_ape.append(counter_sleeps_taus_ape)

    if model_save_folder is not None:
        #model_save_folder = os.path.join(save_base_data_folder, str(modelname))

        if target_tau is not None:
            model_save_folder = os.path.join(model_save_folder, 'perturbed_tau%d' %(target_tau*100))
        elif perturbation is not None:
            model_save_folder = os.path.join(model_save_folder, 'perturbed_tau_perturbation%d' %(perturbation*100))
        else:
            model_save_folder = os.path.join(model_save_folder, 'perturbation%d_target%d' %(perturbation*100, target_tau*100))

        if timestamp is None:
            timestamp = get_timestamp()
            
        os.makedirs(model_save_folder, exist_ok=True)

        pickle.dump(rews_taus_ape, open(os.path.join(model_save_folder, '%s_rewss_taus.pkl' %timestamp), 'wb'))
        pickle.dump(counters_peeks_taus_ape, open(os.path.join(model_save_folder, '%s_perturbed_counters_peeks_taus.pkl' %timestamp), 'wb'))

        if ape:
            pickle.dump(control_errs_taus_ape, open(os.path.join(model_save_folder, '%s_perturbed_control_errs_taus_ape.pkl' %timestamp), 'wb'))    

        if sleep:
            pickle.dump(counters_sleeps_taus_ape, open(os.path.join(model_save_folder, '%s_perturbed_sleep_errs_taus_ape.pkl' %timestamp), 'wb'))    

        pickle.dump(test_taus, open(os.path.join(model_save_folder, '%s_perturbed_test_taus.pkl' %timestamp), 'wb'))

    return rews_taus_ape, counters_peeks_taus_ape, control_errs_taus_ape, counters_sleeps_taus_ape

# %% TEST

if __name__ == '__main__':
    mp.set_start_method('spawn')
    processes = []

    i = 0

    ### EVALUATE LEARNING CURVES
    for ape_model in ape_models:
        if target_taus is not None:
            for target_tau in target_taus:
                ape_model = str(ape_model)
                p = mp.Process(target=sample_perturbed_model_trajectory, args=(ape_model, test_taus, os.path.join(save_results_base, ape_model), True, False, n_repeats_case, '', timestamp, device, os.path.join(model_folder, ape_model), target_tau))
                p.start()
                processes.append(p)

                i += 1
                if i % 5 == 0:
                    for p in processes:
                        p.join()

        if perturbation_taus is not None:
            for perturbation in perturbation_taus:
                ape_model = str(ape_model)
                p = mp.Process(
                    target=sample_perturbed_model_trajectory, 
                    args=(
                        ape_model, 
                        test_taus, 
                        os.path.join(save_results_base, ape_model), 
                        True, 
                        False, 
                        n_repeats_case, 
                        '', 
                        timestamp, 
                        device, 
                        os.path.join(model_folder, ape_model), 
                        None, 
                        perturbation
                        )
                    )
                p.start()
                processes.append(p)

                i += 1
                if i % 5 == 0:
                    for p in processes:
                        p.join()
