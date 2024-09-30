# Kai Sandbrink
# 2023-02-01
# This script samples trajectories across test tau values for the specified models

# %% LIBRARY IMPORTS

import torch
import pandas as pd
import numpy as np
from utils import Config, get_timestamp, flatten
from utils_project import load_config_files
from test_case import test
import os, copy, pickle

import multiprocessing as mp

# %% SPECIFY TEST CASES

from nns.settings_ana import pepe_nn_ape_models as ape_models
from nns.settings_ana import pepe_nn_control_models as control_models
from nns.settings_ana import pepe_human_control_models as human_control_models
from nns.settings_ana import pepe_nn_baseline_models as baseline_models
from nns.settings_ana import pepe_nn_efficacy_at_input_models as efficacy_at_input_models
from nns.settings_ana import pepe_nn_extra_node_models as extra_node_models

# #convert baseline_models from dict to list
baseline_models = flatten(list(baseline_models.values()))
print(baseline_models)
models = efficacy_at_input_models + baseline_models

# %% PARAMETERS

checkpoint = ''
n_repeats_case = 1000
test_taus = np.arange(0,1.01,0.125)
model_folder = 'models'

save_results_base = os.path.join('data', 'eval', 'pepe', )

# %% INITIALIZATIONS

## TORCH
device = "cpu"
print(f"Using {device} device")

## OTHER
timestamp = get_timestamp()

# %% CREATE TEST FUNCTIONS

test_taus = np.arange(0,1.01,0.125)

def sample_model_trajectory(modelname, test_taus, model_save_folder=None, ape=True, sleep=False, n_repeats_case = None, checkpoint='', timestamp = None, device='cpu', model_folder = None):

    config, task_options, ape_nn_options = load_config_files(modelname)

    if n_repeats_case is not None:
        config.n_repeats_case = n_repeats_case

    rews_taus_ape = []
    counters_peeks_taus_ape = []
    control_errs_taus_ape = []
    counters_sleeps_taus_ape = []

    traj_actionss = []
    traj_logitss = []
    traj_controlss = []
    traj_pss = []
    traj_control_errss = []

    for test_tau in test_taus:

        tau_task_options = Config(copy.deepcopy(task_options.__dict__))
        tau_task_options.starting_taus = {'peek': 0, 'take': test_tau}

        (ape_actionss, ape_logitss, _, _, ape_controlss, ape_pss, ape_control_errss), (rews_ape, _, counter_peeks_ape, counter_sleeps_taus_ape, _, _, control_errs_ape) = test(config, ape_nn_options, tau_task_options, checkpoint=checkpoint, device=device, model_folder=model_folder)
        rews_taus_ape.append(rews_ape)
        counters_peeks_taus_ape.append(counter_peeks_ape)
        control_errs_taus_ape.append(control_errs_ape)
        counters_sleeps_taus_ape.append(counter_sleeps_taus_ape)

        traj_actionss.append(ape_actionss)
        traj_logitss.append(ape_logitss)
        traj_controlss.append(ape_controlss)
        traj_pss.append(ape_pss)
        traj_control_errss.append(ape_control_errss)

    if model_save_folder is not None:
        #model_save_folder = os.path.join(save_base_data_folder, str(modelname))

        if timestamp is None:
            timestamp = get_timestamp()
            
        os.makedirs(model_save_folder, exist_ok=True)

        pickle.dump(rews_taus_ape, open(os.path.join(model_save_folder, '%s_rewss_taus.pkl' %timestamp), 'wb'))
        pickle.dump(counters_peeks_taus_ape, open(os.path.join(model_save_folder, '%s_counters_peeks_taus.pkl' %timestamp), 'wb'))

        pickle.dump(traj_actionss, open(os.path.join(model_save_folder, '%s_traj_actionss_taus.pkl' %timestamp), 'wb'))
        pickle.dump(traj_logitss, open(os.path.join(model_save_folder, '%s_traj_logitss_taus.pkl' %timestamp), 'wb'))
        pickle.dump(traj_pss, open(os.path.join(model_save_folder, '%s_traj_pss_taus.pkl' %timestamp), 'wb'))

        if ape:
            pickle.dump(control_errs_taus_ape, open(os.path.join(model_save_folder, '%s_control_errs_taus_ape.pkl' %timestamp), 'wb'))    
            pickle.dump(traj_controlss, open(os.path.join(model_save_folder, '%s_traj_controlss_taus.pkl' %timestamp), 'wb'))
            pickle.dump(traj_control_errss, open(os.path.join(model_save_folder, '%s_traj_control_errss_taus.pkl' %timestamp), 'wb'))

        if sleep:
            pickle.dump(counters_sleeps_taus_ape, open(os.path.join(model_save_folder, '%s_sleep_errs_taus_ape.pkl' %timestamp), 'wb'))    

    return rews_taus_ape, counters_peeks_taus_ape, control_errs_taus_ape, counters_sleeps_taus_ape

# %% TEST

if __name__ == '__main__':
    mp.set_start_method('spawn')
    processes = []

    counter = 0
    n_simultaneous = 5

    ### EVALUATE LEARNING CURVES
    # for ape_model in ape_models:
    #     ape_model = str(ape_model)
    #     p = mp.Process(target=sample_model_trajectory, args=(ape_model, test_taus, os.path.join(save_results_base, ape_model), True, False, n_repeats_case, '', timestamp, device, os.path.join(model_folder, ape_model)))
    #     p.start()
    #     processes.append(p)

    #     counter += 1

    #     if counter % n_simultaneous == 0:
    #         for p in processes:
    #             p.join()

    # for control_model in control_models:
    #     control_model = str(control_model)
    #     p = mp.Process(target=sample_model_trajectory, args=(control_model, test_taus, os.path.join(save_results_base, control_model), False, False, n_repeats_case, '', timestamp, device, os.path.join(model_folder, control_model)))
    #     p.start()
    #     processes.append(p)
    #     counter += 1

    #     if counter % n_simultaneous == 0:
    #         for p in processes:
    #             p.join()

    for model in models:
        model = str(model)
        p = mp.Process(target=sample_model_trajectory, args=(model, test_taus, os.path.join(save_results_base, model), True, False, n_repeats_case, '', timestamp, device, os.path.join(model_folder, model)))
        p.start()
        processes.append(p)

        counter += 1

        if counter % n_simultaneous == 0:
            for p in processes:
                p.join()
