# Kai Sandbrink
# 2023-02-01
# This script computes the learning curves for specified test models

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
#models = control_models + ape_models + efficacy_at_input_models + human_control_models + extra_node_models
#models = efficacy_at_input_models
models = baseline_models
#models = extra_node_models

# %% PARAMETERS

n_checkpoints = 100
n_repeats_case = 1000
test_taus = np.arange(0,1.01,0.125)
model_folder = 'models'

#save_results_folder = os.path.join('results', 'pepe', '%s_eval_learning_curves_%srepeats' %(get_timestamp(), n_repeats_case))
save_results_base = os.path.join('data', 'eval', 'pepe', )

# %% CREATE TEST FUNCTIONS

def test_model_taus(modelname, test_taus, config=None, task_options=None, nn_options=None, n_repeats_case=None, checkpoint='', device='cpu'):
    
    if config is None or task_options is None or nn_options is None:
        config, task_options, nn_options = load_config_files(modelname)
    
    if n_repeats_case is not None:
        config.n_repeats_case = n_repeats_case

    tau_rews = []
    tau_returns_losses = []
    tau_ape_losses = []
    tau_ape_mses = []

    for test_tau in test_taus:

        task_options.starting_taus = {'peek': 0, 'take': test_tau}

        _, (rews_ape, _, _, _, returns_loss_ape, ape_loss_ape, ape_mse_ape) = test(config, nn_options, task_options, device, checkpoint=checkpoint, model_folder = os.path.join('models', str(modelname)))

        tau_rews.append(rews_ape)
        tau_returns_losses.append(returns_loss_ape)
        tau_ape_losses.append(ape_loss_ape)
        tau_ape_mses.append(ape_mse_ape)

    return tau_rews, tau_returns_losses, tau_ape_losses, tau_ape_mses

def test_model(modelname, model_save_folder, n_repeats_case, timestamp, ape=True, device="cpu"):

    config, task_options, ape_nn_options = load_config_files(modelname)
    config.n_repeats_case = n_repeats_case

    ape_test_returns_losses_checkpoints = []
    ape_test_rews_checkpoints = []

    if ape:
        ape_test_apes_mse_checkpoints = []
        ape_test_apes_losses_checkpoints = []

    for checkpoint in [str(int(i/n_checkpoints*100)) for i in range(n_checkpoints)] + ['']:

        tau_rews, tau_returns_losses, tau_ape_losses, tau_ape_mses = test_model_taus(modelname, test_taus, config=config, task_options=task_options, nn_options=ape_nn_options, n_repeats_case=n_repeats_case, checkpoint=checkpoint, device=device)

        ape_test_rews_checkpoints.append(np.array(tau_rews))
        ape_test_returns_losses_checkpoints.append(np.array(tau_returns_losses))

        if ape:
            ape_test_apes_losses_checkpoints.append(np.array(tau_ape_losses))
            ape_test_apes_mse_checkpoints.append(np.array(tau_ape_mses))

    test_episodes = [config.n_episodes * i / n_checkpoints for i in range(n_checkpoints)] + [config.n_episodes]

    os.makedirs(model_save_folder, exist_ok=True)

    pickle.dump(ape_test_returns_losses_checkpoints, open(os.path.join(model_save_folder, '%s_test_learning_curve_returns_losses.pkl' %timestamp), 'wb'))
    pickle.dump(test_episodes, open(os.path.join(model_save_folder, '%s_test_learning_curve_episodes.pkl' %timestamp), 'wb'))
    pickle.dump(ape_test_rews_checkpoints, open(os.path.join(model_save_folder, '%s_test_learning_curve_rews.pkl' %timestamp), 'wb'))

    if ape:
        pickle.dump(ape_test_apes_losses_checkpoints, open(os.path.join(model_save_folder, '%s_test_learning_curve_apes_losses.pkl' %timestamp), 'wb'))
        pickle.dump(ape_test_apes_mse_checkpoints, open(os.path.join(model_save_folder, '%s_test_learning_curve_apes_mses.pkl' %timestamp), 'wb'))

# %% TEST

if __name__ == '__main__':
    
    # %% INITIALIZATIONS

    ## TORCH
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using {device} device")

    ## OTHER
    timestamp = get_timestamp()

    mp.set_start_method('spawn')
    processes = []

    for i, model in enumerate(models):
        print(model)
        model = str(model)
        p = mp.Process(target=test_model, args=(model, os.path.join(save_results_base, model), n_repeats_case, timestamp, True, device))
        p.start()
        processes.append(p)

        if (i+1) % 10 == 0 or i == len(models)-1:
            for p in processes:
                p.join()
            processes = []

    ### EVALUATE LEARNING CURVES
    # for ape_model in ape_models:
    #     ape_model = str(ape_model)
    #     p = mp.Process(target=test_model, args=(ape_model, os.path.join(save_results_base, ape_model), n_repeats_case, timestamp, True, device))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    # for control_model in control_models:
    #     control_model = str(control_model)
    #     p = mp.Process(target=test_model, args=(control_model, os.path.join(save_results_base, control_model), n_repeats_case, timestamp, False, device))
    #     p.start()
    #     processes.append(p)
    
    # for p in processes:
    #     p.join()