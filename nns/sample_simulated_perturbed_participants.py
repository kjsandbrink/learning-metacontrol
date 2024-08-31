# Kai Sandbrink
# 2023-10-07
# This script is used to simulate data corresponding to perturbed participants

# %% LIBRARY IMPORT

import numpy as np
import pandas as pd
import pickle
import os, copy
from tqdm import tqdm

from utils import get_timestamp, Config
from utils_project import load_config_files
from test_case import perturbed_test

import multiprocessing as mp

# %% PARAMETERS

#### PEPE MODELS

# from settings_anal import pepe_human_ape_models as ape_models
# task = 'pepe'

## LEVC
from nns.settings_ana import levc_human_ape_models as ape_models

task = 'levc'

# %% PARAMETERS

#checkpoint = ''
n_repeats_case = 100
n_participants = 150
model_folder = 'models'

save_results_base = os.path.join('data', 'sim_perturbed_participants', task, )

# %% INITIALIZATIONS

## TORCH
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

## OTHER
timestamp = get_timestamp()
perturbations_timestamp = '20240220100914'

sleep = task == 'levc'

n_effs = 9

mag_perturbation = 0.2
original_mag_perturbation = 1
bias_perturbation = -3
# mag_perturbation = 0.4

# bias_perturbation = -0.6


### MAG PERTURBATION VALUES USED IN RESAMPLING FOR 2023 LEVC RUNS
# mag_perturbation = 0.01
# original_mag_perturbation = 3
## ORIGINAL AMOUNT - USED UP TO AND INCLUDING FIRST RUN 12/17 IS 3

# %% SPECIFY PERTURBED DEVIATION CHARACTERISTICS

if perturbations_timestamp is None or task == 'pepe':
    sim_participant_perturbations = np.random.randn(n_participants,)*mag_perturbation
    nostruc_participant_perturbations = np.zeros((n_participants,))
    random_participant_perturbations = np.random.randn(n_participants, n_effs)

#sim_participant_perturbations = np.zeros((150,))
else:
    with open(os.path.join('data', 'sim_perturbed_participants', 'pepe', 'sim', 'mag%d'%(original_mag_perturbation*100), '%s_simulated_participant_perturbations_sim_mag.pkl' %perturbations_timestamp), 'rb') as f:
        sim_participant_perturbations = pickle.load(f) / (original_mag_perturbation / mag_perturbation) + bias_perturbation
    with open(os.path.join('data', 'sim_perturbed_participants', 'pepe', 'nostruc', 'mag%d'%(original_mag_perturbation*100), '%s_simulated_participant_perturbations_nostruc_mag.pkl' %perturbations_timestamp), 'rb') as f:
        nostruc_participant_perturbations = pickle.load(f) / (original_mag_perturbation / mag_perturbation) + bias_perturbation
    with open(os.path.join('data', 'sim_perturbed_participants', 'pepe', 'random', 'mag%d'%(original_mag_perturbation*100), '%s_simulated_participant_perturbations_random_mag.pkl' %perturbations_timestamp), 'rb') as f:
        random_participant_perturbations = pickle.load(f) / (original_mag_perturbation / mag_perturbation)  + bias_perturbation
        np.random.shuffle(random_participant_perturbations)

    print('read in from file, with resize paramter %f' %(mag_perturbation / original_mag_perturbation))

# %% SAMPLE PERTURBED TRAJECTORIES

test_taus = np.arange(0,1.01,0.125)

def simulate_model_perturbed_participants(modelname, test_taus, perturbations, model_save_folder=None, ape=True, sleep=False, n_repeats_case = None, checkpoint='', timestamp = None, device='cpu', model_folder = None):
    ''' Simulates model participants with corresponding efficacy perturbations and saves results to file

    Parameters
    ----------
    modelname : str
        Name of model to be simulated
    test_taus : np.array of shape (n_effs)
        Array of taus to be tested
    perturbations : np.array of shape (n_participants,) or (n_participants, n_effs)
        Array of perturbations to be applied to tau parameters; if 2D, one slice is applied per efficacy parameter in test_taus
    model_save_folder : str, optional
        Path to folder in which to save results
    ape : bool, optional
        Whether to simulate ape model
    sleep : bool, optional
        Whether sleep is included as an action
    n_repeats_case : int, optional
        Number of repeats per case
    checkpoint : str, optional
        Path to checkpoint
    timestamp : str, optional
        Timestamp to be used for saving results
    device : str, optional
        Device to be used for simulation
    model_folder : str, optional
        Path to folder in which model is saved

    Returns
    -------
    rewss_taus_ape : np.array of shape (n_effs, n_participants, n_repeats_case)
        Array of rewards for each participant, efficacy parameter, and repeat
    counters_peeks_taus_ape : np.array of shape (n_effs, n_participants, n_repeats_case)
        Array of peek counters for each participant, efficacy parameter, and repeat
    control_errs_taus_ape : np.array of shape (n_effs, n_participants, n_repeats_case)
        Array of control errors for each participant, efficacy parameter, and repeat
    counters_sleeps_taus_ape : np.array of shape (n_effs, n_participants, n_repeats_case)
        Array of sleep counters for each participant, efficacy parameter, and repeat
    '''

    config, task_options, ape_nn_options = load_config_files(modelname)

    if n_repeats_case is not None:
        config.n_repeats_case = n_repeats_case

    rews_taus_ape = []
    counters_peeks_taus_ape = []
    control_errs_taus_ape = []
    counters_sleeps_taus_ape = []
    control_ests_taus_ape = []

    for i, test_tau in enumerate(test_taus):

        tau_task_options = Config(copy.deepcopy(task_options.__dict__))
        tau_task_options.starting_taus = {'peek': 0, 'take': test_tau}

        rews_taus_ape.append([])
        counters_peeks_taus_ape.append([])
        control_errs_taus_ape.append([])
        counters_sleeps_taus_ape.append([])
        control_ests_taus_ape.append([])
        
        for perturbation in tqdm(perturbations):

            if len(perturbations.shape) == 2:
                pert = perturbation[i]
            else:
                pert = perturbation

            target_tau = test_tau + pert
            (_, _, _, _, controls, _, _), (rews_ape, _, counter_peeks_ape, counter_sleeps_taus_ape, _, _, control_errs_ape) = perturbed_test(config, ape_nn_options, tau_task_options, n_repeats_case = n_repeats_case, device=device, model_folder=model_folder, target_tau=target_tau)
            rews_taus_ape[-1].append(rews_ape)
            counters_peeks_taus_ape[-1].append(counter_peeks_ape)
            control_errs_taus_ape[-1].append(control_errs_ape)
            counters_sleeps_taus_ape[-1].append(counter_sleeps_taus_ape)
            control_ests_taus_ape.append(controls)

    rews_taus_ape = np.array(rews_taus_ape)
    counters_peeks_taus_ape = np.array(counters_peeks_taus_ape)
    control_errs_taus_ape = np.array(control_errs_taus_ape)
    counters_sleeps_taus_ape = np.array(counters_sleeps_taus_ape)

    if model_save_folder is not None:
        #model_save_folder = os.path.join(save_base_data_folder, str(modelname))
        #model_save_folder = os.path.join(model_save_folder, 'perturbed_tau%d' %target_tau)

        if timestamp is None:
            timestamp = get_timestamp()
            
        os.makedirs(model_save_folder, exist_ok=True)

        pickle.dump(rews_taus_ape, open(os.path.join(model_save_folder, '%s_rewss_taus.pkl' %timestamp), 'wb'))
        pickle.dump(counters_peeks_taus_ape, open(os.path.join(model_save_folder, '%s_perturbed_counters_peeks_taus.pkl' %timestamp), 'wb'))

        if ape:
            pickle.dump(control_errs_taus_ape, open(os.path.join(model_save_folder, '%s_perturbed_control_errs_taus_ape.pkl' %timestamp), 'wb'))    
            pickle.dump(control_ests_taus_ape, open(os.path.join(model_save_folder, '%s_perturbed_control_ests_taus_ape.pkl' %timestamp), 'wb'))

        if sleep:
            pickle.dump(counters_sleeps_taus_ape, open(os.path.join(model_save_folder, '%s_perturbed_sleep_errs_taus_ape.pkl' %timestamp), 'wb'))    

        pickle.dump(test_taus, open(os.path.join(model_save_folder, '%s_perturbed_test_taus.pkl' %timestamp), 'wb'))

    return rews_taus_ape, counters_peeks_taus_ape, control_errs_taus_ape, counters_sleeps_taus_ape

# %% MAIN

if __name__ == '__main__':

    mp.set_start_method('spawn')
    processes = []

    for perturbations, perturbation_name in zip([sim_participant_perturbations, nostruc_participant_perturbations, random_participant_perturbations], ['sim', 'nostruc', 'random']):
    #for perturbations, perturbation_name in zip([sim_participant_perturbations], ['sim']): #, 'nostruc', 'random
        print("Running for %s" % perturbation_name)
        print("Perturbations: %s" % perturbations)

        save_results_folder = os.path.join(save_results_base, perturbation_name, 'mag%dbias%d' %(mag_perturbation*100, bias_perturbation*100))
        os.makedirs(save_results_folder, exist_ok=True)
        with open(os.path.join(save_results_folder, '%s_simulated_participant_perturbations_%s_mag.pkl' %(timestamp, perturbation_name)), 'wb') as f:
            pickle.dump(perturbations, f)

        ### EVALUATE LEARNING CURVES
        for ape_model in ape_models:
            ape_model = str(ape_model)
            print("Running for model %s" % ape_model)
            p = mp.Process(target=simulate_model_perturbed_participants, args=(ape_model, test_taus, perturbations, os.path.join(save_results_folder, ape_model), True, sleep, n_repeats_case, '', timestamp, device, os.path.join(model_folder, ape_model)))
            #p = mp.Process(target=simulate_model_perturbed_participants, args=(ape_model, test_taus, sim_participant_perturbations, None, True, False, n_repeats_case, '', timestamp, device, os.path.join(model_folder, ape_model)))
            p.start()
            processes.append(p)

            #rews_taus_ape, counters_peeks_taus_ape, control_errs_taus_ape, counters_sleeps_taus_ape = simulate_model_perturbed_participants(ape_model, test_taus, sim_participant_perturbations, os.path.join(save_results_base, ape_model), True, False, n_repeats_case, '', timestamp, device, os.path.join(model_folder, ape_model))
            #simulate_model_perturbed_participants(ape_model, test_taus, sim_participant_perturbations, os.path.join(save_results_folder, ape_model), True, sleep, n_repeats_case, '', timestamp, device, os.path.join(model_folder, ape_model))

        for p in processes:
            p.join()

# %%
