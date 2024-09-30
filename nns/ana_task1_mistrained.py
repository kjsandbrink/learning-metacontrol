# Kai Sandbrink
# 2023-02-14
# This script analyzes the mistrained model

# %% LIBRARY IMPORTS
import torch
import numpy as np
import os, copy
import pickle
import matplotlib.pyplot as plt

from utils import Config, plot_learning_curve, get_timestamp
from ObserveBetEfficacyTask import ObserveBetEfficacyTask
from PeekTakeTorchRNN import PeekTakeTorchPerturbedAPERNN
from test_case import test

from utils import format_axis, get_timestamp

from test_analyses import policy_barplot, within_episode_efficacy_lineplot, plot_comparison_curves_several_runs, frac_takes_lineplot, frac_correct_takes_lineplot, ape_accuracy_lineplot, plot_behavior_mistrained

# %% SPECIFY MODELS

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

# %% PARAMETERS

#modelname = '20230123122111_mistrained_tau100'
modelname = str(pepe_models[0])
target_tau = 0
timestamp_traj = '20230525015754'
timestamp_model = '20230325003139'
timestamp_original_traj = '20230522205803'
n_models = len(pepe_models)

mistrained_model_folder = os.path.join('data', 'mistrained_models', modelname, timestamp_model + '_mistrained_tau' + str(target_tau))
original_model_folder = os.path.join('models', modelname)
mistrained_traj_folder = os.path.join('data', 'mistrained_trajectories', modelname, 'mistrained_tau%d' %(target_tau*100))
mistrained_traj_base_folder = os.path.join('data', 'mistrained_trajectories')

eval_base_folder = os.path.join('data', 'eval', 'pepe')
analysis_folder = os.path.join('analysis', 'explore-exploit', 'mistrained')

# %% INITIALIZATOINS

#analysis_folder = os.path.join('analysis', modelname)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# %% READ IN DATA COMPARING MULTIPLE MODELS


test_taus = np.arange(0, 1.01,0.25)

models_mis_taus_control_errs = []
models_mis_taus_counters_peeks = []
models_mis_taus_rewss = []

for model in pepe_models:

    mis_taus_control_errs = []
    mis_taus_counters_peeks = []
    mis_taus_rewss = []

    for mistrained_tau in test_taus:

        mistrained_traj_folder = os.path.join(mistrained_traj_base_folder, str(model), 'mistrained_tau%d' %(mistrained_tau*100))

        control_errs_taus_ape = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_control_errs_taus_ape.pkl'), 'rb'))
        counters_peeks_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_counters_peeks_taus.pkl'), 'rb'))
        rewss_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_rewss_taus.pkl'), 'rb'))

        mis_taus_control_errs.append(control_errs_taus_ape)
        mis_taus_counters_peeks.append(counters_peeks_taus)
        mis_taus_rewss.append(rewss_taus)

    models_mis_taus_control_errs.append(mis_taus_control_errs)
    models_mis_taus_counters_peeks.append(mis_taus_counters_peeks)
    models_mis_taus_rewss.append(mis_taus_rewss)

models_mis_taus_control_errs = np.array(models_mis_taus_control_errs)
models_mis_taus_counters_peeks = np.array(models_mis_taus_counters_peeks)
models_mis_taus_rewss = np.array(models_mis_taus_rewss)

# %% READ IN TRAJECTORIES CORRESPONDING TO TRAINED TAU

models_orig_taus_control_errs = []
models_orig_taus_counters_peeks = []
models_orig_taus_rewss = []

for model in pepe_models:

    orig_traj_folder = os.path.join(eval_base_folder, str(model))

    control_errs_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_control_errs_taus_ape.pkl'), 'rb'))
    counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_counters_peeks_taus.pkl'), 'rb'))
    rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_rewss_taus.pkl'), 'rb'))

    models_orig_taus_control_errs.append(control_errs_taus_ape)
    models_orig_taus_counters_peeks.append(counters_peeks_taus)
    models_orig_taus_rewss.append(rewss_taus)

models_orig_taus_control_errs = np.array(models_orig_taus_control_errs)
models_orig_taus_counters_peeks = np.array(models_orig_taus_counters_peeks)
models_orig_taus_rewss = np.array(models_orig_taus_rewss)

# %% PLOT COUNTERS PEEKS ALL TAUS

import matplotlib as mpl

fig = plot_behavior_mistrained(test_taus, np.flip(models_mis_taus_counters_peeks, axis=(0,2)), np.arange(0,1.01,0.125), np.flip(models_orig_taus_counters_peeks, axis=(1)), axis_xlabel='Controllability', axis_ylabel='Number of Observes', cmap=mpl.cm.Blues, figsize=(10.4952, 4.9359), font_size_multiplier=1.4,)

os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_mis_taus_counters_peeks.png'))
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_mis_taus_counters_peeks.svg'))

# %% STATS

model_means_mis = models_mis_taus_rewss.mean(axis=(0,2))

print('max mis', np.max(model_means_mis))
 
# %% ANALYZE ABLATED

## PARAMETERS

timestamp_traj = '20230523222553'
target_taus = np.arange(-3, 4.01, 1)
mistrained_traj_base_folder = os.path.join('data','ablated','pepe')

models_mis_taus_control_errs = []
models_mis_taus_counters_peeks = []
models_mis_taus_rewss = []

for model in pepe_models:

    mis_taus_control_errs = []
    mis_taus_counters_peeks = []
    mis_taus_rewss = []

    for mistrained_tau in target_taus:

        mistrained_traj_folder = os.path.join(mistrained_traj_base_folder, str(model), 'ablated_tau%d' %mistrained_tau)

        control_errs_taus_ape = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_ablated_control_errs_taus_ape.pkl'), 'rb'))
        counters_peeks_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_ablated_counters_peeks_taus.pkl'), 'rb'))
        rewss_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_rewss_taus.pkl'), 'rb'))

        mis_taus_control_errs.append(control_errs_taus_ape)
        mis_taus_counters_peeks.append(counters_peeks_taus)
        mis_taus_rewss.append(rewss_taus)

    models_mis_taus_control_errs.append(mis_taus_control_errs)
    models_mis_taus_counters_peeks.append(mis_taus_counters_peeks)
    models_mis_taus_rewss.append(mis_taus_rewss)

models_mis_taus_control_errs = np.array(models_mis_taus_control_errs)[:,1:-1,:,]
models_mis_taus_counters_peeks = np.array(models_mis_taus_counters_peeks)[:,1:-1,:,]
models_mis_taus_rewss = np.array(models_mis_taus_rewss)[:,1:-1,:]

target_taus = np.arange(-2, 3.01, 1)

# test_taus = pickle.load(open(os.path.join(mistrained_traj_base_folder, str(pepe_models[0]), timestamp_traj + '_test_taus.pkl'), 'rb'))

# %% READ IN TRAJECTORIES CORRESPONDING TO TRAINED TAU

models_orig_taus_control_errs = []
models_orig_taus_counters_peeks = []
models_orig_taus_rewss = []

for model in pepe_models:

    orig_traj_folder = os.path.join(eval_base_folder, str(model))

    control_errs_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_control_errs_taus_ape.pkl'), 'rb'))
    counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_counters_peeks_taus.pkl'), 'rb'))
    rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_rewss_taus.pkl'), 'rb'))

    models_orig_taus_control_errs.append(control_errs_taus_ape)
    models_orig_taus_counters_peeks.append(counters_peeks_taus)
    models_orig_taus_rewss.append(rewss_taus)

models_orig_taus_control_errs = np.array(models_orig_taus_control_errs)
models_orig_taus_counters_peeks = np.array(models_orig_taus_counters_peeks)
models_orig_taus_rewss = np.array(models_orig_taus_rewss)

# %% PLOT COUNTERS PEEKS ALL TAUS
test_taus = np.arange(0, 1.01, 0.125)
fig = plot_behavior_mistrained(test_taus, np.flip(models_mis_taus_counters_peeks, axis=(0,2)), np.arange(0,1.01,0.125), np.flip(models_orig_taus_counters_peeks, axis=(1)), axis_xlabel='Efficacy', axis_ylabel='Number of Observes per Episode', target_taus = target_taus)

os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_ablated_taus_counters_peeks.png'))
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_ablated_taus_counters_peeks.svg'))

# %% STATS

model_means_mis = models_mis_taus_rewss.mean(axis=(0,2))

print('max mis', np.max(model_means_mis))

# %% ANALYZE PERTURBED

## PARAMETERS

timestamp_traj = '20230523220600'
mistrained_traj_base_folder = os.path.join('data','perturbed','pepe')
target_taus = np.arange(-3, 4.01, 1)

models_mis_taus_control_errs = []
models_mis_taus_counters_peeks = []
models_mis_taus_rewss = []

for model in pepe_models:

    mis_taus_control_errs = []
    mis_taus_counters_peeks = []
    mis_taus_rewss = []

    for mistrained_tau in target_taus:

        mistrained_traj_folder = os.path.join(mistrained_traj_base_folder, str(model), 'perturbed_tau%d' %mistrained_tau)

        control_errs_taus_ape = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_perturbed_control_errs_taus_ape.pkl'), 'rb'))
        counters_peeks_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_perturbed_counters_peeks_taus.pkl'), 'rb'))
        rewss_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_rewss_taus.pkl'), 'rb'))

        mis_taus_control_errs.append(control_errs_taus_ape)
        mis_taus_counters_peeks.append(counters_peeks_taus)
        mis_taus_rewss.append(rewss_taus)

    models_mis_taus_control_errs.append(mis_taus_control_errs)
    models_mis_taus_counters_peeks.append(mis_taus_counters_peeks)
    models_mis_taus_rewss.append(mis_taus_rewss)

models_mis_taus_control_errs = np.array(models_mis_taus_control_errs)
models_mis_taus_counters_peeks = np.array(models_mis_taus_counters_peeks)
models_mis_taus_rewss = np.array(models_mis_taus_rewss)

target_taus = np.arange(-2, 3.01, 1)

# %% READ IN TRAJECTORIES CORRESPONDING TO TRAINED TAU

models_orig_taus_control_errs = []
models_orig_taus_counters_peeks = []
models_orig_taus_rewss = []

for model in pepe_models:

    orig_traj_folder = os.path.join(eval_base_folder, str(model))

    control_errs_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_control_errs_taus_ape.pkl'), 'rb'))
    counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_counters_peeks_taus.pkl'), 'rb'))
    rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_rewss_taus.pkl'), 'rb'))

    models_orig_taus_control_errs.append(control_errs_taus_ape)
    models_orig_taus_counters_peeks.append(counters_peeks_taus)
    models_orig_taus_rewss.append(rewss_taus)

models_orig_taus_control_errs = np.array(models_orig_taus_control_errs)
models_orig_taus_counters_peeks = np.array(models_orig_taus_counters_peeks)
models_orig_taus_rewss = np.array(models_orig_taus_rewss)

# %% PLOT COUNTERS PEEKS ALL TAUS

fig = plot_behavior_mistrained(test_taus, np.flip(models_mis_taus_counters_peeks, axis=(0,2)), np.arange(0,1.01,0.125), np.flip(models_orig_taus_counters_peeks, axis=(1)), axis_xlabel='Efficacy', axis_ylabel='Number of Observes per Episode', target_taus = target_taus)

os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_perturbed_taus_counters_peeks.png'))
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_perturbed_taus_counters_peeks.svg'))

# %% STATS

model_means_mis = models_mis_taus_rewss.mean(axis=(0,2))

print('max mis', np.max(model_means_mis))

# %%
