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

### BIAS 0.5 VOL 0.1
### 5/23

ape_models = [
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

#from settings_anal import levc_human_ape_models as ape_models

# %% PARAMETERS

#modelname = '20230123122111_mistrained_tau100'
modelname = str(ape_models[0])
target_tau = 0
timestamp_traj = '20230525143707'
timestamp_model = '20230325003139'
timestamp_original_traj = '20230522212935'
#timestamp_original_traj = '20231015162557'
#timestamp_original_traj = '20240311133201'

n_models = len(ape_models)

mistrained_model_folder = os.path.join('data', 'mistrained_models', modelname, timestamp_model + '_mistrained_tau' + str(target_tau))
original_model_folder = os.path.join('models', modelname)
mistrained_traj_folder = os.path.join('data', 'mistrained_trajectories', modelname, 'mistrained_tau%d' %(target_tau*100))
mistrained_traj_base_folder = os.path.join('data', 'mistrained_trajectories')

eval_base_folder = os.path.join('data', 'eval', 'levc')
analysis_folder = os.path.join('analysis', 'levc', 'mistrained')

# %% INITIALIZATOINS

#analysis_folder = os.path.join('analysis', modelname)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# # %% LOAD MODEL CONFIG FILES

# config = Config({})
# config.load_config_file(os.path.join(original_model_folder, 'config.yaml'))

# task_options = Config({})
# task_options.load_config_file(os.path.join(original_model_folder, 'task_options.yaml'))

# nn_options = Config({})
# nn_options.load_config_file(os.path.join(original_model_folder, 'nn_options.yaml'))

# # %% INITIALIZE ENV AND MODEL

# env = ObserveBetEfficacyTask(**task_options)
# model = PeekTakeTorchPerturbedAPERNN(env.actions, env.encoding_size, **nn_options)

# # %% LOAD MODEL

# model.load_state_dict(torch.load(os.path.join(mistrained_model_folder, 'model.pt')))

# %% LEARNING CURVE - LOAD DATA

rewardss = np.load(os.path.join(mistrained_model_folder, 'logged_returns.npy'))

episodes = np.linspace(0,100000,1001)

# %% PLOT

fig = plot_learning_curve(rewardss, smoothing_window=1, several_runs=False)

os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, 'mistraining_curve.png'))

# %% ANALYZE AMOUNT OF PEEKS ACROSS TRIALS

test_taus = np.arange(0,1.01,0.25)
tau_task_options = Config(copy.deepcopy(task_options.__dict__))
n_repeats_case = 100

ape_actionss_taus = []
ape_logitss_taus = []
ape_controlss_taus = []
ape_pss_taus = []
ape_control_errs_taus = []

for test_tau in test_taus:

    tau_task_options.starting_taus = {'peek': 0, 'take': test_tau}

    (ape_actionss, ape_logitss, _, _, ape_controlss, ape_pss, ape_control_errss), _ = test(config, nn_options, tau_task_options, device, n_repeats_case=n_repeats_case, model_folder = mistrained_model_folder)

    ape_actionss_taus.append(ape_actionss)
    ape_logitss_taus.append(ape_logitss)
    ape_controlss_taus.append(ape_controlss)
    ape_pss_taus.append(ape_pss)
    ape_control_errs_taus.append(ape_control_errss)

# %% PANEL POLICY SINGLE MODEL FRAC TAKES - PLOT

os.makedirs(analysis_folder ,exist_ok = True)
fig = frac_takes_lineplot(test_taus, ape_logitss_taus)
fig.savefig(os.path.join(analysis_folder, '%s_frac_takes_lineplot.png' %get_timestamp()))

# %% PANEL POLICY SINGLE MODEL PERC CORRECT TAKES - PLOT

os.makedirs(analysis_folder ,exist_ok = True)
fig = frac_correct_takes_lineplot(test_taus, ape_logitss_taus, ape_pss_taus)
fig.savefig(os.path.join(analysis_folder, '%s_corr_takes_lineplot.png' %get_timestamp()))

# %% PANEL POLICY SINGLE MODEL ERROR IN APEs - PLOT

os.makedirs(analysis_folder ,exist_ok = True)
fig = ape_accuracy_lineplot(test_taus, ape_control_errs_taus)
fig.savefig(os.path.join(analysis_folder, '%s_ape_accuracy_lineplot.png' %get_timestamp()))

# %% READ IN TRAJECTORIES CORRESPONDING TO MISTRAINED_TAU

control_errs_taus_ape = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_control_errs_taus_ape.pkl'), 'rb'))
counters_peeks_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_counters_peeks_taus.pkl'), 'rb'))
rewss_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_rewss_taus.pkl'), 'rb'))

# %% READ IN DATA COMPARING MULTIPLE MODELS

test_taus = np.arange(0, 1.01,0.25)

models_mis_taus_control_errs = []
models_mis_taus_counters_peeks = []
models_mis_taus_rewss = []
models_mis_taus_sleepss = []

for model in ape_models:

    mis_taus_control_errs = []
    mis_taus_counters_peeks = []
    mis_taus_rewss = []
    mis_taus_sleepss = []

    for mistrained_tau in test_taus:

        mistrained_traj_folder = os.path.join(mistrained_traj_base_folder, str(model), 'mistrained_tau%d' %(mistrained_tau*100))

        control_errs_taus_ape = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_control_errs_taus_ape.pkl'), 'rb'))
        counters_peeks_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_counters_peeks_taus.pkl'), 'rb'))
        rewss_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_rewss_taus.pkl'), 'rb'))
        sleepss_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_sleep_errs_taus_ape.pkl'), 'rb'))

        mis_taus_control_errs.append(control_errs_taus_ape)
        mis_taus_counters_peeks.append(counters_peeks_taus)
        mis_taus_rewss.append(rewss_taus)
        mis_taus_sleepss.append(sleepss_taus)

    models_mis_taus_control_errs.append(mis_taus_control_errs)
    models_mis_taus_counters_peeks.append(mis_taus_counters_peeks)
    models_mis_taus_rewss.append(mis_taus_rewss)
    models_mis_taus_sleepss.append(mis_taus_sleepss)

models_mis_taus_control_errs = np.array(models_mis_taus_control_errs)
models_mis_taus_counters_peeks = np.array(models_mis_taus_counters_peeks)
models_mis_taus_rewss = np.array(models_mis_taus_rewss)
models_mis_taus_sleepss = np.array(models_mis_taus_sleepss)

# %% READ IN TRAJECTORIES CORRESPONDING TO TRAINED TAU

models_orig_taus_control_errs = []
models_orig_taus_counters_peeks = []
models_orig_taus_rewss = []
models_orig_taus_sleepss = []

for model in ape_models:

    orig_traj_folder = os.path.join(eval_base_folder, str(model))

    control_errs_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_control_errs_taus_ape.pkl'), 'rb'))
    counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_counters_peeks_taus.pkl'), 'rb'))
    rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_rewss_taus.pkl'), 'rb'))
    sleepss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_sleep_errs_taus_ape.pkl'), 'rb'))

    models_orig_taus_control_errs.append(control_errs_taus_ape)
    models_orig_taus_counters_peeks.append(counters_peeks_taus)
    models_orig_taus_rewss.append(rewss_taus)
    models_orig_taus_sleepss.append(sleepss_taus)

models_orig_taus_control_errs = np.array(models_orig_taus_control_errs)
models_orig_taus_counters_peeks = np.array(models_orig_taus_counters_peeks)
models_orig_taus_rewss = np.array(models_orig_taus_rewss)
models_orig_taus_sleepss = np.array(models_orig_taus_sleepss)

# %% PLOT COUNTERS PEEKS ALL TAUS

import matplotlib as mpl

fig = plot_behavior_mistrained(test_taus, np.flip(models_mis_taus_sleepss, axis=(0,2)), np.arange(0,1.01,0.125), np.flip(models_orig_taus_sleepss, axis=(1)), axis_xlabel='Controllability', axis_ylabel='Number of Sleeps', cmap=mpl.cm.Blues, figsize=(10.4952, 4.9359), font_size_multiplier=1.4,)

os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_models_mis_taus_counters_sleeps.png'))
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_models_mis_taus_counters_sleeps.svg'))

# %% STATS

model_means_mis = models_mis_taus_rewss.mean(axis=(0,2))

print('max mis', np.max(model_means_mis))

# %% ANALYZE ABLATED

## PARAMETERS

timestamp_traj = '20230523230755'
target_taus = np.arange(-3, 4.01, 1)
mistrained_traj_base_folder = os.path.join('data','ablated','levc')

models_mis_taus_control_errs = []
models_mis_taus_counters_peeks = []
models_mis_taus_rewss = []
models_mis_taus_sleepss = []

for model in ape_models:

    mis_taus_control_errs = []
    mis_taus_counters_peeks = []
    mis_taus_rewss = []
    mis_taus_sleepss = []

    for mistrained_tau in target_taus:

        mistrained_traj_folder = os.path.join(mistrained_traj_base_folder, str(model), 'ablated_tau%d' %mistrained_tau)

        control_errs_taus_ape = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_ablated_control_errs_taus_ape.pkl'), 'rb'))
        counters_peeks_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_ablated_counters_peeks_taus.pkl'), 'rb'))
        rewss_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_rewss_taus.pkl'), 'rb'))
        sleepss_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_ablated_sleep_errs_taus_ape.pkl'), 'rb'))

        mis_taus_control_errs.append(control_errs_taus_ape)
        mis_taus_counters_peeks.append(counters_peeks_taus)
        mis_taus_rewss.append(rewss_taus)
        mis_taus_sleepss.append(sleepss_taus)

    models_mis_taus_control_errs.append(mis_taus_control_errs)
    models_mis_taus_counters_peeks.append(mis_taus_counters_peeks)
    models_mis_taus_rewss.append(mis_taus_rewss)
    models_mis_taus_sleepss.append(mis_taus_sleepss)

models_mis_taus_control_errs = np.array(models_mis_taus_control_errs)[:,1:-1,:]
models_mis_taus_counters_peeks = np.array(models_mis_taus_counters_peeks)[:,1:-1,:]
models_mis_taus_rewss = np.array(models_mis_taus_rewss)[:,1:-1,:]
models_mis_taus_sleepss = np.array(models_mis_taus_sleepss)[:,1:-1,:]

target_taus = np.arange(-2, 3.01, 1)

# test_taus = pickle.load(open(os.path.join(mistrained_traj_base_folder, str(pepe_models[0]), timestamp_traj + '_test_taus.pkl'), 'rb'))

# %% READ IN TRAJECTORIES CORRESPONDING TO TRAINED TAU

models_orig_taus_control_errs = []
models_orig_taus_counters_peeks = []
models_orig_taus_rewss = []
models_orig_taus_sleepss = []

for model in ape_models:

    orig_traj_folder = os.path.join(eval_base_folder, str(model))

    control_errs_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_control_errs_taus_ape.pkl'), 'rb'))
    counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_counters_peeks_taus.pkl'), 'rb'))
    rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_rewss_taus.pkl'), 'rb'))
    sleepss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_sleep_errs_taus_ape.pkl'), 'rb'))

    models_orig_taus_control_errs.append(control_errs_taus_ape)
    models_orig_taus_counters_peeks.append(counters_peeks_taus)
    models_orig_taus_rewss.append(rewss_taus)
    models_orig_taus_sleepss.append(sleepss_taus)

models_orig_taus_control_errs = np.array(models_orig_taus_control_errs)
models_orig_taus_counters_peeks = np.array(models_orig_taus_counters_peeks)
models_orig_taus_rewss = np.array(models_orig_taus_rewss)
models_orig_taus_sleepss = np.array(models_orig_taus_sleepss)

# %% PLOT COUNTERS PEEKS ALL TAUS
test_taus = np.arange(0, 1.01, 0.125)
fig = plot_behavior_mistrained(test_taus, np.flip(models_mis_taus_sleepss, axis=(0,2)), np.arange(0,1.01,0.125), np.flip(models_orig_taus_sleepss, axis=(1)), axis_xlabel='Efficacy', axis_ylabel='Number of Sleeps per Episode', target_taus = target_taus)

os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_ablated_taus_counters_sleeps.png'))
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_ablated_taus_counters_sleeps.svg'))

# %% STATS
model_means_mis = models_mis_taus_rewss.mean(axis=(0,2))

print('max mis', np.max(model_means_mis))

# %% ANALYZE PERTURBED

## PARAMETERS

#timestamp_traj = '20230523232514'
#timestamp_traj = '20231023135947'
timestamp_traj = '20240312141018'
target_taus = np.arange(-3, 4.01, 1)
mistrained_traj_base_folder = os.path.join('data','perturbed','levc')

models_mis_taus_control_errs = []
models_mis_taus_counters_peeks = []
models_mis_taus_rewss = []
models_mis_taus_sleepss = []

for model in ape_models:

    mis_taus_control_errs = []
    mis_taus_counters_peeks = []
    mis_taus_rewss = []
    mis_taus_sleepss = []

    for mistrained_tau in target_taus:

        mistrained_traj_folder = os.path.join(mistrained_traj_base_folder, str(model), 'perturbed_tau%d' %mistrained_tau)

        control_errs_taus_ape = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_perturbed_control_errs_taus_ape.pkl'), 'rb'))
        counters_peeks_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_perturbed_counters_peeks_taus.pkl'), 'rb'))
        rewss_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_rewss_taus.pkl'), 'rb'))
        sleepss_taus = pickle.load(open(os.path.join(mistrained_traj_folder, timestamp_traj + '_perturbed_sleep_errs_taus_ape.pkl'), 'rb'))

        mis_taus_control_errs.append(control_errs_taus_ape)
        mis_taus_counters_peeks.append(counters_peeks_taus)
        mis_taus_rewss.append(rewss_taus)
        mis_taus_sleepss.append(sleepss_taus)

    models_mis_taus_control_errs.append(mis_taus_control_errs)
    models_mis_taus_counters_peeks.append(mis_taus_counters_peeks)
    models_mis_taus_rewss.append(mis_taus_rewss)
    models_mis_taus_sleepss.append(mis_taus_sleepss)

models_mis_taus_control_errs = np.array(models_mis_taus_control_errs)
models_mis_taus_counters_peeks = np.array(models_mis_taus_counters_peeks)
models_mis_taus_rewss = np.array(models_mis_taus_rewss)
models_mis_taus_sleepss = np.array(models_mis_taus_sleepss)

# test_taus = pickle.load(open(os.path.join(mistrained_traj_base_folder, str(pepe_models[0]), timestamp_traj + '_test_taus.pkl'), 'rb'))

# %% READ IN TRAJECTORIES CORRESPONDING TO TRAINED TAU

models_orig_taus_control_errs = []
models_orig_taus_counters_peeks = []
models_orig_taus_rewss = []
models_orig_taus_sleepss = []

for model in ape_models:

    orig_traj_folder = os.path.join(eval_base_folder, str(model))

    control_errs_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_control_errs_taus_ape.pkl'), 'rb'))
    counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_counters_peeks_taus.pkl'), 'rb'))
    rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_rewss_taus.pkl'), 'rb'))
    sleepss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_sleep_errs_taus_ape.pkl'), 'rb'))

    models_orig_taus_control_errs.append(control_errs_taus_ape)
    models_orig_taus_counters_peeks.append(counters_peeks_taus)
    models_orig_taus_rewss.append(rewss_taus)
    models_orig_taus_sleepss.append(sleepss_taus)

models_orig_taus_control_errs = np.array(models_orig_taus_control_errs)
models_orig_taus_counters_peeks = np.array(models_orig_taus_counters_peeks)
models_orig_taus_rewss = np.array(models_orig_taus_rewss)
models_orig_taus_sleepss = np.array(models_orig_taus_sleepss)

# %% PLOT COUNTERS SLEEPS ALL TAUS

### TODO: Figure out where to add offset of +/-1 most elegantly so that legend reflects true delta eff (i.e. currently 1 delta eff is actually 0)

test_taus = np.arange(0, 1.01, 0.125)
fig = plot_behavior_mistrained(test_taus, np.flip(models_mis_taus_sleepss, axis=(0,2)), np.arange(0,1.01,0.125), np.flip(models_orig_taus_sleepss, axis=(1)), axis_xlabel='Efficacy', axis_ylabel='Number of Sleeps per Episode', target_taus = target_taus)

os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_perturbed_taus_counters_sleeps.png'))
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_perturbed_taus_counters_sleeps.svg'))

# %% PLOT COUNTERS PEEKS ALL TAUS

test_taus = np.arange(0, 1.01, 0.125)
fig = plot_behavior_mistrained(test_taus, np.flip(models_mis_taus_counters_peeks, axis=(0,2)), np.arange(0,1.01,0.125), np.flip(models_orig_taus_counters_peeks, axis=(1)), axis_xlabel='Efficacy', axis_ylabel='Number of Peeks per Episode', target_taus = target_taus)

os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_perturbed_taus_counters_peeks.png'))
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_perturbed_taus_counters_peeks.svg'))

# %% PLOT REWARDS ALL TAUS

test_taus = np.arange(0, 1.01, 0.125)
fig = plot_behavior_mistrained(test_taus, np.flip(models_mis_taus_rewss, axis=(0,2)), np.arange(0,1.01,0.125), np.flip(models_orig_taus_rewss, axis=(1)), axis_xlabel='Efficacy', axis_ylabel='Number of Rewards per Episode', target_taus = target_taus)

os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_perturbed_taus_rews.png'))
fig.savefig(os.path.join(analysis_folder, get_timestamp() + '_perturbed_taus_rews.svg'))

# %% STATS

model_means_mis = models_mis_taus_rewss.mean(axis=(0,2))

print('max mis', np.max(model_means_mis))
# %%
