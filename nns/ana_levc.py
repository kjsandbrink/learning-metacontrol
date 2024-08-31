# Kai Sandbrink
# 2023-02-01
# This script performs the analyses for the LEVC task (Task 2)

# %% LIBRARY IMPORTS

import torch
import pandas as pd
import numpy as np
import matplotlib as mpl

from utils import Config, plot_learning_curve, plot_learning_curves_comparison, get_timestamp
from utils_project import load_config_files
from utils_project import calculate_freq_observed_choice_per_t, calculate_freq_observes_per_t, calculate_freq_correct_choice_per_t, calculate_freq_correct_take
from test_case import test, test_helplessness, perturbed_test, convert_control_to_prob_random

from datetime import datetime
import os, copy, pickle

from test_analyses import policy_barplot, within_episode_efficacy_lineplot, plot_comparison_curves_several_runs, frac_takes_lineplot, frac_correct_takes_lineplot, ape_accuracy_lineplot, frac_sleeps_lineplot, plot_evidence_ratios

from scipy.stats import ttest_ind

# %% SPECIFY TEST CASES

### NO HOLDOUT BIAS 0.5, VOL 0.1, 150k ENTROPY ANNEALING
## 12/11

ape_models = [
    20231211181022,
    20231211181020,
    20231211181018,
    20231211181016,
    20231211181015,
    20231211181012,
    20231211181010,
    20231211181009,
    20231211181007,
    20231211181006,
]

control_models = [
    20231211181040,
    20231211181039,
    20231211181038,
    20231211181035,
    20231211181034,
    20231211181033,
    20231211181030,
    20231211181027,
    20231211181026,
    20231211181023
]

# %% OTHER PARAMETERS

base_model_folder = 'models'

# %% CONFIG FILES

# the assumption is all of these have the same config & task files,
# and nn_options are shared within a model type

ape_modelname = str(ape_models[1])
control_modelname = str(control_models[1])

config, task_options, ape_nn_options = load_config_files(ape_models[0])
_, _, control_nn_options = load_config_files(control_models[0])

config.n_repeats_case = 100

ape_config = Config(copy.deepcopy(config.__dict__))
ape_config.model_folder = os.path.join(base_model_folder, ape_modelname)
control_config = Config(copy.deepcopy(config.__dict__))
control_config.model_folder = os.path.join(base_model_folder, control_modelname)

# %% INITIALIZATIONS

## ANALYSIS FOLDER
analysis_folder = os.path.join('analysis', 'levc', 'human' )
ape_analysis_folder = os.path.join('analysis', str(ape_modelname))
control_analysis_folder = os.path.join('analysis', str(control_modelname))
os.makedirs(analysis_folder, exist_ok=True)

## TORCH
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

## OTHER

n_models = len(ape_models)
ape_cmap = mpl.colormaps['Purples']
control_cmap = mpl.colormaps['Oranges']
n_steps = config.n_steps_to_reward

# %% PANEL LEARNING CURVES : LOAD IN LEARNING CURVES

ape_rewardss = []

for ape_modelname in ape_models[1:]:
    ape_rewardss.append(np.load(os.path.join(base_model_folder, str(ape_modelname), 'logged_returns.npy')))

control_rewardss = []

for control_modelname in control_models[1:]:
    control_rewardss.append(np.load(os.path.join(base_model_folder, str(control_modelname), 'logged_returns.npy')))

#episodes = np.arange(0, len(ape_rewardss[0])*100, 100)
k_episodes = np.arange(0, len(ape_rewardss[0])*0.1, 0.1)

# %% PANEL LEARNING CURVES : PLOT CURVE WITH BOTH LCS

import matplotlib.pyplot as plt
from utils import format_axis

fig = plot_learning_curves_comparison(k_episodes, ape_rewardss, control_rewardss, smoothing_window=1000, name_exp="APE-trained")
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves.svg' %get_timestamp()))

# %% PANEL LEARNING CURVES : PLOT CURVE WITH ONLY CONTROL LEARNING CURVES

fig = plot_learning_curves_comparison(k_episodes, None, control_rewardss, smoothing_window=1000, name_exp="APE-trained")
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_control_only.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_control_only.svg' %get_timestamp()))

# %% STATS LEARNING CURVES ON TRAINING

ape_rewardss_mean_final_50 = np.array(ape_rewardss)[:,-50:].mean(axis=1)
control_rewardss_mean_final_50 = np.array(control_rewardss)[:,-50:].mean(axis=1)

print('ape rewards mean', ape_rewardss_mean_final_50.mean())
print('ape rewards std err', ape_rewardss_mean_final_50.std()/np.sqrt(n_models))

print('control rewards mean', control_rewardss_mean_final_50.mean())
print('control rewards std err', control_rewardss_mean_final_50.std()/np.sqrt(n_models))

t = ttest_ind(ape_rewardss_mean_final_50, control_rewardss_mean_final_50, alternative='greater')
print('ttest', t)

# %% STATS LEARNING CURVES

ape_rewardss_mean_final_50 = np.array(ape_rewardss)[:,-50:].mean(axis=1)
control_rewardss_mean_final_50 = np.array(control_rewardss)[:,-50:].mean(axis=1)

print('ape rewards mean', ape_rewardss_mean_final_50.mean())
print('ape rewards std err', ape_rewardss_mean_final_50.std()/np.sqrt(n_models))

print('control rewards mean', control_rewardss_mean_final_50.mean())
print('control rewards std err', control_rewardss_mean_final_50.std()/np.sqrt(n_models))

t = ttest_ind(ape_rewardss_mean_final_50, control_rewardss_mean_final_50, alternative='greater')
print('ttest', t)

# %% PANEL LEARNING CURVES ON TEST : LOAD DATA

# learning_curves_folder = '/home/kai/Documents/Projects/meta-peek-take/results/pepe/20230202005530_eval_learning_curves_10repeats'

# ape_models = pickle.load( open(os.path.join(learning_curves_folder, 'ape_models.pkl'), 'rb'))
# control_models = pickle.load( open(os.path.join(learning_curves_folder, 'control_models.pkl'), 'rb'))
# ape_test_returns_losses = pickle.load( open(os.path.join(learning_curves_folder, 'ape_test_returns_losses.pkl'), 'rb'))
# ape_test_apes_losses = pickle.load( open(os.path.join(learning_curves_folder, 'ape_test_apes_losses.pkl'), 'rb'))
# ape_test_apes_mses = pickle.load( open(os.path.join(learning_curves_folder, 'ape_test_apes_mses.pkl'), 'rb'))
# control_test_returns_losses = pickle.load( open(os.path.join(learning_curves_folder, 'control_test_returns_losses.pkl'), 'rb'))
# test_episodes = pickle.load( open(os.path.join(learning_curves_folder, 'test_episodes.pkl'), 'rb'))
# ape_test_rews = pickle.load( open(os.path.join(learning_curves_folder, 'ape_test_rews.pkl'), 'rb'))
# control_test_rews = pickle.load( open(os.path.join(learning_curves_folder, 'control_test_rews.pkl'), 'rb'))


eval_folder = os.path.join('data', 'eval', 'levc')
#eval_timestamp = '20230324153020'
eval_timestamp = '20231211180601'

ape_test_rews = []
ape_test_apes_losses = []
ape_test_returns_losses = []
ape_test_apes_mses = []

for ape_model in ape_models:
    ape_test_rews.append(pickle.load(open(os.path.join(eval_folder, str(ape_model), eval_timestamp + '_test_learning_curve_rews.pkl'), 'rb')))
    ape_test_returns_losses.append(pickle.load(open(os.path.join(eval_folder, str(ape_model), eval_timestamp + '_test_learning_curve_returns_losses.pkl'), 'rb')))
    ape_test_apes_losses.append(pickle.load(open(os.path.join(eval_folder, str(ape_model), eval_timestamp + '_test_learning_curve_apes_losses.pkl'), 'rb')))
    ape_test_apes_mses.append(pickle.load(open(os.path.join(eval_folder, str(ape_model), eval_timestamp + '_test_learning_curve_apes_mses.pkl'), 'rb')))

control_test_rews = []
control_test_returns_losses = []

for control_model in control_models:
    control_test_rews.append(pickle.load(open(os.path.join(eval_folder, str(control_model), eval_timestamp + '_test_learning_curve_rews.pkl'), 'rb')))
    control_test_returns_losses.append(pickle.load(open(os.path.join(eval_folder, str(control_model), eval_timestamp + '_test_learning_curve_returns_losses.pkl'), 'rb')))

test_episodes = pickle.load(open(os.path.join(eval_folder, str(ape_modelname), eval_timestamp + '_test_learning_curve_episodes.pkl'), 'rb'))


# %% PANEL LEARNING CURVES ON TEST : PLOT REW LCs

fig = plot_comparison_curves_several_runs(test_episodes, np.array(ape_test_rews).T, test_episodes, np.array(control_test_rews).T, title='Learning Curves (Test Reward)', axis_xlabel='Episodes', axis_ylabel="Total Reward per Episode", label_exp="APE-trained", label_control='no APE')
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test.svg' %get_timestamp()))

#fig.savefig(os.path.join(analysis_folder, 'learning_curves_test.png'))

# %% PANEL LEARNING CURVES ON TEST : PLOT PG LOSS

fig = plot_comparison_curves_several_runs(test_episodes, np.array(ape_test_returns_losses).T, test_episodes, np.array(control_test_returns_losses).T, title='Learning Curves (Policy Gradient Test Loss)', axis_xlabel='Episodes', axis_ylabel='Loss (Policy Gradient)', label_exp='APE-trained', label_control='no APE')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_pg_loss.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_pg_loss.svg' %get_timestamp()))

#fig.savefig(os.path.join(analysis_folder, 'learning_curves_test_pg_loss.png'))

# %% PANEL LEARNING CURVES ON TEST : PLOT APEs LOSS

fig = plot_comparison_curves_several_runs(test_episodes, np.array(ape_test_apes_losses).T, title='Learning Curves (APE Test Loss)', axis_xlabel='Episodes', axis_ylabel='Loss (APE)', label_exp='APE-trained')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

#fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_ape_loss.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_ape_loss.svg' %get_timestamp()))

# %% PANEL LEARNING CURVES ON TEST : PLOT APEs MSEs

fig = plot_comparison_curves_several_runs(test_episodes, np.array(ape_test_apes_mses).T, title='Learning Curves (APE Test Mean-Squared Error)', axis_xlabel='Episodes', axis_ylabel='MSE APEs', label_exp='APE-trained')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

#fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_ape_mse.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_ape_mse.svg' %get_timestamp()))

# %% PANEL POLICY REPRESENTATION : SIMULATE TESTS

starting_taus = {'peek': 0, 'take': 0.9}
tau_task_options = Config(copy.deepcopy(task_options.__dict__))
tau_task_options.starting_taus = starting_taus

(ape_actionss, ape_logitss, _, _, ape_controlss, ape_pss, _), _ = test(ape_config, ape_nn_options, tau_task_options, device)
(control_actionss, control_logitss, _, _, _, control_pss, _), _ = test(control_config, control_nn_options, tau_task_options, device)

# %% PANEL POLICY REP: BAR PLOT (for single episode)

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = policy_barplot(ape_logitss, ape_pss, episode=9)
fig.savefig(os.path.join(ape_analysis_folder, '%s_policy_barplot_ape.png' %get_timestamp()))

os.makedirs(control_analysis_folder ,exist_ok = True)
fig = policy_barplot(control_logitss, control_pss, episode=9)
fig.savefig(os.path.join(control_analysis_folder, '%s_policy_barplot_control.png' %get_timestamp()))

# %% LINE PLOT FOR CONTROLS OVER INDIVIDUAL POLICY

'''

## TODO: Get rid of the red line as this is invalid for this LEVC task

os.makedirs(ape_analysis_folder ,exist_ok = True)

tau = starting_taus['take']

fig = within_episode_efficacy_lineplot(convert_control_to_prob_random(ape_controlss), tau=tau, episode=1, )

fig.savefig(os.path.join(ape_analysis_folder, '%s_within_episode_efficacy_plot_tau%d.png' %(get_timestamp(), int(tau*100))))

'''

# %% PANEL POLICY REP: SINGLE MODEL BUT MULTIPLE TAU VALUES AVERAGED - DATA

test_taus = np.arange(0,1.01,0.25)

## TO READ IN

ape_modelname = str(ape_models[6]) 
control_modelname = str(control_models[6])

traj_base = os.path.join('data', 'eval', 'levc', )
traj_timestamp = '20231213153154' ## MATCHING 12/11

orig_traj_folder = os.path.join(traj_base, str(ape_modelname))

ape_control_errs_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_control_errs_taus_ape.pkl'), 'rb'))[::2]

ape_actionss_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_actionss_taus.pkl'), 'rb'))[::2]
ape_logitss_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_logitss_taus.pkl'), 'rb'))[::2]
ape_pss_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_pss_taus.pkl'), 'rb'))[::2]

orig_traj_folder = os.path.join(traj_base, str(control_modelname))

counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_counters_peeks_taus.pkl'), 'rb'))[::2]
rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_rewss_taus.pkl'), 'rb'))[::2]

control_actionss_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_actionss_taus.pkl'), 'rb'))[::2]
control_logitss_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_logitss_taus.pkl'), 'rb'))[::2]
control_pss_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_pss_taus.pkl'), 'rb'))[::2]


# tau_task_options = Config(copy.deepcopy(task_options.__dict__))
# n_repeats_case = 100

# ape_actionss_taus = []
# ape_logitss_taus = []
# ape_controlss_taus = []
# ape_pss_taus = []
# ape_control_errs_taus = []

# control_actionss_taus = []
# control_logitss_taus = []
# control_pss_taus = []

# for test_tau in test_taus:

#     tau_task_options.starting_taus = {'peek': 0, 'take': test_tau}

#     (ape_actionss, ape_logitss, _, _, ape_controlss, ape_pss, ape_control_errss), _ = test(ape_config, ape_nn_options, tau_task_options, device, n_repeats_case=n_repeats_case)
#     (control_actionss, control_logitss, _, _, _, control_pss, _), _ = test(control_config, control_nn_options, tau_task_options, device, n_repeats_case=n_repeats_case)

#     ape_actionss_taus.append(ape_actionss)
#     ape_logitss_taus.append(ape_logitss)
#     ape_controlss_taus.append(ape_controlss)
#     ape_pss_taus.append(ape_pss)
#     ape_control_errs_taus.append(ape_control_errss)

#     control_actionss_taus.append(control_actionss)
#     control_logitss_taus.append(control_logitss)
#     control_pss_taus.append(control_pss)

# %% PANEL POLICY SINGLE MODEL FRAC TAKES - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = frac_takes_lineplot(test_taus, ape_logitss_taus)
fig.savefig(os.path.join(ape_analysis_folder, '%s_frac_takes_lineplot.png' %get_timestamp()))
fig.savefig(os.path.join(ape_analysis_folder, '%s_frac_takes_lineplot.svg' %get_timestamp()))

os.makedirs(control_analysis_folder ,exist_ok = True)
fig = frac_takes_lineplot(test_taus, control_logitss_taus)
fig.savefig(os.path.join(control_analysis_folder, '%s_frac_takes_lineplot.png' %get_timestamp()))
fig.savefig(os.path.join(control_analysis_folder, '%s_frac_takes_lineplot.svg' %get_timestamp()))

# %% PANEL POLICY SINGLE MODEL PERC CORRECT TAKES - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = frac_correct_takes_lineplot(test_taus, ape_logitss_taus, ape_pss_taus, includes_sleep=True)
fig.savefig(os.path.join(ape_analysis_folder, '%s_corr_takes_lineplot.png' %get_timestamp()))
fig.savefig(os.path.join(ape_analysis_folder, '%s_corr_takes_lineplot.svg' %get_timestamp()))

os.makedirs(control_analysis_folder ,exist_ok = True)
fig = frac_correct_takes_lineplot(test_taus, control_logitss_taus, control_pss_taus, includes_sleep=True)
fig.savefig(os.path.join(control_analysis_folder, '%s_corr_takes_lineplot.png'%get_timestamp()))
fig.savefig(os.path.join(control_analysis_folder, '%s_corr_takes_lineplot.svg' %get_timestamp()))

# %% PANEL POLICY SINGLE MODEL PERC SLEEPS - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = frac_sleeps_lineplot(test_taus, ape_logitss_taus, ylim=(-0.025, 1.02 ))
fig.savefig(os.path.join(ape_analysis_folder, '%s_prob_sleeps_lineplot.png' %get_timestamp()))
fig.savefig(os.path.join(ape_analysis_folder, '%s_prob_sleeps_lineplot.svg' %get_timestamp()))

os.makedirs(control_analysis_folder ,exist_ok = True)
fig = frac_sleeps_lineplot(test_taus, control_logitss_taus, ylim=(-0.025, 1.02 ))
fig.savefig(os.path.join(control_analysis_folder, '%s_prob_sleeps_lineplot.png'%get_timestamp()))
fig.savefig(os.path.join(control_analysis_folder, '%s_prob_sleeps_lineplot.svg'%get_timestamp()))

# %% PANEL POLICY SINGLE MODEL ERROR IN APEs - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = ape_accuracy_lineplot(test_taus, ape_control_errs_taus)
fig.savefig(os.path.join(ape_analysis_folder, '%s_ape_accuracy_lineplot.png' %get_timestamp()))
fig.savefig(os.path.join(ape_analysis_folder, '%s_ape_accuracy_lineplot.svg' %get_timestamp()))

# %% PANEL QUANTIFICATION OF REWARDS AND E-E LOAD DATA

#timestamp_original_traj = '20230804105436'
#timestamp_original_traj = '20231008233047'
#timestamp_original_traj = '20231015162557' ##10/13
#timestamp_original_traj = '20231015164013' ##10/10
#timestamp_original_traj = '20231211180601' ##12/11
timestamp_original_traj = '20231213153154' ##12/13 - 150k ANNEALING
eval_base_folder = os.path.join('data', 'eval', 'levc')

test_taus = np.arange(0,1.01,0.125)

rewss_taus_ape = [] #list of lists that will store rews for diff tau values for APE
counterss_peeks_taus_ape = []
control_errss_taus_ape = []
counterss_sleeps_taus_ape = []

trajss_actions_taus_ape = []
trajss_logits_taus_ape = []
trajss_ps_taus_ape = []

rewss_taus_control = []
counterss_peeks_taus_control = []
counterss_sleeps_taus_control = []

trajss_actions_taus_control = []
trajss_logits_taus_control = []
trajss_ps_taus_control = []

for model in ape_models:

    orig_traj_folder = os.path.join(eval_base_folder, str(model))

    control_errs_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_control_errs_taus_ape.pkl'), 'rb'))
    counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_counters_peeks_taus.pkl'), 'rb'))
    rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_rewss_taus.pkl'), 'rb'))
    sleepss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_sleep_errs_taus_ape.pkl'), 'rb'))

    trajs_actions_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_traj_actionss_taus.pkl'), 'rb'))
    trajs_logits_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_traj_logitss_taus.pkl'), 'rb'))
    trajs_ps_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_traj_pss_taus.pkl'), 'rb'))

    control_errss_taus_ape.append(control_errs_taus_ape)
    counterss_peeks_taus_ape.append(counters_peeks_taus)
    rewss_taus_ape.append(rewss_taus)
    counterss_sleeps_taus_ape.append(sleepss_taus)
    
    trajss_actions_taus_ape.append(trajs_actions_taus_ape)
    trajss_logits_taus_ape.append(trajs_logits_taus_ape)
    trajss_ps_taus_ape.append(trajs_ps_taus_ape)


for model in control_models:

    orig_traj_folder = os.path.join(eval_base_folder, str(model))

    counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_counters_peeks_taus.pkl'), 'rb'))
    rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_rewss_taus.pkl'), 'rb'))
    sleepss_taus = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_sleep_errs_taus_ape.pkl'), 'rb'))

    trajs_actions_taus_control = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_traj_actionss_taus.pkl'), 'rb'))
    trajs_logits_taus_control = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_traj_logitss_taus.pkl'), 'rb'))
    trajs_ps_taus_control = pickle.load(open(os.path.join(orig_traj_folder, timestamp_original_traj + '_traj_pss_taus.pkl'), 'rb'))

    counterss_peeks_taus_control.append(counters_peeks_taus)
    rewss_taus_control.append(rewss_taus)
    counterss_sleeps_taus_control.append(sleepss_taus)

    trajss_actions_taus_control.append(trajs_actions_taus_control)
    trajss_logits_taus_control.append(trajs_logits_taus_control)
    trajss_ps_taus_control.append(trajs_ps_taus_control)

control_errss_taus_ape = np.array(control_errss_taus_ape).T
counterss_peeks_taus_ape = np.array(counterss_peeks_taus_ape).T
rewss_taus_ape = np.array(rewss_taus_ape).T
counterss_sleeps_taus_ape = np.array(counterss_sleeps_taus_ape).T

trajss_actions_taus_ape = np.array(trajss_actions_taus_ape)
trajss_logits_taus_ape = np.array(trajss_logits_taus_ape)
trajss_ps_taus_ape = np.array(trajss_ps_taus_ape)

rewss_taus_control = np.array(rewss_taus_control).T
counterss_peeks_taus_control = np.array(counterss_peeks_taus_control).T
counterss_sleeps_taus_control = np.array(counterss_sleeps_taus_control).T

trajss_actions_taus_control = np.array(trajss_actions_taus_control)
trajss_logits_taus_control = np.array(trajss_logits_taus_control)
trajss_ps_taus_control = np.array(trajss_ps_taus_control)

# %% CALCULATE AVERAGE NUMBER OF REWARDS AND OBSERVES, % TAKEN CORRECT PER EFFICACY LEVEL

print("EV of Effs", rewss_taus_ape.mean(axis=1), )
print("stderr EV of Effs", rewss_taus_ape.std(axis=1)/np.sqrt(n_models), )
print("Obs of Effs", counterss_peeks_taus_ape.mean(axis=1), )
print("stderr Obs of Effs", counterss_peeks_taus_ape.std(axis=1)/np.sqrt(n_models), )

print("Sleeps of Effs", counterss_sleeps_taus_ape.mean(axis=1), )
print("stderr Obs of Effs", counterss_sleeps_taus_ape.std(axis=1)/np.sqrt(n_models), )

correct_takes = calculate_freq_correct_take(trajss_logits_taus_ape, trajss_ps_taus_ape, include_sleep=True).mean(axis=(2, 3))
print("Expected correct take probability", correct_takes.mean(axis=0))
print("stderr", correct_takes.std(axis=0)/np.sqrt(n_models))

### 12/11 250k ENTROPY
# EV of Effs [32.643 33.041 32.39  31.817 30.945 29.667 28.225 26.841 25.392]
# stderr EV of Effs [0.39998512 0.34673751 0.42611266 0.37041882 0.26326128 0.21606041
#  0.21752816 0.24142266 0.21831537]
# Obs of Effs [5.531 5.507 5.492 5.488 5.488 5.374 5.42  5.424 5.402]
# stderr Obs of Effs [0.42083833 0.43484032 0.42683205 0.41063804 0.39425068 0.36210827
#  0.33602976 0.31757897 0.31086267]
# Sleeps of Effs [3.255 3.229 3.339 3.639 4.099 4.694 5.349 5.964 6.46 ]
# stderr Obs of Effs [0.29779271 0.30401464 0.27418406 0.22055135 0.21227553 0.28934478
#  0.34460543 0.41471002 0.49888275]
# Expected correct take probability [0.75906872 0.76751221 0.75838994 0.75989741 0.76564176 0.76862511
#  0.77071872 0.76483193 0.76521506]
# stderr [0.01255626 0.01133072 0.01309236 0.01303136 0.00941374 0.01018969
#  0.00945282 0.00853843 0.00714491]

#### 12/11 150k ANNEA

# %% CONTROL MODEL

print("EV of Effs", rewss_taus_control.mean(axis=1), )
print("stderr EV of Effs", rewss_taus_control.std(axis=1)/np.sqrt(n_models), )
print("Obs of Effs", counterss_peeks_taus_control.mean(axis=1), )
print("stderr Obs of Effs", counterss_peeks_taus_control.std(axis=1)/np.sqrt(n_models), )

print("Sleeps of Effs", counterss_sleeps_taus_control.mean(axis=1), )
print("stderr Obs of Effs", counterss_sleeps_taus_control.std(axis=1)/np.sqrt(n_models), )

correct_takes = calculate_freq_correct_take(trajss_logits_taus_control, trajss_ps_taus_control, include_sleep=True).mean(axis=(2, 3))
print("Expected correct take probability", correct_takes.mean(axis=0))
print("stderr", correct_takes.std(axis=0)/np.sqrt(n_models))

### 12/11 250k ENTROPY
# EV of Effs [31.32  31.698 31.427 30.862 30.291 29.349 27.807 26.772 25.382]
# stderr EV of Effs [0.65016921 0.6914359  0.7366601  0.99366473 0.77479346 0.5664635
#  0.5397982  0.32062065 0.21423725]
# Obs of Effs [4.372 4.379 4.339 4.362 4.38  4.41  4.394 4.392 4.332]
# stderr Obs of Effs [0.60818714 0.63209564 0.60664396 0.63751204 0.63482911 0.61538931
#  0.62743318 0.62311604 0.64019966]
# Sleeps of Effs [3.525 3.516 3.545 3.492 3.517 3.501 3.495 3.481 3.491]
# stderr Obs of Effs [0.53033621 0.52223596 0.53152469 0.52892684 0.5245723  0.52519415
#  0.52089778 0.51814178 0.51786185]
# Expected correct take probability [0.71774926 0.72848007 0.72357564 0.71570389 0.72064683 0.72959154
#  0.71722002 0.7243689  0.72335611]
# stderr [0.02505477 0.02628196 0.02656803 0.03321213 0.02594733 0.02654767
#  0.0275857  0.0280971  0.02719329]

# %% PANEL QUANTIFICATION OF REWARDS PLOT

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(rewss_taus_ape)), test_taus, list(reversed(rewss_taus_control)), title='Performance for Different Efficacy Values', axis_xlabel='Efficacy', axis_ylabel='Rewards', label_exp='APE-trained', label_control='no APE')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.svg' %get_timestamp()))

# %% PANEL QUANTIFICATION EXPLORE-EXPLOIT

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(counterss_peeks_taus_ape)), test_taus, list(reversed(counterss_peeks_taus_control)), title='Balance Between Explore-Exploit', axis_xlabel='Efficacy', axis_ylabel='Number of Observes per Episode', label_exp='APE-trained', label_control='no APE')

fig.savefig(os.path.join(analysis_folder, '%s_explore-exploit_ratio.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_explore-exploit_ratio.svg' %get_timestamp()))

# %% PANEL QUANTIFICATION NUMBER OF SLEEPS

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(counterss_sleeps_taus_ape)), test_taus, list(reversed(counterss_sleeps_taus_control)), title='Time Spent Sleeping for Different Efficacy Values', axis_xlabel='Efficacy', axis_ylabel='Number of Sleeps per Episode', label_exp='APE-trained', label_control='no APE')
fig.savefig(os.path.join(analysis_folder, '%s_sleeps_efficacy.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_sleeps_efficacy.svg' %get_timestamp()))

# %% PANEL QUANTIFICATION ERROR APE

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(control_errss_taus_ape)), title="Error in Efficacy Estimation", axis_xlabel= 'Efficacy', axis_ylabel='MSE APEs', label_exp='APE-trained')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_mse_apes.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_mse_apes.svg' %get_timestamp()))

# %% STATS

rewss_taus_ape = np.array(rewss_taus_ape)
counterss_peeks_taus_ape = np.array(counterss_peeks_taus_ape)
control_errs_taus_ape = np.array(control_errs_taus_ape)
counterss_sleeps_taus_ape = np.array(counterss_sleeps_taus_ape)

rewss_taus_control = np.array(rewss_taus_control)
counterss_peeks_taus_control = np.array(counterss_peeks_taus_control)
counterss_sleeps_taus_control = np.array(counterss_sleeps_taus_control)

model_mean_rewss_ape = np.mean(rewss_taus_ape, axis=0)

print("mean ape", np.mean(model_mean_rewss_ape))
print("stderr ape", np.std(model_mean_rewss_ape)/np.sqrt(5))

model_mean_rewss_control = np.mean(rewss_taus_control, axis=0)

print("mean control", np.mean(model_mean_rewss_control))
print("stderr control", np.std(model_mean_rewss_control)/np.sqrt(5))

t = ttest_ind(model_mean_rewss_ape, model_mean_rewss_control, alternative='greater')
print('ttest', t)

model_mean_mse = np.mean(control_errss_taus_ape, axis=0)

print("mean mse ape readout", np.mean(model_mean_mse))
print("stderr mse ape readout", np.std(model_mean_mse)/np.sqrt(5))

# %% EVIDENCE RATIO CALCS

indices_to_plot = [0, 2, 6, 8]

evidence_ratio_observations_ape_models = []
evidence_ratio_takes_observed_ape_models = []
evidence_ratio_takes_correct_ape_models = []

for ape_actionss_taus, ape_logitss_taus, ape_pss_taus in zip(trajss_actions_taus_ape, trajss_logits_taus_ape, trajss_ps_taus_ape):

    evidence_ratio_observations = []
    evidence_ratio_takes_observed = []
    evidence_ratio_takes_correct = []

    for actionss, logitss, pss in zip(ape_actionss_taus, ape_logitss_taus, ape_pss_taus):
        evidence_ratio_observations.append(calculate_freq_observes_per_t(actionss, logitss, pss, include_sleep=True))
        evidence_ratio_takes_observed.append(calculate_freq_observed_choice_per_t(actionss, logitss, pss, include_sleep=True))
        evidence_ratio_takes_correct.append(calculate_freq_correct_choice_per_t(actionss, logitss, pss, include_sleep=True))
        
    evidence_ratio_observations_ape_models.append(evidence_ratio_observations)
    evidence_ratio_takes_observed_ape_models.append(evidence_ratio_takes_observed)
    evidence_ratio_takes_correct_ape_models.append(evidence_ratio_takes_correct)

    # evidence_ratio_observations = np.stack(evidence_ratio_observations)
    # evidence_ratio_takes_observed = np.stack(evidence_ratio_takes_observed)
    # evidence_ratio_takes_correct = np.stack(evidence_ratio_takes_correct)

evidence_ratio_observations_models = np.array(evidence_ratio_observations_ape_models)
evidence_ratio_takes_observed_models = np.array(evidence_ratio_takes_observed_ape_models)
evidence_ratio_takes_correct_models = np.array(evidence_ratio_takes_correct_ape_models)

evidence_ratio_observations_models = np.nanmean(evidence_ratio_observations_models, axis=2)
evidence_ratio_takes_observed_models = np.nanmean(evidence_ratio_takes_observed_models, axis=2)
evidence_ratio_takes_correct_models = np.nanmean(evidence_ratio_takes_correct_models, axis=2)

evidence_ratio_observations_models = np.swapaxes(evidence_ratio_observations_models, 0, 1)
evidence_ratio_takes_observed_models = np.swapaxes(evidence_ratio_takes_observed_models, 0, 1)
evidence_ratio_takes_correct_models = np.swapaxes(evidence_ratio_takes_correct_models, 0, 1)

# %% PANEL EVIDENCE RATIO - PLOT

os.makedirs(analysis_folder ,exist_ok = True)
fig = plot_evidence_ratios(evidence_ratio_observations_models[indices_to_plot], 1- test_taus[indices_to_plot], ape_cmap, ylabel='Probability of Observing', ylim=(0,1), xlim=(0, n_steps))
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_observations_ape.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_observations_ape.svg' %get_timestamp()), dpi=300, transparent=True)

# %% 

os.makedirs(analysis_folder ,exist_ok = True)
fig = plot_evidence_ratios(evidence_ratio_takes_observed_models[indices_to_plot], 1 - test_taus[indices_to_plot], ape_cmap, ylabel='Probability of Intending Observed Arm', ylim=(0,1), xlim=(0, n_steps))
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_takes_observed_ape.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_takes_observed_ape.svg' %get_timestamp()), dpi=300, transparent=True)

# %% 

fig = plot_evidence_ratios(evidence_ratio_takes_correct_models[indices_to_plot], 1 - test_taus[indices_to_plot], ape_cmap, ylabel='Probability of Intending Correct Arm', ylim=(0,1), xlim=(0, n_steps))
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_takes_correct_ape.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_takes_correct_ape.svg' %get_timestamp()), dpi=300, transparent=True)

# %% SLEEP PLOT - ALIGNED TO BEGINNING

sleep_propensity_models = np.exp(trajss_logits_taus_ape[...,1])
sleep_propensity_models = np.nanmean(sleep_propensity_models, axis=2)
sleep_propensity_models = np.swapaxes(sleep_propensity_models, 0, 1)

# %% PLOT

os.makedirs(analysis_folder ,exist_ok = True)
fig = plot_evidence_ratios(sleep_propensity_models[indices_to_plot], 1 - test_taus[indices_to_plot], ape_cmap, xlabel = 'Time since begining of episode', ylabel='Probability of Sleeping', ylim=(0,1), xlim=(0, n_steps))
fig.savefig(os.path.join(analysis_folder, '%s_sleep_propensity_ape.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_sleep_propensity_ape.svg' %get_timestamp()), dpi=300, transparent=True)

# %% CONTROL MODELS

indices_to_plot = [0, 2, 6, 8]

evidence_ratio_observations_control_models = []
evidence_ratio_takes_observed_control_models = []
evidence_ratio_takes_correct_control_models = []

for control_actionss_taus, control_logitss_taus, control_pss_taus in zip(trajss_actions_taus_control, trajss_logits_taus_control, trajss_ps_taus_control):

    evidence_ratio_observations = []
    evidence_ratio_takes_observed = []
    evidence_ratio_takes_correct = []

    for actionss, logitss, pss in zip(control_actionss_taus, control_logitss_taus, control_pss_taus):
        evidence_ratio_observations.append(calculate_freq_observes_per_t(actionss, logitss, pss, include_sleep=True))
        evidence_ratio_takes_observed.append(calculate_freq_observed_choice_per_t(actionss, logitss, pss, include_sleep=True))
        evidence_ratio_takes_correct.append(calculate_freq_correct_choice_per_t(actionss, logitss, pss, include_sleep=True))
        
    evidence_ratio_observations_control_models.append(evidence_ratio_observations)
    evidence_ratio_takes_observed_control_models.append(evidence_ratio_takes_observed)
    evidence_ratio_takes_correct_control_models.append(evidence_ratio_takes_correct)

    # evidence_ratio_observations = np.stack(evidence_ratio_observations)
    # evidence_ratio_takes_observed = np.stack(evidence_ratio_takes_observed)
    # evidence_ratio_takes_correct = np.stack(evidence_ratio_takes_correct)

evidence_ratio_observations_models = np.array(evidence_ratio_observations_control_models)
evidence_ratio_takes_observed_models = np.array(evidence_ratio_takes_observed_control_models)
evidence_ratio_takes_correct_models = np.array(evidence_ratio_takes_correct_control_models)

evidence_ratio_observations_models = np.nanmean(evidence_ratio_observations_models, axis=2)
evidence_ratio_takes_observed_models = np.nanmean(evidence_ratio_takes_observed_models, axis=2)
evidence_ratio_takes_correct_models = np.nanmean(evidence_ratio_takes_correct_models, axis=2)

evidence_ratio_observations_models = np.swapaxes(evidence_ratio_observations_models, 0, 1)
evidence_ratio_takes_observed_models = np.swapaxes(evidence_ratio_takes_observed_models, 0, 1)
evidence_ratio_takes_correct_models = np.swapaxes(evidence_ratio_takes_correct_models, 0, 1)

# %% PANEL EVIDENCE RATIO - PLOT

os.makedirs(analysis_folder ,exist_ok = True)
fig = plot_evidence_ratios(evidence_ratio_observations_models[indices_to_plot], 1- test_taus[indices_to_plot], control_cmap, ylabel='Probability of Observing', ylim=(0,1), xlim=(0, n_steps))
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_observations_control.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_observations_control.svg' %get_timestamp()), dpi=300, transparent=True)

# %% 

os.makedirs(analysis_folder ,exist_ok = True)
fig = plot_evidence_ratios(evidence_ratio_takes_observed_models[indices_to_plot], 1 - test_taus[indices_to_plot], control_cmap, ylabel='Probability of Intending Observed Arm', ylim=(0,1), xlim=(0, n_steps))
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_takes_observed_control.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_takes_observed_control.svg' %get_timestamp()), dpi=300, transparent=True)

# %% 

fig = plot_evidence_ratios(evidence_ratio_takes_correct_models[indices_to_plot], 1 - test_taus[indices_to_plot], control_cmap, ylabel='Probability of Intending Correct Arm', ylim=(0,1), xlim=(0, n_steps))
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_takes_correct_control.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_evidence_ratio_takes_correct_control.svg' %get_timestamp()), dpi=300, transparent=True)

# %% SLEEP PLOT - ALIGNED TO BEGINNING

sleep_propensity_models = np.exp(trajss_logits_taus_control[...,1])
sleep_propensity_models = np.nanmean(sleep_propensity_models, axis=2)
sleep_propensity_models = np.swapaxes(sleep_propensity_models, 0, 1)

# %% PLOT

os.makedirs(analysis_folder ,exist_ok = True)
fig = plot_evidence_ratios(sleep_propensity_models[indices_to_plot], 1 - test_taus[indices_to_plot], control_cmap, xlabel = 'Time since begining of episode', ylabel='Probability of Sleeping', ylim=(0,1), xlim=(0, n_steps))
fig.savefig(os.path.join(analysis_folder, '%s_sleep_propensity_control.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_sleep_propensity_control.svg' %get_timestamp()), dpi=300, transparent=True)

# %% PERTURBED APERNN TESTING SINGLE TAU

starting_taus = {'peek': 0, 'take': 0.25}
tau_task_options = Config(copy.deepcopy(task_options.__dict__))
tau_task_options.starting_taus = starting_taus

target_tau = 1

(ape_actionss, ape_logitss, _, _, ape_controlss, ape_pss, _), _ = perturbed_test(ape_config, ape_nn_options, tau_task_options, device, target_tau=target_tau)

# %% PERTURBED APERNN POLICY BARPLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = policy_barplot(ape_logitss, ape_pss, episode=3)
fig.savefig(os.path.join(ape_analysis_folder, '%s_perturbed_policy_barplot.png' %get_timestamp()))

# %% PEPRTURBED APERNN TESTING ALL VALUES

#(task_options.starting_taus['take'], 1 - task_options.starting_taus['take'], task_options.alphas['take'], rewardss.mean(), counter_frac_correct_takess.mean(), counter_peekss.mean(), counter_sleepss.mean())

### TODO: READ IN FROM FILE

test_taus = np.arange(0, 1.01, 0.125)

# test_taus = np.arange(0,1.01,0.125)
# target_tau = 3

# rewss_taus_ape = [] #list of lists that will store rews for diff tau values for APE
# counterss_peeks_taus_ape = []
# control_errss_taus_ape = []
# counterss_sleeps_taus_ape = []

# rewss_taus_perturbed_ape = [] #list of lists that will store rews for diff tau values for APE
# counterss_peeks_taus_perturbed_ape = []
# control_errss_taus_perturbed_ape = []
# counterss_sleeps_taus_perturbed_ape = []

# for test_tau in test_taus:

#     rews_taus_ape = []
#     counters_peeks_taus_ape = []
#     control_errs_taus_ape = []
#     counters_sleeps_taus_ape = []

#     rews_taus_perturbed_ape = []
#     counters_peeks_taus_perturbed_ape = []
#     control_errs_taus_perturbed_ape = []
#     counters_sleeps_taus_perturbed_ape = []

#     tau_task_options = Config(copy.deepcopy(task_options.__dict__))
#     tau_task_options.starting_taus = {'peek': 0, 'take': test_tau}

#     for ape_model in ape_models:

#         _, (rews_ape, _, counter_peeks_ape, counter_sleeps_ape, _, _, control_errs_ape,) = perturbed_test(ape_config, ape_nn_options, tau_task_options, device, model_folder = os.path.join(base_model_folder , str(ape_model)))
#         rews_taus_ape.append(rews_ape)
#         counters_peeks_taus_ape.append(counter_peeks_ape)
#         control_errs_taus_ape.append(control_errs_ape)
#         counters_sleeps_taus_ape.append(counter_sleeps_ape)

#         _, (rews_ape, _, counter_peeks_ape, counter_sleeps_ape, _, _, control_errs_ape,) = perturbed_test(ape_config, ape_nn_options, tau_task_options, device, model_folder = os.path.join(base_model_folder , str(ape_model)), target_tau = target_tau)
#         rews_taus_perturbed_ape.append(rews_ape)
#         counters_peeks_taus_perturbed_ape.append(counter_peeks_ape)
#         control_errs_taus_perturbed_ape.append(control_errs_ape)
#         counters_sleeps_taus_perturbed_ape.append(counter_sleeps_ape)

#     rewss_taus_ape.append(rews_taus_ape)
#     counterss_peeks_taus_ape.append(counters_peeks_taus_ape)
#     control_errss_taus_ape.append(control_errs_taus_ape)
#     counterss_sleeps_taus_ape.append(counters_sleeps_taus_ape)

#     rewss_taus_perturbed_ape.append(rews_taus_perturbed_ape)
#     counterss_peeks_taus_perturbed_ape.append(counters_peeks_taus_perturbed_ape)
#     control_errss_taus_perturbed_ape.append(control_errs_taus_perturbed_ape)
#     counterss_sleeps_taus_perturbed_ape.append(counters_sleeps_taus_perturbed_ape)

# %%

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(rewss_taus_ape)), test_taus, list(reversed(rewss_taus_perturbed_ape)), title='Performance for Different Efficacy Values', axis_xlabel='Efficacy', axis_ylabel='Rewards', label_exp='APE-trained', label_control='perturbed APE')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy_perturbed_tau%d.png' %(get_timestamp(), int(target_tau*100))))

# %% PANEL QUANTIFICATION EXPLORE-EXPLOIT

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(counterss_peeks_taus_ape)), test_taus, list(reversed(counterss_peeks_taus_perturbed_ape)), title='Balance Between Explore-Exploit', axis_xlabel='Efficacy', axis_ylabel='Number of Observes per Episode', label_exp='APE-trained', label_control='perturbed APE')

fig.savefig(os.path.join(analysis_folder, '%s_explore-exploit_ratio_perturbed_tau%d.png' %(get_timestamp(), int(target_tau*100))))

# %% PANEL QUANTIFICATION NUMBER OF SLEEPS

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(counterss_sleeps_taus_ape)), test_taus, list(reversed(counterss_sleeps_taus_perturbed_ape)), title='Time Spent Sleeping for Different Efficacy Values', axis_xlabel='Efficacy', axis_ylabel='Number of Sleeps per Episode', label_exp='APE-trained', label_control='perturbed APE')
fig.savefig(os.path.join(analysis_folder, '%s_sleeps_efficacy_perturbed_tau%d.png' %(get_timestamp(), int(target_tau*100))))

# %%
