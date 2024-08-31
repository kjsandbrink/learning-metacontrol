# Kai Sandbrink
# 2022-10-27
# This script completes the analyses the pure explore vs. pure exploit task (Task 1)

# %% LIBRARY IMPORTS

import torch
import pandas as pd
import numpy as np
from utils import Config, plot_learning_curve, plot_learning_curves_comparison, get_timestamp
from test_case import test, test_helplessness, perturbed_test, convert_control_to_prob_random

from datetime import datetime
import os, copy, pickle
import matplotlib as mpl

from test_analyses import policy_barplot, within_episode_efficacy_lineplot, plot_comparison_curves_several_runs, frac_takes_lineplot, frac_correct_takes_lineplot, ape_accuracy_lineplot

from utils_project import calculate_freq_observed_choice_per_t, calculate_freq_observes_per_t, calculate_freq_correct_choice_per_t, calculate_freq_correct_take
from test_analyses import plot_evidence_ratios

from scipy.stats import ttest_ind

# %% SPECIFY TEST CASES

from nns.settings_ana import pepe_nn_ape_models as ape_models
from nns.settings_ana import pepe_nn_control_models as control_models

# %% CONFIG FILES

# the assumption is all of these have the same config & task files,
# and nn_options are shared within a model type

ape_modelname = str(ape_models[0]) 
control_modelname = str(control_models[0])
base_results_folder = 'models'

ape_results_folder = os.path.join(base_results_folder,ape_modelname)
control_results_folder = os.path.join(base_results_folder, control_modelname)

config = Config({})
config.load_config_file(os.path.join(ape_results_folder, 'config.yaml'))

## config-specific options
config.n_repeats_case = 100

ape_config = Config(copy.deepcopy(config.__dict__))
ape_config.model_folder = os.path.join('models', ape_modelname)
control_config = Config(copy.deepcopy(config.__dict__))
control_config.model_folder = os.path.join('models', control_modelname)

task_options = Config({})
task_options.load_config_file(os.path.join(ape_results_folder, 'task_options.yaml'))

ape_nn_options = Config({})
ape_nn_options.load_config_file(os.path.join(ape_results_folder, 'nn_options.yaml'))

control_nn_options = Config({})
control_nn_options.load_config_file(os.path.join(control_results_folder, 'nn_options.yaml'))

# %% INITIALIZATIONS

## ANALYSIS FOLDER
analysis_folder = os.path.join('analysis', 'explore-exploit', 'human_task')
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

for ape_modelname in ape_models[0:]:
    ape_rewardss.append(np.load(os.path.join(base_results_folder, str(ape_modelname), 'logged_returns.npy')))

control_rewardss = []

for control_modelname in control_models[0:]:
    control_rewardss.append(np.load(os.path.join(base_results_folder, str(control_modelname), 'logged_returns.npy')))

#episodes = np.arange(0, len(ape_rewardss[0])*100, 100)
k_episodes = np.arange(0, len(ape_rewardss[0])*0.1, 0.1)

# %% PANEL LEARNING CURVES : PLOT CURVE WITH BOTH LCS

fig = plot_learning_curves_comparison(k_episodes, ape_rewardss, control_rewardss, smoothing_window=1000, name_exp="APE-trained")
#fig = plot_learning_curves_comparison(k_episodes, ape_rewardss, control_rewardss, smoothing_window=1, name_exp="APE-trained")
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves.svg' %get_timestamp()))

# %% STATS LEARNING CURVES ON TRAINING

ape_rewardss_mean_final_50 = np.array(ape_rewardss)[:,-50:].mean(axis=1)
control_rewardss_mean_final_50 = np.array(control_rewardss)[:,-50:].mean(axis=1)

print('ape rewards mean', ape_rewardss_mean_final_50.mean())
print('ape rewards std err', ape_rewardss_mean_final_50.std()/np.sqrt(n_models))

print('control rewards mean', control_rewardss_mean_final_50.mean())
print('control rewards std err', control_rewardss_mean_final_50.std()/np.sqrt(n_models))

t = ttest_ind(ape_rewardss_mean_final_50, control_rewardss_mean_final_50, alternative='greater')
print('ttest', t)

# %% PANEL LEARNING CURVES : PLOT CURVE WITH ONLY CONTROL LEARNING CURVES

fig = plot_learning_curves_comparison(k_episodes, None, control_rewardss, smoothing_window=1000, name_exp="APE-trained", axis_ylim = (21, 28))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_control_only.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_control_only.svg' %get_timestamp()))

# %% PANEL LEARNING CURVES ON TEST : LOAD DATA

eval_folder = os.path.join('data', 'eval', 'pepe')
#eval_timestamp = '20230430051946'
eval_timestamp = '20230707124801'

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

fig = plot_comparison_curves_several_runs(test_episodes, np.array(ape_test_rews).T, test_episodes, np.array(control_test_rews).T, title='Learning Curves (Test Reward)', axis_xlabel='Episodes', axis_ylabel="Total Reward per Episode", label_exp="APE-trained", label_control='no APE', x_units = 'k')
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test.png' %get_timestamp()))
#fig.savefig(os.path.join(analysis_folder, 'learning_curves_test.png'))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test.svg' %get_timestamp()))

# %% PANEL LEARNING CURVES ON TEST : PLOT PG LOSS

fig = plot_comparison_curves_several_runs(test_episodes, np.array(ape_test_returns_losses).T, test_episodes, np.array(control_test_returns_losses).T, title='Learning Curves (Policy Gradient Test Loss)', axis_xlabel='Episodes', axis_ylabel='Loss (Policy Gradient)', label_exp='APE-trained', label_control='no APE', x_units='k')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_pg_loss.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_pg_loss.svg' %get_timestamp()))
#fig.savefig(os.path.join(analysis_folder, 'learning_curves_test_pg_loss.png'))

# %% PANEL LEARNING CURVES ON TEST : PLOT APEs LOSS

fig = plot_comparison_curves_several_runs(test_episodes, np.array(ape_test_apes_losses).T, title='Learning Curves (APE Test Loss)', axis_xlabel='Episodes', axis_ylabel='Loss (APE)', label_exp='APE-trained', x_units='k')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

#fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_ape_loss.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_ape_loss.svg' %get_timestamp()))

# %% PANEL LEARNING CURVES ON TEST : PLOT APEs MSEs

fig = plot_comparison_curves_several_runs(test_episodes, np.array(ape_test_apes_mses).T, title='Learning Curves (APE Test Mean-Squared Error)', axis_xlabel='Episodes', axis_ylabel='MSE APEs', label_exp='APE-trained', x_units='k')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

#fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_ape_mse.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_learning_curves_test_ape_mse.svg' %get_timestamp()))

# %% TEST DATA STATS

ape_test_rews = np.array(ape_test_rews)
ape_rewards_final = np.array(ape_test_rews)[:,-1]

control_test_rews = np.array(control_test_rews)
control_rewards_final = np.array(control_test_rews)[:,-1]

print('ape rewards mean', ape_rewards_final.mean())
print('ape rewards std err', ape_rewards_final.std()/np.sqrt(len(ape_rewards_final)))

print('control rewards mean', control_rewards_final.mean())
print('control rewards std err', control_rewards_final.std()/np.sqrt(len(control_rewards_final)))

t = ttest_ind(ape_rewards_final, control_rewards_final, alternative='greater')
print('ttest', t)

# %% PANEL POLICY REPRESENTATION : SIMULATE TESTS

starting_taus = {'peek': 0, 'take': 1}
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

os.makedirs(ape_analysis_folder ,exist_ok = True)

tau = starting_taus['take']

fig = within_episode_efficacy_lineplot(convert_control_to_prob_random(ape_controlss), tau=tau, episode=4, )

fig.savefig(os.path.join(ape_analysis_folder, '%s_within_episode_efficacy_plot_tau%d.png' %(get_timestamp(), int(tau*100))))

# %% PANEL POLICY REP: SINGLE MODEL BUT MULTIPLE TAU VALUES AVERAGED - DATA

test_taus = np.arange(0,1.01,0.25)
tau_task_options = Config(copy.deepcopy(task_options.__dict__))
#n_repeats_case = 100
n_repeats_case = 500

## TO READ IN

ape_modelname = str(ape_models[0]) 
control_modelname = str(control_models[0])

traj_base = os.path.join('data', 'eval', 'pepe', )
traj_timestamp = '20231006143445' ## MATCHING 10/06/23

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


# %% PANEL POLICY SINGLE MODEL FRAC TAKES - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = frac_takes_lineplot(test_taus, ape_logitss_taus, ylim=(-0.02, 0.82))
fig.savefig(os.path.join(ape_analysis_folder, '%s_frac_takes_lineplot.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(ape_analysis_folder, '%s_frac_takes_lineplot.svg' %get_timestamp()), dpi=300, transparent=True)

os.makedirs(control_analysis_folder ,exist_ok = True)
fig = frac_takes_lineplot(test_taus, control_logitss_taus, ylim=(-0.02, 0.82))
fig.savefig(os.path.join(control_analysis_folder, '%s_frac_takes_lineplot.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(control_analysis_folder, '%s_frac_takes_lineplot.svg' %get_timestamp()), dpi=300, transparent=True)

# %% PANEL POLICY SINGLE MODEL PERC CORRECT TAKES - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = frac_correct_takes_lineplot(test_taus, ape_logitss_taus, ape_pss_taus, ylim=(0,1))
fig.savefig(os.path.join(ape_analysis_folder, '%s_corr_takes_lineplot.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(ape_analysis_folder, '%s_corr_takes_lineplot.svg' %get_timestamp()), dpi=300, transparent=True)

os.makedirs(control_analysis_folder ,exist_ok = True)
fig = frac_correct_takes_lineplot(test_taus, control_logitss_taus, control_pss_taus, ylim=(0,1))
fig.savefig(os.path.join(control_analysis_folder, '%s_corr_takes_lineplot.png'%get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(control_analysis_folder, '%s_corr_takes_lineplot.svg'%get_timestamp()), dpi=300, transparent=True)

# %% PANEL POLICY SINGLE MODEL ERROR IN APEs - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = ape_accuracy_lineplot(test_taus, ape_control_errs_taus)
fig.savefig(os.path.join(ape_analysis_folder, '%s_ape_accuracy_lineplot.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(ape_analysis_folder, '%s_ape_accuracy_lineplot.svg' %get_timestamp()), dpi=300, transparent=True)

# %% EVIDENCE RATIO CALCS

evidence_ratio_observations = []
evidence_ratio_takes_observed = []
evidence_ratio_takes_correct = []

for actionss, logitss, pss in zip(ape_actionss_taus, ape_logitss_taus, ape_pss_taus):
    evidence_ratio_observations.append(calculate_freq_observes_per_t(actionss, logitss, pss))
    evidence_ratio_takes_observed.append(calculate_freq_observed_choice_per_t(actionss, logitss, pss))
    evidence_ratio_takes_correct.append(calculate_freq_correct_choice_per_t(actionss, logitss, pss))
    
evidence_ratio_observations = np.stack(evidence_ratio_observations)
evidence_ratio_takes_observed = np.stack(evidence_ratio_takes_observed)
evidence_ratio_takes_correct = np.stack(evidence_ratio_takes_correct)

# ape_logitss_taus = np.stack(ape_logitss_taus)
# control_logitss_taus = np.stack(control_logitss_taus)
# evidence_ratio_observations = ape_logitss_taus[:,:,:,0].mean(axis=1)

# %% PANEL EVIDENCE RATIO - PLOT

indices_to_plot = [0, 1, 3, 4]

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = plot_evidence_ratios(evidence_ratio_observations[indices_to_plot], 1- test_taus[indices_to_plot], ape_cmap, ylabel='Probability of Observing', ylim=(0,1), xlim=(0, n_steps))
fig.savefig(os.path.join(ape_analysis_folder, '%s_evidence_ratio_observations.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(ape_analysis_folder, '%s_evidence_ratio_observations.svg' %get_timestamp()), dpi=300, transparent=True)

# %% 

indices_to_plot = [0, 1, 3, 4]

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = plot_evidence_ratios(evidence_ratio_takes_observed[indices_to_plot], 1 - test_taus[indices_to_plot], ape_cmap, ylabel='Probability of Intending Observed Arm', ylim=(0,1), xlim=(0, n_steps))
fig.savefig(os.path.join(ape_analysis_folder, '%s_evidence_ratio_takes_observed.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(ape_analysis_folder, '%s_evidence_ratio_takes_observed.svg' %get_timestamp()), dpi=300, transparent=True)

# %% 

indices_to_plot = [0, 1, 3, 4]

fig = plot_evidence_ratios(evidence_ratio_takes_correct[indices_to_plot], 1 - test_taus[indices_to_plot], ape_cmap, ylabel='Probability of Intending Correct Arm', ylim=(0,1), xlim=(0, n_steps))
fig.savefig(os.path.join(ape_analysis_folder, '%s_evidence_ratio_takes_correct.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(ape_analysis_folder, '%s_evidence_ratio_takes_correct.svg' %get_timestamp()), dpi=300, transparent=True)

# %% CONTROL MODELS

evidence_ratio_observations = []
evidence_ratio_takes_observed = []
evidence_ratio_takes_correct = []

for actionss, logitss, pss in zip(control_actionss_taus, control_logitss_taus, control_pss_taus):
    evidence_ratio_observations.append(calculate_freq_observes_per_t(actionss, logitss, pss))
    evidence_ratio_takes_observed.append(calculate_freq_observed_choice_per_t(actionss, logitss, pss))
    evidence_ratio_takes_correct.append(calculate_freq_correct_choice_per_t(actionss, logitss, pss))
    
evidence_ratio_observations = np.stack(evidence_ratio_observations)
evidence_ratio_takes_observed = np.stack(evidence_ratio_takes_observed)
evidence_ratio_takes_correct = np.stack(evidence_ratio_takes_correct)

# ape_logitss_taus = np.stack(ape_logitss_taus)
# control_logitss_taus = np.stack(control_logitss_taus)
# evidence_ratio_observations = ape_logitss_taus[:,:,:,0].mean(axis=1)

# %% PANEL EVIDENCE RATIO - PLOT

indices_to_plot = [0, 1, 3, 4]

os.makedirs(control_analysis_folder ,exist_ok = True)
fig = plot_evidence_ratios(evidence_ratio_observations[indices_to_plot], 1- test_taus[indices_to_plot], control_cmap, ylabel='Probability of Observing', ylim=(0,1), xlim=(0, evidence_ratio_observations.shape[2]))
fig.savefig(os.path.join(control_analysis_folder, '%s_evidence_ratio_observations.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(control_analysis_folder, '%s_evidence_ratio_observations.svg' %get_timestamp()), dpi=300, transparent=True)

# %% 

indices_to_plot = [0, 1, 3, 4]

os.makedirs(control_analysis_folder ,exist_ok = True)
fig = plot_evidence_ratios(evidence_ratio_takes_observed[indices_to_plot], 1 - test_taus[indices_to_plot], control_cmap, ylabel='Probability of Intending Observed Arm', ylim=(0,1), xlim=(0, evidence_ratio_observations.shape[2]))
fig.savefig(os.path.join(control_analysis_folder, '%s_evidence_ratio_takes_observed.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(control_analysis_folder, '%s_evidence_ratio_takes_observed.svg' %get_timestamp()), dpi=300, transparent=True)

# %% 

indices_to_plot = [0, 1, 3, 4]

fig = plot_evidence_ratios(evidence_ratio_takes_correct[indices_to_plot], 1 - test_taus[indices_to_plot], control_cmap, ylabel='Probability of Intending Correct Arm', ylim=(0,1), xlim=(0, evidence_ratio_observations.shape[2]))
fig.savefig(os.path.join(control_analysis_folder, '%s_evidence_ratio_takes_correct.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(control_analysis_folder, '%s_evidence_ratio_takes_correct.svg' %get_timestamp()), dpi=300, transparent=True)

# %% PANEL QUANTIFICATION OF REWARDS AND E-E LOAD DATA

traj_base = os.path.join('data', 'eval', 'pepe', )
traj_timestamp = '20231006143445' ## MATCHING 10/06/23

test_taus = np.arange(0,1.01,0.125)

rewss_taus_ape = [] #list of lists that will store rews for diff tau values for APE
counterss_peeks_taus_ape = []
control_errss_taus_ape = []

trajss_actions_taus_ape = []
trajss_logits_taus_ape = []
trajss_ps_taus_ape = []

rewss_taus_control = []
counterss_peeks_taus_control = []

trajss_actions_taus_control = []
trajss_logits_taus_control = []
trajss_ps_taus_control = []

for model in ape_models:

    orig_traj_folder = os.path.join(traj_base, str(model))

    control_errs_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_control_errs_taus_ape.pkl'), 'rb'))
    counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_counters_peeks_taus.pkl'), 'rb'))
    rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_rewss_taus.pkl'), 'rb'))

    trajs_actions_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_actionss_taus.pkl'), 'rb'))
    trajs_logits_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_logitss_taus.pkl'), 'rb'))
    trajs_ps_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_pss_taus.pkl'), 'rb'))

    control_errss_taus_ape.append(control_errs_taus_ape)
    counterss_peeks_taus_ape.append(counters_peeks_taus)
    rewss_taus_ape.append(rewss_taus)

    trajss_actions_taus_ape.append(trajs_actions_taus_ape)
    trajss_logits_taus_ape.append(trajs_logits_taus_ape)
    trajss_ps_taus_ape.append(trajs_ps_taus_ape)

for model in control_models:

    orig_traj_folder = os.path.join(traj_base, str(model))

    counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_counters_peeks_taus.pkl'), 'rb'))
    rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_rewss_taus.pkl'), 'rb'))

    trajs_actions_taus_control = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_actionss_taus.pkl'), 'rb'))
    trajs_logits_taus_control = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_logitss_taus.pkl'), 'rb'))
    trajs_ps_taus_control = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_pss_taus.pkl'), 'rb'))

    counterss_peeks_taus_control.append(counters_peeks_taus)
    rewss_taus_control.append(rewss_taus)

    trajss_actions_taus_control.append(trajs_actions_taus_control)
    trajss_logits_taus_control.append(trajs_logits_taus_control)
    trajss_ps_taus_control.append(trajs_ps_taus_control)

control_errss_taus_ape = np.array(control_errss_taus_ape).T
counterss_peeks_taus_ape = np.array(counterss_peeks_taus_ape).T
rewss_taus_ape = np.array(rewss_taus_ape).T

trajss_actions_taus_ape = np.array(trajss_actions_taus_ape)
trajss_logits_taus_ape = np.array(trajss_logits_taus_ape)
trajss_ps_taus_ape = np.array(trajss_ps_taus_ape)

counterss_peeks_taus_control = np.array(counterss_peeks_taus_control).T
rewss_taus_control = np.array(rewss_taus_control).T

trajss_actions_taus_control = np.array(trajss_actions_taus_control)
trajss_logits_taus_control = np.array(trajss_logits_taus_control)
trajss_ps_taus_control = np.array(trajss_ps_taus_control)

# %% CALCULATE AVERAGE NUMBER OF REWARDS AND OBSERVES, % TAKEN CORRECT PER EFFICACY LEVEL

print("EV of Effs", rewss_taus_ape.mean(axis=1), )
print("stderr EV of Effs", rewss_taus_ape.std(axis=1)/np.sqrt(n_models), )
print("Obs of Effs", counterss_peeks_taus_ape.mean(axis=1), )
print("stderr Obs of Effs", counterss_peeks_taus_ape.std(axis=1)/np.sqrt(n_models), )

correct_takes = calculate_freq_correct_take(trajss_logits_taus_ape, trajss_ps_taus_ape, include_sleep=False).mean(axis=(2, 3))
print("Expected correct take probability", correct_takes.mean(axis=0))
print("stderr", correct_takes.std(axis=0)/np.sqrt(n_models))

### FOR HUMAN-MATCHING DATASET
# EV of Effs [30.137 29.282 28.337 26.76  25.597 25.078 24.64  24.693 24.553]
# stderr EV of Effs [0.55636328 0.4134847  0.32582219 0.20409802 0.12408908 0.13327265
#  0.11044456 0.08972235 0.16780971]
# Obs of Effs [6.561 6.334 5.741 4.829 3.641 2.531 1.8   1.324 1.089]
# stderr Obs of Effs [0.80767004 0.78759025 0.73035122 0.65029909 0.56589124 0.38641545
#  0.2786862  0.2229852  0.13387644]
# Expected correct take probability [0.68109297 0.68260722 0.67462006 0.64070059 0.60635619 0.58156844
#  0.5455282  0.54777943 0.53220974]
# stderr [0.02063236 0.0176135  0.01929194 0.01756656 0.0139558  0.01069981
#  0.00754262 0.00725149 0.0067924 ]

# %% FOR CONTROL MODEL

print("EV of Effs", rewss_taus_control.mean(axis=1), )
print("stderr EV of Effs", rewss_taus_control.std(axis=1)/np.sqrt(n_models), )
print("Obs of Effs", counterss_peeks_taus_control.mean(axis=1), )
print("stderr Obs of Effs", counterss_peeks_taus_control.std(axis=1)/np.sqrt(n_models), )

correct_takes = calculate_freq_correct_take(trajss_logits_taus_control, trajss_ps_taus_control, include_sleep=False).mean(axis=(2, 3))
print("Expected correct take probability", correct_takes.mean(axis=0))
print("stderr", correct_takes.std(axis=0)/np.sqrt(n_models))

# EV of Effs [25.932 26.542 25.644 25.68  25.577 24.96  25.081 24.574 24.453]
# stderr EV of Effs [0.67923457 0.54275556 0.47197712 0.37860005 0.30084897 0.1523089
#  0.21404883 0.24092821 0.20654806]
# Obs of Effs [1.243 1.201 1.206 1.232 1.174 1.197 1.123 1.141 1.142]
# stderr Obs of Effs [0.58808171 0.56375784 0.55877938 0.55218258 0.53688956 0.53908079
#  0.50891856 0.51853341 0.50112833]
# Expected correct take probability [0.53184911 0.54825911 0.53155858 0.54346746 0.54686225 0.54018698
#  0.54959084 0.53630105 0.54202072]
# stderr [0.01929597 0.01875204 0.02015837 0.01751587 0.02030193 0.01888387
#  0.01479076 0.01712247 0.01732436]

# %% PANEL QUANTIFICATION OF REWARDS PLOT

#fig = plot_comparison_curves_several_runs(test_taus, list(reversed(rewss_taus_ape)), test_taus, list(reversed(rewss_taus_control)), baseline_tau_values, list(reversed(rewss_taus_baselines)), 'Performance for Different Efficacy Values', 'Efficacy', 'Rewards', 'APE-trained', 'no APE', 'Single Settings')
fig = plot_comparison_curves_several_runs(test_taus, list(reversed(rewss_taus_ape)), test_taus, list(reversed(rewss_taus_control)), axis_xlabel='Efficacy', axis_ylabel='Rewards', label_exp='APE-trained', label_control='no APE')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.svg' %get_timestamp()), dpi=300, transparent=True)

# %%

#fig = plot_comparison_curves_several_runs(test_taus, list(reversed(rewss_taus_ape)), test_taus, list(reversed(rewss_taus_control)), baseline_tau_values, list(reversed(rewss_taus_baselines)), 'Performance for Different Efficacy Values', 'Efficacy', 'Rewards', 'APE-trained', 'no APE', 'Single Settings')
fig = plot_comparison_curves_several_runs(test_taus, list(reversed(rewss_taus_ape)), axis_xlabel='Efficacy', axis_ylabel='Rewards', label_exp='APE-trained', ylim=(6, 46))
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy_ape.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy_ape.svg' %get_timestamp()), dpi=300, transparent=True)

# %%

#fig = plot_comparison_curves_several_runs(test_taus, list(reversed(rewss_taus_ape)), test_taus, list(reversed(rewss_taus_control)), baseline_tau_values, list(reversed(rewss_taus_baselines)), 'Performance for Different Efficacy Values', 'Efficacy', 'Rewards', 'APE-trained', 'no APE', 'Single Settings')
fig = plot_comparison_curves_several_runs(None, None, test_taus, list(reversed(rewss_taus_control)), axis_xlabel='Efficacy', axis_ylabel='Rewards', label_control='no APE', ylim=(6, 46))
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy_noape.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy_noape.svg' %get_timestamp()), dpi=300, transparent=True)

# %% PANEL QUANTIFICATION EXPLORE-EXPLOIT

#fig = plot_comparison_curves_several_runs(test_taus, list(reversed(counterss_peeks_taus_ape)), test_taus, list(reversed(counterss_peeks_taus_control)), baseline_tau_values, list(reversed(counterss_peeks_taus_baselines)), 'Balance Between Explore-Exploit', 'Efficacy', 'Number of Observes per Episode', 'APE-trained', 'no APE', 'Single Settings')
fig = plot_comparison_curves_several_runs(test_taus, list(reversed(counterss_peeks_taus_ape)), test_taus, list(reversed(counterss_peeks_taus_control)), axis_xlabel='Efficacy', axis_ylabel='Number of Observes per Episode', label_exp='APE-trained', label_control='no APE')

fig.savefig(os.path.join(analysis_folder, '%s_explore-exploit_ratio.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_explore-exploit_ratio.svg' %get_timestamp()), dpi=300, transparent=True)

# %% PANEL QUANTIFICATION EXPLORE-EXPLOIT

#fig = plot_comparison_curves_several_runs(test_taus, list(reversed(counterss_peeks_taus_ape)), test_taus, list(reversed(counterss_peeks_taus_control)), baseline_tau_values, list(reversed(counterss_peeks_taus_baselines)), 'Balance Between Explore-Exploit', 'Efficacy', 'Number of Observes per Episode', 'APE-trained', 'no APE', 'Single Settings')
fig = plot_comparison_curves_several_runs(test_taus, list(reversed(counterss_peeks_taus_ape)), axis_xlabel='Efficacy', axis_ylabel='Number of Observes per Episode', label_exp='APE-trained', label_control='no APE', ylim=(-1, 26))

fig.savefig(os.path.join(analysis_folder, '%s_explore-exploit_ratio_ape.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_explore-exploit_ratio_ape.svg' %get_timestamp()), dpi=300, transparent=True)


# %% PANEL QUANTIFICATION EXPLORE-EXPLOIT

#fig = plot_comparison_curves_several_runs(test_taus, list(reversed(counterss_peeks_taus_ape)), test_taus, list(reversed(counterss_peeks_taus_control)), baseline_tau_values, list(reversed(counterss_peeks_taus_baselines)), 'Balance Between Explore-Exploit', 'Efficacy', 'Number of Observes per Episode', 'APE-trained', 'no APE', 'Single Settings')
fig = plot_comparison_curves_several_runs(None, None, test_taus, list(reversed(counterss_peeks_taus_control)), axis_xlabel='Efficacy', axis_ylabel='Number of Observes per Episode', label_exp='APE-trained', label_control='no APE', ylim=(-1, 26))

fig.savefig(os.path.join(analysis_folder, '%s_explore-exploit_ratio_noape.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_explore-exploit_ratio_noape.svg' %get_timestamp()), dpi=300, transparent=True)

# %% PANEL QUANTIFICATION ERROR APE

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(control_errss_taus_ape)), title="Error in Efficacy Estimation", axis_xlabel= 'Efficacy', axis_ylabel='MSE APEs', label_exp='APE-trained')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_mse_apes.png' %get_timestamp()), dpi=300, transparent=True)
fig.savefig(os.path.join(analysis_folder, '%s_mse_apes.svg' %get_timestamp()), dpi=300, transparent=True)

# %% EVIDENCE RATIO CALCULATIONS

indices_to_plot = [0, 2, 6, 8]

evidence_ratio_observations_ape_models = []
evidence_ratio_takes_observed_ape_models = []
evidence_ratio_takes_correct_ape_models = []

for ape_actionss_taus, ape_logitss_taus, ape_pss_taus in zip(trajss_actions_taus_ape, trajss_logits_taus_ape, trajss_ps_taus_ape):

    evidence_ratio_observations = []
    evidence_ratio_takes_observed = []
    evidence_ratio_takes_correct = []

    for actionss, logitss, pss in zip(ape_actionss_taus, ape_logitss_taus, ape_pss_taus):
        evidence_ratio_observations.append(calculate_freq_observes_per_t(actionss, logitss, pss))
        evidence_ratio_takes_observed.append(calculate_freq_observed_choice_per_t(actionss, logitss, pss))
        evidence_ratio_takes_correct.append(calculate_freq_correct_choice_per_t(actionss, logitss, pss))
        
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
        evidence_ratio_observations.append(calculate_freq_observes_per_t(actionss, logitss, pss))
        evidence_ratio_takes_observed.append(calculate_freq_observed_choice_per_t(actionss, logitss, pss))
        evidence_ratio_takes_correct.append(calculate_freq_correct_choice_per_t(actionss, logitss, pss))
        
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

# %%  STATS
rewss_taus_ape = np.array(rewss_taus_ape)
counterss_peeks_taus_ape = np.array(counterss_peeks_taus_ape)
control_errs_taus_ape = np.array(control_errs_taus_ape)

rewss_taus_control = np.array(rewss_taus_control)
counterss_peeks_taus_control = np.array(counterss_peeks_taus_control)

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

test_taus = np.arange(0,1.01,0.125)
target_tau = 3

rewss_taus_ape = [] #list of lists that will store rews for diff tau values for APE
counterss_peeks_taus_ape = []
control_errss_taus_ape = []

rewss_taus_perturbed_ape = [] #list of lists that will store rews for diff tau values for APE
counterss_peeks_taus_perturbed_ape = []
control_errss_taus_perturbed_ape = []

for test_tau in test_taus:

    rews_taus_ape = []
    counters_peeks_taus_ape = []
    control_errs_taus_ape = []

    rews_taus_perturbed_ape = []
    counters_peeks_taus_perturbed_ape = []
    control_errs_taus_perturbed_ape = []

    tau_task_options = Config(copy.deepcopy(task_options.__dict__))
    tau_task_options.starting_taus = {'peek': 0, 'take': test_tau}

    for ape_model in ape_models:

        _, (rews_ape, _, counter_peeks_ape, _, _, _, control_errs_ape,) = perturbed_test(ape_config, ape_nn_options, tau_task_options, device, model_folder = os.path.join(base_results_folder , str(ape_model)))
        rews_taus_ape.append(rews_ape)
        counters_peeks_taus_ape.append(counter_peeks_ape)
        control_errs_taus_ape.append(control_errs_ape)

        _, (rews_ape, _, counter_peeks_ape, _, _, _, control_errs_ape,) = perturbed_test(ape_config, ape_nn_options, tau_task_options, device, model_folder = os.path.join(base_results_folder , str(ape_model)), target_tau = target_tau)
        rews_taus_perturbed_ape.append(rews_ape)
        counters_peeks_taus_perturbed_ape.append(counter_peeks_ape)
        control_errs_taus_perturbed_ape.append(control_errs_ape)

    rewss_taus_ape.append(rews_taus_ape)
    counterss_peeks_taus_ape.append(counters_peeks_taus_ape)
    control_errss_taus_ape.append(control_errs_taus_ape)

    rewss_taus_perturbed_ape.append(rews_taus_perturbed_ape)
    counterss_peeks_taus_perturbed_ape.append(counters_peeks_taus_perturbed_ape)
    control_errss_taus_perturbed_ape.append(control_errs_taus_perturbed_ape)

# %%

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(rewss_taus_ape)), test_taus, list(reversed(rewss_taus_perturbed_ape)), title='Performance for Different Efficacy Values', axis_xlabel='Efficacy', axis_ylabel='Rewards', label_exp='APE-trained', label_control='perturbed APE')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy_perturbed_tau%d.png' %(get_timestamp(), int(target_tau*100))))

# %% PANEL QUANTIFICATION EXPLORE-EXPLOIT

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(counterss_peeks_taus_ape)), test_taus, list(reversed(counterss_peeks_taus_perturbed_ape)), title='Balance Between Explore-Exploit', axis_xlabel='Efficacy', axis_ylabel='Number of Observes per Episode', label_exp='APE-trained', label_control='perturbed APE')

fig.savefig(os.path.join(analysis_folder, '%s_explore-exploit_ratio_perturbed_tau%d.png' %(get_timestamp(), int(target_tau*100))))

# %% L2 REGULARIZATION MODEL ON APE

ape_model_readout_l2reg = 20230128195013
ape_readout_l2reg_analysis_folder = os.path.join('analysis', str(ape_model_readout_l2reg))

# %% RUN TEST CASES

starting_taus = {'peek': 0, 'take': 0.8}
tau_task_options = Config(copy.deepcopy(task_options.__dict__))
tau_task_options.starting_taus = starting_taus

(ape_actionss, ape_logitss, _, _, ape_controlss, ape_pss, _), _ = test(config, ape_nn_options, tau_task_options, device, model_folder=os.path.join(base_results_folder, str(ape_model_readout_l2reg)))

# %% PANEL POLICY REP: BAR PLOT (for single episode)

os.makedirs(ape_readout_l2reg_analysis_folder ,exist_ok = True)
fig = policy_barplot(ape_logitss, ape_pss, episode=9)
fig.savefig(os.path.join(ape_readout_l2reg_analysis_folder, '%s_policy_barplot_ape.png' %get_timestamp()))

# %% LINE PLOT FOR CONTROLS OVER INDIVIDUAL POLICY

os.makedirs(ape_analysis_folder ,exist_ok = True)

tau = starting_taus['take']

fig = within_episode_efficacy_lineplot(convert_control_to_prob_random(ape_controlss), tau=tau, episode=9, )

fig.savefig(os.path.join(ape_readout_l2reg_analysis_folder, '%s_within_episode_efficacy_plot_tau%d.png' %(get_timestamp(), int(tau*100))))

# %% SINGLE MODEL TEST DATA

test_taus = np.arange(0,1.01,0.125)
tau_task_options = Config(copy.deepcopy(task_options.__dict__))
n_repeats_case = 100

ape_actionss_taus = []
ape_logitss_taus = []
ape_controlss_taus = []
ape_pss_taus = []
ape_control_errs_taus = []

control_actionss_taus = []
control_logitss_taus = []
control_pss_taus = []

ape_rews = []

for test_tau in test_taus:

    tau_task_options.starting_taus = {'peek': 0, 'take': test_tau}

    (ape_actionss, ape_logitss, _, _, ape_controlss, ape_pss, ape_control_errss), (rews, _, _, _, _, _, _) = test(config, ape_nn_options, tau_task_options, device, model_folder=os.path.join(base_results_folder, str(ape_model_readout_l2reg)),  n_repeats_case=n_repeats_case)

    ape_actionss_taus.append(ape_actionss)
    ape_logitss_taus.append(ape_logitss)
    ape_controlss_taus.append(ape_controlss)
    ape_pss_taus.append(ape_pss)
    ape_control_errs_taus.append(ape_control_errss)

    ape_rews.append(rews)

# %% PANEL POLICY SINGLE MODEL FRAC TAKES - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = frac_takes_lineplot(test_taus, ape_logitss_taus)
fig.savefig(os.path.join(ape_readout_l2reg_analysis_folder, '%s_frac_takes_lineplot.png' %get_timestamp()))

# %% PANEL POLICY SINGLE MODEL PERC CORRECT TAKES - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = frac_correct_takes_lineplot(test_taus, ape_logitss_taus, ape_pss_taus)
fig.savefig(os.path.join(ape_readout_l2reg_analysis_folder, '%s_corr_takes_lineplot.png' %get_timestamp()))

# %% PANEL POLICY SINGLE MODEL ERROR IN APEs - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = ape_accuracy_lineplot(test_taus, ape_control_errs_taus)
fig.savefig(os.path.join(ape_readout_l2reg_analysis_folder, '%s_ape_accuracy_lineplot.png' %get_timestamp()))

# %% PLOT AS BASELINE IN REWARDS PLOT FOR COMPARISON

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(rewss_taus_ape)), test_taus, list(reversed(rewss_taus_control)), test_taus, list(reversed(ape_rews)), 'Performance for Different Efficacy Values', 'Efficacy', 'Rewards', 'APE-trained', 'no APE', 'APE L2 Reg')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy_apel2reg_comp.png' %get_timestamp()))

# %% WEIGHT DECAY MODEL

ape_model_weight_decay = 20230128193949
ape_weight_decay_analysis_folder = os.path.join('analysis', str(ape_model_weight_decay))

# %% RUN TEST CASES

starting_taus = {'peek': 0, 'take': 0.5}
tau_task_options = Config(copy.deepcopy(task_options.__dict__))
tau_task_options.starting_taus = starting_taus

(ape_actionss, ape_logitss, _, _, ape_controlss, ape_pss, _), _ = test(config, ape_nn_options, tau_task_options, device, model_folder=os.path.join(base_results_folder, str(ape_model_weight_decay)))

# %% PANEL POLICY REP: BAR PLOT (for single episode)

os.makedirs(ape_weight_decay_analysis_folder ,exist_ok = True)
fig = policy_barplot(ape_logitss, ape_pss, episode=9)
fig.savefig(os.path.join(ape_weight_decay_analysis_folder, '%s_policy_barplot_ape.png' %get_timestamp()))

# %% LINE PLOT FOR CONTROLS OVER INDIVIDUAL POLICY

os.makedirs(ape_analysis_folder ,exist_ok = True)

tau = starting_taus['take']

fig = within_episode_efficacy_lineplot(convert_control_to_prob_random(ape_controlss), tau=tau, episode=9, )

fig.savefig(os.path.join(ape_weight_decay_analysis_folder, '%s_within_episode_efficacy_plot_tau%d.png' %(get_timestamp(), int(tau*100))))

# %%

test_taus = np.arange(0,1.01,0.125)
tau_task_options = Config(copy.deepcopy(task_options.__dict__))
n_repeats_case = 100

ape_actionss_taus = []
ape_logitss_taus = []
ape_controlss_taus = []
ape_pss_taus = []
ape_control_errs_taus = []

control_actionss_taus = []
ape_rews_weight_decay = []

for test_tau in test_taus:

    tau_task_options.starting_taus = {'peek': 0, 'take': test_tau}

    (ape_actionss, ape_logitss, _, _, ape_controlss, ape_pss, ape_control_errss), (rews, _, _, _, _, _, _,) = test(config, ape_nn_options, tau_task_options, device, model_folder=os.path.join(base_results_folder, str(ape_model_weight_decay)),  n_repeats_case=n_repeats_case)

    ape_actionss_taus.append(ape_actionss)
    ape_logitss_taus.append(ape_logitss)
    ape_controlss_taus.append(ape_controlss)
    ape_pss_taus.append(ape_pss)
    ape_control_errs_taus.append(ape_control_errss)

    ape_rews_weight_decay.append(rews)

# %% PANEL POLICY SINGLE MODEL FRAC TAKES - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = frac_takes_lineplot(test_taus, ape_logitss_taus)
fig.savefig(os.path.join(ape_weight_decay_analysis_folder, '%s_frac_takes_lineplot.png' %get_timestamp()))

# %% PANEL POLICY SINGLE MODEL PERC CORRECT TAKES - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = frac_correct_takes_lineplot(test_taus, ape_logitss_taus, ape_pss_taus)
fig.savefig(os.path.join(ape_weight_decay_analysis_folder, '%s_corr_takes_lineplot.png' %get_timestamp()))

# %% PANEL POLICY SINGLE MODEL ERROR IN APEs - PLOT

os.makedirs(ape_analysis_folder ,exist_ok = True)
fig = ape_accuracy_lineplot(test_taus, ape_control_errs_taus)
fig.savefig(os.path.join(ape_weight_decay_analysis_folder, '%s_ape_accuracy_lineplot.png' %get_timestamp()))

os.makedirs(ape_analysis_folder ,exist_ok = True)

tau = starting_taus['take']

fig = within_episode_efficacy_lineplot(convert_control_to_prob_random(ape_controlss), tau=tau, episode=9, )

fig.savefig(os.path.join(ape_weight_decay_analysis_folder, '%s_within_episode_efficacy_plot_tau%d.png' %(get_timestamp(), int(tau*100))))

# %% PLOT AS BASELINE IN REWARDS PLOT FOR COMPARISON

fig = plot_comparison_curves_several_runs(test_taus, list(reversed(rewss_taus_ape)), test_taus, list(reversed(rewss_taus_control)), test_taus, list(reversed(ape_rews_weight_decay)), 'Performance for Different Efficacy Values', 'Efficacy', 'Rewards', 'APE-trained', 'no APE', 'Weight Decay')
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy_weight_decay_comp.png' %get_timestamp()))
6