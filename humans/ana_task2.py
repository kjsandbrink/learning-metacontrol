# Kai Sandbrink
# 2023-06-23
# This script analyzes the task 1 behavior for a compiled DF or responses

# %% LIBRARY IMPORT

import numpy as np
import pandas as pd
import scipy.fft
import os, pickle

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from human_utils_project import sort_train_test, plot_train_test_comp, calculate_freq_sleeps
from human_utils_project import get_clean_data, calculate_freq_observed_choice_per_t, calculate_freq_observes_per_t, calculate_freq_correct_choice_per_t, plot_evidence_ratios
from human_utils_project import calculate_nns_freq_observed_choice_per_t, calculate_nns_freq_observes_per_t, calculate_nns_freq_sleeps_per_t

from human_plot_traj_analyses import plot_violin, plot_line_scatter, plot_line_scatter_humans_ape_noape, plot_line_scatter_humans_ape_noape_group, plot_line_scatter_group

from utils import format_axis, get_timestamp

from human_utils_behavioral_analysis import compute_2D_correlation, compute_2D_correlation_matrices
from human_utils_behavioral_analysis import get_factor_analysis_details, compute_similarity, bootstrap_similarity
from human_utils_behavioral_analysis import load_simulated_participants_across_models, upper_tri_masking, competitive_corr_regression, competitive_ridge_corr_regression

import statsmodels.api as sm
import statsmodels.formula.api as smf

# %% PARAMETERS

#day1_test_mask_cutoff = 10
day1_test_mask_cutoff = {
    "groupA": {"lower": 10, "upper": 90},
    "groupB": {"lower": 8, "upper": 72}
}

day = 'day3'
#exp_date = '619-706'
#exp_date = '518-525-619-706'
#exp_date = '12-11'
exp_date = '24-01-22-29'

#group = 'groupB'
group = None

df, effs_train, effs_test, test_start = get_clean_data(day=3, exp_date=exp_date, day1_test_mask_cutoff=day1_test_mask_cutoff, group=group)

effs = np.arange(0, 1.01, 0.125)
n_steps = 50

analysis_folder = os.path.join('analysis', 'traj_diff_efficacy', day, exp_date)

if day1_test_mask_cutoff is not None and type(day1_test_mask_cutoff) is int:
    analysis_folder = os.path.join(analysis_folder, 'day1_test_cutoff_%i' %day1_test_mask_cutoff)
elif day1_test_mask_cutoff is not None and type(day1_test_mask_cutoff) is dict:
    analysis_folder = os.path.join(analysis_folder, 'day1_test_cutoff_%i' %(day1_test_mask_cutoff["groupA"]['lower']))

if group is not None:
    analysis_folder = os.path.join(analysis_folder, group)

os.makedirs(analysis_folder, exist_ok=True)

cmap_train = mpl.colormaps['Greens']
cmap_test = mpl.colormaps['Blues']

cmaps = {
    'train': cmap_train,
    'test': cmap_test,
}
# %% NEURAL NETWORK ANALYSIS

### NO HOLDOUT BIAS 0.5, VOL 0.1, 250k ENTROPY ANNEALING
## 12/11
ape_models = [
    20231017233748,
    20231017233746,
    20231017233744,
    20231017233742,
    20231017233741,
    20231017233738,
    20231017233737,
    20231017233735,
    20231017233734,
    20231017233732
]

control_models = [
    20231023003506,
    20231023003504,
    20231023003503,
    20231023003501,
    20231023003500, 
    20231023003457,
    20231023003456,
    20231023003454,
    20231023003453,
    20231023003451
]


ape_cmap = mpl.colormaps['Purples']
control_cmap = mpl.colormaps['Oranges']

# %% CALCULATE TIME BETWEEN OBSERVATIONS

def calculate_time_between_actions(traj, action=0.5):
    ''' Calculate the time between observations 
    
    Arguments
    ---------
    traj : np.array, trajectory with 0.5 corresponding to an observation

    Returns
    -------
    list, time between observations in the trajectory
    '''
    
    # Get indices of observations
    obs_idx = np.where(traj == action)[0]

    # Calculate time between observations
    time_between_obs = np.diff(obs_idx)

    return time_between_obs

# Calculate time between observations for all trajectories
time_between_obss = df['transitions_ep_rightwrong'].apply(lambda x : [calculate_time_between_actions(traj) for traj in x])

# %% PLOT HISTOGRAM OF TIMES

# Flatten array
intervals = [np.concatenate(time_between_obs) for time_between_obs in time_between_obss]
intervals = np.concatenate(intervals)

sns.histplot(intervals)

# %% PLOT HISTOGRAM OF MEANS
meanss = time_between_obss.apply(lambda x : [np.mean(time_between_obs) for time_between_obs in x])

means = np.concatenate(meanss)

min_val = np.floor(np.nanmin(means))
max_val = np.ceil(np.nanmax(means))

# generate bin edges from min_val to max_val
bins = np.arange(min_val, max_val + 1, 1) 

sns.histplot(means, bins=bins)

# %% SPLIT MEANS INTO TWO GROUPS
meanss = time_between_obss.apply(lambda x : [np.mean(time_between_obs) for time_between_obs in x])

meanss_train, meanss_test = sort_train_test(meanss, df['effs'], test_start)

fig = plot_violin(meanss_test, effs_test, meanss_train, effs_train, yjitter=0, median_over_mean=True)
fig.savefig(os.path.join(analysis_folder, '%s_intervals_means_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_intervals_means_violin.svg' %get_timestamp()))

# %% CALCULATE STD FOR EACH PARTICIPANT

stdss = time_between_obss.apply(lambda x : [np.std(time_between_obs) for time_between_obs in x])

# %% VIOLIN PLOTS OF STD

stdss_train, stdss_test = sort_train_test(stdss, df['effs'], test_start)

fig = plot_violin(stdss_test, effs_test, stdss_train, effs_train, ylabel='Standard Deviation per Participant', yjitter=False, median_over_mean = True)

fig.savefig(os.path.join(analysis_folder, '%s_intervals_std_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_intervals_std_violin.svg' %get_timestamp()))

# %% HISTOGRAM OF STD

stdss = np.concatenate(stdss)

min_val = np.floor(np.nanmin(stdss))
max_val = np.ceil(np.nanmax(stdss))

# generate bin edges from min_val to max_val
bins = np.arange(min_val, max_val + 1, 1) 

sns.histplot(stdss, bins=bins)

# %% ASSIGN EXPECTED VALUES

## 10/08
# EV of Effs [30.524 29.69  29.074 27.747 26.416 25.185 24.81  24.296 23.756]
# stderr EV of Effs [0.31885169 0.20487557 0.13855107 0.23871343 0.14069968 0.11478894
#  0.11116654 0.143779   0.15853202]
# Obs of Effs [8.687 8.31  7.736 6.944 5.938 4.989 4.074 3.387 2.774]
# stderr Obs of Effs [0.40396547 0.43047416 0.48587076 0.55212354 0.63212625 0.70667737
#  0.72862089 0.69987577 0.62862103]
# Sleeps of Effs [0.993 0.975 0.961 1.009 1.002 0.99  1.047 1.    1.005]
# stderr Obs of Effs [0.31270769 0.30560514 0.30935562 0.30521615 0.31302971 0.29990665
#  0.31413071 0.31657858 0.3143032 ]
# Expected correct take probability [0.72725754 0.71611237 0.7133871  0.69412615 0.66972148 0.6427785
# 0.63022032 0.60111341 0.59176638]
# stderr [0.00603112 0.00764887 0.00900832 0.01148982 0.01454364 0.01576637
#  0.01608729 0.01830126 0.01864268]
# erews = np.flip([30.524, 29.69,  29.074, 27.747, 26.416, 25.185, 24.81,  24.296, 23.756])
# eobs = np.flip([8.687, 8.31,  7.736, 6.944, 5.938, 4.989, 4.074, 3.387, 2.774])
# esleeps = np.flip([0.993, 0.975, 0.961, 1.009, 1.002, 0.99,  1.047, 1,    1.005])
# ecorr = np.flip([0.72725754, 0.71611237, 0.7133871,  0.69412615, 0.66972148, 0.6427785, 0.63022032, 0.60111341, 0.59176638])

## 10/13
# erews = np.flip([29.018, 29.151, 28.507, 27.885, 26.787, 26.056, 25.08,  23.99,  23.308])
# eobs = np.flip([7.08,  7.045, 6.945, 6.722, 6.308, 6.014, 5.578, 5.117, 4.772])
# esleeps = np.flip([2.425, 2.469, 2.496, 2.53,  2.525, 2.515, 2.657, 2.594, 2.684])
# ecorr = np.flip([0.69129279, 0.69593419, 0.68788904, 0.6827596,  0.67559874, 0.67257292, 0.66747279, 0.65227933, 0.64441181])

## 12/15
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

## 12/15
erews = np.flip([32.643, 33.041, 32.39,  31.817, 30.945, 29.667, 28.225, 26.841, 25.392])
stderrrews = np.flip([0.39998512, 0.34673751, 0.42611266, 0.37041882, 0.26326128, 0.21606041, 0.21752816, 0.24142266, 0.21831537])
eobs = np.flip([5.531, 5.507, 5.492, 5.488, 5.488, 5.374, 5.42,  5.424, 5.402])
stderrobs = np.flip([0.42083833, 0.43484032, 0.42683205, 0.41063804, 0.39425068, 0.36210827, 0.33602976, 0.31757897, 0.31086267])
esleeps = np.flip([3.255, 3.229, 3.339, 3.639, 4.099, 4.694, 5.349, 5.964, 6.46])
stderrsleeps = np.flip([0.29779271, 0.30401464, 0.27418406, 0.22055135, 0.21227553, 0.28934478, 0.34460543, 0.41471002, 0.49888275])
ecorr = np.flip([0.75906872, 0.76751221, 0.75838994, 0.75989741, 0.76564176, 0.76862511, 0.77071872, 0.76483193, 0.76521506])
stderrcorr = np.flip([0.01255626, 0.01133072, 0.01309236, 0.01303136, 0.00941374, 0.01018969, 0.00945282, 0.00853843, 0.00714491])

### 12/11 250k ENTROPY -- CONTROL
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

erews_noape = np.flip([31.32,  31.698, 31.427, 30.862, 30.291, 29.349, 27.807, 26.772, 25.382])
stderrrews_noape = np.flip([0.65016921, 0.6914359,  0.7366601,  0.99366473, 0.77479346, 0.5664635, 0.5397982,  0.32062065, 0.21423725])
eobs_noape = np.flip([4.372, 4.379, 4.339, 4.362, 4.38,  4.41,  4.394, 4.392, 4.332])
stderrobs_noape = np.flip([0.60818714, 0.63209564, 0.60664396, 0.63751204, 0.63482911, 0.61538931, 0.62743318, 0.62311604, 0.64019966])
esleeps_noape = np.flip([3.525, 3.516, 3.545, 3.492, 3.517, 3.501, 3.495, 3.481, 3.491])
stderrsleeps_noape = np.flip([0.53033621, 0.52223596, 0.53152469, 0.52892684, 0.5245723,  0.52519415, 0.52089778, 0.51814178, 0.51786185])
ecorr_noape = np.flip([0.71774926, 0.72848007, 0.72357564, 0.71570389, 0.72064683, 0.72959154, 0.71722002, 0.7243689,  0.72335611])
stderrcorr_noape = np.flip([0.02505477, 0.02628196, 0.02656803, 0.03321213, 0.02594733, 0.02654767, 0.0275857,  0.0280971,  0.02719329])


# %% NUMBER OF REWARDS

# n_train, n_test = sort_train_test(df['rewards_tallies'], df['effs'], test_start)
# fig = plot_violin(n_test, effs_test, n_train, effs_train, ylabel='Number of Rewards per Participant', median_over_mean = True)
# fig.savefig(os.path.join(analysis_folder, '%s_nrews_violin.png' %get_timestamp()))
# fig.savefig(os.path.join(analysis_folder, '%s_nrews_violin.svg' %get_timestamp()))

# # %% 

# n_train, n_test = sort_train_test(df['intended_correct'], df['effs'], test_start)
# fig = plot_violin(n_test, effs_test, n_train, effs_train, ylabel='Probability Intended Correct Bet', median_over_mean = True)
# fig.savefig(os.path.join(analysis_folder, '%s_pc_violin.png' %get_timestamp()))
# fig.savefig(os.path.join(analysis_folder, '%s_pc_violin.svg' %get_timestamp()))

# %%

n_train, n_test = sort_train_test(df['rewards_tallies'], df['effs'], test_start)
fig = plot_line_scatter(n_test, effs_test, n_train, effs_train, ylabel='Number of Rewards', xjitter=0.025, yjitter=0.5, median_over_mean=True, true_values=erews, effs_true=effs, true_label='APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_rews_line_scatter.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_rews_line_scatter.svg' %get_timestamp()))

# %% 

if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['rewards_tallies'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['rewards_tallies'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Number of Rewards', xjitter=0.025, yjitter=0.5, median_over_mean=True,)
    fig.savefig(os.path.join(analysis_folder, '%s_rews_line_scatter_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_rews_line_scatter_bothgroups.svg' %get_timestamp()))

# %% 

n_train, n_test = sort_train_test(df['rewards_tallies'], df['effs'], test_start)
fig = plot_line_scatter_humans_ape_noape(n_test, effs_test, n_train, effs_train, ylabel='Number of Rewards', ylim=(20, 40), xjitter=0.025, yjitter=1, median_over_mean=True, true_values=erews, true_stderr=stderrrews, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=erews_noape, noape_stderr = stderrrews_noape, effs_noape=effs, noape_label='No APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_nrews_line_scatter_ape_noape.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_nrews_line_scatter_ape_noape.svg' %get_timestamp()))

# %% FOR BOTH GROUPS

if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['rewards_tallies'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['rewards_tallies'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_humans_ape_noape_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Number of Rewards', ylim=(15, 40), xjitter=0.025, yjitter=1, median_over_mean=True, true_values=erews, true_stderr=stderrrews, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=erews_noape, noape_stderr = stderrrews_noape, effs_noape=effs, noape_label='No APE-NNs')
    fig.savefig(os.path.join(analysis_folder, '%s_nrews_line_scatter_ape_noape_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_nrews_line_scatter_ape_noape_bothgroups.svg' %get_timestamp()))


# %% NUMBER OF OBSERVES

# nobs_train, nobs_test = sort_train_test(df['n_observes'], df['effs'], test_start)
# fig = plot_violin(nobs_test, effs_test, nobs_train, effs_train, ylabel='Number of Observes per Participant', median_over_mean = True)
# fig.savefig(os.path.join(analysis_folder, '%s_nobs_violin.png' %get_timestamp()))
# fig.savefig(os.path.join(analysis_folder, '%s_nobs_violin.svg' %get_timestamp()))

# %% 

nobs_train, nobs_test = sort_train_test(df['n_observes'], df['effs'], test_start)

fig = plot_line_scatter(nobs_test, effs_test, nobs_train, effs_train, ylabel='Number of Observes', xjitter=0.025, yjitter=0.5, ylim=(-2, 16), median_over_mean=True, true_values=eobs, effs_true=effs, true_label='APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter.svg' %get_timestamp()))

# %% 

if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['n_observes'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['n_observes'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Number of Observes', xjitter=0.025, yjitter=0.5, median_over_mean=True)
    fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter_bothgroups.svg' %get_timestamp()))

# %%

nobs_train, nobs_test = sort_train_test(df['n_observes'], df['effs'], test_start)
fig = plot_line_scatter_humans_ape_noape(nobs_test, effs_test, nobs_train, effs_train, ylabel='Number of Observes', xjitter=0.025, yjitter=1, ylim=(-1, 16), median_over_mean=True, true_values=eobs, true_stderr=stderrobs, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=eobs_noape, noape_stderr = stderrobs_noape, effs_noape=effs, noape_label='No APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter_ape_noape.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter_ape_noape.svg' %get_timestamp()))

# %% FOR BOTH GROUPS

if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['n_observes'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['n_observes'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_humans_ape_noape_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Number of Observes', ylim=(-1, 16), xjitter=0.025, yjitter=1, median_over_mean=True, true_values=eobs, true_stderr=stderrobs, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=eobs_noape, noape_stderr = stderrobs_noape, effs_noape=effs, noape_label='No APE-NNs')
    fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter_ape_noape_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter_ape_noape_bothgroups.svg' %get_timestamp()))


# %%

n_train, n_test = sort_train_test(df['intended_correct'], df['effs'], test_start)
fig = plot_line_scatter(n_test, effs_test, n_train, effs_train, ylabel='Probability Intended Correct Bet', xjitter=0.025, yjitter=0.01, median_over_mean=True, true_values=ecorr, effs_true=effs, true_label='APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter.svg' %get_timestamp()))

# %% 

if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['intended_correct'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['intended_correct'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Probability Intended Correct Bet', xjitter=0.025, yjitter=0.01, median_over_mean=True)
    fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_bothgroups.svg' %get_timestamp()))

# %%


n_train, n_test = sort_train_test(df['intended_correct'], df['effs'], test_start)
fig = plot_line_scatter_humans_ape_noape(n_test, effs_test, n_train, effs_train, ylabel='Probability Intended Correct', ylim=(0, 1), xjitter=0.025, median_over_mean=True, true_values=ecorr, true_stderr=stderrcorr, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=ecorr_noape, noape_stderr = stderrcorr_noape, effs_noape=effs, noape_label='No APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_ape_noape.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_ape_noape.svg' %get_timestamp()))

# %% FOR BOTH GROUPS


if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['intended_correct'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['intended_correct'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_humans_ape_noape_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Probability Intended Correct', ylim=(0, 1), xjitter=0.025, yjitter=0, median_over_mean=True, true_values=ecorr, true_stderr=stderrcorr, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=ecorr_noape, noape_stderr = stderrcorr_noape, effs_noape=effs, noape_label='No APE-NNs')
    fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_ape_noape_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_ape_noape_bothgroups.svg' %get_timestamp()))


# %%  SLEEPS

# nsleeps_train, nsleeps_test = sort_train_test(df['n_sleeps'], df['effs'], test_start)
# fig = plot_violin(nsleeps_test, effs_test, nsleeps_train, effs_train, ylabel='Number of Sleeps per Participant', median_over_mean=True)
# fig.savefig(os.path.join(analysis_folder, '%s_nsleeps_violin.png' %get_timestamp()))
# fig.savefig(os.path.join(analysis_folder, '%s_nsleeps_violin.svg' %get_timestamp()))

# %% SLEEPS LINE SCATTER

nobs_train, nobs_test = sort_train_test(df['n_sleeps'], df['effs'], test_start)

#fig = plot_line_scatter(nobs_test, effs_test, nobs_train, effs_train, ylabel='Number of Observes', xjitter=0.025, yjitter=1, ylim=(-2, 16), median_over_mean=True, true_values=eobs, effs_true=effs, true_label='APE-NNs')
fig = plot_line_scatter(nobs_test, effs_test, nobs_train, effs_train, ylabel='Number of Sleeps', xjitter=0.025, yjitter=1, median_over_mean=True, true_values=esleeps, effs_true=effs, true_label='APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_nsleeps_line_scatter.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_nsleeps_line_scatter.svg' %get_timestamp()))

# %%

if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['n_sleeps'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['n_sleeps'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Number of Sleeps', xjitter=0.025, yjitter=1, median_over_mean=True)
    fig.savefig(os.path.join(analysis_folder, '%s_nsleeps_line_scatter_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_nsleeps_line_scatter_bothgroups.svg' %get_timestamp()))

# %%

n_train, n_test = sort_train_test(df['n_sleeps'], df['effs'], test_start)
fig = plot_line_scatter_humans_ape_noape(n_test, effs_test, n_train, effs_train, ylabel='Number of Sleeps', xjitter=0.025, yjitter=1, median_over_mean=True, true_values=esleeps, true_stderr=stderrsleeps, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=esleeps_noape, noape_stderr = stderrsleeps_noape, effs_noape=effs, noape_label='No APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_nsleeps_line_scatter_ape_noape.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_nsleeps_line_scatter_ape_noape.svg' %get_timestamp()))



# %% FOR BOTH GROUPS

if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['n_sleeps'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['n_sleeps'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_humans_ape_noape_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Number of Sleeps', ylim=(-1, 16), xjitter=0.025, yjitter=1, median_over_mean=True, true_values=esleeps, true_stderr=stderrsleeps, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=esleeps_noape, noape_stderr = stderrsleeps_noape, effs_noape=effs, noape_label='No APE-NNs')
    fig.savefig(os.path.join(analysis_folder, '%s_nsleeps_line_scatter_ape_noape_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_nsleeps_line_scatter_ape_noape_bothgroups.svg' %get_timestamp()))

# %% TIME OF FIRST OBSERVE

first_observes = df['transitions_ep_rightwrong'].apply(lambda x: [np.where(traj == 0.5)[0][0] if len(traj) > 0 and len(np.where(traj == 0.5)[0]) > 0 else np.nan for traj in x])

first_observes_train, first_observes_test = sort_train_test(first_observes, df['effs'], test_start)

fig = plot_violin(first_observes_test, effs_test, first_observes_train, effs_train, ylabel='Time of First Observe per Participant', yjitter=True)
fig.savefig(os.path.join(analysis_folder, '%s_first_observes_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_first_observes_violin.svg' %get_timestamp()))

# %% TIME OF MEAN OBSERVE

mean_obs_time = df['transitions_ep_rightwrong'].apply(lambda x: [np.mean(np.where(traj == 0.5)[0]) if len(traj) > 0 and len(np.where(traj == 0.5)[0]) > 0 else np.nan for traj in x])

train, test = sort_train_test(mean_obs_time, df['effs'], test_start)

fig = plot_violin(test, effs_test, train, effs_train, ylabel='Mean Observation Time per Participant', median_over_mean=True)
fig.savefig(os.path.join(analysis_folder, '%s_mean_observes_time_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_mean_observes_time_violin.svg' %get_timestamp()))

# %% FINAL OBSERVATION TIME

final_obs_time = df['transitions_ep_rightwrong'].apply(lambda x: [np.where(traj == 0.5)[0][-1] if len(traj) > 0 and len(np.where(traj == 0.5)[0]) > 0 else np.nan for traj in x])

train, test = sort_train_test(final_obs_time, df['effs'], test_start)

fig = plot_violin(test, effs_test, train, effs_train, ylabel='Last Observation Time per Participant', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_last_observes_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_last_observes_violin.svg' %get_timestamp()))

# %% IF SLEEP THEN GIVE TIMES

time_between_sleeps = df['transitions_ep_rightwrong'].apply(lambda x : [calculate_time_between_actions(traj, -1) for traj in x])
meanss = time_between_sleeps.apply(lambda x : [np.nanmean(tb) for tb in x])

meanss_train, meanss_test = sort_train_test(meanss, df['effs'], test_start)

fig = plot_violin(meanss_test, effs_test, meanss_train, effs_train, ylabel='Mean Sleep Interval per Participant', median_over_mean=True)
fig.savefig(os.path.join(analysis_folder, '%s_sleep_intervals_means_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_sleep_intervals_means_violin.svg' %get_timestamp()))

# %%

stdss = time_between_sleeps.apply(lambda x : [np.std(time_between_sleeps) for time_between_sleeps in x])
stdss_train, stdss_test = sort_train_test(stdss, df['effs'], test_start)

fig = plot_violin(stdss_test, effs_test, stdss_train, effs_train, ylabel='Sleep Interval SD per Participant', median_over_mean=True)

fig.savefig(os.path.join(analysis_folder, '%s_sleep_intervals_std_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_sleep_intervals_std_violin.svg' %get_timestamp()))

# %% TIME OF FIRST SLEEPERVE

first_sleep = df['transitions_ep_rightwrong'].apply(lambda x: [np.where(traj == -1)[0][0] if len(traj) > 0 and len(np.where(traj == -1)[0]) > 0 else np.nan for traj in x])

first_sleep_train, first_sleep_test = sort_train_test(first_sleep, df['effs'], test_start)

fig = plot_violin(first_sleep_test, effs_test, first_sleep_train, effs_train, ylabel='Time of First Sleep per Participant', yjitter=True)
fig.savefig(os.path.join(analysis_folder, '%s_first_sleep_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_first_sleep_violin.svg' %get_timestamp()))

# %% TIME OF MEAN SLEEPERVE

mean_sleep_time = df['transitions_ep_rightwrong'].apply(lambda x: [np.mean(np.where(traj == -1)[0]) if len(traj) > 0 and len(np.where(traj == -1)[0]) > 0 else np.nan for traj in x])

train, test = sort_train_test(mean_sleep_time, df['effs'], test_start)

fig = plot_violin(test, effs_test, train, effs_train, ylabel='Mean Sleep Time per Participant', median_over_mean=True)
fig.savefig(os.path.join(analysis_folder, '%s_mean_sleep_time_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_mean_sleep_time_violin.svg' %get_timestamp()))

# %% FINAL SLEEPERVATION TIME

final_sleep_time = df['transitions_ep_rightwrong'].apply(lambda x: [np.where(traj == -1)[0][-1] if len(traj) > 0 and len(np.where(traj == -1)[0]) > 0 else np.nan for traj in x])

train, test = sort_train_test(final_sleep_time, df['effs'], test_start)

fig = plot_violin(test, effs_test, train, effs_train, ylabel='Last Sleep Time per Participant', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_last_sleep_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_last_sleep_violin.svg' %get_timestamp()))

# %% EVIDENCE RATIO CALCS

df['frac_takes_obs_t_since'] = df['transitions_ep'].apply(calculate_freq_observed_choice_per_t, args=(n_steps,))
df['frac_observes_t_since'] = df['transitions_ep'].apply(calculate_freq_observes_per_t, args=(n_steps,))
df['frac_corr_takes_t_since'] = df['transitions_ep'].apply(calculate_freq_correct_choice_per_t, args=(n_steps,))

os.makedirs(os.path.join(analysis_folder, 'evidence_ratio'), exist_ok=True)

for metric_name, metric in zip(['Fraction of Bets on Observed', 'Fraction of Observe Actions', 'Fraction of Bets on Correct'], ['frac_takes_obs_t_since', 'frac_observes_t_since', 'frac_corr_takes_t_since']):
#for metric in ['frac_takes_obs_t_since']:
    er_sorted_train, er_sorted_test = sort_train_test(df[metric].values, df['effs'].values, test_start)

    fig = plot_evidence_ratios(er_sorted_train, effs_train, cmap_train, metric_name, jitter=True, ylim=(0, 1))
    fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_train.png' %(get_timestamp(), metric)))
    fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_train.svg' %(get_timestamp(), metric)))

    fig = plot_evidence_ratios(er_sorted_test, effs_test, cmap_test, metric_name, jitter=True, ylim=(0,1))
    fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_test.png' %(get_timestamp(), metric)))
    fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_test.svg' %(get_timestamp(), metric)))

# %% SLEEP

df['traj_sleeps'] = df['transitions_ep'].apply(calculate_freq_sleeps, args=(n_steps,))

er_sorted_train, er_sorted_test = sort_train_test(df['traj_sleeps'].values, df['effs'].values, test_start)

fig = plot_evidence_ratios(er_sorted_train, effs_train, cmap_train, 'Frequency of Sleeping', xlabel='Time since beginning of episode', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_train.png' %(get_timestamp(), 'traj_sleeps')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_train.svg' %(get_timestamp(), 'traj_sleeps')))

fig = plot_evidence_ratios(er_sorted_test, effs_test, cmap_test, 'Frequency of Sleeping', xlabel='Time since beginning of episode', jitter=True, ylim=(0,1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_test.png' %(get_timestamp(), 'traj_sleeps')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_test.svg' %(get_timestamp(), 'traj_sleeps')))


# %% TEACHER-FORCED EFFICACY RATIO PLOTS - FRACTION OF TIME OBSERVES

df['ape_nns_frac_observes_t_since'] = df.apply(calculate_nns_freq_observes_per_t, args=(n_steps, ape_models), axis=1)
os.makedirs(os.path.join(analysis_folder, 'evidence_ratio'), exist_ok=True)
# %%

er_sorted_train, er_sorted_test = sort_train_test(df['ape_nns_frac_observes_t_since'].values, df['effs'].values, test_start)


fig = plot_evidence_ratios(er_sorted_train, effs_train, ape_cmap, 'Observe probability', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_train.png' %(get_timestamp(), 'nns_frac_observes_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_train.svg' %(get_timestamp(), 'nns_frac_observes_t_since')))


fig = plot_evidence_ratios(er_sorted_test, effs_test, ape_cmap, 'Observe probability', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_test.png' %(get_timestamp(), 'nns_frac_observes_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_test.svg' %(get_timestamp(), 'nns_frac_observes_t_since')))

# %% NO APE MODELS

df['no_ape_nns_frac_observes_t_since'] = df.apply(calculate_nns_freq_observes_per_t, args=(n_steps, control_models), axis=1)

# %%

er_sorted_train, er_sorted_test = sort_train_test(df['no_ape_nns_frac_observes_t_since'].values, df['effs'].values, test_start)

fig = plot_evidence_ratios(er_sorted_train, effs_train, control_cmap, 'Observe probability', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_train.png' %(get_timestamp(), 'nns_frac_observes_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_train.svg' %(get_timestamp(), 'nns_frac_observes_t_since')))

fig = plot_evidence_ratios(er_sorted_test, effs_test, control_cmap, 'Observe probability', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_test.png' %(get_timestamp(), 'nns_frac_observes_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_test.svg' %(get_timestamp(), 'nns_frac_observes_t_since')))

# %% TEACHER-FORCED EFFICACY RATIO PLOTS - FRACTION OF TIME TAKES LAST OBSERVED

df['ape_nns_frac_takes_obs_t_since'] = df.apply(calculate_nns_freq_observed_choice_per_t, args=(n_steps, ape_models, True), axis=1)

# %%

er_sorted_train, er_sorted_test = sort_train_test(df['ape_nns_frac_takes_obs_t_since'].values, df['effs'].values, test_start)

fig = plot_evidence_ratios(er_sorted_train, effs_train, ape_cmap, 'bet on last observed', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_train.png' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_train.svg' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))

fig = plot_evidence_ratios(er_sorted_test, effs_test, ape_cmap, 'bet on last observed', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_test.png' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_test.svg' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))

# %% NO APE MODELS

df['no_ape_nns_frac_takes_obs_t_since'] = df.apply(calculate_nns_freq_observed_choice_per_t, args=(n_steps, control_models, True), axis=1)

# %%

er_sorted_train, er_sorted_test = sort_train_test(df['no_ape_nns_frac_takes_obs_t_since'].values, df['effs'].values, test_start)

fig = plot_evidence_ratios(er_sorted_train, effs_train, control_cmap, 'bet on last observed', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_train.png' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_train.svg' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))

fig = plot_evidence_ratios(er_sorted_test, effs_test, control_cmap, 'bet on last observed', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_test.png' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_test.svg' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))


# %% TEACHER-FORCED EFFICACY RATIO PLOTS - SLEEPS

df['ape_nns_frac_sleeps_per_t'] = df.apply(calculate_nns_freq_sleeps_per_t, args=(n_steps, ape_models), axis=1)

# %%

er_sorted_train, er_sorted_test = sort_train_test(df['ape_nns_frac_sleeps_per_t'].values, df['effs'].values, test_start)

fig = plot_evidence_ratios(er_sorted_train, effs_train, ape_cmap, 'sleep', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_train.png' %(get_timestamp(), 'sleep')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_train.svg' %(get_timestamp(), 'sleep')))

fig = plot_evidence_ratios(er_sorted_test, effs_test, ape_cmap, 'sleep', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_test.png' %(get_timestamp(), 'sleep')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape_test.svg' %(get_timestamp(), 'sleep')))

# %% NO APE MODELS

df['no_ape_nns_frac_sleeps_per_t'] = df.apply(calculate_nns_freq_sleeps_per_t, args=(n_steps, control_models), axis=1)

# %%

er_sorted_train, er_sorted_test = sort_train_test(df['no_ape_nns_frac_sleeps_per_t'].values, df['effs'].values, test_start)

fig = plot_evidence_ratios(er_sorted_train, effs_train, control_cmap, 'sleep', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_train.png' %(get_timestamp(), 'sleep')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_train.svg' %(get_timestamp(), 'sleep')))

fig = plot_evidence_ratios(er_sorted_test, effs_test, control_cmap, 'sleep', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_test.png' %(get_timestamp(), 'sleep')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape_test.svg' %(get_timestamp(), 'sleep')))

# %% ANOVA

import statsmodels.api as sm
from statsmodels.formula.api import ols

#rews_anova = do_anova(df['signed_dev_rews_test'])

series = df['n_sleeps']

df_exploded = pd.DataFrame(series.explode())

df_exploded['efficacy_index'] = df_exploded.groupby(df_exploded.index).cumcount()
df_exploded['pid'] = df_exploded.index
df_exploded['n_sleeps'] = pd.to_numeric(df_exploded['n_sleeps'], errors='coerce')

#print(df_exploded)
# Fit the model
#model = ols('signed_dev_rews_test ~ C(pid) + C(efficacy_index) + C(pid):C(efficacy_index)', data=df_exploded).fit()
model = ols('n_sleeps ~ C(pid) + efficacy_index', data=df_exploded).fit()
# Run the ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
#anova_table = sm.stats.anova_lm(model, typ=2, robust='hc3')

print(anova_table)

## REWARDS
# .
#                          sum_sq     df         F    PR(>F)
# C(pid)              4953.026846  148.0  1.264388  0.035810
# C(efficacy_index)     39.628399    3.0  0.499065  0.683112
# Residual           11751.993289  444.0       NaN       NaN

# Between-subject Variability (variance of individual mean scores)
between_subject_var = df_exploded.groupby('pid')['n_sleeps'].mean().var()

# Within-subject Variability (average of individual variances)
within_subject_var = df_exploded.groupby('pid')['n_sleeps'].var().mean()

print("Between-subject Variability:", between_subject_var)
print("Within-subject Variability:", within_subject_var)

### OBSERVATION
#                         sum_sq     df          F        PR(>F)
# C(pid)             7069.439597  148.0  11.516788  1.258800e-89
# C(efficacy_index)  1197.441887    3.0  96.236822  5.365951e-48
# Residual           1841.513423  444.0        NaN           NaN
# Between-subject Variability: 11.941620941411207
# Within-subject Variability: 6.798557739932886

# %% MIXED EFFECTS MODEL

md = smf.mixedlm('n_sleeps ~ efficacy_index', df_exploded, groups=df_exploded["pid"], re_formula="~efficacy_index")#, re_formula="~Time")

# Fit the model
mdf = md.fit()

# Print the summary
print(mdf.summary())

residuals = mdf.resid
df_exploded['residuals'] = residuals

grouped = df_exploded.groupby('pid')
print("Mean residual per individual:", grouped['residuals'].mean().mean())
print("Std residual per individual:", grouped['residuals'].std().mean())

# %% CHECK SIGNIFICANCE

# Fit a model without the random slope
md_null = smf.mixedlm('n_sleeps ~ efficacy_index', df_exploded, groups=df_exploded["pid"])
mdf_null = md_null.fit()

# Fit a model with the random slope
md_alt = smf.mixedlm('n_sleeps ~ efficacy_index', df_exploded, groups=df_exploded["pid"], re_formula="~efficacy_index")
mdf_alt = md_alt.fit()

# Perform the likelihood ratio test
import scipy.stats as stats
lr = -2 * (mdf_null.llf - mdf_alt.llf)
p_value = stats.chi2.sf(lr, df=1)  # df is 1 because we are testing one additional parameter (random slope)

print("Likelihood Ratio:", lr)
print("p-value:", p_value)

# %% PERFORM FFT ON TRAJECTORIES

from scipy.fft import fft, fftfreq

### compute observations

def get_obs_traj(traj):
    ''' Return trajectory with 1 for observes 0 else 
    
    Arguments
    ---------
    traj : np.array, trajectory with 0.5 corresponding to an observation

    Returns
    -------
    obs_traj : np.array, trajectory with 1 for observes 0 else 
    '''

    obs_traj = np.where(traj == 0.5, 1, 0)

    return obs_traj

obs_trajss = df['transitions_ep_rightwrong'].apply(lambda x : [get_obs_traj(traj) for traj in x])

# ## apply fft to each trajectory

# def apply_fft(traj):
#     ''' Apply fft to trajectory 
    
#     Arguments
#     ---------
#     traj : np.array, trajectory with 0.5 corresponding to an observation

#     Returns
#     -------
#     fft_traj : np.array, fft of trajectory
#     '''

#     # Get length of trajectory
#     n = len(traj)

#     # Apply fft
#     fft_traj = fft(traj)

#     # Get frequencies
#     freqs = fftfreq(n)

#     return fft_traj, freqs

# fft_trajss = obs_trajss.apply(lambda x : [apply_fft(traj)[0] for traj in x])
# fft_freqss = obs_trajss.apply(lambda x : [apply_fft(traj)[1] for traj in x])

def entropy_of_trajectory(trajectory):
    """
    This function computes the entropy of a trajectory based on the normalized
    power spectral density.
    
    Parameters:
    trajectory (np.array): Input trajectory represented as a 1D numpy array.

    Returns:
    float: The entropy of the trajectory.
    """

    # Compute FFT of the trajectory
    Y = scipy.fft.fft(trajectory)
    
    # Compute power spectral density (PSD)
    # The PSD is the square of the absolute value of the FFT
    psd = np.abs(Y) ** 2

    # Normalize the PSD
    # Normalization makes the PSD act like a probability distribution
    psd_norm = psd / np.sum(psd)

    # Compute entropy
    # Use a very small constant to prevent taking log of zero
    entropy = -np.sum(psd_norm * np.log2(psd_norm + np.finfo(float).eps))

    return entropy

fft_entropies = obs_trajss.apply(lambda x : [entropy_of_trajectory(traj) for traj in x])

# %% PLOT HISTOGRAM OF ENTROPIES

#entropies = [np.concatenate(time_between_obs) for time_between_obs in time_between_obss]
entropies = np.concatenate(fft_entropies)

sns.histplot(entropies, bins=100)

# %% COMPARE LOG LIKELIHOODS PER EFFICACY LEVEL

lls_ape_train = []
lls_ape_test = []

# lls_control_train = []
# lls_control_test = []

comparisons_train = []
comparisons_test = []

for ape_model in ape_models:

    ll_train_ape, ll_test_ape = sort_train_test(df['ll_' + str(ape_model)].values, df['effs'].values, test_start)

    lls_ape_train.append(ll_train_ape)
    lls_ape_test.append(ll_test_ape)

    for control_model in control_models:
        
        ll_train_control, ll_test_control = sort_train_test(df['ll_' + str(control_model)].values, df['effs'].values, test_start)

        comparisons_train.append(ll_train_ape - ll_train_control)
        comparisons_test.append(ll_test_ape - ll_test_control)

mean_lls_ape_train = np.stack(lls_ape_train).mean(axis=0)
mean_lls_ape_test = np.stack(lls_ape_test).mean(axis=0)

comparisons_train = np.stack(comparisons_train).mean(axis=0)
comparisons_test = np.stack(comparisons_test).mean(axis=0)

# %%

fig = plot_train_test_comp(effs_train, comparisons_train, effs_test, comparisons_test, y_label = "Log Likelihood Ratio")

fig.savefig(os.path.join(analysis_folder, '%s_log_likelihood_ratio_efficacy.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_log_likelihood_ratio_efficacy.svg' %get_timestamp()))

# %% LLR VIOLIN PLOT

fig = plot_violin(comparisons_test, effs_test, comparisons_train, effs_train, ylabel='Log Likelihood Ratio', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_llr_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_llr_violin.svg' %get_timestamp()))


# %%

fig = plot_violin(mean_lls_ape_test, effs_test, mean_lls_ape_train, effs_train, ylabel='APE-trained Mean Log Likelihood', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_mean_lls_ape_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_mean_lls_ape_violin.svg' %get_timestamp()))

# %%

stepmax_ll_ape_train, stepmax_ll_ape_test = sort_train_test(df['step_max_ll_ape'], df['effs'], test_start)

# %%
fig = plot_violin(stepmax_ll_ape_test, effs_test, stepmax_ll_ape_train, effs_train, ylabel='Stepmax Log Likelihoods', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_stepmax_ll_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_stepmax_ll_violin.svg' %get_timestamp()))


# %% STEPWISE MAXIMUM

comparisons = np.stack(df['step_max_ll_ape'].values) - np.stack(df['step_max_ll_control'].values)

comparisons_train, comparisons_test = sort_train_test(comparisons, df['effs'].values, test_start)

# %% 

fig = plot_train_test_comp(effs_train, comparisons_train, effs_test, comparisons_test, y_label = "Log Likelihood Ratio")

fig.savefig(os.path.join(analysis_folder, '%s_stepmax_log_likelihood_ratio_efficacy.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_stepmax_log_likelihood_ratio_efficacy.svg' %get_timestamp()))

# %%

fig = plot_violin(comparisons_test, effs_test, comparisons_train, effs_train, ylabel='Log Likelihood Ratio', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_stepmax_llr_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_stepmax_llr_violin.svg' %get_timestamp()))



# %% OUTPUT INDICES OF BEST WORST ETC

stepmax_ll_ape_train, stepmax_ll_ape_test = sort_train_test(df['step_max_ll_ape'], df['effs'], test_start)

idx_train_sorted = np.argsort(stepmax_ll_ape_train, axis=0)
min_idx = idx_train_sorted[0, 2]
median_idx = idx_train_sorted[len(idx_train_sorted)//2, 2]
max_idx = idx_train_sorted[-1, 2]
print("eff 05", df.index[min_idx], df.index[median_idx], df.index[max_idx])
print("lls", stepmax_ll_ape_train[min_idx, 2], stepmax_ll_ape_train[median_idx, 2], stepmax_ll_ape_train[max_idx, 2])

## RESULTS FOR 6-19 7-06

# eff 05 5c0ca233632102000147c7db 617032f133b49cddf96ad317 5e904f36cc3d6c355bbf48fa
# lls -381.73687040805817 -20.845579081412552 -4.251381540662351

# %%

idx_sorted = np.argsort(stepmax_ll_ape_test, axis=0)
min_idx = idx_sorted[0, 0]
median_idx = idx_sorted[len(idx_sorted)//2, 0]
max_idx = idx_sorted[-1, 0]
print("eff 0", df.index[min_idx], df.index[median_idx], df.index[max_idx])
print("lls", stepmax_ll_ape_test[min_idx, 0], stepmax_ll_ape_test[median_idx, 0], stepmax_ll_ape_test[max_idx, 0])


min_idx = idx_sorted[0, 3]
median_idx = idx_sorted[len(idx_sorted)//2, 3]
max_idx = idx_sorted[-1, 3]
print("eff 1", df.index[min_idx], df.index[median_idx], df.index[max_idx])
print("lls", stepmax_ll_ape_test[min_idx, 3], stepmax_ll_ape_test[median_idx, 3], stepmax_ll_ape_test[max_idx, 3])

## RESULTS FOR 6-19 7-06

# eff 0 640785ce903930be7597e7c1 60ff47c59f3aff4127e3f234 5ba988cc4a16920001d48564
# lls -116.1805382714374 -22.681773337559207 -3.76787719585991
# eff 1 5f4407f84d55e40dd21a5032 5b2a72f7c293b90001732b89 55da1c4669dbc30010b67569
# lls -166.81286415457726 -15.184147065063598 -1.29612773275698

# %% NN PERTURBATIONS

## read in df from pickle
df_nns_perturbations = pd.read_pickle('results/perturbation_only_NN/day3/518-525-619-706/20231027024138_perturbation_only_nns_df_lr05.pkl')

nns_perturbations_analysis_folder = os.path.join(analysis_folder, 'nns_perturbations', )
os.makedirs(nns_perturbations_analysis_folder, exist_ok=True)

# %% PLOT INITIAL LIKELIHOODS

fig = plot_violin(np.stack(df_nns_perturbations['initial_liks'].values).mean(axis=2), effs, ylabel='initial likelihood', median_over_mean = True, xlabel='efficacy', ylim=(0,1)) 
fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_violin_initial_lik.png' %(get_timestamp())), dpi=300)
fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_violin_initial_lik.svg' %(get_timestamp())), dpi=300)

# %% PLOT FINAL LIKELIHOODS

fig = plot_violin(np.stack(df_nns_perturbations['final_liks'].values).mean(axis=2), effs, ylabel='final likelihood', median_over_mean = True, xlabel='efficacy', ylim=(0,1)) 
fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_violin_final_lik.png' %(get_timestamp())), dpi=300)
fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_violin_final_lik.svg' %(get_timestamp())), dpi=300)

# %% PLOT CORRELATION OF FINAL LIKELIHOODS

final_liks = np.stack(df_nns_perturbations['final_liks'].values).mean(axis=2)

corr_fig, pvs_fig = compute_2D_correlation(final_liks, final_liks, effs, effs, col1name = 'final likelihood', col2name = 'final likelihood')
corr_fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_2D_correlation_final_lik.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_2D_correlation_pvs_final_lik.svg' %(get_timestamp())), dpi=300)

# %% PLOT PERTURBATIONS

fig = plot_violin(np.stack(df_nns_perturbations['perturbation'].values).squeeze(), effs, ylabel='perturbation', median_over_mean = True, xlabel='efficacy', ylim=(-5, 5)) 
fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_violin_perturbations.png' %(get_timestamp())), dpi=300)
fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_violin_perturbations.svg' %(get_timestamp())), dpi=300)

# %% PLOT CORRELATION MATRIX OF PERTURBATIONS

perturbations = np.stack(df_nns_perturbations['perturbation'].values).squeeze()

corr_fig, pvs_fig = compute_2D_correlation(perturbations, perturbations, effs, effs, col1name = 'perturbations', col2name = 'perturbations')
corr_fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_2D_correlation_perturbations.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_2D_correlation_pvs_perturbations.svg' %(get_timestamp())), dpi=300)

# %% PLOT OF CORRELATIONS PERTURBATIONS WITH NUMBER OF OBSERVES

perturbations = np.stack(df_nns_perturbations['perturbation'].values).squeeze()

corr_fig, pvs_fig = compute_2D_correlation(perturbations, np.stack(df_nns_perturbations['n_observes'].values), effs, effs, col1name = 'perturbations', col2name = 'observes')
corr_fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_2D_correlation_perturbations_n_observes.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_2D_correlation_pvs_perturbations_n_observes.png' %(get_timestamp())), dpi=300)

# %% PLOT OF CORRELATIONS PERTURBATIONS WITH NUMBER OF SLEEPS

corr_fig, pvs_fig = compute_2D_correlation(perturbations, np.stack(df_nns_perturbations['n_sleeps'].values), effs, effs, col1name = 'perturbations', col2name = 'sleeps')
corr_fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_2D_correlation_perturbations_n_sleeps.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(nns_perturbations_analysis_folder, '%s_2D_correlation_pvs_perturbations_n_sleeps.png' %(get_timestamp())), dpi=300)

# %% STATISTICAL ANALYSIS

corr_matrix, pvs_matrix = compute_2D_correlation_matrices(perturbations, np.stack(df_nns_perturbations['n_sleeps'].values), effs, effs)
print(corr_matrix.sum())

#-4.342215320109962

# %% PLOT GROUND TRUTH CORRELATION MATRIX

corr_fig, pvs_fig = compute_2D_correlation(df['rewards_tallies'], df['rewards_tallies'], df['effs'], df['effs'], col1name = 'Rewards', col2name='Rewards', annot=False)

corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_rews.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_rews.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% 

corr_fig, pvs_fig = compute_2D_correlation(df['n_observes'], df['n_observes'], df['effs'], df['effs'], col1name = 'Observes', col2name='Observes', annot=False)

corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_rews.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_rews.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% 

corr_fig, pvs_fig = compute_2D_correlation(df['n_sleeps'], df['n_sleeps'], df['effs'], df['effs'], col1name = 'Sleeps', col2name='Sleeps', annot=False)

corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_rews.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_rews.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% IMPORT SIMULATED PARTICIPANT TRAJECTORIES

#simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/sim/mag1/'
#simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/sim/mag100'
simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/sim/mag10'
#simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/sim/mag1'
#simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/sim/mag10'
#/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/sim/mag1/20231017233738/20231218104238_perturbed_control_errs_taus_ape.pkl
#modelname = '20230922111413'
#modelname = '20230923060013'

## WITH VOLATILITY 0.1 
#modelname = ape_models[0]

#sim_timestamp = '20231007230234'
#sim_timestamp = '20231008220417'
#sim_timestamp = '20231008220818'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
#sim_timestamp = '20231016001832'
#sim_timestamp = '20231218104238'
#sim_timestamp = '20231218172807'
#sim_timestamp = '20240220165535'
sim_timestamp = '20240225232718' ## MAG 10 - WEIRD BUT BEST?
#sim_timestamp = '20240225232749' ## MAG 100 - TOO STRONG
#sim_timestamp = '20240220165535' ## MAG 1 - WEAK


sim_rewss, sim_obss, sim_sleepss = load_simulated_participants_across_models(simulated_participants_folder, ape_models, sim_timestamp, include_sleep=True)
sim_rewss = sim_rewss.mean(axis=0)
sim_obss = sim_obss.mean(axis=0)
sim_sleepss = sim_sleepss.mean(axis=0)

sim_parts_analysis_folder = os.path.join(analysis_folder, 'simulated_participants', 'across_models')
os.makedirs(sim_parts_analysis_folder, exist_ok=True)

# %% SIMULATED OBSERVATIONS

corr_fig, pvs_fig = compute_2D_correlation(sim_obss.T, sim_obss.T, effs, effs, "simulated observations", "simulated observations", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% SIMULATED REWARDS

corr_fig, pvs_fig = compute_2D_correlation(sim_rewss.T, sim_rewss.T, effs, effs, "simulated rewards", "simulated rewards", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_rews.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_rews.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_rews.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_rews.svg' %(get_timestamp())), dpi=300)

# %% SIMULATED SLEEPS

corr_fig, pvs_fig = compute_2D_correlation(sim_sleepss.T, sim_sleepss.T, effs, effs, "simulated sleeps", "simulated sleeps", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_sleeps.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_sleeps.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_sleeps.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_sleeps.svg' %(get_timestamp())), dpi=300)

# %% NN SIMULATED PARTICIPANTS W/ NO STRUCTURE

simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/nostruc/mag10'
#modelname = '20230922111413'
modelname = '20230923060013'
#sim_timestamp = '20231008220307'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
#sim_timestamp = '20231016001832'
sim_timestamp = '20231218172807'

nostruc_rewss, nostruc_obss, nostruc_sleepss = load_simulated_participants_across_models(simulated_participants_folder, ape_models, sim_timestamp, include_sleep=True)

nostruc_rewss = nostruc_rewss.mean(axis=0)
nostruc_obss = nostruc_obss.mean(axis=0)
nostruc_sleepss = nostruc_sleepss.mean(axis=0)

sim_parts_analysis_folder = os.path.join(analysis_folder, 'nostruc_simulated_participants', 'across models', sim_timestamp)
os.makedirs(sim_parts_analysis_folder, exist_ok=True)

# %% SIMULATED OBSERVATIONS

corr_fig, pvs_fig = compute_2D_correlation(nostruc_obss.T, nostruc_obss.T, effs, effs, "simulated observations", "simulated observations", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% SIMULATED REWARDS

corr_fig, pvs_fig = compute_2D_correlation(nostruc_rewss.T, nostruc_rewss.T, effs, effs, "simulated rewards", "simulated rewards", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_rews.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_rews.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% SIMULATED SLEEPS

corr_fig, pvs_fig = compute_2D_correlation(nostruc_sleepss.T, nostruc_sleepss.T, effs, effs, "simulated sleeps", "simulated sleeps", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_sleeps.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_sleeps.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_sleeps.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_sleeps.svg' %(get_timestamp())), dpi=300)



# %% RANDOM SIMULATED PARTICIPANTS (i.e. with perturbations but not structured)

simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/random/mag10'
#sim_timestamp = '20231008220307'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
#sim_timestamp = '20231016001832'
#sim_timestamp = '20231218172807'
#sim_timestamp = '20231217163240'
sim_timestamp = '20231218172807'

random_rewss, random_obss, random_sleepss = load_simulated_participants_across_models(simulated_participants_folder, ape_models, sim_timestamp, include_sleep=True)

random_rewss = random_rewss.mean(axis=0)
random_obss = random_obss.mean(axis=0)
random_sleepss = random_sleepss.mean(axis=0)

sim_parts_analysis_folder = os.path.join(analysis_folder, 'random_simulated_participants', 'across models', sim_timestamp)
os.makedirs(sim_parts_analysis_folder, exist_ok=True)

# %% SIMULATED OBSERVATIONS

corr_fig, pvs_fig = compute_2D_correlation(random_obss.T, random_obss.T, effs, effs, "simulated observations", "simulated observations", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% SIMULATED REWARDS

corr_fig, pvs_fig = compute_2D_correlation(random_rewss.T, random_rewss.T, effs, effs, "simulated rewards", "simulated rewards", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_rews.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_rews.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% SIMULATED REWARDS

corr_fig, pvs_fig = compute_2D_correlation(random_sleepss.T, random_sleepss.T, effs, effs, "simulated sleeps", "simulated sleeps", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_sleeps.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_sleeps.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% COMPETITIVE REGRESSION

if group is not None:
    corr_obs_data = np.corrcoef(np.stack(df['n_observes'].values).T)
    corr_sleeps_data = np.corrcoef(np.stack(df['n_sleeps'].values).T)
    corr_rews_data = np.corrcoef(np.stack(df['rewards_tallies'].values).T)
else:
    data_obs_corr_g1, data_obs_pvs_g1 = compute_2D_correlation_matrices(np.stack(df[~df['group']]['n_observes'].values), np.stack(df[~df['group']]['n_observes'].values), np.stack(df[~df['group']]['effs'].values), np.stack(df[~df['group']]['effs'].values),)
    data_obs_corr_g2, data_obs_pvs_g2 = compute_2D_correlation_matrices(np.stack(df[df['group']]['n_observes'].values), np.stack(df[df['group']]['n_observes'].values), np.stack(df[df['group']]['effs'].values), np.stack(df[df['group']]['effs'].values),)

    data_rews_corr_g1, data_rews_pvs_g1 = compute_2D_correlation_matrices(np.stack(df[~df['group']]['rewards_tallies'].values), np.stack(df[~df['group']]['rewards_tallies'].values), np.stack(df[~df['group']]['effs'].values), np.stack(df[~df['group']]['effs'].values),)
    data_rews_corr_g2, data_rews_pvs_g2 = compute_2D_correlation_matrices(np.stack(df[df['group']]['rewards_tallies'].values), np.stack(df[df['group']]['rewards_tallies'].values), np.stack(df[df['group']]['effs'].values), np.stack(df[df['group']]['effs'].values),)

    data_sleeps_corr_g1, data_sleeps_pvs_g1 = compute_2D_correlation_matrices(np.stack(df[~df['group']]['n_sleeps'].values), np.stack(df[~df['group']]['n_sleeps'].values), np.stack(df[~df['group']]['effs'].values), np.stack(df[~df['group']]['effs'].values),)
    data_sleeps_corr_g2, data_sleeps_pvs_g2 = compute_2D_correlation_matrices(np.stack(df[df['group']]['n_sleeps'].values), np.stack(df[df['group']]['n_sleeps'].values), np.stack(df[df['group']]['effs'].values), np.stack(df[df['group']]['effs'].values),)

    data_obs_corr = (~df['group']).sum() / len(df) * data_obs_corr_g1 + (df['group']).sum() / len(df) * data_obs_corr_g2
    data_obs_pvs = (~df['group']).sum() / len(df) * data_obs_pvs_g1 + (df['group']).sum() / len(df) * data_obs_pvs_g2

    data_rews_corr = (~df['group']).sum() / len(df) * data_rews_corr_g1 + (df['group']).sum() / len(df) * data_rews_corr_g2
    data_rews_pvs = (~df['group']).sum() / len(df) * data_rews_pvs_g1 + (df['group']).sum() / len(df) * data_rews_pvs_g2

    data_sleeps_corr = (~df['group']).sum() / len(df) * data_sleeps_corr_g1 + (df['group']).sum() / len(df) * data_sleeps_corr_g2
    data_sleeps_pvs = (~df['group']).sum() / len(df) * data_sleeps_pvs_g1 + (df['group']).sum() / len(df) * data_sleeps_pvs_g2

sim_obs_corr, sim_obs_pvs = compute_2D_correlation_matrices(sim_obss.T, sim_obss.T, effs, effs,)
sim_rews_corr, sim_rews_pvs = compute_2D_correlation_matrices(sim_rewss.T, sim_rewss.T, effs, effs,)
sim_sleeps_corr, sim_sleeps_pvs = compute_2D_correlation_matrices(sim_sleepss.T, sim_sleepss.T, effs, effs,)
nostruc_obs_corr, nostruc_obs_pvs = compute_2D_correlation_matrices(nostruc_obss.T, nostruc_obss.T, effs, effs,)
nostruc_rews_corr, nostruc_rews_pvs = compute_2D_correlation_matrices(nostruc_rewss.T, nostruc_rewss.T, effs, effs,)
nostruc_sleeps_corr, nostruc_sleeps_pvs = compute_2D_correlation_matrices(nostruc_sleepss.T, nostruc_sleepss.T, effs, effs,)
random_obs_corr, random_obs_pvs = compute_2D_correlation_matrices(random_obss.T, random_obss.T, effs, effs,)
random_rews_corr, random_rews_pvs = compute_2D_correlation_matrices(random_rewss.T, random_rewss.T, effs, effs,)
random_sleeps_corr, random_sleeps_pvs = compute_2D_correlation_matrices(random_sleepss.T, random_sleepss.T, effs, effs,)
null_obs_corr, null_obs_pvs = np.eye(len(effs)), np.eye(len(effs))
null_rews_corr, null_rews_pvs = np.eye(len(effs)), np.eye(len(effs))
null_sleeps_corr, null_sleeps_pvs = np.eye(len(effs)), np.eye(len(effs))


# %% 

competitive_corr_regression(upper_tri_masking(data_obs_corr), [upper_tri_masking(sim_obs_corr), upper_tri_masking(nostruc_obs_corr), upper_tri_masking(random_obs_corr)], do_fisher_transform=True)

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.132
# Model:                            OLS   Adj. R-squared:                  0.051
# Method:                 Least Squares   F-statistic:                     1.629
# Date:                Mon, 12 Feb 2024   Prob (F-statistic):              0.202
# Time:                        21:24:04   Log-Likelihood:                 12.411
# No. Observations:                  36   AIC:                            -16.82
# Df Residuals:                      32   BIC:                            -10.49
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.6394      0.480      1.332      0.192      -0.338       1.617
# x1             0.3375      0.330      1.022      0.315      -0.335       1.010
# x2             0.0238      0.370      0.064      0.949      -0.729       0.777
# x3             0.6577      0.350      1.881      0.069      -0.055       1.370
# ==============================================================================
# Omnibus:                        0.792   Durbin-Watson:                   1.356
# Prob(Omnibus):                  0.673   Jarque-Bera (JB):                0.223
# Skew:                          -0.161   Prob(JB):                        0.894
# Kurtosis:                       3.212   Cond. No.                         33.9
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f19e2c154d0>

# %%
competitive_corr_regression(upper_tri_masking(data_rews_corr), [upper_tri_masking(sim_rews_corr), upper_tri_masking(nostruc_rews_corr), upper_tri_masking(random_obs_corr)])

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.244
# Model:                            OLS   Adj. R-squared:                  0.173
# Method:                 Least Squares   F-statistic:                     3.441
# Date:                Mon, 12 Feb 2024   Prob (F-statistic):             0.0282
# Time:                        21:24:04   Log-Likelihood:                 29.876
# No. Observations:                  36   AIC:                            -51.75
# Df Residuals:                      32   BIC:                            -45.42
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.0645      0.053      1.221      0.231      -0.043       0.172
# x1             0.2254      0.086      2.628      0.013       0.051       0.400
# x2            -0.1295      0.258     -0.501      0.619      -0.656       0.397
# x3            -0.5093      0.229     -2.228      0.033      -0.975      -0.044
# ==============================================================================
# Omnibus:                        0.770   Durbin-Watson:                   1.859
# Prob(Omnibus):                  0.680   Jarque-Bera (JB):                0.847
# Skew:                          -0.277   Prob(JB):                        0.655
# Kurtosis:                       2.493   Cond. No.                         17.3
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f19e2adfe50>

# %%
competitive_corr_regression(upper_tri_masking(data_sleeps_corr), [upper_tri_masking(sim_sleeps_corr), upper_tri_masking(nostruc_sleeps_corr), upper_tri_masking(random_sleeps_corr)])

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.028
# Model:                            OLS   Adj. R-squared:                 -0.063
# Method:                 Least Squares   F-statistic:                    0.3086
# Date:                Mon, 12 Feb 2024   Prob (F-statistic):              0.819
# Time:                        21:24:04   Log-Likelihood:                 1.2228
# No. Observations:                  36   AIC:                             5.554
# Df Residuals:                      32   BIC:                             11.89
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.1614      0.674      0.239      0.812      -1.212       1.535
# x1             0.1347      0.248      0.543      0.591      -0.370       0.640
# x2             0.1718      0.465      0.369      0.714      -0.775       1.119
# x3             0.2693      0.542      0.497      0.623      -0.835       1.374
# ==============================================================================
# Omnibus:                        2.740   Durbin-Watson:                   1.408
# Prob(Omnibus):                  0.254   Jarque-Bera (JB):                1.454
# Skew:                          -0.136   Prob(JB):                        0.483
# Kurtosis:                       2.054   Cond. No.                         52.0
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f19e6263fd0>

# %% 

competitive_ridge_corr_regression(upper_tri_masking(data_obs_corr), [upper_tri_masking(sim_obs_corr), upper_tri_masking(nostruc_obs_corr), upper_tri_masking(random_obs_corr)], do_fisher_transform=True, alpha=0.05)

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.044
# Model:                            OLS   Adj. R-squared:                 -0.045
# Method:                 Least Squares   F-statistic:                    0.4948
# Date:                Mon, 12 Feb 2024   Prob (F-statistic):              0.688
# Time:                        21:42:37   Log-Likelihood:                 10.669
# No. Observations:                  36   AIC:                            -13.34
# Df Residuals:                      32   BIC:                            -7.004
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.3708      0.504      0.736      0.467      -0.655       1.397
# x1             0.5132      0.347      1.480      0.149      -0.193       1.219
# x2            -0.0147      0.388     -0.038      0.970      -0.805       0.776
# x3             0.0895      0.367      0.244      0.809      -0.658       0.837
# ==============================================================================
# Omnibus:                        0.794   Durbin-Watson:                   1.347
# Prob(Omnibus):                  0.672   Jarque-Bera (JB):                0.858
# Skew:                          -0.298   Prob(JB):                        0.651
# Kurtosis:                       2.535   Cond. No.                         33.9
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# <statsmodels.regression.linear_model.OLS at 0x7f1a05042250>

# %%

competitive_ridge_corr_regression(upper_tri_masking(data_rews_corr), [upper_tri_masking(sim_rews_corr), upper_tri_masking(nostruc_rews_corr), upper_tri_masking(random_rews_corr)], do_fisher_transform=True, alpha=0.05)


#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.244
# Model:                            OLS   Adj. R-squared:                  0.173
# Method:                 Least Squares   F-statistic:                     3.441
# Date:                Mon, 12 Feb 2024   Prob (F-statistic):             0.0282
# Time:                        21:24:04   Log-Likelihood:                 29.876
# No. Observations:                  36   AIC:                            -51.75
# Df Residuals:                      32   BIC:                            -45.42|
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.0645      0.053      1.221      0.231      -0.043       0.172
# x1             0.2254      0.086      2.628      0.013       0.051       0.400
# x2            -0.1295      0.258     -0.501      0.619      -0.656       0.397
# x3            -0.5093      0.229     -2.228      0.033      -0.975      -0.044
# ==============================================================================
# Omnibus:                        0.770   Durbin-Watson:                   1.859
# Prob(Omnibus):                  0.680   Jarque-Bera (JB):                0.847
# Skew:                          -0.277   Prob(JB):                        0.655
# Kurtosis:                       2.493   Cond. No.                         17.3
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f19e2adfe50>

# %%
competitive_ridge_corr_regression(upper_tri_masking(data_sleeps_corr), [upper_tri_masking(sim_sleeps_corr), upper_tri_masking(nostruc_sleeps_corr), upper_tri_masking(random_sleeps_corr)], alpha=0.01)

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.044
# Model:                            OLS   Adj. R-squared:                 -0.045
# Method:                 Least Squares   F-statistic:                    0.4948
# Date:                Mon, 12 Feb 2024   Prob (F-statistic):              0.688
# Time:                        21:42:58   Log-Likelihood:                 10.669
# No. Observations:                  36   AIC:                            -13.34
# Df Residuals:                      32   BIC:                            -7.004
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.3708      0.504      0.736      0.467      -0.655       1.397
# x1             0.5132      0.347      1.480      0.149      -0.193       1.219
# x2            -0.0147      0.388     -0.038      0.970      -0.805       0.776
# x3             0.0895      0.367      0.244      0.809      -0.658       0.837
# ==============================================================================
# Omnibus:                        0.794   Durbin-Watson:                   1.347
# Prob(Omnibus):                  0.672   Jarque-Bera (JB):                0.858
# Skew:                          -0.298   Prob(JB):                        0.651
# Kurtosis:                       2.535   Cond. No.                         33.9
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# <statsmodels.regression.linear_model.OLS at 0x7f1a08682690>

# %% 

# %% COMPUTE DISTANCES

for corr in [sim_obs_corr, nostruc_obs_corr, null_obs_corr]:
    print(np.linalg.norm(corr - corr_obs_data))

# # DISTANCES
# 0.8629554457453036
# 6.616740650597982
# 6.811573422849496

# %%

for corr in [sim_sleeps_corr, nostruc_sleeps_corr, null_sleeps_corr]:
    print(np.linalg.norm(corr - corr_sleeps_data))

# 2.1630080117534645
# 3.523561686089782
# 3.565371822792874

# %%

for corr in [sim_rews_corr, nostruc_rews_corr, null_rews_corr]:
    print(np.linalg.norm(corr - corr_rews_data))

# # DISTANCES
# 1.4245006951786598
# 1.3836799146096148
# 1.2666250896610731

# %% CORRELATIONS

for corr in [sim_obs_corr, nostruc_obs_corr, null_obs_corr]:
    print(np.corrcoef(corr.flatten(), corr_obs_data.flatten())[0,1])

# 0.618431684897603
# 0.8070591300504857
# 0.8078704079662324

# %%

for corr in [sim_rews_corr, nostruc_rews_corr, null_rews_corr]:
    print(np.corrcoef(corr.flatten(), corr_rews_data.flatten())[0,1])

# 0.9072275219296325
# 0.9190937584925356
# 0.9379331754021807

# %% 

for corr in [sim_sleeps_corr, nostruc_sleeps_corr, null_sleeps_corr]:
    print(np.corrcoef(corr.flatten(), corr_sleeps_data.flatten())[0,1])

# 0.754169366626593
# 0.8066902940484921
# 0.8005166020318416

# %% WITH MASKING

## True process
data = np.stack(df['rewards_tallies'].values).T
corr_data = np.corrcoef(np.stack(df['rewards_tallies'].values).T)

p_value_matrix = np.zeros_like(corr_data)
n_vars = corr_data.shape[1]

for i in range(n_vars):
    for j in range(n_vars):
        _, p_value_matrix[i, j] = scipy.stats.pearsonr(data[:, i], data[:, j])

# Step 2: Mask non-significant correlations
significance_level = 0.05  # or your chosen alpha level
rews_mask = (p_value_matrix > significance_level)
corr_rews_masked = np.where(rews_mask, np.nan, corr_data)

# %%

## True process
data = np.stack(df['n_observes'].values).T
corr_data = np.corrcoef(np.stack(df['n_observes'].values).T)

p_value_matrix = np.zeros_like(corr_data)
n_vars = corr_data.shape[1]

for i in range(n_vars):
    for j in range(n_vars):
        _, p_value_matrix[i, j] = scipy.stats.pearsonr(data[:, i], data[:, j])

# Step 2: Mask non-significant correlations
significance_level = 0.05  # or your chosen alpha level
obs_mask = (p_value_matrix > significance_level)
corr_obs_masked = np.where(obs_mask, np.nan, corr_data)

# %% DISTANCES

for corr in [sim_obs_corr, nostruc_obs_corr, null_obs_corr]:
    print(np.linalg.norm(corr[obs_mask] - corr_obs_data[obs_mask]))

#1.3549024216031404
# 5.29027797619724
# 5.382296652387779

# %% REWARDS

for corr in [sim_rews_corr, nostruc_rews_corr, null_rews_corr]:
    print(np.linalg.norm(corr[rews_mask] - corr_rews_data[rews_mask]))

# 0.9411864485358982
# 0.7574183296366597
# 0.5378447066427565

# %% CORRELATIONS

for corr in [sim_obs_corr, nostruc_obs_corr, null_obs_corr]:
    print(np.corrcoef(corr[obs_mask].flatten(), corr_obs_data[obs_mask].flatten())[0,1])

# 0.2625685527576698
# 0.547161973771874
# nan

# %%

for corr in [sim_rews_corr, nostruc_rews_corr, null_rews_corr]:
    print(np.corrcoef(corr[rews_mask].flatten(), corr_rews_data[rews_mask].flatten())[0,1])

# -0.24626161056142484
# 0.3608235612917617
# nan

# %% ANALYZING DATA AT THE LEVEL OF OVERALL FACTORS

data_obss = np.stack(df['n_observes'].values).T
data_rewss = np.stack(df['rewards_tallies'].values).T
data_sleepss = np.stack(df['n_sleeps'].values).T

sim_obss = np.flip(sim_obss, axis=1)
sim_rewss = np.flip(sim_rewss, axis=1)
sim_sleepss = np.flip(sim_sleepss, axis=1)

nostruc_obss = np.flip(nostruc_obss, axis=1)
nostruc_rewss = np.flip(nostruc_rewss, axis=1)
nostruc_sleepss = np.flip(nostruc_sleepss, axis=1)

random_obss = np.flip(random_obss, axis=1)
random_rewss = np.flip(random_rewss, axis=1)
random_sleepss = np.flip(random_sleepss, axis=1)

# %%

# Extract factor analysis details for each dataset
loadings_data, n_factors_data, variance_data = get_factor_analysis_details(data_obss.T)
loadings_sim, n_factors_sim, variance_sim = get_factor_analysis_details(sim_obss.T)
loadings_nostruc, n_factors_nostruc, variance_nostruc = get_factor_analysis_details(nostruc_obss.T)
loadings_random, n_factors_random, variance_random = get_factor_analysis_details(random_obss.T)

# Compute cosine similarities
similarity_sim = compute_similarity(loadings_data, loadings_sim)
similarity_nostruc = compute_similarity(loadings_data, loadings_nostruc)
similarity_random = compute_similarity(loadings_data, loadings_random)

# Bootstrap confidence intervals
ci_sim = bootstrap_similarity(data_obss.T, sim_obss.T)
ci_nostruc = bootstrap_similarity(data_obss.T, nostruc_obss.T)
ci_random = bootstrap_similarity(data_obss.T, random_obss.T)

# Print results
print("Data Info for data_obss:", "Number of Factors:", n_factors_data, "Variance Explained:", variance_data)
print("Data Info for sim_obss:", "Number of Factors:", n_factors_sim, "Variance Explained:", variance_sim)
print("Data Info for nostruc_obss:", "Number of Factors:", n_factors_nostruc, "Variance Explained:", variance_nostruc)
print("Data Info for random_obss:", "Number of Factors:", n_factors_random, "Variance Explained:", variance_random)
print("\n")
print("Similarity with sim_obss:", similarity_sim, "Confidence Interval:", ci_sim)
print("Similarity with nostruc_obss:", similarity_nostruc, "Confidence Interval:", ci_nostruc)
print("Similarity with random_obss:", similarity_random, "Confidence Interval:", ci_random)

# Determine which process is the most similar to the ground truth
max_similarity = max(similarity_sim, similarity_nostruc, similarity_random)
if max_similarity == similarity_sim:
    print("\nsim_obss is the most similar to the ground truth.")
elif max_similarity == similarity_nostruc:
    print("\nnostruc_obss is the most similar to the ground truth.")
else:
    print("\nrandom_obss is the most similar to the ground truth.")

# Data Info for data_obss: Number of Factors: 3 Variance Explained: 0.861169562968622
# Data Info for sim_obss: Number of Factors: 3 Variance Explained: 0.9984385409284339
# Data Info for nostruc_obss: Number of Factors: 3 Variance Explained: 0.2250476423361595
# Data Info for random_obss: Number of Factors: 3 Variance Explained: 0.20639216745104383


# Similarity with sim_obss: 0.639091787572715 Confidence Interval: [0.61267952 0.6537671 ]
# Similarity with nostruc_obss: 0.20290440587878358 Confidence Interval: [0.04568397 0.34188618]
# Similarity with random_obss: 0.2884843659395418 Confidence Interval: [0.08340743 0.49705282]

# sim_obss is the most similar to the ground truth.

# %% REWARDS

# Extract factor analysis details for each dataset
loadings_data, n_factors_data, variance_data = get_factor_analysis_details(data_rewss.T)
loadings_sim, n_factors_sim, variance_sim = get_factor_analysis_details(sim_rewss.T)
loadings_nostruc, n_factors_nostruc, variance_nostruc = get_factor_analysis_details(nostruc_rewss.T)
loadings_random, n_factors_random, variance_random = get_factor_analysis_details(random_rewss.T)

# Compute cosine similarities
similarity_sim = compute_similarity(loadings_data, loadings_sim)
similarity_nostruc = compute_similarity(loadings_data, loadings_nostruc)
similarity_random = compute_similarity(loadings_data, loadings_random)

# Bootstrap confidence intervals
ci_sim = bootstrap_similarity(data_rewss.T, sim_rewss.T, n_iterations=10)
ci_nostruc = bootstrap_similarity(data_rewss.T, nostruc_rewss.T, n_iterations=10)
ci_random = bootstrap_similarity(data_rewss.T, random_rewss.T, n_iterations=10)

# Print results
print("Data Info for data_obss:", "Number of Factors:", n_factors_data, "Variance Explained:", variance_data)
print("Data Info for sim_obss:", "Number of Factors:", n_factors_sim, "Variance Explained:", variance_sim)
print("Data Info for nostruc_obss:", "Number of Factors:", n_factors_nostruc, "Variance Explained:", variance_nostruc)
print("Data Info for random_obss:", "Number of Factors:", n_factors_random, "Variance Explained:", variance_random)
print("\n")
print("Similarity with sim_obss:", similarity_sim, "Confidence Interval:", ci_sim)
print("Similarity with nostruc_obss:", similarity_nostruc, "Confidence Interval:", ci_nostruc)
print("Similarity with random_obss:", similarity_random, "Confidence Interval:", ci_random)

# Determine which process is the most similar to the ground truth
max_similarity = max(similarity_sim, similarity_nostruc, similarity_random)
if max_similarity == similarity_sim:
    print("\nsim_obss is the most similar to the ground truth.")
elif max_similarity == similarity_nostruc:
    print("\nnostruc_obss is the most similar to the ground truth.")
else:
    print("\nrandom_obss is the most similar to the ground truth.")

# Data Info for data_obss: Number of Factors: 3 Variance Explained: 0.3174890588375574
# Data Info for sim_obss: Number of Factors: 3 Variance Explained: 0.9977769057471351
# Data Info for nostruc_obss: Number of Factors: 3 Variance Explained: 0.2422745937927186
# Data Info for random_obss: Number of Factors: 3 Variance Explained: 0.21537707757619778


# Similarity with sim_obss: 0.4179065482688999 Confidence Interval: [0.30335636 0.47166546]
# Similarity with nostruc_obss: 0.11602141823711891 Confidence Interval: [-0.04743573  0.4021409 ]
# Similarity with random_obss: 0.560081917146543 Confidence Interval: [-0.05922754  0.511746  ]

# random_obss is the most similar to the ground truth.


# %% SLEEPS

# Extract factor analysis details for each dataset
loadings_data, n_factors_data, variance_data = get_factor_analysis_details(data_sleepss.T)
loadings_sim, n_factors_sim, variance_sim = get_factor_analysis_details(sim_sleepss.T)
loadings_nostruc, n_factors_nostruc, variance_nostruc = get_factor_analysis_details(nostruc_sleepss.T)
loadings_random, n_factors_random, variance_random = get_factor_analysis_details(random_sleepss.T)

# Compute cosine similarities
similarity_sim = compute_similarity(loadings_data, loadings_sim)
similarity_nostruc = compute_similarity(loadings_data, loadings_nostruc)
similarity_random = compute_similarity(loadings_data, loadings_random)

# Bootstrap confidence intervals
ci_sim = bootstrap_similarity(data_sleepss.T, sim_sleepss.T, n_iterations=10)
ci_nostruc = bootstrap_similarity(data_sleepss.T, nostruc_sleepss.T, n_iterations=10)
ci_random = bootstrap_similarity(data_sleepss.T, random_sleepss.T, n_iterations=10)

# Print results
print("Data Info for data_obss:", "Number of Factors:", n_factors_data, "Variance Explained:", variance_data)
print("Data Info for sim_obss:", "Number of Factors:", n_factors_sim, "Variance Explained:", variance_sim)
print("Data Info for nostruc_obss:", "Number of Factors:", n_factors_nostruc, "Variance Explained:", variance_nostruc)
print("Data Info for random_obss:", "Number of Factors:", n_factors_random, "Variance Explained:", variance_random)
print("\n")
print("Similarity with sim_obss:", similarity_sim, "Confidence Interval:", ci_sim)
print("Similarity with nostruc_obss:", similarity_nostruc, "Confidence Interval:", ci_nostruc)
print("Similarity with random_obss:", similarity_random, "Confidence Interval:", ci_random)

# Determine which process is the most similar to the ground truth
max_similarity = max(similarity_sim, similarity_nostruc, similarity_random)
if max_similarity == similarity_sim:
    print("\nsim_obss is the most similar to the ground truth.")
elif max_similarity == similarity_nostruc:
    print("\nnostruc_obss is the most similar to the ground truth.")
else:
    print("\nrandom_obss is the most similar to the ground truth.")

# Data Info for data_obss: Number of Factors: 3 Variance Explained: 0.5756081613858208
# Data Info for sim_obss: Number of Factors: 3 Variance Explained: 0.9988109007213377
# Data Info for nostruc_obss: Number of Factors: 3 Variance Explained: 0.22745411418951378
# Data Info for random_obss: Number of Factors: 3 Variance Explained: 0.29951503956802816

# Similarity with sim_obss: 0.5695001573214561 Confidence Interval: [0.54656533 0.59882841]
# Similarity with nostruc_obss: 0.22958124578179584 Confidence Interval: [0.08371386 0.44871028]
# Similarity with random_obss: 0.1607595528509367 Confidence Interval: [-0.15451689  0.37790529]

# sim_obss is the most similar to the ground truth.

# %%
