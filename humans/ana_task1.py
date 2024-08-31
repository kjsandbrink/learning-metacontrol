# Kai Sandbrink
# 2023-06-23
# This script analyzes the task 1 behavior for a compiled DF or responses

# %% LIBRARY IMPORT

import numpy as np
import pandas as pd
import scipy.fft
import os
import pickle
from sklearn.linear_model import Ridge, Lasso

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from human_utils_project import get_clean_data, sort_train_test, get_mean_ll, plot_train_test_comp, calculate_freq_observed_choice_per_t, calculate_freq_observes_per_t, calculate_freq_correct_choice_per_t, plot_evidence_ratios
from human_utils_project import calculate_nns_freq_observed_choice_per_t, calculate_nns_freq_observes_per_t

from human_plot_traj_analyses import plot_violin, plot_line_scatter, plot_line_scatter_humans_ape_noape, plot_line_scatter_humans_ape_noape_group, frac_takes_lineplot, frac_correct_takes_lineplot, plot_line_scatter_group

from utils import format_axis, get_timestamp
from human_utils_behavioral_analysis import get_evs_wanted, mean_slope_train_test, calc_dev_behavior, compute_2D_correlation, compute_2D_correlation_matrices
from human_utils_behavioral_analysis import get_factor_analysis_details, compute_similarity, bootstrap_similarity
from human_utils_behavioral_analysis import load_simulated_participants_across_models
from human_utils_behavioral_analysis import compute_partial_2D_correlation, compute_partial_2D_correlation_matrices
from human_utils_behavioral_analysis import fisher_transform, competitive_corr_regression, combine_train_test, competitive_lasso_corr_regression, competitive_ridge_corr_regression, upper_tri_masking

import statsmodels.api as sm
import statsmodels.formula.api as smf

# %% PARAMETERS & DATA

day = 'day2'
#exp_date = '24-01-22'
exp_date = '24-01-22-29'

#exp_date = '518-525-619-706'
#exp_date = '12-11'

#day1_test_mask_cutoff = 10
#day1_test_mask_cutoff = None
day1_test_mask_cutoff = {
    "groupA": {"lower": 10, "upper": 90},
    "groupB": {"lower": 8, "upper": 72}
}

#group = 'groupA' # Options: None, 'groupA', 'groupB'
#group = 'groupB'
group = None

df, effs_train, effs_test, test_start = get_clean_data(day = int(day[-1]), exp_date = exp_date, day1_test_mask_cutoff=day1_test_mask_cutoff, group=group)

effs = np.arange(0, 1.125, 0.125)
n_steps = 50

cmap_train = mpl.colormaps['Greens']
cmap_test = mpl.colormaps['Blues']

cmaps = {
    'train': cmap_train,
    'test': cmap_test,
}

results_folder = 'results'
analysis_folder = os.path.join('analysis', 'traj_diff_efficacy', day, exp_date)

if group is not None:
    analysis_folder = os.path.join(analysis_folder, group)

if day1_test_mask_cutoff is not None:
    #analysis_folder = os.path.join(analysis_folder, 'day1_test_cutoff_%d' %day1_test_mask_cutoff)
    analysis_folder = os.path.join(analysis_folder, 'day1_test_cutoff_%d' %day1_test_mask_cutoff['groupA']['lower'])

os.makedirs(analysis_folder, exist_ok=True)

# %% NEURAL NETWORK ANALYSES

## VOLATILITY 0.1
# ape_models = [
#     20230427201627,
#     20230427201629,
#     20230427201630,
#     20230427201632,
#     20230427201633,
#     20230427201644,
#     20230427201646,
#     20230427201647,
#     20230427201648,
#     20230427201649
# ]

# control_models = [
#     20230427201636,
#     20230427201637,
#     20230427201639,
#     20230427201640,
#     20230427201642,
#     20230427201657,
#     20230427201656,
#     20230427201655,
#     20230427201653,
#     20230427201652
# ]

#### WITH BIAS 0.5, VOLATILITY 0.2, AND NO MANUAL SEED
#### 7/7/23

# ape_models = [
#      20230704231542,
#      20230704231540,
#      20230704231539,
#      20230704231537,
#      20230704231535,
#      20230704231525,
#      20230704231524,
#      20230704231522,
#      20230704231521,
#      20230704231519
# ]

# control_models = [
#      20230704231549,
#      20230704231548,
#      20230704231546,
#      20230704231545,
#      20230704231543,
#      20230704231534,
#      20230704231533,
#      20230704231531,
#      20230704231529,
#      20230704231528
# ]

#### WITH BIAS 0.5, VOLATILITY 0.2, AND NO HELDOUT TEST REGION
#### 10/06

ape_models = [
	20230923060019,
	20230923060017,
	20230923060016,
	20230923060014,
	20230923060013,
	20230922111420,
	20230922111418,
	20230922111417,
	20230922111415,
	20230922111413,
]

control_models = [
	20230923023710,
	20230923023709,
	20230923023707,
	20230923023706,
	20230923023704,
	20230922110530,
	20230922110528,
	20230922110527,
	20230922110525,
	20230922110524,
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

fig = plot_violin(meanss_test, effs_test, meanss_train, effs_train, median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_intervals_means_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_intervals_means_violin.svg' %get_timestamp()))

# %% CALCULATE STD FOR EACH PARTICIPANT

stdss = time_between_obss.apply(lambda x : [np.std(time_between_obs) for time_between_obs in x])

# %% VIOLIN PLOTS OF STD

stdss_train, stdss_test = sort_train_test(stdss, df['effs'], test_start)

fig = plot_violin(stdss_test, effs_test, stdss_train, effs_train, ylabel='Standard Deviation per Participant', median_over_mean = True)

fig.savefig(os.path.join(analysis_folder, '%s_intervals_std_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_intervals_std_violin.svg' %get_timestamp()))

# %% HISTOGRAM OF STD

stdss = np.concatenate(stdss)

min_val = np.floor(np.nanmin(stdss))
max_val = np.ceil(np.nanmax(stdss))

# generate bin edges from min_val to max_val
bins = np.arange(min_val, max_val + 1, 1) 

sns.histplot(stdss, bins=bins)

# %% NUMBER OF OBSERVES

# nobs_train, nobs_test = sort_train_test(df['n_observes'], df['effs'], test_start)
# fig = plot_violin(nobs_test, effs_test, nobs_train, effs_train, ylabel='Number of Observes per Participant', median_over_mean = True)
# fig.savefig(os.path.join(analysis_folder, '%s_nobs_violin.png' %get_timestamp()))
# fig.savefig(os.path.join(analysis_folder, '%s_nobs_violin.svg' %get_timestamp()))

# fig = plot_violin(nobs_test[mask], effs_test, nobs_train[mask], effs_train, ylabel='Number of Observes per Participant', median_over_mean = True)
# fig.savefig(os.path.join(analysis_folder, '%s_masked_nobs_violin.png' %get_timestamp()))
# fig.savefig(os.path.join(analysis_folder, '%s_masked_nobs_violin.svg' %get_timestamp()))

# %% SIMPLE HISTOGRAM NUMBER OF OBSERVES

nobs = df['n_observes'].apply(lambda x : np.sum(x))

sns.histplot(nobs, bins=100)
plt.xticks(np.arange(0, 250, 10))

plt.savefig(os.path.join(analysis_folder, '%s_nobs_hist.png' %get_timestamp()))

# %% HISTOGRAM FOR TEST SET ONLY

nobs_train, nobs_test = sort_train_test(df['n_observes'], df['effs'], test_start)

nobs_test_sum = nobs_test.sum(axis=1)

sns.histplot(nobs_test_sum, bins=100)
plt.xticks(np.arange(0, 250, 10))

plt.savefig(os.path.join(analysis_folder, '%s_nobs_test_hist.png' %get_timestamp()))

# %%

## Calculated on 8/16
erew = np.flip(np.array([30.137, 29.282, 28.337, 26.76,  25.597, 25.078, 24.64,  24.693, 24.553]))
stderrrew = np.flip(np.array([0.55636328, 0.4134847,  0.32582219, 0.20409802 ,0.12408908 ,0.13327265, 0.11044456, 0.08972235, 0.16780971]))
eobs = np.flip(np.array([6.561, 6.334, 5.741, 4.829, 3.641, 2.531, 1.8,   1.324, 1.089]))
stderrobs = np.flip(np.array([0.80767004, 0.78759025, 0.73035122, 0.65029909 ,0.56589124, 0.38641545, 0.2786862,  0.2229852,  0.13387644]))
ecorr = np.flip(np.array([0.68109297, 0.68260722, 0.67462006, 0.64070059, 0.60635619, 0.58156844, 0.5455282,  0.54777943, 0.53220974]))
stderrcorr = np.flip(np.array([0.02063236, 0.0176135,  0.01929194, 0.01756656, 0.0139558,  0.01069981, 0.00754262, 0.00725149, 0.0067924 ]))

erew_noape = np.flip(np.array([25.932, 26.542, 25.644, 25.68,  25.577, 24.96,  25.081, 24.574, 24.453]))
stderrrew_noape = np.flip(np.array([0.67923457, 0.54275556, 0.47197712, 0.37860005, 0.30084897 ,0.1523089, 0.21404883, 0.24092821, 0.20654806]))
eobs_noape = np.flip(np.array([1.243, 1.201, 1.206, 1.232, 1.174, 1.197, 1.123, 1.141, 1.142]))
stderrobs_noape = np.flip(np.array([0.58808171, 0.56375784, 0.55877938, 0.55218258, 0.53688956, 0.53908079,  0.50891856, 0.51853341, 0.50112833]))
ecorr_noape = np.flip(np.array([0.53184911, 0.54825911, 0.53155858, 0.54346746 , 0.54686225, 0.54018698, 0.54959084, 0.53630105 , 0.54202072]))
stderrcorr_noape = np.flip(np.array([0.01929597, 0.01875204, 0.02015837, 0.01751587, 0.02030193, 0.01888387, 0.01479076, 0.01712247, 0.01732436]))

# %% 

nobs_train, nobs_test = sort_train_test(df['n_observes'], df['effs'], test_start)
fig = plot_line_scatter(nobs_test, effs_test, nobs_train, effs_train, ylabel='Number of Observes', xjitter=0.025, yjitter=1, ylim=(-2, 16), median_over_mean=True, true_values=eobs, effs_true=effs, true_label='APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter.svg' %get_timestamp()))

# %% FOR BOTH GROUPS

if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['n_observes'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['n_observes'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Number of Observes', xjitter=0.025, yjitter=1, median_over_mean=True,)
    fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_nobs_line_scatter_bothgroups.svg' %get_timestamp()))

# %% PLOT COMPARISONS

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


# %% NUMBER OF REWARDS

# n_train, n_test = sort_train_test(df['rewards_tallies'], df['effs'], test_start)
# fig = plot_violin(nobs_test, effs_test, nobs_train, effs_train, ylabel='Number of Rewards per Participant', median_over_mean = True)
# fig.savefig(os.path.join(analysis_folder, '%s_nrews_violin.png' %get_timestamp()))
# fig.savefig(os.path.join(analysis_folder, '%s_nrews_violin.svg' %get_timestamp()))

# %% PLOT COMPARISONS

n_train, n_test = sort_train_test(df['rewards_tallies'], df['effs'], test_start)
fig = plot_line_scatter_humans_ape_noape(n_test, effs_test, n_train, effs_train, ylabel='Number of Rewards', ylim=(15, 40), xjitter=0.025, yjitter=1, median_over_mean=True, true_values=erew, true_stderr=stderrrew, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=erew_noape, noape_stderr = stderrrew_noape, effs_noape=effs, noape_label='No APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_nrews_line_scatter_ape_noape.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_nrews_line_scatter_ape_noape.svg' %get_timestamp()))


# %% FOR BOTH GROUPS

if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['rewards_tallies'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['rewards_tallies'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_humans_ape_noape_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Number of Rewards', ylim=(15, 40), xjitter=0.025, yjitter=1, median_over_mean=True, true_values=erew, true_stderr=stderrrew, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=erew_noape, noape_stderr = stderrrew_noape, effs_noape=effs, noape_label='No APE-NNs')
    fig.savefig(os.path.join(analysis_folder, '%s_nrews_line_scatter_ape_noape_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_nrews_line_scatter_ape_noape_bothgroups.svg' %get_timestamp()))

# %%

n_train, n_test = sort_train_test(df['rewards_tallies'], df['effs'], test_start)
fig = plot_line_scatter(n_test, effs_test, n_train, effs_train, ylabel='Number of Rewards', xjitter=0.025, yjitter=1, median_over_mean=True, true_values=erew, effs_true=effs, true_label='APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_rews_line_scatter.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_rews_line_scatter.svg' %get_timestamp()))

# %% 

# n_train, n_test = sort_train_test(df['intended_correct'], df['effs'], test_start)
# fig = plot_violin(n_test, effs_test, n_train, effs_train, ylabel='Probability Intended Correct Bet', median_over_mean = True)
# fig.savefig(os.path.join(analysis_folder, '%s_pc_violin.png' %get_timestamp()))
# fig.savefig(os.path.join(analysis_folder, '%s_pc_violin.svg' %get_timestamp()))

# %% 

n_train, n_test = sort_train_test(df['intended_correct'], df['effs'], test_start)
fig = plot_line_scatter(n_test, effs_test, n_train, effs_train, ylabel='Probability Intended Correct Bet', xjitter=0.025, yjitter=0.01, median_over_mean=True, true_values=ecorr, effs_true=effs, true_label='APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter.svg' %get_timestamp()))


# %% 

if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['intended_correct'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['intended_correct'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Probability Intended Correct Bet', xjitter=0.025, yjitter=0.01, median_over_mean=True, ylim=(0, 1))
    fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_bothgroups.svg' %get_timestamp()))

# %% 

n_train, n_test = sort_train_test(df['intended_correct'], df['effs'], test_start)
fig = plot_line_scatter_humans_ape_noape(n_test, effs_test, n_train, effs_train, ylabel='Probability Intended Correct', ylim=(0, 1), xjitter=0.025, yjitter=1, median_over_mean=True, true_values=ecorr, true_stderr=stderrcorr, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=ecorr_noape, noape_stderr = stderrcorr_noape, effs_noape=effs, noape_label='No APE-NNs')
fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_ape_noape.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_ape_noape.svg' %get_timestamp()))


# %% FOR BOTH GROUPS

if group is None:
    n_trainA, n_testA = sort_train_test(df[~df['group']]['intended_correct'], df[~df['group']]['effs'], test_start[0])
    n_trainB, n_testB = sort_train_test(df[df['group']]['intended_correct'], df[df['group']]['effs'], test_start[1])

    fig = plot_line_scatter_humans_ape_noape_group((n_testA, n_testB), effs_test, (n_trainA, n_trainB), effs_train, ylabel='Probability Intended Correct', ylim=(0, 1), xjitter=0.025, yjitter=0, median_over_mean=True, true_values=ecorr, true_stderr=stderrcorr, effs_true=effs, true_label='APE-NNs', true_color='C0', noape_color='C1', noape_values=ecorr_noape, noape_stderr = stderrcorr_noape, effs_noape=effs, noape_label='No APE-NNs')
    fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_ape_noape_bothgroups.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_corr_line_scatter_ape_noape_bothgroups.svg' %get_timestamp()))


# %% WITHIN-EPISODE PLOTS FOR GROUP A

transitions_train, transitions_test = sort_train_test(df['transitions_ep_rightwrong'], df['effs'], test_start)
observations = transitions_test == 0.5
smoothing_window = 5
#observations = np.mean(observations, axis=0)

fig = frac_takes_lineplot(1 - np.array(effs_test), observations, smoothing_window=smoothing_window)
fig.savefig(os.path.join(analysis_folder, '%s_obs_lineplot_smooth%d.png' %(get_timestamp(), smoothing_window)))
fig.savefig(os.path.join(analysis_folder, '%s_obs_lineplot_smooth%d.svg' %(get_timestamp(), smoothing_window)))

# %%

transitions_train, transitions_test = sort_train_test(df['transitions_ep_rightwrong'], df['effs'], test_start)
smoothing_window = 4

fig = frac_correct_takes_lineplot(1 - np.array(effs_test), transitions_test, smoothing_window=smoothing_window, ylim=(0,1))
fig.savefig(os.path.join(analysis_folder, '%s_corr_lineplot_smooth%d.png' %(get_timestamp(), smoothing_window)))
fig.savefig(os.path.join(analysis_folder, '%s_corr_lineplot_smooth%d.svg' %(get_timestamp(), smoothing_window)))

#plt.close('all')

# %% EFFICACY ESTIMATES

df_estimates = df[df['efficacy_estimates'].notnull()].copy()

n_train, n_test = sort_train_test(df_estimates['efficacy_estimates'], df_estimates['effs'], test_start)
fig = plot_violin(n_test, effs_test, n_train, effs_train, ylabel='Efficacy Estimate', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_estimate_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_estimate_violin.svg' %get_timestamp()))

# %% SUBTRACTED VERSION

n_train_sub = n_train - effs_train
n_test_sub = n_test - effs_test

fig = plot_violin(n_test_sub, effs_test, n_train_sub, effs_train, ylabel='Efficacy Estimate Deviation', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_dev_estimate_violin_method2.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_dev_estimate_violin_method2.svg' %get_timestamp()))

# %% TIME OF FIRST OBSERVE

first_observes = df['transitions_ep_rightwrong'].apply(lambda x: [np.where(traj == 0.5)[0][0] if len(traj) > 0 and len(np.where(traj == 0.5)[0]) > 0 else np.nan for traj in x])

first_observes_train, first_observes_test = sort_train_test(first_observes, df['effs'], test_start)

fig = plot_violin(first_observes_test, effs_test, first_observes_train, effs_train, ylabel='Time of First Observe per Participant', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_first_observes_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_first_observes_violin.svg' %get_timestamp()))

# %% TIME OF FIRST OBSERVE

mean_obs_time = df['transitions_ep_rightwrong'].apply(lambda x: [np.mean(np.where(traj == 0.5)[0]) if len(traj) > 0 and len(np.where(traj == 0.5)[0]) > 0 else np.nan for traj in x])

train, test = sort_train_test(mean_obs_time, df['effs'], test_start)

fig = plot_violin(test, effs_test, train, effs_train, ylabel='Mean Observation Time per Participant', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_mean_observes_time_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_mean_observes_time_violin.svg' %get_timestamp()))

# %% FINAL OBSERVATION TIME

final_obs_time = df['transitions_ep_rightwrong'].apply(lambda x: [np.where(traj == 0.5)[0][-1] if len(traj) > 0 and len(np.where(traj == 0.5)[0]) > 0 else np.nan for traj in x])

train, test = sort_train_test(final_obs_time, df['effs'], test_start)

fig = plot_violin(test, effs_test, train, effs_train, ylabel='Last Observation Time per Participant', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_last_observes_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_last_observes_violin.svg' %get_timestamp()))

# %% EVIDENCE RATIO PLOTS

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

# %% TEACHER-FORCED EFFICACY RATIO PLOTS - FRACTION OF TIME OBSERVES

df['ape_nns_frac_observes_t_since'] = df.apply(calculate_nns_freq_observes_per_t, args=(n_steps, ape_models), axis=1)

# %%

er_sorted_train, er_sorted_test = sort_train_test(df['ape_nns_frac_observes_t_since'].values, df['effs'].values, test_start)

fig = plot_evidence_ratios(er_sorted_train, effs_train, ape_cmap, 'Observe probability', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape.png' %(get_timestamp(), 'nns_frac_observes_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape.svg' %(get_timestamp(), 'nns_frac_observes_t_since')))

# %% NO APE MODELS

df['no_ape_nns_frac_observes_t_since'] = df.apply(calculate_nns_freq_observes_per_t, args=(n_steps, control_models), axis=1)

# %%

er_sorted_train, er_sorted_test = sort_train_test(df['no_ape_nns_frac_observes_t_since'].values, df['effs'].values, test_start)

fig = plot_evidence_ratios(er_sorted_train, effs_train, control_cmap, 'Observe probability', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape.png' %(get_timestamp(), 'nns_frac_observes_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape.svg' %(get_timestamp(), 'nns_frac_observes_t_since')))


# %% TEACHER-FORCED EFFICACY RATIO PLOTS - FRACTION OF TIME TAKES LAST OBSERVED

df['ape_nns_frac_takes_obs_t_since'] = df.apply(calculate_nns_freq_observed_choice_per_t, args=(n_steps, ape_models), axis=1)

# %%

er_sorted_train, er_sorted_test = sort_train_test(df['ape_nns_frac_takes_obs_t_since'].values, df['effs'].values, test_start)

fig = plot_evidence_ratios(er_sorted_train, effs_train, ape_cmap, 'bet on last observed', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape.png' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_ape.svg' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))

# %% NO APE MODELS

df['no_ape_nns_frac_takes_obs_t_since'] = df.apply(calculate_nns_freq_observed_choice_per_t, args=(n_steps, control_models), axis=1)

# %%

er_sorted_train, er_sorted_test = sort_train_test(df['no_ape_nns_frac_takes_obs_t_since'].values, df['effs'].values, test_start)

fig = plot_evidence_ratios(er_sorted_train, effs_train, control_cmap, 'bet on last observed', jitter=True, ylim=(0, 1))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape.png' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))
fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_noape.svg' %(get_timestamp(), 'nns_frac_takes_obs_t_since')))

# %% HOW WELL THE PARTICIPANTS MATCHED NN MEANS ON BEHAVIORAL METRICS?

## Calculated on 8/16
erew = np.flip(np.array([30.137, 29.282, 28.337, 26.76,  25.597, 25.078, 24.64,  24.693, 24.553]))
eobs = np.flip(np.array([6.561, 6.334, 5.741, 4.829, 3.641, 2.531, 1.8,   1.324, 1.089]))
ecorr = np.flip(np.array([0.68109297, 0.68260722, 0.67462006, 0.64070059, 0.60635619, 0.58156844, 0.5455282,  0.54777943, 0.53220974]))

df['signed_dev_rews_train'], df['signed_dev_rews_test'] = calc_dev_behavior(df['rewards_tallies'], df['effs'], erew, aggregate_efficacies=False, use_abs = False, effs_train=effs_train, effs_test=effs_test)
df['signed_dev_obs_train'], df['signed_dev_obs_test'] = calc_dev_behavior(df['n_observes'], df['effs'], eobs, aggregate_efficacies=False, use_abs = False, effs_train=effs_train, effs_test=effs_test)
df['signed_dev_corr_train'], df['signed_dev_corr_test'] = calc_dev_behavior(df['intended_correct'], df['effs'], ecorr, aggregate_efficacies=False, use_abs = False, effs_train=effs_train, effs_test=effs_test)

#df['signed_dev_corr_train'], df['abs_dev_corr_test'] = calc_dev_behavior(df['n_observes'], df['effs'], eobs, use_abs = False, effs_train=effs_train, effs_test=effs_test)

# %% REWARDS

fig = plot_violin(np.stack(df['signed_dev_rews_test'].values), effs_test, np.stack(df['signed_dev_rews_train'].values), effs_train, ylabel='Deviation from NN Rewards', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_dev_rews_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_dev_rews_violin.svg' %get_timestamp()))

# %% OBSERVES

fig = plot_violin(np.stack(df['signed_dev_obs_test'].values), effs_test, np.stack(df['signed_dev_obs_train'].values), effs_train, ylabel='Deviation from NN Observes', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_dev_obs_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_dev_obs_violin.svg' %get_timestamp()))

# %% CORRECT TAKES

fig = plot_violin(np.stack(df['signed_dev_corr_test'].values), effs_test, np.stack(df['signed_dev_corr_train'].values), effs_train, ylabel='Deviation from NN Correct Takes', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_dev_corr_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_dev_corr_violin.svg' %get_timestamp()))

# %% EFFICACY ESTIMATES

df_estimates = df[df['efficacy_estimates'].notnull()].copy()
df_estimates['signed_dev_estimate_train'], df_estimates['signed_dev_estimate_test'] = calc_dev_behavior(df_estimates['efficacy_estimates'], df_estimates['effs'], effs, aggregate_efficacies=False, use_abs = False, effs_train=effs_train, effs_test=effs_test)

# %% 

fig = plot_violin(np.stack(df_estimates['signed_dev_estimate_test'].values), effs_test, np.stack(df_estimates['signed_dev_estimate_train'].values), effs_train, ylabel='Efficacy Estimate Deviation', median_over_mean = True)

fig.savefig(os.path.join(analysis_folder, '%s_dev_estimate_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_dev_estimate_violin.svg' %get_timestamp()))

# %% 

print("Variance between subject", np.stack(df['signed_dev_rews_test'].values).mean(axis=1).var() )
print("Variance between efficacies", np.stack(df['signed_dev_rews_test'].values).mean(axis=0).var() )
print("Variance within subject", np.stack(df['signed_dev_rews_test'].values).var(axis=1).mean() )


# %% 

print("Variance between subject obs ", np.stack(df['signed_dev_obs_test'].values).mean(axis=1).var() )
print("Variance between efficacies", np.stack(df['signed_dev_obs_test'].values).mean(axis=0).var() )
print("Variance within subject", np.stack(df['signed_dev_obs_test'].values).var(axis=1).mean() )

# %% ANOVA

import statsmodels.api as sm
from statsmodels.formula.api import ols

#rews_anova = do_anova(df['signed_dev_rews_test'])'

series = df['signed_dev_obs_test']

df_exploded = pd.DataFrame(series.explode())

df_exploded['efficacy'] = df['effs'].explode().astype(float) ## TODO: FIX REMAINING LINES
df_exploded['pid'] = df_exploded.index

df_exploded['signed_dev_obs_test'] = pd.to_numeric(df_exploded['signed_dev_obs_test'], errors='coerce')

#print(df_exploded)
# Fit the model
#model = ols('signed_dev_rews_test ~ C(pid) + C(efficacy_index) + C(pid):C(efficacy_index)', data=df_exploded).fit()
model = ols('signed_dev_obs_test ~ C(pid) + efficacy_index', data=df_exploded).fit()
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
between_subject_var = df_exploded.groupby('pid')['signed_dev_obs_test'].mean().var()

# Within-subject Variability (average of individual variances)
within_subject_var = df_exploded.groupby('pid')['signed_dev_obs_test'].var().mean()

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

md = smf.mixedlm('signed_dev_obs_test ~ efficacy_index', df_exploded, groups=df_exploded["pid"], re_formula="~efficacy_index")#, re_formula="~Time")

# Fit the model
mdf = md.fit()

# Print the summary
print(mdf.summary())

residuals = mdf.resid
df_exploded['residuals'] = residuals


sns.lmplot(x='efficacy_index', y='residuals', col='pid', col_wrap=4, data=df_exploded, fit_reg=False)
plt.show()

grouped = df_exploded.groupby('pid')
print("Mean residual per individual:", grouped['residuals'].mean().mean())
print("Std residual per individual:", grouped['residuals'].std().mean())

# %% CHECK SIGNIFICANCE

# Fit a model without the random slope
md_null = smf.mixedlm('signed_dev_obs_test ~ efficacy_index', df_exploded, groups=df_exploded["pid"])
mdf_null = md_null.fit()

# Fit a model with the random slope
md_alt = smf.mixedlm('signed_dev_obs_test ~ efficacy_index', df_exploded, groups=df_exploded["pid"], re_formula="~efficacy_index")
mdf_alt = md_alt.fit()

# Perform the likelihood ratio test
import scipy.stats as stats
lr = -2 * (mdf_null.llf - mdf_alt.llf)
p_value = stats.chi2.sf(lr, df=1)  # df is 1 because we are testing one additional parameter (random slope)

print("Likelihood Ratio:", lr)
print("p-value:", p_value)

# %% ANOVA

import statsmodels.api as sm
from statsmodels.formula.api import ols

#rews_anova = do_anova(df['signed_dev_rews_test'])

series = df['rewards_tallies']

df_exploded = pd.DataFrame(series.explode())

df_exploded['efficacy_index'] = df_exploded.groupby(df_exploded.index).cumcount()
df_exploded['pid'] = df_exploded.index
df_exploded['rewards_tallies'] = pd.to_numeric(df_exploded['rewards_tallies'], errors='coerce')

#print(df_exploded)
# Fit the model
#model = ols('signed_dev_rews_test ~ C(pid) + C(efficacy_index) + C(pid):C(efficacy_index)', data=df_exploded).fit()
model = ols('rewards_tallies ~ C(pid) + efficacy_index', data=df_exploded).fit()
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
between_subject_var = df_exploded.groupby('pid')['rewards_tallies'].mean().var()

# Within-subject Variability (average of individual variances)
within_subject_var = df_exploded.groupby('pid')['rewards_tallies'].var().mean()

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

md = smf.mixedlm('rewards_tallies ~ efficacy_index', df_exploded, groups=df_exploded["pid"], re_formula="~efficacy_index")#, re_formula="~Time")

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
md_null = smf.mixedlm('rewards_tallies ~ efficacy_index', df_exploded, groups=df_exploded["pid"])
mdf_null = md_null.fit()

# Fit a model with the random slope
md_alt = smf.mixedlm('rewards_tallies ~ efficacy_index', df_exploded, groups=df_exploded["pid"], re_formula="~efficacy_index")
mdf_alt = md_alt.fit()

# Perform the likelihood ratio test
import scipy.stats as stats
lr = -2 * (mdf_null.llf - mdf_alt.llf)
p_value = stats.chi2.sf(lr, df=1)  # df is 1 because we are testing one additional parameter (random slope)

print("Likelihood Ratio:", lr)
print("p-value:", p_value)

# %% ORDINAL GLMS

target = 'rewards_tallies'

series = df[target]

df_glm = pd.DataFrame(series.explode())

df_glm['efficacy'] = df['effs'].explode().astype(float)
df_glm['pid'] = df_glm.index.astype(str)

df_glm[target] = df_glm[target].astype(int)

# %%

df_glm['efficacy_C'] = pd.Categorical(df_glm['efficacy'], ordered=True)

# Define the model
# Note: This is a simplification. You might need to adjust the formula to fit your specific needs.
#model_formula = target + ' ~ C(efficacy_C) + (1|pid)'

# Fit the model
#mixed_glm = smf.mixedlm(model_formula, df_glm, family=sm.families.Poisson(), groups=df_glm['pid'],).fit()
mixed_glm = smf.mixedlm(target + " ~ C(efficacy_C)", df_glm, groups=df_glm["pid"]).fit()

# Print the summary
print(mixed_glm.summary())

# %% ORDINAL GLMS FACTORING IN GROUPS

target = 'rewards_tallies'

series = df[target]

df_glm = pd.DataFrame(series.explode())

df_glm['efficacy'] = df['effs'].explode().astype(float)

## merge with group
df_glm['group'] = df['group']

df_glm['pid'] = df_glm.index.astype(str)

df_glm[target] = df_glm[target].astype(int)

# %%

df_glm['efficacy_C'] = pd.Categorical(df_glm['efficacy'], ordered=True)
df_glm['group_C'] = pd.Categorical(df_glm['group'], ordered=False)

# Define the model
# Note: This is a simplification. You might need to adjust the formula to fit your specific needs.
#model_formula = target + ' ~ C(efficacy_C) + (1|pid)'

# Fit the model
#mixed_glm = smf.mixedlm(model_formula, df_glm, family=sm.families.Poisson(), groups=df_glm['pid'],).fit()
mixed_glm = smf.mixedlm(target + " ~ C(efficacy_C) + C(group_C)", df_glm, groups=df_glm["pid"]).fit()

# Print the summary
print(mixed_glm.summary())

# %% WITH CONTINUOUS EFFICACY

target = 'rewards_tallies'

series = df[target]
df_glm = pd.DataFrame(series.explode())

df_glm['efficacy'] = df['effs'].explode().astype(float)
df_glm['pid'] = df_glm.index.astype(str)
df_glm[target] = df_glm[target].astype(int)

# No need to convert 'efficacy' to categorical for a continuous variable
# Define the model treating 'efficacy' as a continuous variable
model_formula = target + ' ~ efficacy'

# Fit the model
mixed_glm = smf.mixedlm(model_formula, df_glm, groups=df_glm['pid']).fit(method='nm')

# Print the summary
print(mixed_glm.summary())

# %% 

# Group by 'efficacy_index' and calculate mean and SEM
grouped_stats = df_glm.groupby('efficacy_C')['rewards_tallies'].agg(['mean', 'sem'])

# Resetting index for better formatting
grouped_stats = grouped_stats.reset_index()

# Display the resulting DataFrame
print(grouped_stats)

 # %% CLASSIFICATION INTO GROUPS BASED ON HOW WELL THEY MATCH NN'S OBSERVATION RATES

eobs = np.flip(np.array([7.10666667, 6.76555556, 6.22444444, 5.22,       3.95444444, 2.76777778,
    1.89555556, 1.27333333, 1.10666667]))

nobs_train, nobs_test = sort_train_test(df['n_observes'], df['effs'], test_start)

eobs_train = get_evs_wanted(eobs, effs, effs_train)
eobs_test = get_evs_wanted(eobs, effs, effs_test)

signed_dev_obs_train = np.mean(nobs_train - eobs_train, axis=1)
signed_dev_obs_test = np.mean(nobs_test - eobs_test, axis=1)

abs_dev_obs_train = np.mean(np.abs(nobs_train - eobs_train), axis=1)
abs_dev_obs_test = np.mean(np.abs(nobs_test - eobs_test), axis=1)

bins = np.linspace(-5, 25, 30)

# %%

fig = plt.figure(dpi=300)
ax = sns.histplot(signed_dev_obs_train, color='blue', label='Train', bins=bins)
sns.histplot(signed_dev_obs_test, color='red', label='Test', bins=bins)
plt.legend()

ax.set_xlabel("Mean Signed Deviation")

format_axis(ax)

fig.savefig(os.path.join(analysis_folder, '%s_signed_dev_obs_histogram.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_signed_dev_obs_histogram.svg' %get_timestamp()))

# %% 

fig = plt.figure(dpi=300)
ax = sns.histplot(abs_dev_obs_train, color='blue', label='Train', bins=bins)
sns.histplot(abs_dev_obs_test, color='red', label='Test', bins=bins)
plt.legend()
ax.set_xlabel("Mean Absolute Deviation")

format_axis(ax)

fig.savefig(os.path.join(analysis_folder, '%s_abs_dev_obs_histogram.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_abs_dev_obs_histogram.svg' %get_timestamp()))

# %% DIFFERENCE IN OBS RATES

fig = plt.figure(dpi=300)
ax = sns.histplot(abs_dev_obs_train - abs(signed_dev_obs_train), color='blue', label='Train', bins=bins)
sns.histplot(abs_dev_obs_test - abs(signed_dev_obs_test), color='red', label='Test', bins=bins)
plt.legend()
ax.set_xlabel("Mean Difference Deviations")

format_axis(ax)

fig.savefig(os.path.join(analysis_folder, '%s_diffss_dev_obs_histogram.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_diffss_dev_obs_histogram.svg' %get_timestamp()))


# %% CLASSIFICATION INTO GROUPS

signed_dev_cutoff_lower = -3
signed_dev_cutoff_higher = 10

mask_group_low = signed_dev_obs_test < signed_dev_cutoff_lower
mask_group_high = signed_dev_obs_test > signed_dev_cutoff_higher

mask = ~mask_group_low & ~mask_group_high

print("Size of low group: %i" %np.sum(mask_group_low))
print("Size of high group: %i" %np.sum(mask_group_high))

train_mask_group_low = signed_dev_obs_train < signed_dev_cutoff_lower
train_mask_group_high = signed_dev_obs_train > signed_dev_cutoff_higher

## calculate percent overlap between train and test groups

print("Percent overlap between train and test groups: %.2f" %((np.sum(mask_group_low & train_mask_group_low) + np.sum(mask_group_high & train_mask_group_high) )/ (np.sum(mask_group_low) + np.sum(mask_group_high)) * 100))

# RESULTS FOR JOINED 518-525-618-706 DATASET

# Size of low group: 38
# Size of high group: 1
# Percent overlap between train and test low groups: 69.23

# %% 

fig = plt.figure(dpi=300)
ax = sns.histplot(abs_dev_obs_train[mask], color='blue', label='Train', bins=bins)
sns.histplot(abs_dev_obs_test[mask], color='red', label='Test', bins=bins)
plt.legend()
ax.set_xlabel("Mean Absolute Deviation")

format_axis(ax)

fig.savefig(os.path.join(analysis_folder, '%s_masked_abs_dev_obs_histogram.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_masked_abs_dev_obs_histogram.svg' %get_timestamp()))

# %% 

high_abs_cutoff = 4

mask_group_high_abs = (abs_dev_obs_test > high_abs_cutoff) & mask

print("Size of high abs group for threshold %d: %i" %(high_abs_cutoff, np.sum(mask_group_high_abs)))

# Size of high abs group for threshold 5: 9
# Size of high abs group for threshold 4: 17

mask = mask & ~mask_group_high_abs


# %% 

fig = plt.figure(dpi=300)
ax = sns.histplot(abs_dev_obs_train[~mask_group_low & ~mask_group_high] - abs(signed_dev_obs_train[~mask_group_low & ~mask_group_high]), color='blue', label='Train', bins=bins)
sns.histplot(abs_dev_obs_test[~mask_group_low & ~mask_group_high] - abs(signed_dev_obs_test[~mask_group_low & ~mask_group_high]), color='red', label='Test', bins=bins)
plt.legend()
ax.set_xlabel("Mean Difference Deviations")

format_axis(ax)

# fig.savefig(os.path.join(analysis_folder, '%s_masked_diffss_dev_obs_histogram.png' %get_timestamp()))
# fig.savefig(os.path.join(analysis_folder, '%s_masked_diffss_dev_obs_histogram.svg' %get_timestamp()))

# %% CLASSIFICATION BASED ON SLOPES IN OBSERVATION RATES

mso_train, mso_test = mean_slope_train_test(df['n_observes'], df['effs'], effs_train, effs_test)

fig = plt.figure(dpi=300)
# ax = sns.histplot(mso_train - np.diff(eobs_train).mean(), color='blue', label='Train', bins=bins)
# sns.histplot(mso_test - np.diff(eobs_test).mean(), color='red', label='Test', bins=bins)
ax = sns.histplot(mso_train[mask], color='blue', label='Train', bins=bins)
sns.histplot(mso_test[mask], color='red', label='Test', bins=bins)

plt.legend()
ax.set_xlabel("Mean Slope of Observation Rates")

format_axis(ax)

## FOR NNS:
# DIFF OVER TRAIN: 1.373
# DIFF OVER TEST: 2

# %% CLASSIFY DIFFERENCE IN DEVIATION OF OBSERVATION RATES

dev_slope_train = mso_train - np.diff(eobs_train).mean()
dev_slope_test = mso_test - np.diff(eobs_test).mean()

## t-test between train and test

print(ttest_rel(dev_slope_train, dev_slope_test))
print("DoF", len(dev_slope_train) - 1 )

# %% CHECK AVERAGE TOTAL REWARD FOR MASKED AND UNMASKED

rews_test, rews_train = sort_train_test(df['rewards_tallies'], df['effs'], test_start)
rews_test = rews_test.mean(axis=1)
rews_train = rews_train.mean(axis=1)
outliers_test, outliers_train = rews_test[~mask], rews_train[~mask]
ingroup_test, ingroup_train = rews_test[mask], rews_train[mask]

print("Average total reward for outliers in test: %.2f +/- %.2f" %(outliers_test.mean(), outliers_test.std()/np.sqrt(len(outliers_test))))
print("Average total reward for outliers in train: %.2f +/- %.2f" %(outliers_train.mean(), outliers_train.std()/np.sqrt(len(outliers_train))))
print("Average total reward for ingroup in test: %.2f +/- %.2f" %(ingroup_test.mean(), ingroup_test.std()/np.sqrt(len(ingroup_test))))
print("Average total reward for ingroup in train: %.2f +/- %.2f" %(ingroup_train.mean(), ingroup_train.std()/np.sqrt(len(ingroup_train))))

## perform t-test on train

print("Ttest training", ttest_ind(outliers_train, ingroup_train, equal_var=False))
print("DoF", len(outliers_train) + len(ingroup_train) - 2)

## perform t-test on test

print("Ttest test", ttest_ind(outliers_test, ingroup_test, equal_var=False))
print("DoF", len(outliers_test) + len(ingroup_test) - 2)

# Average total reward for outliers in test: 24.76 +/- 0.30
# Average total reward for outliers in train: 24.59 +/- 0.31
# Average total reward for ingroup in test: 26.33 +/- 0.20
# Average total reward for ingroup in train: 27.32 +/- 0.28
# Ttest training Ttest_indResult(statistic=-6.4518802206513755, pvalue=2.080225799658255e-09)
# DoF 147
# Ttest test Ttest_indResult(statistic=-4.361069428701785, pvalue=3.059660425171836e-05)
# DoF 147

# %% CHECK SLIDER CORRELATION WITH REWS TEST AND TRAIN

#from scipy.stats import spearmanr

df_with_slider = df[mask][df[mask]['efficacy_estimates'	].notnull()]  # remove nans

## get slider values

slider_train, slider_test = sort_train_test(df_with_slider['efficacy_estimates'], df_with_slider['effs'], test_start)
#slider_mse_test = abs(slider_test - effs_test).flatten()
#slider_mse_train = abs(slider_train - effs_train).flatten()

slider_mse_test = ((slider_test - effs_test)**2).mean(axis=1)
slider_mse_train = ((slider_train - effs_train)**2).mean(axis=1)

## get rews

rews_train, rews_test = sort_train_test(df_with_slider['rewards_tallies'], df_with_slider['effs'], test_start)
#rews_train = rews_train.flatten()
#rews_test = rews_test.flatten()

rews_train = rews_train.mean(axis=1)
rews_test = rews_test.mean(axis=1)

## get correlation

print("Correlation slider and rews test", pearsonr(slider_mse_test, rews_test))
print("Correlation slider and rews train", pearsonr(slider_mse_train, rews_train))
# print("Correlation slider and rews test", spearmanr(slider_mse_test, rews_test))
# print("Correlation slider and rews train", spearmanr(slider_mse_train, rews_train))
print("Number of subjects", len(slider_mse_test))

## RESULTS FOR DAYS 5-18 5-25 6-19- 7-06
## WHOLE GROUP
# Correlation slider and rews test PearsonRResult(statistic=-0.16472829327741084, pvalue=0.11458931160037428)
# Correlation slider and rews train PearsonRResult(statistic=0.003550006405161843, pvalue=0.9730588815508745)
# Number of subjects 93

## MASKED SUBPORTION
# Correlation slider and rews test PearsonRResult(statistic=-0.28600575228328434, pvalue=0.04406306913447283)
# Correlation slider and rews train PearsonRResult(statistic=0.16219412791416937, pvalue=0.26044133104146383)
# Number of subjects 50

# %% SCATTER

fig = plt.figure(dpi=300)

ax = sns.scatterplot(x=slider_mse_train, y=rews_train, color='blue', label='Train')
sns.scatterplot(x=slider_mse_test, y=rews_test, color='red', label='Test')

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

# %% MEAN LOGITS APE PLOT

fig = plot_line_scatter(np.stack(df['mean_lik_ape_test'].values), effs_test, np.stack(df['mean_lik_ape_train'].values), effs_train, ylabel='Mean Likelihood APE-trained', median_over_mean = True, xjitter=0.025, yjitter=0, true_values=[1/3]*9, effs_true=effs, true_label='Random', true_color='red', ylim=(0,1))
fig.savefig(os.path.join(analysis_folder, '%s_ml_line_scatter.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_ml_line_scatter.svg' %get_timestamp()))

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

fig = plot_line_scatter(comparisons_test, effs_test, comparisons_train, effs_train, ylabel='Log Likelihood Ratio', median_over_mean = True, xjitter=0.025, yjitter=0, true_values=[0]*9, effs_true=effs, true_label='Even LLR', true_color='red')
fig.savefig(os.path.join(analysis_folder, '%s_llr_line_scatter.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_llr_line_scatter.svg' %get_timestamp()))

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

# %% MEAN LOGIT - STEPWISE MAX

msml_train, msml_test = sort_train_test(df['mean_stepmax_l_ape'], df['effs'], test_start)

# %% 

#fig = plot_line_scatter(np.stack(np.array(msml_test)), effs_test, np.stack(np.array(msml_train)), effs_train, ylabel='Mean Likelihood APE-trained', median_over_mean = True, xjitter=0.025, yjitter=0, true_values=[1/3]*9, effs_true=effs, true_label='Random', true_color='red')
fig = plot_line_scatter(np.stack(np.array(msml_test)), effs_test, np.stack(np.array(msml_train)), effs_train, ylabel='Mean Likelihood APE-trained', median_over_mean = True, xjitter=0.025, yjitter=0, true_values=[1/3]*9, effs_true=effs, true_label='Random', true_color='red', ylim=(0,1))
fig.savefig(os.path.join(analysis_folder, '%s_mstepmaxl_line_scatter.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_mstepmaxl_line_scatter.svg' %get_timestamp()))

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

# %% experimenting with alternate method of stepwise maximum

stepmax_ll_ape_train, stepmax_ll_ape_test = sort_train_test(df['step_max_ll_ape'], df['effs'], test_start)

# = sort_train_test(df['step_max_ll_control'], df['effs'], test_start)

# %%

fig = plot_violin(stepmax_ll_ape_test, effs_test, stepmax_ll_ape_train, effs_train, ylabel='Stepmax Log Likelihoods', median_over_mean = True)
fig.savefig(os.path.join(analysis_folder, '%s_stepmax_ll_violin.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_stepmax_ll_violin.svg' %get_timestamp()))

# %%

idx_train_sorted = np.argsort(stepmax_ll_ape_train, axis=0)
min_idx = idx_train_sorted[0, 2]
median_idx = idx_train_sorted[len(idx_train_sorted)//2, 2]
max_idx = idx_train_sorted[-1, 2]
print("eff 05", df.index[min_idx], df.index[median_idx], df.index[max_idx])
print("lls", stepmax_ll_ape_train[min_idx, 2], stepmax_ll_ape_train[median_idx, 2], stepmax_ll_ape_train[max_idx, 2])

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

# %% find LLR

stepmax_ll_control_train, stepmax_ll_control_test = sort_train_test(df['step_max_ll_control'], df['effs'], test_start)

# %% find top 

mean_ll_ape_train, mean_ll_ape_test = get_mean_ll(df, ape_models, test_start, aggregate_efficacies=False)
mean_ll_control_train, mean_ll_control_test = get_mean_ll(df, control_models, test_start, aggregate_efficacies=False)
#mean_ll_ape_train, mean_ll_ape_test = get_max_ll(df, ape_models, test_start, aggregate_efficacies=False)
df['mean_ll_ape_train'], df['mean_ll_ape_test'] = mean_ll_ape_train.tolist(), mean_ll_ape_test.tolist()

# %%

idx_train_sorted = np.argsort(mean_ll_ape_train, axis=0)
min_idx = idx_train_sorted[0, 2]
median_idx = idx_train_sorted[len(idx_train_sorted)//2, 2]
max_idx = idx_train_sorted[-1, 2]
print("eff 05", df.index[min_idx], df.index[median_idx], df.index[max_idx])
print("lls", stepmax_ll_ape_train[min_idx, 2], stepmax_ll_ape_train[median_idx, 2], stepmax_ll_ape_train[max_idx, 2])

# %%

idx_sorted = np.argsort(mean_ll_ape_test, axis=0)
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

# %% PCA OF OBSERVATIONS AND REWARDS

n_obss = np.stack(df['n_observes'].values)

# Standardize the data
scaler = StandardScaler()
n_obss_std = scaler.fit_transform(n_obss)

# Initialize and apply PCA
pca = PCA(n_components=3)
red_obss = pca.fit_transform(n_obss_std)

# Output the reduced data
print(red_obss)

# Output proportion of variance explained
print("Proportion of Variance Explained: ", pca.explained_variance_ratio_)

# Proportion of Variance Explained:  [0.70965795 0.05995176 0.04936782]
# %%

# For each principal component
for i, component in enumerate(pca.components_):
    print(f"Principal Component {i+1}")
    # For each loading in the component
    for j, loading in enumerate(component):
        print(f"Variable {j+1}: {loading}")
    print()

## FIRST PC COULD ROUGHLY CORRESPOND TO "GENERAL OBSERVATION LEVEL",
## 2, 3 ETC TO ADAPTIVENESS (ALONG DIFFERENT DIMENSIONS)

# Principal Component 1
# Variable 1: 0.3340541137666766
# Variable 2: 0.3010618382128582
# Variable 3: 0.35935235226542805
# Variable 4: 0.32738893047071904
# Variable 5: 0.3429119770062947
# Variable 6: 0.3403953791801037
# Variable 7: 0.34303652587265093
# Variable 8: 0.3209656934391217
# Variable 9: 0.32756880374179637

# Principal Component 2
# Variable 1: -0.22820427447966565
# Variable 2: -0.7818236618398504
# Variable 3: 0.15003779227197106
# Variable 4: 0.3220942027951023
# Variable 5: 0.1616344438716502
# Variable 6: 0.12251999097202927
# Variable 7: -0.05310167218230862
# Variable 8: -0.15620571449682455
# Variable 9: 0.3769107345897335

# Principal Component 3
# Variable 1: 0.06558900136921308
# Variable 2: 0.26087342202041586
# Variable 3: -0.11587447325380625
# Variable 4: 0.12035047752828268
# Variable 5: -0.36869086422292774
# Variable 6: 0.1632180280581869
# Variable 7: 0.26098698439409923
# Variable 8: -0.7335111770367004
# Variable 9: 0.361947618702032

# %% RECONSTRUCT DATA

# Reconstructing the original data
reconstructed_obss_std = np.dot(red_obss, pca.components_) + scaler.mean_

# If you want the original scale (before standardization) 
reconstructed_obss = scaler.inverse_transform(reconstructed_obss_std)

# %% PLOT RECONSTRUCTED CORRELATION MATRIX

fig = compute_2D_correlation(reconstructed_obss, reconstructed_obss, effs, effs, col1name = 'Reconstructed Observations', col2name = 'Reconstructed Observations')

# %%

# Plot the first two principal components
plt.figure(figsize=(8,6))
plt.scatter(red_obss[:, 0], red_obss[:, 1])

# Adding biplot vectors
for i, (x, y) in enumerate(zip(pca.components_[0], pca.components_[1])):
    plt.arrow(0, 0, x, y, color='r', alpha=0.5)
    plt.text(x * 1.1, y * 1.1, f"Variable {i+1}", color='b', ha='center', va='center')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Biplot")
plt.grid(True)
plt.show()

# %% REWARDS

rewss = np.stack(df['rewards_tallies'].values)

# Standardize the data
scaler = StandardScaler()
rewss_std = scaler.fit_transform(rewss)

# Initialize and apply PCA
pca = PCA(n_components=3)
rew_red = pca.fit_transform(rewss_std)

# Output the reduced data
print(rew_red)

# Output proportion of variance explained
print("Proportion of Variance Explained: ", pca.explained_variance_ratio_)
# Proportion of Variance Explained:  [0.15497668 0.14084374 0.13483612]

# %%

# For each principal component
for i, component in enumerate(pca.components_):
    print(f"Principal Component {i+1}")
    # For each loading in the component
    for j, loading in enumerate(component):
        print(f"Variable {j+1}: {loading}")
    print()

# Principal Component 1
## First one generally could be a "no rewards on sides" / general reward level in high and low settings
# Variable 1: -0.3592975757424255
# Variable 2: -0.4044441961212522
# Variable 3: -0.4728349608051196
# Variable 4: -0.07658001135635409
# Variable 5: 0.08532221926141036
# Variable 6: -0.5395032656482263
# Variable 7: -0.2206528073712699
# Variable 8: -0.32061825269983
# Variable 9: -0.16752720937395538

# Principal Component 2
# Variable 1: 0.5850505172062533
# Variable 2: -0.3139731004928142
# Variable 3: -0.03820880565797973
# Variable 4: -0.06496115048826705
# Variable 5: 0.4588826568780206
# Variable 6: -0.1896284691874646
# Variable 7: -0.2552059162178333
# Variable 8: 0.4814322569695077
# Variable 9: -0.10008793008103918

# Principal Component 3
# Variable 1: 0.10742015232351257
# Variable 2: 0.11398203473346535
# Variable 3: 0.04322357153589782
# Variable 4: -0.7111443158744636
# Variable 5: -0.07356327509932205
# Variable 6: 0.22992899197026478
# Variable 7: -0.5658773418468399
# Variable 8: -0.2558738386294568
# Variable 9: 0.15461891020672838

# %% RECONSTRUCTION

# Reconstructing the original data
reconstructed_rew_red_std = np.dot(rew_red, pca.components_) + scaler.mean_

# If you want the original scale (before standardization) 
reconstructed_rew_red = scaler.inverse_transform(reconstructed_rew_red_std)

# %% PLOT

fig = compute_2D_correlation(reconstructed_rew_red, reconstructed_rew_red, effs, effs, col1name = 'Reconstructed Rewards', col2name = 'Reconstructed Rewards')

# %% MANUAL LOADING ONTO TWO VARIABLES IN OBSERVATION/REWARDS SPACE:
# 1. GENERAL OBSERVATION LEVEL
# 2. ADAPTIVENESS

#i.e. we start from mean response in each efficacy level
#

# %% READ IN STAN MODEL PARAMETERS

stan_model_folder = os.path.join(results_folder, 'stan_model')
stan_model_timestamp = '20230928142828'
df_stan = df.copy()

keys = ['alpha', 'c', 'd0', 'd1', 'd_rate', 'sigma', 'likelihood']

# Define a function to extract values from the pickled dictionary file
def extract_values(row, keys, stan_model_timestamp, stan_model_folder):
    # Construct the filename from the row, e.g., '1.pkl' for ID=1
    values = {key: [] for key in keys}

    for eff in effs:
        filename = os.path.join(stan_model_folder, 'subj_%s' %row.name, 'eff%d' %(eff*1000), '%s_analysis_variables.pkl' %stan_model_timestamp)
        
        # Load the dictionary from the pickle file
        with open(filename, 'rb') as file:
            dictionary = pickle.load(file)
        
        # Extract values for the keys and return them as a Series
        for key in keys:
            values[key].append(dictionary.get(key))

    return pd.Series([np.array(values.get(key)) for key in keys], index=keys)

# Apply the function to each row and join the resulting DataFrame with the original one
df_stan = df_stan.join(df.apply(extract_values, axis=1, keys=keys, stan_model_timestamp=stan_model_timestamp, stan_model_folder=stan_model_folder))
    #print(df.apply(extract_values, axis=1, keys=keys, eff=eff, stan_model_timestamp=stan_model_timestamp, stan_model_folder=stan_model_folder))

## sum likelihood so that it can be plotted in the same way as the rest
df_stan['likelihood_mean'] = df_stan['likelihood'].apply(np.mean, args=(1,))
keys[-1] = 'likelihood_mean'

print("DataFrame with New Columns:")
print(df_stan)

stan_analysis_folder = os.path.join(analysis_folder, 'stan', )
os.makedirs(stan_analysis_folder, exist_ok=True)


# %% OVERALL DISTRIBUTIONS OF FITTED PARAMETERS - HISTOGRAMS

#for key in keys:
if True:
    key = keys[-2]

    alphas = np.stack(df_stan[key].values)
    num_columns = alphas.shape[1]

    # Create a color map to color each histogram differently.
    colors = plt.cm.viridis(np.linspace(0, 1, num_columns))

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    # Calculate the overall range of the data
    data_min = np.min(alphas)
    data_max = np.max(alphas)

    # Define the number of bins and calculate the bin edges
    num_bins = 20
    bin_edges = np.linspace(data_min, data_max, num_bins + 1)

    # Plot a histogram for each column.
    for i in range(num_columns):
        plt.hist(alphas[:, i], bins=bin_edges, color=colors[i], alpha=0.3, label=f'Eff {effs[i]}')

    # Optional: Add a legend to label each histogram.
    plt.legend(loc='upper right')

    # Show the plot.
    fig.savefig(os.path.join(stan_analysis_folder, '%s_hist_%s.png' %(get_timestamp(), key)), dpi=300)
    fig.savefig(os.path.join(stan_analysis_folder, '%s_hist_%s.svg' %(get_timestamp(), key)), dpi=300)

# %% OVERALL DISTRIBUTIONS OF FITTED PARAMETERS - VIOLINS

#for key in keys:

key = keys[-2]
fig = plot_violin(np.stack(df_stan[key].values), effs, ylabel=key, median_over_mean = True) 
fig.savefig(os.path.join(stan_analysis_folder, '%s_violin_%s.png' %(get_timestamp(), key)), dpi=300)
fig.savefig(os.path.join(stan_analysis_folder, '%s_violin_%s.svg' %(get_timestamp(), key)), dpi=300)

# %%

df_stan['effs_sorted'] = [list(effs)]*len(df_stan)

for key in keys:
    corr_fig, pvs_fig = compute_2D_correlation(df_stan[key], df_stan[key], df_stan['effs_sorted'],  df_stan['effs_sorted'], key, key)
    corr_fig.savefig(os.path.join(stan_analysis_folder, '%s_2D_correlation_%s.png' %(get_timestamp(), key)), dpi=300)
    corr_fig.savefig(os.path.join(stan_analysis_folder, '%s_2D_correlation_%s.svg' %(get_timestamp(), key)), dpi=300)
    pvs_fig.savefig(os.path.join(stan_analysis_folder, '%s_2D_pvs_%s.png' %(get_timestamp(), key)), dpi=300)
    pvs_fig.savefig(os.path.join(stan_analysis_folder, '%s_2D_pvs_%s.svg' %(get_timestamp(), key)), dpi=300)

plt.close('all')
# %% ANALYZE HIERARCHICAL STAN MODEL

hierarchical_stan_model_folder = os.path.join(results_folder, 'hierarchical_stan_model')
stan_model_timestamp = '20230928161418'
df_stan_h = df.copy()

pars = ['alpha', 'c', 'd0', 'd1', 'sigma']
hierarchical_pars = ['m_', 'b_']

keys = []

for par in pars:
    for hierarchical_par in hierarchical_pars:
        keys.append(hierarchical_par + par)

keys.append('likelihood')

# Define a function to extract values from the pickled dictionary file
def extract_values(row, keys, stan_model_timestamp, stan_model_folder):
    # Construct the filename from the row, e.g., '1.pkl' for ID=1
    values = []

    filename = os.path.join(stan_model_folder, 'subj_%s' %row.name, '%s_analysis_variables.pkl' %stan_model_timestamp)

    # Load the dictionary from the pickle file
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)
    
    # Extract values for the keys and return them as a Series
    for key in keys:
        values.append(dictionary.get(key))

    return pd.Series(values, index=keys)

# Apply the function to each row and join the resulting DataFrame with the original one
df_stan_h = df_stan_h.join(df.apply(extract_values, axis=1, keys=keys, stan_model_timestamp=stan_model_timestamp, stan_model_folder=hierarchical_stan_model_folder))
    #print(df.apply(extract_values, axis=1, keys=keys, eff=eff, stan_model_timestamp=stan_model_timestamp, stan_model_folder=stan_model_folder))

## sum likelihood so that it can be plotted in the same way as the rest
df_stan_h['likelihood_mean'] = df_stan_h['likelihood'].apply(np.mean)
keys[-1] = 'likelihood_mean'

print("DataFrame with New Columns:")
print(df_stan_h)

h_stan_analysis_folder = os.path.join(analysis_folder, 'hierarchical_stan', )
os.makedirs(h_stan_analysis_folder, exist_ok=True)

# %%

for key in keys:

    fig = plot_violin(np.stack(df_stan_h[key].values), [1], ylabel=key, median_over_mean = True, xlabel='All Efficacies') 
    fig.savefig(os.path.join(h_stan_analysis_folder, '%s_violin_%s.png' %(get_timestamp(), key)), dpi=300)
    fig.savefig(os.path.join(h_stan_analysis_folder, '%s_violin_%s.svg' %(get_timestamp(), key)), dpi=300)

# %% ANALYZE NEURAL NETWORK FITS - SIMPLE MODELS

def convert_to_numpy(c):
    try: 
        c = c.detach().numpy()
    except AttributeError as e:
        print('attribute error', e, c)
        c = np.array(c)
    return c

df_nns = pd.read_pickle('results/NN/day2/518-525-619-706/20231003112434_fitted_nns_df_lr01.pkl')
df_nns_simplified = df_nns.copy()
df_nns_simplified['raw_coeffss'] = df_nns['raw_coeffss'].apply(lambda x: [c.detach().numpy() for c in x])
df_nns_simplified['coeffss'] = df_nns['coeffss'].apply(lambda x: [c.detach().numpy() for c in x])
df_nns_simplified['final_losses'] = df_nns['final_losses'].apply(lambda x: [c.detach().numpy() for c in x])
df_nns_simplified['initial_losses'] = df_nns['initial_losses'].apply(lambda x: [convert_to_numpy(c) for d in x for c in d])

df_nns_simplified.to_pickle('results/NN/day2/518-525-619-706/20231003112434_fitted_nns_df_lr01_simplified.pkl')

# %%

df_nns_simplified = pd.read_pickle('results/NN/day2/518-525-619-706/20231003112434_fitted_nns_df_lr01_simplified.pkl')

# %% NNS ANALYSIS

nns_analysis_folder = os.path.join(analysis_folder, 'nns', )
os.makedirs(nns_analysis_folder, exist_ok=True)

# %%

for key in ['initial_liks', 'final_liks']:
    fig = plot_violin(np.stack(df_nns_simplified[key].values).mean(axis=2), effs, ylabel=key, median_over_mean = True, xlabel='Efficacies') 
    fig.savefig(os.path.join(nns_analysis_folder, '%s_violin_%s.png' %(get_timestamp(), key)), dpi=300)
    fig.savefig(os.path.join(nns_analysis_folder, '%s_violin_%s.svg' %(get_timestamp(), key)), dpi=300)

# %%

from scipy.stats import entropy

coeffs = np.stack(df_nns_simplified['coeffss'].values)
entropies = np.apply_along_axis(entropy, 2, coeffs)

# %%

key = 'policy_entropy'
fig = plot_violin(entropies, effs, ylabel=key, median_over_mean = True) 
fig.savefig(os.path.join(nns_analysis_folder, '%s_violin_%s.png' %(get_timestamp(), key)), dpi=300)
fig.savefig(os.path.join(nns_analysis_folder, '%s_violin_%s.svg' %(get_timestamp(), key)), dpi=300)

# %% NN PERTURBATIONS

## read in df from pickle
df_nns_perturbations = pd.read_pickle('results/perturbation_only_NN/day2/518-525-619-706/20231023013412_perturbation_only_nns_df_lr05.pkl')

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

# %% PARTIALED CORRELATION OF PERTURBATIONS

df_nns_perturbations['sorted_effs'] = [effs.copy() for _ in range(len(df))]

corr_fig, pvs_fig = compute_partial_2D_correlation(df_nns_perturbations['perturbation'], df_nns_perturbations['perturbation'], df['rewards_tallies'], df_nns_perturbations['sorted_effs'], df_nns_perturbations['sorted_effs'], df['effs'], semi=False)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.png' %(get_timestamp(), 'perturbation', 'perturbation', 'n_rewards')), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.svg' %(get_timestamp(), 'perturbation', 'perturbation', 'n_rewards')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.png' %(get_timestamp(), 'perturbation', 'perturbation', 'n_rewards')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.svg' %(get_timestamp(), 'perturbation', 'perturbation', 'n_rewards')), dpi=300)

# %% PARTIALED CORRELATION N_OBSREVES

corr_fig, pvs_fig = compute_partial_2D_correlation(df['n_observes'], df['n_observes'], df['rewards_tallies'], df['effs'], df['effs'], df['effs'], semi=False)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.png' %(get_timestamp(), 'n_observes', 'n_observes', 'n_rewards')), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.svg' %(get_timestamp(), 'n_observes', 'n_observes', 'n_rewards')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.png' %(get_timestamp(), 'n_observes', 'n_observes', 'n_rewards')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.svg' %(get_timestamp(), 'n_observes', 'n_observes', 'n_rewards')), dpi=300)

# %% PARTIALED CORRELATION PERTURBATIONS WITH NETWORK LIKELIHOODS

df_nns_perturbations['mean_ll_ape'] = df_nns_perturbations.apply(lambda x: combine_train_test(x['mean_ll_ape_train'], x['mean_ll_ape_test'], effs_train, effs_test), axis=1)

# %%

corr_fig, pvs_fig = compute_partial_2D_correlation(df_nns_perturbations['perturbation'], df_nns_perturbations['perturbation'], df_nns_perturbations['mean_ll_ape_train'], df_nns_perturbations['sorted_effs'], df_nns_perturbations['sorted_effs'], df_nns_perturbations['sorted_effs'], semi=False)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.png' %(get_timestamp(), 'perturbation', 'perturbation', 'n_rewards')), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.svg' %(get_timestamp(), 'perturbation', 'perturbation', 'n_rewards')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.png' %(get_timestamp(), 'perturbation', 'perturbation', 'n_rewards')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.svg' %(get_timestamp(), 'perturbation', 'perturbation', 'n_rewards')), dpi=300)

# %% CALCULATE CORRELATION STRUCTURE FOR N_OBSERVES AND SEE IF IT CAN BE EXPLAINED BY 

## True process
corr_data = np.corrcoef(np.stack(df['n_observes'].values).T)

## Set up candidate correlation matrices
corr_hyp = np.ones((len(effs), len(effs)))*0.5 + np.eye(len(effs))*0.5
corr_null = np.eye(len(effs))

# %% 
competitive_corr_regression(corr_data, [corr_hyp, corr_null])

# %% ANALYZE CORRELATION STRUCTURE FOR N_REWARDS

## True process
data = np.stack(df['rewards_tallies'].values)
corr_data = np.corrcoef(np.stack(df['rewards_tallies'].values).T)

## Set up candidate correlation matrices
corr_hyp = np.block([[np.ones((4,4)), np.zeros((4,1)), np.ones((4,4))*-1], 
                     [np.zeros((1,4)), np.ones((1,1)), np.zeros((1,4))],
                     [np.ones((4,4))*-1, np.zeros((4,1)), np.ones((4,4))]])
corr_hyp = corr_hyp * 0.5 + 0.5 * np.eye((9))
corr_null = np.eye(len(effs))

# %%

# Fisher z-transform function
def fisher_transform(correlation_matrix):
    return 0.5 * np.log((1 + correlation_matrix + np.finfo(float).eps) / (1 - correlation_matrix + np.finfo(float).eps))

# Applying the transform to each matrix
z_observed = fisher_transform(corr_data)
z_candidate_1 = fisher_transform(corr_hyp)
z_candidate_2 = fisher_transform(corr_null)

# Flattening the matrices to 1D arrays, removing the diagonal elements [assuming symmetric matrix and 1s on the diagonal]
#z_observed_flat = z_observed[np.triu_indices(z_observed.shape[0], k = 1)]
#z_candidate_1_flat = z_candidate_1[np.triu_indices(z_candidate_1.shape[0], k = 1)]
#z_candidate_2_flat = z_candidate_2[np.triu_indices(z_candidate_2.shape[0], k = 1)]
z_observed_flat = z_observed.flatten()
z_candidate_1_flat = z_candidate_1.flatten()
z_candidate_2_flat = z_candidate_2.flatten()

# Building the regression model
X = np.column_stack(( z_candidate_1_flat, z_candidate_2_flat))#np.random.random((36,))))#z_candidate_2_flat, z_candidate_1_flat,))
X = sm.add_constant(X)  # Adding a constant term to the predictors
model = sm.OLS(z_observed_flat, X).fit()

# Displaying the results
print(model.summary())

# %%

# Applying ridge regression
ridge_model = Ridge(alpha=1.0)  # alpha is the regularization strength
ridge_model.fit(X[:, 1:], z_observed_flat)  # note: Ridge automatically adds a constant term

# Display results
print("Ridge coefficients: ", ridge_model.coef_)
# %% WITH MASKING

## True process
data = np.stack(df['rewards_tallies'].values)
corr_data = np.corrcoef(np.stack(df['rewards_tallies'].values).T)

p_value_matrix = np.zeros_like(corr_data)
n_vars = corr_data.shape[1]

for i in range(n_vars):
    for j in range(n_vars):
        _, p_value_matrix[i, j] = scipy.stats.pearsonr(data[:, i], data[:, j])

# Step 2: Mask non-significant correlations
significance_level = 0.05  # or your chosen alpha level
mask = (p_value_matrix > significance_level)
corr_data_masked = np.where(mask, np.nan, corr_data)
corr_data = corr_data_masked

## Set up candidate correlation matrices
corr_hyp = np.block([[np.ones((4,4)), np.zeros((4,1)), np.ones((4,4))*-1], 
                     [np.zeros((1,4)), np.ones((1,1)), np.zeros((1,4))],
                     [np.ones((4,4))*-1, np.zeros((4,1)), np.ones((4,4))]])
corr_hyp = corr_hyp * 0.5 + 0.5 * np.eye((9))
corr_null = np.eye(len(effs))

# %%

# Create mask for NaN values
nan_mask = ~np.isnan(corr_data)

# Apply mask to vectors (only keeping non-NaN instances)
corr_data_vec = corr_data[nan_mask].flatten()
corr_hyp1_vec = corr_hyp[nan_mask].flatten()
corr_hyp2_vec = corr_null[nan_mask].flatten()

# Data for OLS model
X = np.vstack((corr_hyp1_vec, corr_hyp2_vec)).T
X = sm.add_constant(X)  # Add a constant term to the independent variables
y = corr_data_vec

# Fit OLS model
model = sm.OLS(y, X, missing='drop').fit()

# Show results
print(model.summary())

# %% PLOT GROUND TRUTH CORRELATION MATRIX

corr_fig, pvs_fig = compute_2D_correlation(df['rewards_tallies'], df['rewards_tallies'], df['effs'], df['effs'], col1name = 'Rewards', col2name='Rewards', annot=False, resize_colorbar = True)

corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_rews.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_rews.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% 

corr_fig, pvs_fig = compute_2D_correlation(df['n_observes'], df['n_observes'], df['effs'], df['effs'], col1name = 'Observes', col2name='Observes', annot=False, resize_colorbar = True)

corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_rews.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_rews.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)


# %% IMPORT SIMULATED PARTICIPANT TRAJECTORIES

### DIFFERENT MAG LEVELS
#simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/pepe/sim'
simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/pepe/sim1000/mag100'
#simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/pepe/sim/mag50'
#simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/pepe/sim/mag200'

#modelname = '20230922111413'
#modelname = '20230923060013'

### TIMESTAMP
#sim_timestamp = '20231007230234'
#sim_timestamp = '20231008220417'
#sim_timestamp = '20231008220818'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
#sim_timestamp = '20231015161128' ## Matching mag 3
sim_timestamp = '20240219163433' ## Matching mag 1
#sim_timestamp = '20240219225152' ## matching mag 0.5
#sim_timestamp = '20240219224009' ## matching mag 2

sim_rewss, sim_obss, _ = load_simulated_participants_across_models(simulated_participants_folder, ape_models, sim_timestamp)
sim_rewss = sim_rewss.mean(axis=0)
sim_obss = sim_obss.mean(axis=0)

sim_parts_analysis_folder = os.path.join(analysis_folder, 'simulated_participants', 'across_models')
os.makedirs(sim_parts_analysis_folder, exist_ok=True)

# %% SIMULATED OBSERVATIONS

corr_fig, pvs_fig = compute_2D_correlation(sim_obss.T, sim_obss.T, effs, effs, "simulated observations", "simulated observations", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% SIMULATED REWARDS

corr_fig, pvs_fig = compute_2D_correlation(sim_rewss.T, sim_rewss.T, effs, effs, "Rewards", "Rewards", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_rew.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_rew.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs.svg' %(get_timestamp())), dpi=300)

# %% NN SIMULATED PARTICIPANTS W/ NO STRUCTURE

simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/pepe/nostruc'
#modelname = '20230922111413'
#modelname = '20230923060013'

#sim_timestamp = '20231008220307'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
sim_timestamp = '20231015161128'

nostruc_rewss, nostruc_obss, _ = load_simulated_participants_across_models(simulated_participants_folder, ape_models, sim_timestamp)

nostruc_rewss = nostruc_rewss.mean(axis=0)
nostruc_obss = nostruc_obss.mean(axis=0)

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

# %% RANDOM SIMULATED PARTICIPANTS (i.e. with perturbations but not structured)

simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/pepe/random'
#modelname = '20230922111413'
modelname = '20230923060013'
#sim_timestamp = '20231008220307'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
sim_timestamp = '20231015161128'

random_rewss, random_obss, _ = load_simulated_participants_across_models(simulated_participants_folder, ape_models, sim_timestamp)

random_rewss = random_rewss.mean(axis=0)
random_obss = random_obss.mean(axis=0)

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


# %% COMPETITIVE REGRESSION

if group is not None:
    data_obs_corr, data_obs_pvs = compute_2D_correlation_matrices(np.stack(df['n_observes'].values), np.stack(df['n_observes'].values), np.stack(df['effs'].values), np.stack(df['effs'].values),)
    data_rews_corr, data_rews_pvs = compute_2D_correlation_matrices(np.stack(df['rewards_tallies'].values), np.stack(df['rewards_tallies'].values), np.stack(df['effs'].values), np.stack(df['effs'].values),)
else:
    data_obs_corr_g1, data_obs_pvs_g1 = compute_2D_correlation_matrices(np.stack(df[~df['group']]['n_observes'].values), np.stack(df[~df['group']]['n_observes'].values), np.stack(df[~df['group']]['effs'].values), np.stack(df[~df['group']]['effs'].values),)
    data_obs_corr_g2, data_obs_pvs_g2 = compute_2D_correlation_matrices(np.stack(df[df['group']]['n_observes'].values), np.stack(df[df['group']]['n_observes'].values), np.stack(df[df['group']]['effs'].values), np.stack(df[df['group']]['effs'].values),)

    data_rews_corr_g1, data_rews_pvs_g1 = compute_2D_correlation_matrices(np.stack(df[~df['group']]['rewards_tallies'].values), np.stack(df[~df['group']]['rewards_tallies'].values), np.stack(df[~df['group']]['effs'].values), np.stack(df[~df['group']]['effs'].values),)
    data_rews_corr_g2, data_rews_pvs_g2 = compute_2D_correlation_matrices(np.stack(df[df['group']]['rewards_tallies'].values), np.stack(df[df['group']]['rewards_tallies'].values), np.stack(df[df['group']]['effs'].values), np.stack(df[df['group']]['effs'].values),)

    data_obs_corr = (~df['group']).sum() / len(df) * data_obs_corr_g1 + (df['group']).sum() / len(df) * data_obs_corr_g2
    data_obs_pvs = (~df['group']).sum() / len(df) * data_obs_pvs_g1 + (df['group']).sum() / len(df) * data_obs_pvs_g2

    data_rews_corr = (~df['group']).sum() / len(df) * data_rews_corr_g1 + (df['group']).sum() / len(df) * data_rews_corr_g2
    data_rews_pvs = (~df['group']).sum() / len(df) * data_rews_pvs_g1 + (df['group']).sum() / len(df) * data_rews_pvs_g2
sim_obs_corr, sim_obs_pvs = compute_2D_correlation_matrices(sim_obss.T, sim_obss.T, effs, effs,)
sim_rews_corr, sim_rews_pvs = compute_2D_correlation_matrices(sim_rewss.T, sim_rewss.T, effs, effs,)
nostruc_obs_corr, nostruc_obs_pvs = compute_2D_correlation_matrices(nostruc_obss.T, nostruc_obss.T, effs, effs,)
nostruc_rews_corr, nostruc_rews_pvs = compute_2D_correlation_matrices(nostruc_rewss.T, nostruc_rewss.T, effs, effs,)
random_obs_corr, random_obs_pvs = compute_2D_correlation_matrices(random_obss.T, random_obss.T, effs, effs,)
random_rews_corr, random_rews_pvs = compute_2D_correlation_matrices(random_rewss.T, random_rewss.T, effs, effs,)
null_obs_corr, null_pvs_corr = np.eye(len(effs)), np.eye(len(effs))
null_rews_corr, null_rews_corr = np.eye(len(effs)), np.eye(len(effs))

# %% 

competitive_corr_regression(upper_tri_masking(data_obs_corr), [upper_tri_masking(sim_obs_corr), upper_tri_masking(nostruc_obs_corr), upper_tri_masking(random_obs_corr)], do_fisher_transform=True)

# %%
competitive_corr_regression(upper_tri_masking(data_rews_corr), [upper_tri_masking(sim_rews_corr), upper_tri_masking(nostruc_rews_corr), upper_tri_masking(random_obs_corr)])

##### OBSERVATIONS
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.471
# Model:                            OLS   Adj. R-squared:                  0.421
# Method:                 Least Squares   F-statistic:                     9.479
# Date:                Mon, 22 Jan 2024   Prob (F-statistic):           0.000125
# Time:                        14:27:56   Log-Likelihood:                 16.117
# No. Observations:                  36   AIC:                            -24.23
# Df Residuals:                      32   BIC:                            -17.90
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.4911      0.108      4.562      0.000       0.272       0.710
# x1             0.2070      0.046      4.466      0.000       0.113       0.301
# x2            -0.6162      0.320     -1.928      0.063      -1.267       0.035
# x3             0.6681      0.365      1.830      0.077      -0.076       1.412
# ==============================================================================
# Omnibus:                        3.918   Durbin-Watson:                   1.169
# Prob(Omnibus):                  0.141   Jarque-Bera (JB):                2.594
# Skew:                           0.462   Prob(JB):                        0.273
# Kurtosis:                       3.936   Cond. No.                         33.5
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

#### OBSERVATIONS - PRE-REGISTERED REPLICATION
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.603
# Model:                            OLS   Adj. R-squared:                  0.566
# Method:                 Least Squares   F-statistic:                     16.19
# Date:                Fri, 09 Feb 2024   Prob (F-statistic):           1.40e-06
# Time:                        10:41:50   Log-Likelihood:                 24.917
# No. Observations:                  36   AIC:                            -41.83
# Df Residuals:                      32   BIC:                            -35.50
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.4199      0.084      4.981      0.000       0.248       0.592
# x1             0.2371      0.036      6.530      0.000       0.163       0.311
# x2            -0.3234      0.250     -1.292      0.206      -0.833       0.186
# x3             0.4456      0.286      1.559      0.129      -0.137       1.028
# ==============================================================================
# Omnibus:                        4.620   Durbin-Watson:                   1.478
# Prob(Omnibus):                  0.099   Jarque-Bera (JB):                3.229
# Skew:                           0.670   Prob(JB):                        0.199
# Kurtosis:                       3.598   Cond. No.                         33.5
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7fcd8a09e0d0>

##### REWARDS
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.663
# Model:                            OLS   Adj. R-squared:                  0.631
# Method:                 Least Squares   F-statistic:                     20.96
# Date:                Mon, 22 Jan 2024   Prob (F-statistic):           1.07e-07
# Time:                        14:28:25   Log-Likelihood:                 48.856
# No. Observations:                  36   AIC:                            -89.71
# Df Residuals:                      32   BIC:                            -83.38
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -0.0654      0.021     -3.083      0.004      -0.109      -0.022
# x1             0.1068      0.016      6.843      0.000       0.075       0.139
# x2             0.3794      0.137      2.767      0.009       0.100       0.659
# x3             0.0538      0.148      0.364      0.718      -0.247       0.355
# ==============================================================================
# Omnibus:                        2.603   Durbin-Watson:                   2.632
# Prob(Omnibus):                  0.272   Jarque-Bera (JB):                1.472
# Skew:                          -0.180   Prob(JB):                        0.479
# Kurtosis:                       2.077   Cond. No.                         22.3
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

###### REWARDS - PRE-REGISTERED REPLICATION_
#                            OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.163
# Model:                            OLS   Adj. R-squared:                  0.085
# Method:                 Least Squares   F-statistic:                     2.081
# Date:                Fri, 09 Feb 2024   Prob (F-statistic):              0.122
# Time:                        10:42:06   Log-Likelihood:                 44.089
# No. Observations:                  36   AIC:                            -80.18
# Df Residuals:                      32   BIC:                            -73.84
# Df Model:                           3                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.1334      0.024      5.506      0.000       0.084       0.183
# x1             0.0407      0.018      2.285      0.029       0.004       0.077
# x2             0.0836      0.157      0.534      0.597      -0.235       0.402
# x3             0.0451      0.169      0.268      0.791      -0.299       0.389
# ==============================================================================
# Omnibus:                        0.764   Durbin-Watson:                   1.857
# Prob(Omnibus):                  0.682   Jarque-Bera (JB):                0.187
# Skew:                           0.134   Prob(JB):                        0.911
# Kurtosis:                       3.230   Cond. No.                         22.3
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7fcd82a9f410>

# %%

competitive_ridge_corr_regression(upper_tri_masking(data_obs_corr), [upper_tri_masking(sim_obs_corr), upper_tri_masking(nostruc_obs_corr), upper_tri_masking(random_obs_corr)], alpha=None)


# %% RIDGE REGRESSION

competitive_ridge_corr_regression(upper_tri_masking(data_rews_corr), [upper_tri_masking(sim_rews_corr), upper_tri_masking(nostruc_rews_corr), upper_tri_masking(random_rews_corr)], alpha=None)


# %%

competitive_lasso_corr_regression(upper_tri_masking(data_obs_corr), [upper_tri_masking(sim_obs_corr), upper_tri_masking(nostruc_obs_corr), upper_tri_masking(random_obs_corr)], alpha=None)

# %% LASSO REGRESSION

#competitive_lasso_corr_regression(corr_obs_data, [sim_obs_corr, nostruc_obs_corr, null_obs_corr], alpha=1)
competitive_lasso_corr_regression(upper_tri_masking(data_rews_corr), [upper_tri_masking(sim_rews_corr), upper_tri_masking(nostruc_rews_corr), upper_tri_masking(random_rews_corr)], alpha=None)

# Lasso coefficients:  [0.         0.62461017 0.30081531]

# %% COMPUTE DISTANCES

for corr in [sim_obs_corr, nostruc_obs_corr, random_obs_corr]:
    print(np.linalg.norm(np.triu(corr) - np.triu(data_obs_corr), 'fro'))

# %%

for corr in [sim_rews_corr, nostruc_rews_corr, random_obs_corr]:
    print(np.linalg.norm(np.triu(corr) - np.triu(data_obs_corr)))

# %% CORRELATIONS

for corr in [sim_obs_corr, nostruc_obs_corr, random_obs_corr]:
    print(np.corrcoef(upper_tri_masking(corr), upper_tri_masking(data_obs_corr))[0,1])

## CORRELATIONS
# 0.712710234997705
# -0.1942416799941877
# 0.3009799241576838

# %%

for corr in [sim_rews_corr, nostruc_rews_corr, random_rews_corr]:
    print(np.corrcoef(upper_tri_masking(corr), upper_tri_masking(data_rews_corr))[0,1])

## CORRELATIONS
# 0.7148900326956457
# 0.4059866419425452
# -0.3933543750457362

# %% WITH MASKING

rews_mask = data_rews_pvs < 0.05
obs_mask = data_obs_pvs < 0.05

# %% DISTANCES

for corr in [sim_obs_corr, nostruc_obs_corr, random_obs_corr]:
    print(np.linalg.norm(upper_tri_masking(corr)[upper_tri_masking(obs_mask)] - upper_tri_masking(data_obs_corr)[upper_tri_masking(obs_mask)]))

# 1.0490662147347491
# 4.059187400761608
# 4.081772805029239

# %% REWARDS

for corr in [sim_rews_corr, nostruc_rews_corr, random_rews_corr]:
    print(np.linalg.norm(upper_tri_masking(corr)[upper_tri_masking(rews_mask)] - upper_tri_masking(data_rews_corr)[upper_tri_masking(rews_mask)]))

# 1.0444056272122102
# 0.9040236130544311
# 0.7936308525878248

# %%

for corr in [sim_obs_corr, nostruc_obs_corr, random_obs_corr]:
    print(np.corrcoef(upper_tri_masking(corr)[upper_tri_masking(obs_mask)], upper_tri_masking(data_obs_corr)[upper_tri_masking(obs_mask)])[0,1])

# %%

for corr in [sim_rews_corr, nostruc_rews_corr, random_rews_corr]:
    print(np.corrcoef(upper_tri_masking(corr)[upper_tri_masking(rews_mask)], upper_tri_masking(data_rews_corr)[upper_tri_masking(rews_mask)])[0,1])

# %% CORRELATIONS

from scipy.stats import spearmanr

for corr in [sim_obs_corr, nostruc_obs_corr, random_obs_corr]:

    #print(np.corrcoef(corr[obs_mask].flatten(), corr_obs_data[obs_mask].flatten())[0,1])
    print(spearmanr(corr[obs_mask].flatten(), data_obs_corr[obs_mask].flatten()))

# 0.13212928630783283
# 0.5277414201944145
# nan

# %%

for corr in [sim_rews_corr, nostruc_rews_corr, random_rews_corr]:
    #print(np.corrcoef(corr[rews_mask].flatten(), corr_rews_data[rews_mask].flatten())[0,1])
    print(spearmanr(corr[obs_mask].flatten(), data_obs_corr[obs_mask].flatten()))

# -0.05792832423429532
# 0.4010079690827608
# nan

# %% ANALYZING DATA AT THE LEVEL OF OVERALL FACTORS

data_obss = np.stack(df['n_observes'].values).T
data_rewss = np.stack(df['rewards_tallies'].values).T

sim_obss = np.flip(sim_obss, axis=1)
sim_rewss = np.flip(sim_rewss, axis=1)

nostruc_obss = np.flip(nostruc_obss, axis=1)
nostruc_rewss = np.flip(nostruc_rewss, axis=1)

random_obss = np.flip(random_obss, axis=1)
random_rewss = np.flip(random_rewss, axis=1)

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

# Data Info for data_obss: Number of Factors: 3 Variance Explained: 0.7532025184581258
# Data Info for sim_obss: Number of Factors: 3 Variance Explained: 0.997551971318038
# Data Info for nostruc_obss: Number of Factors: 3 Variance Explained: 0.24954223203240072
# Data Info for random_obss: Number of Factors: 3 Variance Explained: 0.24954225444676376

## BOOTSTRAPPING RESULTS WITH 10 REPS
# Similarity with sim_obss: 0.820527255032471 Confidence Interval: [0.80820532 0.91279088]
# Similarity with nostruc_obss: 0.15466291039043167 Confidence Interval: [0.01811479 0.38422398]
# Similarity with random_obss: 0.15466290202735877 Confidence Interval: [0.09337074 0.27988125]

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
ci_sim = bootstrap_similarity(data_rewss.T, sim_rewss.T, n_iterations=1000)
ci_nostruc = bootstrap_similarity(data_rewss.T, nostruc_rewss.T, n_iterations=1000)
ci_random = bootstrap_similarity(data_rewss.T, random_rewss.T, n_iterations=1000)

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

# Data Info for data_obss: Number of Factors: 3 Variance Explained: 0.29839298911712425
# Data Info for sim_obss: Number of Factors: 3 Variance Explained: 0.9868887693907871
# Data Info for nostruc_obss: Number of Factors: 3 Variance Explained: 0.1539957656219977
# Data Info for random_obss: Number of Factors: 3 Variance Explained: 0.15399576948941146

## BOOTSTRAPPING WITH 10 REPS
# Similarity with sim_obss: 0.2710691746245606 Confidence Interval: [0.15279135 0.57624111]
# Similarity with nostruc_obss: 0.026054022804085624 Confidence Interval: [-0.0779228   0.22902728]
# Similarity with random_obss: 0.026054032764129004 Confidence Interval: [-0.17786156  0.24722162]

## BOOTSTRAPPING WITH 100 REPS
# Similarity with sim_obss: 0.2710691746245606 Confidence Interval: [0.10283291 0.55299088]
# Similarity with nostruc_obss: 0.026054022804085624 Confidence Interval: [-0.25414224  0.42140825]
# Similarity with random_obss: 0.026054032764129004 Confidence Interval: [-0.24633863  0.41406571]

# sim_obss is the most similar to the ground truth.

# %%
