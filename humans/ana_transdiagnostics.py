# Kai Sandbrink
# 2023-05-27
# This script compares behavior and transdiagnostic scores

# %% LIBRARY IMPORT

from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.decomposition import PCA

from utils import format_axis, get_timestamp
from human_utils_project import sort_overall, sort_train_test

from human_utils_project import plot_reward_tally, plot_n_observes, plot_prob_intended_correct_takes, plot_train_test_comp, plot_scatter_linefit, plot_train_test_td

from human_plot_traj_analyses import plot_violin_binarization

from human_utils_transdiagnostics import get_clean_combined_data, compute_2D_correlation_transdiagnostics, compute_2D_correlation_matrices_transdiagnostics, plot_quantile_analysis_results, plot_td_quantile_analysis_results

import statsmodels.api as sm
import statsmodels.formula.api as smf

# %% PARAMETERS

#mask = 'defaultmask'
mask = 'nomask'
iday = 2
day = 'day%d' %iday
exp_date = '24-01-22-29'
#exp_date = '518-525-619-706'
with_nets = False
include_sleep = False
include_estimates = False
#day1_test_mask_cutoff = 10
day1_test_mask_cutoff = None
group = 'groupA'
#group = 'groupB'
#group = None

analysis_folder = os.path.join('analysis', 'transdiagnostics', day, exp_date, mask)

if group is not None:
    analysis_folder = os.path.join(analysis_folder, group)

if day1_test_mask_cutoff is not None:
    analysis_folder = os.path.join(analysis_folder, 'mask%d' %day1_test_mask_cutoff)

os.makedirs(analysis_folder, exist_ok=True)

effs = np.arange(0, 1.01, 0.125)

# %% DATA READ IN

df, effs_sorted_train, effs_sorted_test, test_start = get_clean_combined_data(day = iday, group = group, exp_date=exp_date, day1_test_mask_cutoff=day1_test_mask_cutoff)
n_train = len(effs_sorted_train)
n_test = len(effs_sorted_test)


# %% DATA PREPROCESSING OF BEHAVIOR DATA

# Convert the rewards_tallies from behavior_df to a 1D list, then calculate the sum
def parse_float_list(s):
    return [float(x) for x in s.strip('[]').split()]

df['total_rewards'] = df['rewards_tallies'].apply(
    lambda x: np.nansum(parse_float_list(x)) if isinstance(x, str) else np.nansum(np.array(x))
)

def parse_float_nested_list(s):
    s = s.replace("[", "").replace("]", "")
    outer_list = s.split(', ')
    return [[float(x) for x in inner_list.split()] for inner_list in outer_list]

# Using the new function to count observe actions
df['observe_count'] = df['transitions_ep_rightwrong'].apply(
    lambda x: np.count_nonzero(np.array(parse_float_nested_list(x)) == 0.5) if isinstance(x, str) else np.count_nonzero(np.array(x) == 0.5)
)

nobs_tr, nobs_te = sort_train_test(df['n_observes'].values, df['effs'].values, test_start)
df['n_observes_train'], df['n_observes_test'] = nobs_tr.sum(axis=1), nobs_te.sum(axis=1)

nrews_tr, nrews_te = sort_train_test(df['rewards_tallies'].values, df['effs'].values, test_start)
df['total_rewards_train'], df['total_rewards_test'] = nrews_tr.sum(axis=1), nrews_te.sum(axis=1)

#nst_tr /= nst_tr.sum(axis=0)
#nst_te /= nst_te.sum(axis=0)
#if 'with_nets' in behavior_file:
if with_nets:
    nst_tr, nst_te = sort_train_test(df['step_max_ll_ape'].values, df['effs'].values, test_start)

    df['total_step_max_ll_ape_train'], df['total_step_max_ll_ape_test'] = nst_tr.sum(axis=1), nst_te.sum(axis=1)

    df['total_mean_odl_ape_train'], df['total_mean_odl_ape_test'] = df['mean_odl_ape_train'].apply(sum), df['mean_odl_ape_test'].apply(sum)
    df['total_mean_odl_control_train'], df['total_mean_odl_control_test'] = df['mean_odl_control_train'].apply(sum), df['mean_odl_control_test'].apply(sum)
    df['total_mean_ll_ape_train'], df['total_mean_ll_ape_test'] = df['mean_ll_ape_train'].apply(sum), df['mean_ll_ape_test'].apply(sum)
    df['total_mean_ll_control_train'], df['total_mean_ll_control_test'] = df['mean_ll_control_train'].apply(sum), df['mean_ll_control_test'].apply(sum)

    df['total_mean_odl_ape_loweff_train'], df['total_mean_odl_ape_higheff_train'] = np.stack(df['mean_odl_ape_train'].values)[:,:2].sum(axis=1).tolist(), np.stack(df['mean_odl_ape_train'].values)[:,-2:].sum(axis=1).tolist()
    df['total_mean_odl_ape_loweff_test'], df['total_mean_odl_ape_higheff_test'] = np.stack(df['mean_odl_ape_test'].values)[:,:2].sum(axis=1).tolist(), np.stack(df['mean_odl_ape_test'].values)[:,-2:].sum(axis=1).tolist()

    if include_sleep:
        df['total_mean_sdl_ape_train'], df['total_mean_sdl_ape_test'] = df['mean_sdl_ape_train'].apply(sum), df['mean_sdl_ape_test'].apply(sum)
        df['total_mean_sdl_ape_loweff_train'], df['total_mean_sdl_ape_higheff_train'] = np.stack(df['mean_sdl_ape_train'].values)[:,:2].sum(axis=1).tolist(), np.stack(df['mean_odl_ape_train'].values)[:,-2:].sum(axis=1).tolist()
        df['total_mean_sdl_ape_loweff_test'], df['total_mean_sdl_ape_higheff_test'] = np.stack(df['mean_sdl_ape_test'].values)[:,:2].sum(axis=1).tolist(), np.stack(df['mean_odl_ape_test'].values)[:,-2:].sum(axis=1).tolist()

df['n_observes_loweff_train'], df['n_observes_higheff_train'] = nobs_tr[:,:2].sum(axis=1), nobs_tr[:,-2:].sum(axis=1)
df['n_observes_loweff_test'], df['n_observes_higheff_test'] = nobs_te[:,:2].sum(axis=1), nobs_te[:,-2:].sum(axis=1)

if include_sleep:
    nsl_tr, nsl_te = sort_train_test(df['n_sleeps'].values, df['effs'].values, test_start)
    df['total_sleeps_train'], df['total_sleeps_test'] = nsl_tr.sum(axis=1), nsl_te.sum(axis=1)

    df['total_sleeps'] = df['n_sleeps'].apply(sum)

# %% ADDITIONAL OPERATIONS TO COUNT TOTAL DEVIATION ETC

def total_deviation_train_test(metric_name, test_start=5):
    obs_sorted_train, obs_sorted_test = sort_train_test(df[metric_name].values, df['effs'].values, test_start)
    mean_ostr, mean_oste = obs_sorted_train.mean(axis=0), obs_sorted_test.mean(axis=0)

    dev_obs_ostr, dev_obs_oste = obs_sorted_train - mean_ostr, obs_sorted_test - mean_oste

    total_dev_obs_ostr, total_dev_obs_oste = dev_obs_ostr.sum(axis=1), dev_obs_oste.sum(axis=1)

    return total_dev_obs_ostr, total_dev_obs_oste

df['total_dev_obs_train'], df['total_dev_obs_test'] = total_deviation_train_test('n_observes', 5)
df['total_dev_rews_train'], df['total_dev_rews_test'] = total_deviation_train_test('rewards_tallies', 5)
#behavior_df['total_dev_obs_train'], behavior_df['total_deviation_obs_test'] = total_deviation_train_test('n_observes', 5)

#behavior_df['obs_sorted_train'], behavior_df['obs_sorted_test'] = sort_train_test(behavior_df['n_observes'].values, behavior_df['effs'].values, 5)
#behavior_df.apply(lambda row: sort_train_test(row['n_observes'], row['effs'], test_start=5), axis=1)

if include_sleep:
    df['total_dev_sleeps_train'], df['total_dev_sleeps_test'] = total_deviation_train_test('n_sleeps', 5)

# %% AVERAGE SLOPE METRIC

def mean_slope_train_test(metric_name, effs_sorted_train, effs_sorted_test):
    test_start = len(effs_sorted_train)

    obs_sorted_train, obs_sorted_test = sort_train_test(df[metric_name].values, df['effs'].values, test_start)
    #mean_slope_ostr, mean_slope_oste = np.diff(obs_sorted_train, axis=1).mean(axis=1), np.diff(obs_sorted_test, axis=1).mean(axis=1)

    delta_effs_train = np.diff(effs_sorted_train)
    delta_effs_test = np.diff(effs_sorted_test)

    delta_ostr = np.diff(obs_sorted_train, axis=1)
    delta_oste = np.diff(obs_sorted_test, axis=1)

    mean_slope_ostr, mean_slope_oste = (delta_ostr / delta_effs_train).mean(axis=1), (delta_oste / delta_effs_test).mean(axis=1)

    return mean_slope_ostr, mean_slope_oste

df['mean_slope_obs_train'], df['mean_slope_obs_test'] = mean_slope_train_test('n_observes', effs_sorted_train, effs_sorted_test)
df['mean_slope_rews_train'], df['mean_slope_rews_test'] = mean_slope_train_test('rewards_tallies', effs_sorted_train, effs_sorted_test)

if include_sleep:
    df['mean_slope_sleeps_train'], df['mean_slope_sleeps_test'] = mean_slope_train_test('rewards_tallies', effs_sorted_train, effs_sorted_test)

# %% DATA PREPROCESSING FOR TD METRICS

X = df[['AD', 'Compul']]

pca = PCA(n_components=1)  # We only need the first principal component
df['Control_PC1'] = pca.fit_transform(X)

variance_explained = pca.explained_variance_ratio_[0]
print(f'Variance explained by the first principal component: {variance_explained*100:.2f}%')

# %% DIVIDE MODEL METRICS INTO TRAIN AND TEST

# test_start = len(effs_sorted_train)

# for model_metric in ['taken_logits', 'observe_deviation_logits']:
#     obs_sorted_train, obs_sorted_test = sort_train_test(behavior_df[model_metric].values, behavior_df['effs'].values, test_start)

#     behavior_df[f'{model_metric}_train'], behavior_df[f'{model_metric}_test'] = obs_sorted_train.tolist(), obs_sorted_test.tolist()

#     behavior_df[f'mean_{model_metric}_train'], behavior_df[f'mean_{model_metric}_test'] = np.mean(obs_sorted_train, axis=1).tolist(), np.mean(obs_sorted_test, axis=1).tolist()

# %% MERGE AND PLOT

# Merge the two dataframes on the index (participant id)
#df = pd.merge(behavior_df, td_df, left_index=True, right_index=True)

if include_estimates:

    from sklearn.metrics import mean_squared_error

    def calculate_mse(list1, list2):
        return mean_squared_error(list1, list2)

    df['estimates_mse'] = df.apply(lambda row: calculate_mse(row['efficacy_estimates'], row['effs']), axis=1)
    df['neg_estimates_mse'] = - df['estimates_mse']

    df['dev_mse'] = df.apply(lambda row: np.array(row['efficacy_estimates']) - np.array(row['effs']), axis=1)

    dmtr, dmte =  sort_train_test(df['dev_mse'], df['effs'], test_start)
    df['dev_mse_train'], df['dev_mse_test'] = dmtr.tolist(), dmte.tolist()

    df['total_dev_mse'] = df['dev_mse'].apply(sum)

    df['total_dev_mse_train'] = df['dev_mse_train'].apply(sum)
    df['total_dev_mse_test'] = df['dev_mse_test'].apply(sum)

    dmtr, dmte = sort_train_test(df['dev_mse'], df['effs'], test_start)
    df['dev_mse_train'] = dmtr.tolist()
    df['dev_mse_test'] = dmte.tolist()

# %% PLOT 

# Create scatter plots
fig = plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(df['total_rewards'], df['AD'])
plt.xlabel('Total Rewards')
plt.ylabel('AD Score')
format_axis(plt.gca())

plt.subplot(132)
plt.scatter(df['total_rewards'], df['Compul'])
plt.xlabel('Total Rewards')
plt.ylabel('Compul Score')
format_axis(plt.gca())

plt.subplot(133)
plt.scatter(df['total_rewards'], df['SW'])
plt.xlabel('Total Rewards')
plt.ylabel('SW Score')
format_axis(plt.gca())

fig.savefig(os.path.join(analysis_folder, '%s_td_rews_%s.png' %(get_timestamp(), mask)))
fig.savefig(os.path.join(analysis_folder, '%s_td_rews_%s.svg' %(get_timestamp(), mask)))

plt.tight_layout()
plt.show()
# %%

# Create scatter plots
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(df['observe_count'], df['AD'])
plt.xlabel('Number of Observe Actions')
plt.ylabel('AD Score')
format_axis(plt.gca())

plt.subplot(132)
plt.scatter(df['observe_count'], df['Compul'])
plt.xlabel('Number of Observe Actions')
plt.ylabel('Compul Score')
format_axis(plt.gca())

plt.subplot(133)
plt.scatter(df['observe_count'], df['SW'])
plt.xlabel('Number of Observe Actions')
plt.ylabel('SW Score')
format_axis(plt.gca())

fig.savefig(os.path.join(analysis_folder, '%s_td_obs_%s.png' %(get_timestamp(), mask)))
fig.savefig(os.path.join(analysis_folder, '%s_td_obs_%s.svg' %(get_timestamp(), mask)))

plt.tight_layout()
plt.show()
# %% PLOT PC 1 VS BEH METRICS

for metric in ['total_rewards', 'n_observes', 'total_dev_obs', 'mean_slope_obs', 'mean_slope_rews',]:
    #'n_sleeps']:
               #, 'mean_taken_logits', 'mean_observe_deviation_logits']:
#for metric in ['mean_taken_logits', 'mean_observe_deviation_logits']:
    fig = plot_scatter_linefit(df['Control_PC1'], df[metric + "_train"], df[metric + "_test"], x_label='Control PC1', y_label=metric, xlim=(df['Control_PC1'].min(), df['Control_PC1'].max()))
    fig.savefig(os.path.join(analysis_folder, '%s_PC1_%s_scatterline_%s.png' %(get_timestamp(), metric, mask)))
    fig.savefig(os.path.join(analysis_folder, '%s_PC1_%s_scatterline_%s.svg' %(get_timestamp(), metric, mask)))

# %% TRAIN TEST DEVIAITION PLOT

# %% TOTAL NUMBER OF REWARDS AND OBSERVES

fig = plot_train_test_td(df, 'n_observes', 'Number of Observe Actions', ylim=(0, 65), group=df['group'] if group is not None else None)
#fig = plot_train_test_td(df, 'n_observes', 'Number of Observe Actions')
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_fig_%s.png' %(get_timestamp(), mask)))
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_fig_%s.svg' %(get_timestamp(), mask)))

# %%

fig = plot_train_test_td(df, 'n_observes_loweff', 'Number of Observe Actions for Low Efficacy', ylim=(0,40))
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_loweff_fig_%s.png' %(get_timestamp(), mask)))
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_loweff_fig_%s.svg' %(get_timestamp(), mask)))

# %% 

fig = plot_train_test_td(df, 'n_observes_higheff', 'Number of Observe Actions for High Efficacy', ylim=(0,40))
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_higheff_fig_%s.png' %(get_timestamp(), mask)))
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_higheff_fig_%s.svg' %(get_timestamp(), mask)))

# %% 

fig = plot_train_test_td(df, 'total_dev_mse', 'Total Deviation Efficacy Estimate')
fig.savefig(os.path.join(analysis_folder, '%s_total_mse_estimate_fig_%s.png' %(get_timestamp(), mask)))
fig.savefig(os.path.join(analysis_folder, '%s_total_mse_estimate_fig_%s.svg' %(get_timestamp(), mask)))

# %%

fig = plot_train_test_td(df, 'total_rewards', 'Number of Rewards')
fig.savefig(os.path.join(analysis_folder, '%s_total_rews_fig_%s.png' %(get_timestamp(), mask)))
fig.savefig(os.path.join(analysis_folder, '%s_total_rews_fig_%s.svg' %(get_timestamp(), mask)))

# %%

if include_sleep:
    fig = plot_train_test_td(df, 'total_sleeps', 'Number of Sleeps')
    fig.savefig(os.path.join(analysis_folder, '%s_total_sleeps_fig_%s.png' %(get_timestamp(), mask)))
    fig.savefig(os.path.join(analysis_folder, '%s_total_sleeps_fig_%s.svg' %(get_timestamp(), mask)))

# %%

dev_obs_fig = plot_train_test_td(df, 'total_dev_obs', 'Dev Mean Observes for Efficacy')
dev_obs_fig.savefig(os.path.join(analysis_folder, '%s_dev_obs_fig_%s.png' %(get_timestamp(), mask)))
dev_obs_fig.savefig(os.path.join(analysis_folder, '%s_dev_obs_fig_%s.svg' %(get_timestamp(), mask)))

# %%

dev_rews_fig = plot_train_test_td(df, 'total_dev_rews', 'Dev Mean Rewards for Efficacy')
dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_dev_rews_fig_%s.png' %(get_timestamp(), mask)))
dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_dev_rews_fig_%s.svg' %(get_timestamp(), mask)))

# %%

dev_rews_fig = plot_train_test_td(df, 'total_dev_rews', 'Dev Mean Rewards for Efficacy')
dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_dev_sleeps_fig_%s.png' %(get_timestamp(), mask)))
dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_dev_sleeps_fig_%s.svg' %(get_timestamp(), mask)))

# %%

slope_obs_fig = plot_train_test_td(df, 'mean_slope_obs', 'Mean Slope of Obs Change over Eff')
slope_obs_fig.savefig(os.path.join(analysis_folder, '%s_mean_slope_obs.png' %get_timestamp()))
slope_obs_fig.savefig(os.path.join(analysis_folder, '%s_mean_slope_obs.svg' %get_timestamp()))

# %%

dev_rews_fig = plot_train_test_td(df, 'mean_slope_rews', 'Mean Slope of Rews Change over Eff')
dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_mean_slope_rews.png' %get_timestamp()))
dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_mean_slope_rews.svg' %get_timestamp()))

# %%

if include_sleep:
    dev_rews_fig = plot_train_test_td(df, 'mean_slope_sleeps', 'Mean Slope of Sleeps Change over Eff')
    dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_mean_slope_sleep.png' %get_timestamp()))
    dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_mean_slope_sleep.svg' %get_timestamp()))

# %% NN VERSION

fig = plot_train_test_td (df, 'total_step_max_ll_ape', 'Stepwise Max Log-Likelihood')
fig.savefig(os.path.join(analysis_folder, '%s_total_stepmaxllape_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_total_stepmaxllape_fig.svg' %get_timestamp()))

# %%

fig = plot_train_test_td (df, 'total_mean_ll_ape', 'Mean Log-Likelihood')
fig.savefig(os.path.join(analysis_folder, '%s_total_meanllape_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_total_meanllape_fig.svg' %get_timestamp()))
# %%

fig = plot_train_test_td (df, 'total_mean_ll_control', 'Mean Log-Likelihood')
fig.savefig(os.path.join(analysis_folder, '%s_total_meanllcontrol_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_total_meanllcontrol_fig.svg' %get_timestamp()))


# %%

fig = plot_train_test_td (df, 'total_mean_odl_ape', 'Mean Deviation from Observe Likelihoods')
fig.savefig(os.path.join(analysis_folder, '%s_total_odl_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_total_odl_fig.svg' %get_timestamp()))

# %%

fig = plot_train_test_td (df, 'total_mean_odl_control', 'Mean Deviation from Observe Likelihoods')
fig.savefig(os.path.join(analysis_folder, '%s_total_odl_control_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_total_odl_control_fig.svg' %get_timestamp()))

# %%

fig = plot_train_test_td (df, 'total_mean_odl_ape_loweff', 'Mean Deviation from Observe Likelihoods')
fig.savefig(os.path.join(analysis_folder, '%s_mean_odl_ape_loweff_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_mean_odl_ape_loweff_fig.svg' %get_timestamp()))

# %%

fig = plot_train_test_td (df, 'total_mean_odl_ape_higheff', 'Mean Deviation from Observe Likelihoods')
fig.savefig(os.path.join(analysis_folder, '%s_mean_odl_ape_higheff_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_mean_odl_ape_higheff_fig.svg' %get_timestamp()))

# %%

fig = plot_train_test_td (df, 'total_mean_sdl_ape_loweff', 'Mean Deviation from Sleep Likelihoods')
fig.savefig(os.path.join(analysis_folder, '%s_mean_sdl_ape_loweff_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_mean_sdl_ape_loweff_fig.svg' %get_timestamp()))

# %%

fig = plot_train_test_td (df, 'total_mean_sdl_ape_higheff', 'Mean Deviation from Sleep Likelihoods')
fig.savefig(os.path.join(analysis_folder, '%s_mean_sdl_ape_higheff_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_mean_sdl_ape_higheff_fig.svg' %get_timestamp()))

# %% 

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['n_observes'].values).squeeze(), df[['AD', 'Compul', 'SW']], df["effs"], "n_observes", ["AD", "Compul", "SW"], groups = df["group"])

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_observes_tds.png" %get_timestamp()))
corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_observes_tds.svg" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_observes_tds.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_observes_tds.svg" %get_timestamp()))

# %% 

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['rewards_tallies'].values).squeeze(), df[['AD', 'Compul', 'SW']], df["effs"], "rewards", ["AD", "Compul", "SW"], groups=df["group"])

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_rews_tds.png" %get_timestamp()))
corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_rews_tds.svg" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_rews_tds.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_rews_tds.svg" %get_timestamp()))
# %% 

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['intended_correct'].values).squeeze(), df[['AD', 'Compul', 'SW']], effs, "intended correct", ["AD", "Compul", "SW"], groups = df["group"])

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_corr_tds.png" %get_timestamp()))
corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_corr_tds.svg" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_corr_tds.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_corr_tds.svg" %get_timestamp()))

# %% CORRELATE AD WITH EFFICACY LEVEL

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['n_observes'].values).squeeze(), df['AD'], effs, "n_observes", "AD")

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_observes_AD.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_observes_AD.png" %get_timestamp()))

# %% CORRELATE COMPUL WITH EFFICACY LEVEL

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['n_observes'].values).squeeze(), df['Compul'], effs, "n_observes", "Compul")

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_observes_Compul.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_observes_Compul.png" %get_timestamp()))

# %% CORRELATE COMPUL WITH EFFICACY LEVEL

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['n_observes'].values).squeeze(), df['SW'], effs, "n_observes", "SW")

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_observes_SW.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_observes_SW.png" %get_timestamp()))

# %% QUANTILE ANALYSIS

quantile_fig = plot_quantile_analysis_results(np.stack(df['n_observes'].values).squeeze(), df['AD'], df["effs"], "n_observes", groups = df["group"])
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_observes_AD.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_observes_AD.svg" %get_timestamp()))

# %% 

quantile_fig = plot_quantile_analysis_results(np.stack(df['n_observes'].values).squeeze(), df['Compul'], df["effs"], "n_observes", groups = df["group"])
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_observes_Compul.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_observes_Compul.svg" %get_timestamp()))

# %% 

quantile_fig = plot_quantile_analysis_results(np.stack(df['n_observes'].values).squeeze(), df['SW'], df["effs"], "n_observes", groups = df["group"])
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_observes_SW.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_observes_SW.svg" %get_timestamp()))

# %% QUANTILE ANALYSIS

quantile_fig = plot_quantile_analysis_results(np.stack(df['rewards_tallies'].values).squeeze(), df['AD'], df["effs"], "rewards_tallies", groups = df["group"])
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_rewards_tallies_AD.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_rewards_tallies_AD.svg" %get_timestamp()))

# %% 

quantile_fig = plot_quantile_analysis_results(np.stack(df['rewards_tallies'].values).squeeze(), df['Compul'], df["effs"], "rewards_tallies", groups = df["group"])
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_rewards_tallies_Compul.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_rewards_tallies_Compul.svg" %get_timestamp()))

# %% 

quantile_fig = plot_quantile_analysis_results(np.stack(df['rewards_tallies'].values).squeeze(), df['SW'], df["effs"], "rewards_tallies", groups = df["group"])
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_rewards_tallies_SW.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_rewards_tallies_SW.svg" %get_timestamp()))


# %% QUANTILE ANALYSIS

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['n_observes'].values).squeeze(), df['AD'], df["effs"], "n_observes", groups = df["group"], num_bins=9)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_AD.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_AD.svg" %get_timestamp()))

# %% 

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['n_observes'].values).squeeze(), df['Compul'], df["effs"], "n_observes", groups = df["group"], num_bins =9)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_Compul.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_Compul.svg" %get_timestamp()))

# %% 

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['n_observes'].values).squeeze(), df['SW'], df["effs"], "n_observes", groups = df["group"], num_bins=9)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_SW.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_SW.svg" %get_timestamp()))

# %% QUANTILE ANALYSIS

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['rewards_tallies'].values).squeeze(), df['AD'], df["effs"], "rewards_tallies", groups = df["group"],num_bins=9)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_AD.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_AD.svg" %get_timestamp()))

# %% 

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['rewards_tallies'].values).squeeze(), df['Compul'], df["effs"], "rewards_tallies", groups = df["group"], num_bins=9)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_Compul.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_Compul.svg" %get_timestamp()))

# %% 

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['rewards_tallies'].values).squeeze(), df['SW'], df["effs"], "rewards_tallies", groups = df["group"], num_bins=9)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_SW.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_SW.svg" %get_timestamp()))


# %% COMBINING GROUPS

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['n_observes'].values).squeeze(), df['AD'], df["effs"], "Observes", col2name="AD", groups = df["group"], num_bins=9, combine_groups = True, annot=False)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_AD.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_AD.svg" %get_timestamp()))

# %% 

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['n_observes'].values).squeeze(), df['Compul'], df["effs"], "Observes", col2name = "Compul", groups = df["group"], num_bins =9, combine_groups = True, annot=False)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_Compul.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_Compul.svg" %get_timestamp()))

# %% 

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['n_observes'].values).squeeze(), df['SW'], df["effs"], "n_observes", groups = df["group"], num_bins=9, combine_groups = True)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_SW.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_observes_SW.svg" %get_timestamp()))

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['rewards_tallies'].values).squeeze(), df['AD'], df["effs"], "rewards_tallies", groups = df["group"],num_bins=9, combine_groups = True)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_AD.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_AD.svg" %get_timestamp()))

# %% 

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['rewards_tallies'].values).squeeze(), df['Compul'], df["effs"], "rewards_tallies", groups = df["group"], num_bins=9, combine_groups = True)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_Compul.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_Compul.svg" %get_timestamp()))

# %% 

quantile_fig = plot_td_quantile_analysis_results(np.stack(df['rewards_tallies'].values).squeeze(), df['SW'], df["effs"], "rewards_tallies", groups = df["group"], num_bins=9, combine_groups = True)
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_SW.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_td_quantile_rewards_tallies_SW.svg" %get_timestamp()))


# %% QUANTILE ANALYSIS

plot_quantile_analysis_results(np.stack(df['n_observes'].values).squeeze(), df['AD'], df["effs"], "n_observes", groups = df["group"])


# %% OUTPUT TRAJECTORIES OF MAXIMUM AND MINIMUM PERFORMERS IN AD FOR HIGH-EFFICACY SETTINGS

sorted_df = df.sort_values(by='AD')

## save the corresponding AD scores

top_5pct_AD = sorted_df['AD'][-int(len(sorted_df)*0.05):]
bottom_5pct_AD = sorted_df['AD'][:int(len(sorted_df)*0.05)]

## print indices and scores side-by-side for top performers

print("top 5% of AD", top_5pct_AD)

## print indices and scores side-by-side for bottom performers

print("bottom 5% of AD", bottom_5pct_AD)

# %% BINARIZE PARTICIPANTS BASED ON VALUES

def binarize_df(df: pd.DataFrame, columns: list):
    for col in columns:
        median = df[col].median()
        df[col + '_binarized'] = df[col].apply(lambda x: 1 if x > median else 0)
    return df

cols = ['AD', 'Compul', 'SW']

binarize_df(df, cols)

# %%

binarized_cols = ['AD_binarized', 'Compul_binarized', 'SW_binarized']

for bin_col in binarized_cols:
    for bin_val in [0, 1]:  # lower and upper halves
        # create a new dataframe for each half
        temp_df = df[df[bin_col] == bin_val]

        # Call your existing sorting and plotting functions here
        rewardss_tallies_sorted_train, rewardss_tallies_sorted_test = sort_train_test(temp_df['rewards_tallies'], temp_df['effs'].values, test_start)
        fig = plot_train_test_comp(effs_sorted_train, rewardss_tallies_sorted_train, effs_sorted_test, rewardss_tallies_sorted_test, ylim=(20, 33))
        fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_rewards_tr_te_comp.png'))
        fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_rewards_tr_te_comp.svg'))

        obs_sorted_train, obs_sorted_test = sort_train_test(temp_df['n_observes'], temp_df['effs'].values, test_start)
        fig = plot_train_test_comp(effs_sorted_train, obs_sorted_train, effs_sorted_test, obs_sorted_test, y_label="Number of Observes", ylim=(1, 8))
        fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_n_observes_efficacy_trte.png'))
        fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_n_observes_efficacy_trte.svg'))

        ic_sorted_train, ic_sorted_test = sort_train_test(temp_df['intended_correct'], temp_df['effs'].values, test_start)
        fig = plot_train_test_comp(effs_sorted_train, ic_sorted_train, effs_sorted_test, ic_sorted_test, y_label="Probability of Intended Correct Bet", ylim=(0.4, 0.9))
        fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_ic_tr_te_comp.png'))
        fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_ic_tr_te_comp.svg'))

        if include_sleep:
            ic_sorted_train, ic_sorted_test = sort_train_test(temp_df['n_sleeps'], temp_df['effs'].values, test_start)
            fig = plot_train_test_comp(effs_sorted_train, ic_sorted_train, effs_sorted_test, ic_sorted_test, y_label="Number of Sleeps")
            fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_nsleeps_tr_te_comp.png'))
            fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_nsleeps_tr_te_comp.svg'))

binarized_cols = ['AD_binarized', 'Compul_binarized', 'SW_binarized']

# %% VIOLIN PLOT

n_tr, n_te = sort_train_test(df['n_observes'], df['effs'].values, test_start)
fig = plot_violin_binarization(n_te, effs_sorted_test, n_tr, effs_sorted_train, ylabel='Number of Observes', median_over_mean=True, binarization = df['AD_binarized'], binarization_name = 'AD', ylim=(0, 17.5))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_n_observes_binarized_AD_violin.png'))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_n_observes_binarized_AD_violin.svg'))

# %%

n_tr, n_te = sort_train_test(df['n_observes'], df['effs'].values, test_start)
fig = plot_violin_binarization(n_te, effs_sorted_test, n_tr, effs_sorted_train, ylabel='Number of Observes', median_over_mean=True, binarization = df['Compul_binarized'], binarization_name = 'Compul', ylim=(0, 17.5))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_n_observes_binarized_Compul_violin.png'))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_n_observes_binarized_Compul_violin.svg'))

# %%

n_tr, n_te = sort_train_test(df['n_observes'], df['effs'].values, test_start)
fig = plot_violin_binarization(n_te, effs_sorted_test, n_tr, effs_sorted_train, ylabel='Number of Observes', median_over_mean=True, binarization = df['SW_binarized'], binarization_name = 'SW', ylim=(0, 17.5))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_n_observes_binarized_SW_violin.png'))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_n_observes_binarized_SW_violin.svg'))

# %% ANOVA TEST



# %% NN METRICS

# from utils_project import get_mean_ll

# ape_models = [
#     20230704231542,
#     20230704231540,
#     20230704231539,
#     20230704231537,
#     20230704231535,
#     20230704231525,
#     20230704231524,
#     20230704231522,
#     20230704231521,
#     20230704231519
# ]

# control_models = [
#     20230704231549,
#     20230704231548,
#     20230704231546,
#     20230704231545,
#     20230704231543,
#     20230704231534,
#     20230704231533,
#     20230704231531,
#     20230704231529,
#     20230704231528
# ]

# df['mean_odl_ape_train'],df['mean_odl_ape_test'] = get_mean_ll(df, ape_models, test_start, 'odl', aggregate_efficacies=False)
# df['mean_odl_control_train'],df['mean_odl_control_test'] = get_mean_ll(df, control_models, test_start, 'odl', aggregate_efficacies=False)

fig = plot_violin_binarization(np.stack(df['mean_odl_ape_test'].values), effs_sorted_test, np.stack(df['mean_odl_ape_train'].values), effs_sorted_train, ylabel='Mean Dev from APE Obs Lik', median_over_mean=True, binarization = df['AD_binarized'], binarization_name = 'AD', ylim=(0, 17.5))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_meanodl_binarized_AD_violin.png'))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_meanodl_binarized_AD_violin.svg'))

# %%

fig = plot_violin_binarization(np.stack(df['mean_odl_ape_test'].values), effs_sorted_test, np.stack(df['mean_odl_ape_train'].values), effs_sorted_train, ylabel='Mean Dev from APE Obs Lik', median_over_mean=True, binarization = df['Compul_binarized'], binarization_name = 'Compul', ylim=(0, 17.5))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_meanodl_binarized_Compul_violin.png'))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_meanodl_binarized_Compul_violin.svg'))

# %%

fig = plot_violin_binarization(np.stack(df['mean_odl_ape_test'].values), effs_sorted_test, np.stack(df['mean_odl_ape_train'].values), effs_sorted_train, ylabel='Mean Dev from APE Obs Lik', median_over_mean=True, binarization = df['SW_binarized'], binarization_name = 'SW', ylim=(0, 17.5))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_meanodl_binarized_SW_violin.png'))
fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_meanodl_binarized_SW_violin.svg'))

# %% PLOT BINARIZATIONS TOGETHER

## TODO : Implement this

# for bin_col in binarized_cols:
#     for bin_val in [0, 1]:  # lower and upper halves
#         # create a new dataframe for each half
#         temp_df = df[df[bin_col] == bin_val]

#         # Call your existing sorting and plotting functions here
#         rewardss_tallies_sorted_train, rewardss_tallies_sorted_test = sort_train_test(temp_df['rewards_tallies'], temp_df['effs'].values, test_start)
#         fig = plot_train_test_comp(effs_sorted_train, rewardss_tallies_sorted_train, effs_sorted_test, rewardss_tallies_sorted_test, ylim=(20, 33))
#         fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_rewards_tr_te_comp.png'))
#         fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_rewards_tr_te_comp.svg'))

#         obs_sorted_train, obs_sorted_test = sort_train_test(temp_df['n_observes'], temp_df['effs'].values, test_start)
#         fig = plot_train_test_comp(effs_sorted_train, obs_sorted_train, effs_sorted_test, obs_sorted_test, y_label="Number of Observes", ylim=(1, 8))
#         fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_n_observes_efficacy_trte.png'))
#         fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_n_observes_efficacy_trte.svg'))

#         ic_sorted_train, ic_sorted_test = sort_train_test(temp_df['intended_correct'], temp_df['effs'].values, test_start)
#         fig = plot_train_test_comp(effs_sorted_train, ic_sorted_train, effs_sorted_test, ic_sorted_test, y_label="Probability of Intended Correct Bet", ylim=(0.4, 0.9))
#         fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_ic_tr_te_comp.png'))
#         fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_ic_tr_te_comp.svg'))

#         if include_sleep:
#             obs_sorted_train, obs_sorted_test = sort_train_test(temp_df['n_sleeps'], temp_df['effs'].values, test_start)
#             fig = plot_train_test_comp(effs_sorted_train, obs_sorted_train, effs_sorted_test, obs_sorted_test, y_label="Number of Sleeps", ylim=(0, 8))
#             fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_n_sleeps_efficacy_trte.png'))
#             fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_{bin_col}_{bin_val}_n_sleeps_efficacy_trte.svg'))

# %% CORRELATION MATRIX

# Select columns
cols = ['AD', 'Compul', 'SW', 'n_observes_train', 'n_observes_test', 'total_rewards_train', 'total_rewards_test', 
        'mean_slope_obs_train', 'mean_slope_obs_test', 'mean_slope_rews_train', 'mean_slope_rews_test', 
        'total_step_max_ll_ape_train', 'total_step_max_ll_ape_test', 
        'total_mean_ll_ape_train', 'total_mean_ll_ape_test', 'total_mean_odl_ape_train', 'total_mean_odl_ape_test',
        ]

if include_sleep:
    cols.extend(['n_sleeps_train', 'n_sleeps_test', 'mean_slope_sleeps_train', 'mean_slope_sleeps_test'])

# Subset dataframe with selected columns
df_subset = df[cols]

# Calculate correlation matrix
corr = df_subset.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

# Create a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

ax.set_title("Correlation Matrix")

#plt.show()

fig.savefig(os.path.join(analysis_folder, "%s_correlation_matrix.png" %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, "%s_correlation_matrix.svg" %get_timestamp()))

# %% MLS ALL 3 REGRESSORS

import statsmodels.api as sm

dependent_vars = ['n_observes_train', 'n_observes_test', 'total_rewards_train', 'total_rewards_test', 
                  'mean_slope_obs_train', 'mean_slope_obs_test', 'mean_slope_rews_train', 'mean_slope_rews_test',
                  'total_step_max_ll_ape_train', 'total_step_max_ll_ape_test', 
                 #'mean_observe_deviation_logits_train','mean_observe_deviation_logits_test', 'mean_taken_logits_train',  'mean_taken_logits_test'
                 'total_mean_ll_ape_train', 'total_mean_ll_ape_test', 'total_mean_odl_ape_train', 'total_mean_odl_ape_test',
                 ]

if include_sleep:
    dependent_vars.extend(['n_sleeps_train', 'n_sleeps_test', 'mean_slope_sleeps_train', 'mean_slope_sleeps_test'])
# Create an empty DataFrame to store the summary statistics
summary_df = pd.DataFrame()

# For each dependent variable, we perform a regression analysis
for dv in dependent_vars:
    # Define the dependent variable
    Y = df[dv]
    
    # Define the independent variables
    X = df[['AD', 'Compul', 'SW']]
    
    # Add a constant to the independent variables matrix
    X = sm.add_constant(X)
    
    # Perform the multiple linear regression
    model = sm.OLS(Y, X)
    results = model.fit()

    # Store the p-values of the individual regressors and the overall p-value and R-squared
        # Store the p-values of the individual regressors and the overall p-value and R-squared
    summary_df = summary_df.append(pd.Series({
        'Dependent Variable': dv,
        'AD_p-value': results.pvalues['AD'],
        'Compul_p-value': results.pvalues['Compul'],
        'SW_p-value': results.pvalues['SW'],
        'Overall_p-value': results.f_pvalue,
        'R-squared': results.rsquared,
        'AD_coefficient': results.params['AD'],
        'Compul_coefficient': results.params['Compul'],
        'SW_coefficient': results.params['SW']
    }), ignore_index=True)

# Print the summary statistics
print(summary_df)

summary_df.to_csv(os.path.join(analysis_folder ,"%s_MLR_statistics.csv" %get_timestamp()))

# %% MLR ONLY 2 REGRESSORS

dependent_vars = ['n_observes_train', 'n_observes_test', 'total_rewards_train', 'total_rewards_test', 
                  'mean_slope_obs_train', 'mean_slope_obs_test', 'mean_slope_rews_train', 'mean_slope_rews_test', 
                  'total_step_max_ll_ape_train', 'total_step_max_ll_ape_test', 
                  #'mean_observe_deviation_logits_train','mean_observe_deviation_logits_test', 'mean_taken_logits_train',  'mean_taken_logits_test'
                  'total_mean_ll_ape_train', 'total_mean_ll_ape_test', 'total_mean_odl_ape_train', 'total_mean_odl_ape_test',
                  ]

if include_sleep:
    dependent_vars.extend(['n_sleeps_train', 'n_sleeps_test', 'mean_slope_sleeps_train', 'mean_slope_sleeps_test'])

# Create an empty DataFrame to store the summary statistics
summary_df = pd.DataFrame()

# For each dependent variable, we perform a regression analysis
for dv in dependent_vars:
    # Define the dependent variable, plot_train_test_td
    Y = df[dv]
    
    # Define the independent variables
    X = df[['AD', 'Compul']]
    
    # Add a constant to the independent variables matrix
    X = sm.add_constant(X)
    
    # Perform the multiple linear regression
    model = sm.OLS(Y, X)
    results = model.fit()

    # Store the p-values of the individual regressors and the overall p-value and R-squared
        # Store the p-values of the individual regressors and the overall p-value and R-squared
    summary_df = summary_df.append(pd.Series({
        'Dependent Variable': dv,
        'AD_p-value': results.pvalues['AD'],
        'Compul_p-value': results.pvalues['Compul'],
        'Overall_p-value': results.f_pvalue,
        'R-squared': results.rsquared,
        'AD_coefficient': results.params['AD'],
        'Compul_coefficient': results.params['Compul'],
    }), ignore_index=True)

# Print the summary statistics
print(summary_df)

summary_df.to_csv(os.path.join(analysis_folder ,"%s_MLR_statistics.csv" %get_timestamp()))

# %% PLOT RELATIONSHIP WITH T LOGITS

# for td in ['AD', 'Compul', 'SW']:
#     for metric in ['taken_logits', 'observe_deviation_logits']:
#         beh_train = np.stack(df[metric + "_train"].values).mean(axis=1)
#         beh_test = np.stack(df[metric + "_test"].values).mean(axis=1)
#         fig = plot_scatter_linefit(df[td], beh_train, beh_test, x_label=td, y_label=metric, xlim=(df[td].min(), df[td].max()))
#         fig.savefig(os.path.join(analysis_folder, '%s_%s_%s_scatterline.png' %(get_timestamp(), td, metric)))
#         fig.savefig(os.path.join(analysis_folder, '%s_%s_%s_scatterline.svg' %(get_timestamp(), td, metric)))

# %% CREATE NEW COLUMNS FOR INDIVIDUAL N_OBSERVES AND EPISODE-WISE MODEL SCORES

nobs_tr, nobs_te = sort_train_test(df['n_observes'].values, df['effs'].values, test_start)
df[['n_observes_train_ep%d' %i for i in range(n_train) ]] = pd.DataFrame(nobs_tr.tolist(), index=df.index)
df[['n_observes_test_ep%d' %i for i in range(n_test) ]] = pd.DataFrame(nobs_te.tolist(), index=df.index)

tr, te = sort_train_test(df['step_max_ll_ape'].values, df['effs'].values, test_start)
df[['step_max_ll_ape_train_ep%d' %i for i in range(n_train) ]] = pd.DataFrame(tr.tolist(), index=df.index)
df[['step_max_ll_ape_test_ep%d' %i for i in range(n_test) ]] = pd.DataFrame(te.tolist(), index=df.index)

df[['mean_odl_ape_train_ep%d' %i for i in range(n_train) ]] = pd.DataFrame(df['mean_odl_ape_train'].tolist(), index=df.index)
df[['mean_odl_ape_test_ep%d' %i for i in range(n_test) ]] = pd.DataFrame(df['mean_odl_ape_test'].tolist(), index=df.index)

# %% PLOT CORRELATION MATRIX FOR MODEL SCORES, WITHIN-EPISODE N_OBSERVES, AND TRANSDIAGNOSTIC FACTORS

# Select columns
cols = ['AD', 'Compul', 'SW', 'n_observes_train', 'n_observes_test', 'total_step_max_ll_ape_train', 'total_step_max_ll_ape_test', 'total_mean_odl_ape_train', 'total_mean_odl_ape_test'] + ['n_observes_train_ep%d' %i for i in range(n_train) ] + \
        ['n_observes_test_ep%d' %i for i in range(n_test) ] + ['step_max_ll_ape_train_ep%d' %i for i in range(n_train) ] + \
        ['step_max_ll_ape_test_ep%d' %i for i in range(n_test) ] + ['mean_odl_ape_train_ep%d' %i for i in range(n_train) ] + \
        ['mean_odl_ape_test_ep%d' %i for i in range(n_test) ]

if include_sleep:
    cols.extend(['n_sleeps_train', 'n_sleeps_test', 'mean_slope_sleeps_train', 'mean_slope_sleeps_test'])

# Subset dataframe with selected columns
df_subset = df[cols]

# Calculate correlation matrix
corr = df_subset.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

# Create a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax,)

ax.set_title("Correlation Matrix")

#plt.show()

fig.savefig(os.path.join(analysis_folder, "%s_correlation_matrix_td_nn.png" %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, "%s_correlation_matrix_td_nn.svg" %get_timestamp()))

# %% SORT PARTICIPANTS BASED ON AD

df.sort_values(by='AD', inplace=False).index

# %% ANOVA FOR OBSERVES

import statsmodels.api as sm
from statsmodels.formula.api import ols

#rews_anova = do_anova(df['signed_dev_rews_test'])

# Step 1: Reset the index
df_reset = df.reset_index()

# Step 2: Explode the 'n_observes' column while keeping the original index as a column
df_exploded = df_reset[['index', 'n_observes']].explode('n_observes')

# Step 3: Create the 'efficacy_index' and convert 'n_observes' to numeric type
df_exploded['efficacy_index'] = df_exploded.groupby('index').cumcount()
df_exploded['n_observes'] = pd.to_numeric(df_exploded['n_observes'], errors='coerce')

# Step 4: Merge the 'AD' column from the original DataFrame
df_exploded = pd.merge(df_exploded, df_reset[['index', 'AD', 'Compul', 'SW']], on='index', how='left')


# Step 5: If you want to recreate the 'pid' column to be the same as the original index
df_exploded['pid'] = df_exploded['index']
df_exploded.drop('index', axis=1, inplace=True)  # Optional: Remove the temporary 'index' column


columns_to_keep = ['n_observes', 'efficacy_index', 'pid', 'AD', 'Compul', 'SW']
df_exploded = df_exploded[columns_to_keep]

print(df_exploded)

# %%  ANOVA FOR REWARDS

# Step 1: Reset the index
df_reset = df.reset_index()

# Step 2: Explode the 'n_observes' column while keeping the original index as a column
df_exploded = df_reset[['index', 'rewards_tallies']].explode('rewards_tallies')

# Step 3: Create the 'efficacy_index' and convert 'n_observes' to numeric type
df_exploded['efficacy_index'] = df_exploded.groupby('index').cumcount()
df_exploded['rewards_tallies'] = pd.to_numeric(df_exploded['rewards_tallies'], errors='coerce')

# Step 4: Merge the 'AD' column from the original DataFrame
df_exploded = pd.merge(df_exploded, df_reset[['index', 'AD', 'Compul', 'SW']], on='index', how='left')


# Step 5: If you want to recreate the 'pid' column to be the same as the original index
df_exploded['pid'] = df_exploded['index']
df_exploded.drop('index', axis=1, inplace=True)  # Optional: Remove the temporary 'index' column


columns_to_keep = ['rewards_tallies', 'efficacy_index', 'pid', 'AD', 'Compul', 'SW']
df_exploded = df_exploded[columns_to_keep]

print(df_exploded)

# %%% ANOVA

# Fit the model
#model = ols('signed_dev_rews_test ~ C(pid) + C(efficacy_index) + C(pid):C(efficacy_index)', data=df_exploded).fit()
model = ols('rewards_tallies ~ C(pid) + efficacy_index  + AD + Compul + SW', data=df_exploded).fit()
# Run the ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)


## OBSERVES:
#                       sum_sq      df          F         PR(>F)
# C(pid)          20439.216281   147.0  24.820776  6.795795e-273
# efficacy_index     37.232545     1.0   6.646472   1.005531e-02
# AD                 21.874146     1.0   3.904807   4.838020e-02
# Compul             22.858839     1.0   4.080587   4.360364e-02
# SW                 48.824306     1.0   8.715745   3.216983e-03
# Residual         6626.989677  1183.0        NaN            NaN
# Between-subject Variability: 11.961595949691192
# Within-subject Variability: 5.628566066066066

## REWARDS
#                        sum_sq      df          F    PR(>F)
# C(pid)          162978.334289   147.0  39.592860  0.000000
# efficacy_index     131.351351     1.0   4.690713  0.030525
# AD                  19.175655     1.0   0.684785  0.408111
# Compul               4.353009     1.0   0.155451  0.693451
# SW                  98.803303     1.0   3.528384  0.060572
# Residual         33126.870871  1183.0        NaN       NaN
# %% 

# Between-subject Variability (variance of individual mean scores)
between_subject_var = df_exploded.groupby('pid')['n_observes'].mean().var()

# Within-subject Variability (average of individual variances)
within_subject_var = df_exploded.groupby('pid')['n_observes'].var().mean()

print("Between-subject Variability:", between_subject_var)
print("Within-subject Variability:", within_subject_var)

# %% MIXED EFFECTS MODEL

md = smf.mixedlm('n_observes ~ efficacy_index + AD + Compul + SW', df_exploded, groups=df_exploded["pid"], re_formula="~efficacy_index")#, re_formula="~Time")

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

# %% SUBDIVIDE INTO QUINTILES AND COMPARE PAIRWISE

td = 'Compul'

### create quintiles
df['Quintile'] = pd.qcut(df[td], 5, labels=[1, 2, 3, 4, 5])

quintile_stats = {}

# Calculate and store the mean and count for each quintile group
quintile_edges = [df['AD'].quantile(i/5) for i in range(6)]

# Calculate and store the mean and count for each quintile group
for quintile in range(1, 6):
    group_data = df[df['Quintile'] == quintile]['n_observes'].apply(sum)
    quintile_stats[quintile] = {
        'mean': group_data.mean(),
        'count': group_data.count(),
        'start': quintile_edges[quintile - 1],
        'stop': quintile_edges[quintile]
    }

# Print the mean, count, and quintile edges for each quintile group
for quintile, qstats in quintile_stats.items():
    print(f"Quintile {quintile} (Start: {qstats['start']}, Stop: {qstats['stop']}): mean = {qstats['mean']}, count = {qstats['count']}")

# Generate all pairwise combinations of the quintile groups
from itertools import combinations

pairwise_combinations = list(combinations(range(1, 6), 2))

# Run t-tests for each pair and collect results
t_test_results = {}

for combo in pairwise_combinations:
    group1 = df[df['Quintile'] == combo[0]]['n_observes'].apply(sum)
    group2 = df[df['Quintile'] == combo[1]]['n_observes'].apply(sum)
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    t_test_results[combo] = {
        't_stat': t_stat,
        'p_value': p_value
    }

# Print out the t-test results
for combo, results in t_test_results.items():
    print(f"Quintile {combo[0]} vs Quintile {combo[1]}: t_stat = {results['t_stat']}, p_value = {results['p_value']}")


###
# Quintile 1 (Start: -2.4025104016484526, Stop: -1.3194367445638844): mean = 42.6, count = 30
# Quintile 2 (Start: -1.3194367445638844, Stop: -0.9078507838200628): mean = 37.62068965517241, count = 29
# Quintile 3 (Start: -0.9078507838200628, Stop: -0.5445808901112904): mean = 42.0, count = 30
# Quintile 4 (Start: -0.5445808901112904, Stop: -0.20022177788629056): mean = 40.793103448275865, count = 29
# Quintile 5 (Start: -0.20022177788629056, Stop: 0.4930754123935381): mean = 43.166666666666664, count = 30
# Quintile 1 vs Quintile 2: t_stat = 0.5294690383217131, p_value = 0.598536321118978
# Quintile 1 vs Quintile 3: t_stat = 0.06475976033586843, p_value = 0.9485879726982026
# Quintile 1 vs Quintile 4: t_stat = 0.19603383157576879, p_value = 0.8452809030634572
# Quintile 1 vs Quintile 5: t_stat = -0.06215871482841368, p_value = 0.9506501523987465
# Quintile 2 vs Quintile 3: t_stat = -0.5768964843245451, p_value = 0.5662809017757466
# Quintile 2 vs Quintile 4: t_stat = -0.4253860566621055, p_value = 0.6721855711673287
# Quintile 2 vs Quintile 5: t_stat = -0.7491898555640161, p_value = 0.45682357245324656
# Quintile 3 vs Quintile 4: t_stat = 0.1640223929305861, p_value = 0.87029375048889
# Quintile 3 vs Quintile 5: t_stat = -0.15974524029720405, p_value = 0.873636781841983
# Quintile 4 vs Quintile 5: t_stat = -0.33134274369722055, p_value = 0.7416007418038328

# %% POWER ANALYSIS 

from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.power import TTestIndPower


# Step 1: Calculate the Pearson's correlation coefficient
#corr_coefficient, p_value = pearsonr(df['AD'], df['n_observes'].apply(sum))
corr_coefficient, p_value = spearmanr(df['AD'], df['n_observes'].apply(sum))

print(f"Pearson's correlation coefficient: {corr_coefficient}")
print(f"P-value: {p_value}")

# Step 2: Test if the correlation is significantly different from 0
alpha = 0.05  # significance level

if p_value < alpha:
    print("The correlation is statistically significant.")
else:
    print("The correlation is not statistically significant.")

# Step 3: Power estimation and sample size calculation
effect_size = np.abs(corr_coefficient)  # Pearson's r can be used as the effect size
alpha = 0.05  # significance level
power = 0.8  # desired power

# analysis = TTestIndPower()
# required_sample_size = analysis.solve_power(effect_size=effect_size,
#                                             power=power,
#                                             alpha=alpha,
#                                             ratio=1.0  # equal number of samples in both groups
#                                             )
# print(f"Required sample size: {np.ceil(required_sample_size)}")

## RESULTS AD
# Pearson's correlation coefficient: -0.08787998189601195
# P-value: 0.2881913737757674
# The correlation is not statistically significant.
# Required sample size (for power of 0.8, alpha=0.05): 1014

## RESULTS COMPUL
# Pearson's correlation coefficient: 0.0707816226144559
# P-value: 0.392624618204298
# The correlation is not statistically significant.
# Required sample size: 1564

## RESULTS SW
# Pearson's correlation coefficient: 0.00656573546339401
# P-value: 0.9368741739283399
# The correlation is not statistically significant.
# Required sample size: 6182387

# %%

td = 'AD'

df['total_observes'] = df['n_observes'].apply(sum)
num_bins = 5

# Calculate bin edges based on the range of 'AD'
min_val = df[td].min()
max_val = df[td].max()
bin_width = (max_val - min_val) / num_bins
bin_edges = np.linspace(min_val, max_val, num_bins+1)

# Divide the participants into bins based on 'AD'
df['Bin'] = pd.cut(df['AD'], bins=bin_edges, labels=range(1, num_bins+1))

# Create a dictionary to store bin group stats
bin_stats = {}

# Calculate and store the mean and count for each bin group
for bin_label in range(1, num_bins+1):
    group_data = df[df['Bin'] == bin_label]['total_observes']
    bin_stats[bin_label] = {
        'mean': group_data.mean(),
        'count': group_data.count(),
        'start': bin_edges[bin_label-1],
        'stop': bin_edges[bin_label]
    }

# Print the mean, count, and bin edges for each bin group
for bin_label, qstats in bin_stats.items():
    print(f"Bin {bin_label} (Start: {qstats['start']}, Stop: {qstats['stop']}): mean = {qstats['mean']}, count = {qstats['count']}")

# Generate all pairwise combinations of the bin groups
pairwise_combinations = list(combinations(range(1, num_bins+1), 2))

# Run t-tests for each pair and collect results
t_test_results = {}

for combo in pairwise_combinations:
    group1 = df[df['Bin'] == combo[0]]['total_observes']
    group2 = df[df['Bin'] == combo[1]]['total_observes']

    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    t_test_results[combo] = {
        't_stat': t_stat,
        'p_value': p_value
    }

# Print out the t-test results
for combo, results in t_test_results.items():
    print(f"Bin {combo[0]} vs Bin {combo[1]}: t_stat = {results['t_stat']}, p_value = {results['p_value']}")

### AD
# Bin 1 (Start: -2.4025104016484526, Stop: -1.8233932388400544): mean = 45.22222222222222, count = 9
# Bin 2 (Start: -1.8233932388400544, Stop: -1.2442760760316562): mean = 44.166666666666664, count = 24
# Bin 3 (Start: -1.2442760760316562, Stop: -0.665158913223258): mean = 45.026315789473685, count = 38
# Bin 4 (Start: -0.665158913223258, Stop: -0.08604175041485984): mean = 39.94, count = 50
# Bin 5 (Start: -0.08604175041485984, Stop: 0.4930754123935381): mean = 34.19047619047619, count = 21
# Bin 1 vs Bin 2: t_stat = 0.1125765533572071, p_value = 0.9110921200943851
# Bin 1 vs Bin 3: t_stat = 0.014621352763461553, p_value = 0.9883988978650932
# Bin 1 vs Bin 4: t_stat = 0.5276896419349482, p_value = 0.5997629815240348
# Bin 1 vs Bin 5: t_stat = 1.0086584944746892, p_value = 0.3217760432814402
# Bin 2 vs Bin 3: t_stat = -0.09514776468192708, p_value = 0.9245145224288154
# Bin 2 vs Bin 4: t_stat = 0.6058710011652111, p_value = 0.5465057265759237
# Bin 2 vs Bin 5: t_stat = 1.1810758322290398, p_value = 0.2440641268379236
# Bin 3 vs Bin 4: t_stat = 0.7009549436244646, p_value = 0.48522343498707576
# Bin 3 vs Bin 5: t_stat = 1.096377298927846, p_value = 0.2775242815270125
# Bin 4 vs Bin 5: t_stat = 0.7492767706376023, p_value = 0.4562373252788744

# %% CORRELATE TRANSDIAGNOSTICS WITH FITTED PERTURBATIONS

df_nns_perturbations = pd.read_pickle('results/perturbation_only_NN/day2/518-525-619-706/20231023013412_perturbation_only_nns_df_lr05.pkl')

df['perturbation'] = df_nns_perturbations['perturbation'].loc[df.index]

# %% CORRELATE AD WITH EFFICACY LEVEL

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['perturbation'].values).squeeze(), df['AD'], effs, "perturbations", "AD")

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_perturbations_AD.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_perturbations_AD.png" %get_timestamp()))

# %% CORRELATE COMPUL WITH EFFICACY LEVEL

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['perturbation'].values).squeeze(), df['Compul'], effs, "perturbations", "Compul")

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_perturbations_Compul.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_perturbations_Compul.png" %get_timestamp()))

# %%
