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

from human_utils_transdiagnostics import get_clean_combined_data, compute_2D_correlation_transdiagnostics, plot_quantile_analysis_results

import statsmodels.api as sm
import statsmodels.formula.api as smf

# %% PARAMETERS


iday = 3
day = 'day%d' %iday
#exp_date = '518-525-619-706'
exp_date = '24-01-22-29'
with_nets = False
include_sleep = True
include_estimates = True
#group = 'groupB'
group = None

day1_test_mask_cutoff=None
mask = 'mask%d' %(day1_test_mask_cutoff if day1_test_mask_cutoff is not None else 0)
#mask = 'nomask'

analysis_folder = os.path.join('analysis', 'transdiagnostics', day, exp_date, mask)

if group is not None:
    analysis_folder = os.path.join(analysis_folder, group)

os.makedirs(analysis_folder, exist_ok=True)

# %% DATA READ IN

df, effs_sorted_train, effs_sorted_test, test_start = get_clean_combined_data(day = iday, group=group, day1_test_mask_cutoff=day1_test_mask_cutoff)
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

fig.savefig(os.path.join(analysis_folder, '%s_td_rews.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_td_rews.svg' %get_timestamp()))

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

fig.savefig(os.path.join(analysis_folder, '%s_td_obs.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_td_obs.svg' %get_timestamp()))

plt.tight_layout()
plt.show()

# %%
if 'day3' in day:
        
    # Create scatter plots
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.scatter(df['total_sleeps'], df['AD'])
    plt.xlabel('Number of Sleep Actions')
    plt.ylabel('AD Score')
    format_axis(plt.gca())

    plt.subplot(132)
    plt.scatter(df['total_sleeps'], df['Compul'])
    plt.xlabel('Number of Sleep Actions')
    plt.ylabel('Compul Score')
    format_axis(plt.gca())
    plt.subplot(133)


    plt.scatter(df['total_sleeps'], df['SW'])
    plt.xlabel('Number of Sleep Actions')
    plt.ylabel('SW Score')
    format_axis(plt.gca())

    fig.savefig(os.path.join(analysis_folder, '%s_td_sleeps.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_td_sleeps.svg' %get_timestamp()))

    plt.tight_layout()
    plt.show()

# %% PLOT PC 1 VS BEH METRICS

for metric in ['total_rewards', 'n_observes', 'total_dev_obs', 'mean_slope_obs', 'mean_slope_rews',#]:
    'total_sleeps']:
               #, 'mean_taken_logits', 'mean_observe_deviation_logits']:
#for metric in ['mean_taken_logits', 'mean_observe_deviation_logits']:
    fig = plot_scatter_linefit(df['Control_PC1'], df[metric + "_train"], df[metric + "_test"], x_label='Control PC1', y_label=metric, xlim=(df['Control_PC1'].min(), df['Control_PC1'].max()))
    fig.savefig(os.path.join(analysis_folder, '%s_PC1_%s_scatterline.png' %(get_timestamp(), metric)))
    fig.savefig(os.path.join(analysis_folder, '%s_PC1_%s_scatterline.svg' %(get_timestamp(), metric)))

# %% TRAIN TEST DEVIAITION PLOT

# %% TOTAL NUMBER OF REWARDS AND OBSERVES

fig = plot_train_test_td(df, 'n_observes', 'Number of Observe Actions', ylim=(0, 40))
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_fig.svg' %get_timestamp()))

# %%

fig = plot_train_test_td(df, 'n_observes_loweff', 'Number of Observe Actions for Low Efficacy', ylim=(0,40))
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_loweff_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_loweff_fig.svg' %get_timestamp()))

# %% 

fig = plot_train_test_td(df, 'n_observes_higheff', 'Number of Observe Actions for High Efficacy', ylim=(0,40))
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_higheff_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_total_obs_higheff_fig.svg' %get_timestamp()))

# %% 

if include_estimates:
    fig = plot_train_test_td(df, 'total_dev_mse', 'Total Deviation Efficacy Estimate')
    fig.savefig(os.path.join(analysis_folder, '%s_total_mse_estimate_fig.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_total_mse_estimate_fig.svg' %get_timestamp()))

# %%

fig = plot_train_test_td(df, 'total_rewards', 'Number of Rewards')
fig.savefig(os.path.join(analysis_folder, '%s_total_rews_fig.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_total_rews_fig.svg' %get_timestamp()))

# %%

if include_sleep:
    fig = plot_train_test_td(df, 'total_sleeps', 'Number of Sleeps')
    fig.savefig(os.path.join(analysis_folder, '%s_total_sleeps_fig.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_total_sleeps_fig.svg' %get_timestamp()))

# %% 

dev_obs_fig = plot_train_test_td(df, 'total_dev_obs', 'Dev Mean Observes for Efficacy')
dev_obs_fig.savefig(os.path.join(analysis_folder, '%s_dev_obs_fig.png' %get_timestamp()))
dev_obs_fig.savefig(os.path.join(analysis_folder, '%s_dev_obs_fig.svg' %get_timestamp()))

# %%

dev_rews_fig = plot_train_test_td(df, 'total_dev_rews', 'Dev Mean Rewards for Efficacy')
dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_dev_rews_fig.png' %get_timestamp()))
dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_dev_rews_fig.svg' %get_timestamp()))

# %%

dev_rews_fig = plot_train_test_td(df, 'total_dev_rews', 'Dev Mean Rewards for Efficacy')
dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_dev_sleeps_fig.png' %get_timestamp()))
dev_rews_fig.savefig(os.path.join(analysis_folder, '%s_dev_sleeps_fig.svg' %get_timestamp()))

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

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['n_observes'].values).squeeze(), df[['AD', 'Compul', 'SW']], df["effs"], "n_observes", ["AD", "Compul", "SW"], groups = df["group"] if group is None else None)

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_observes_tds.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_observes_tds.png" %get_timestamp()))

# %% 

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['n_sleeps'].values).squeeze(), df[['AD', 'Compul', 'SW']], df["effs"], "n_sleeps", ["AD", "Compul", "SW"], groups = df["group"] if group is None else None)

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_sleeps_tds.png" %get_timestamp()))
corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_sleeps_tds.svg" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_sleeps_tds.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_sleeps_tds.svg" %get_timestamp()))

# %% 

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['rewards_tallies'].values).squeeze(), df[['AD', 'Compul', 'SW']], effs, "rewards", ["AD", "Compul", "SW"])

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_rews_tds.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_rews_tds.png" %get_timestamp()))
# %% 

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['intended_correct'].values).squeeze(), df[['AD', 'Compul', 'SW']], effs, "intended correct", ["AD", "Compul", "SW"])

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_corr_tds.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_corr_tds.png" %get_timestamp()))

# %% PARTIALED

corr_fig, pvs_fig = compute_2D_correlation_transdiagnostics(np.stack(df['n_sleeps'].values).squeeze(), df[['AD', 'Compul', 'SW']], df["effs"], "n_sleeps", ["AD", "Compul", "SW"], groups = df["group"] if group is None else None, partial=np.stack(df['n_observes'].values))

corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_sleeps_tds_partialed.png" %get_timestamp()))
corr_fig.savefig(os.path.join(analysis_folder, "%s_corr_sleeps_tds_partialed.svg" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_sleeps_tds_partialed.png" %get_timestamp()))
pvs_fig.savefig(os.path.join(analysis_folder, "%s_pvalues_sleeps_tds_partialed.svg" %get_timestamp()))


# %% QUANTILE ANALYSIS

quantile_fig = plot_quantile_analysis_results(np.stack(df['n_sleeps'].values).squeeze(), df['AD'], df["effs"], "n_observes", groups = df["group"])
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_sleeps_AD.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_sleeps_AD.svg" %get_timestamp()))

# %% 

quantile_fig = plot_quantile_analysis_results(np.stack(df['n_sleeps'].values).squeeze(), df['Compul'], df["effs"], "n_observes", groups = df["group"])
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_sleeps_Compul.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_sleeps_Compul.svg" %get_timestamp()))

# %% 

quantile_fig = plot_quantile_analysis_results(np.stack(df['n_sleeps'].values).squeeze(), df['SW'], df["effs"], "n_observes", groups = df["group"])
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_sleeps_SW.png" %get_timestamp()))
quantile_fig.savefig(os.path.join(analysis_folder, "%s_quantile_sleeps_SW.svg" %get_timestamp()))

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


# %% VIOLIN PLOT

for td in ['AD', 'Compul', 'SW']:

    n_tr, n_te = sort_train_test(df['n_sleeps'], df['effs'].values, test_start)
    fig = plot_violin_binarization(n_te, effs_sorted_test, n_tr, effs_sorted_train, ylabel='Number of Sleeps', median_over_mean=True, binarization = df['%s_binarized' %td], binarization_name = td, ylim=(0, 17.5))
    fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_n_sleeps_binarized_{td}_violin.png'))
    fig.savefig(os.path.join(analysis_folder, f'{get_timestamp()}_n_sleeps_binarized_{td}_violin.svg'))

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
    cols.extend(['total_sleeps_train', 'total_sleeps_test', 'mean_slope_sleeps_train', 'mean_slope_sleeps_test'])

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
    dependent_vars.extend(['total_sleeps_train', 'total_sleeps_test', 'mean_slope_sleeps_train', 'mean_slope_sleeps_test'])
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

# %%


# %% ANOVA

import statsmodels.api as sm
from statsmodels.formula.api import ols

#rews_anova = do_anova(df['signed_dev_rews_test'])

# Step 1: Reset the index
df_reset = df.reset_index()

# Step 2: Explode the 'n_observes' column while keeping the original index as a column
df_exploded = df_reset[['index', 'n_sleeps']].explode('n_sleeps')

# Step 3: Create the 'efficacy_index' and convert 'n_observes' to numeric type
df_exploded['efficacy_index'] = df_exploded.groupby('index').cumcount()
df_exploded['n_sleeps'] = pd.to_numeric(df_exploded['n_sleeps'], errors='coerce')

# Step 4: Merge the 'AD' column from the original DataFrame
df_exploded = pd.merge(df_exploded, df_reset[['index', 'AD', 'Compul', 'SW']], on='index', how='left')


# Step 5: If you want to recreate the 'pid' column to be the same as the original index
df_exploded['pid'] = df_exploded['index']
df_exploded.drop('index', axis=1, inplace=True)  # Optional: Remove the temporary 'index' column


columns_to_keep = ['n_sleeps', 'efficacy_index', 'pid', 'AD', 'Compul', 'SW']
df_exploded = df_exploded[columns_to_keep]

print(df_exploded)

# %%% ANOVA

# Fit the model
#model = ols('signed_dev_rews_test ~ C(pid) + C(efficacy_index) + C(pid):C(efficacy_index)', data=df_exploded).fit()
model = ols('n_sleeps ~ C(pid) + efficacy_index  + AD + Compul + SW', data=df_exploded).fit()
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

md = smf.mixedlm('n_sleeps ~ efficacy_index + AD + Compul + SW', df_exploded, groups=df_exploded["pid"], re_formula="~efficacy_index")#, re_formula="~Time")

# Fit the model
mdf = md.fit()

# Print the summary
print(mdf.summary())

residuals = mdf.resid
df_exploded['residuals'] = residuals

grouped = df_exploded.groupby('pid')
print("Mean residual per individual:", grouped['residuals'].mean().mean())
print("Std residual per individual:", grouped['residuals'].std().mean())
# %%

# %% ANOVA

import statsmodels.api as sm
from statsmodels.formula.api import ols

#rews_anova = do_anova(df['signed_dev_rews_test'])

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

md = smf.mixedlm('rewards_tallies ~ efficacy_index + AD + Compul + SW', df_exploded, groups=df_exploded["pid"], re_formula="~efficacy_index")#, re_formula="~Time")

# Fit the model
mdf = md.fit()

# Print the summary
print(mdf.summary())

residuals = mdf.resid
df_exploded['residuals'] = residuals

grouped = df_exploded.groupby('pid')
print("Mean residual per individual:", grouped['residuals'].mean().mean())
print("Std residual per individual:", grouped['residuals'].std().mean())
# %%
