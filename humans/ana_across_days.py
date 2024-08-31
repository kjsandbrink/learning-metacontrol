# Kai Sandbrink
# 2023-06-23
# Analyze behavior across days

# %% LIBRARY IMPORT

import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error

import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from human_utils_project import sort_train_test, get_max_ll, plot_single_train_test_td, get_clean_data
from utils import get_timestamp

from human_utils_behavioral_analysis import get_mask_behav, calc_dev_behavior, compute_2D_correlation

from human_utils_behavioral_analysis import compute_2D_correlation, compute_2D_correlation_matrices, compute_partial_2D_correlation, compute_partial_2D_correlation_matrices
from human_utils_behavioral_analysis import get_factor_analysis_details, compute_similarity, bootstrap_similarity
from human_utils_behavioral_analysis import load_simulated_participants_across_models
from human_utils_behavioral_analysis import combine_train_test, competitive_corr_regression, competitive_lasso_corr_regression

# %% PARAMETERS


#### WITH BIAS 0.5, VOLATILITY 0.2, AND NO HELDOUT TEST REGION
#### 10/06

ape_models_task1 = [
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

control_models_task1 = [
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


### NO HOLDOUT BIAS 0.5, VOL 0.1, 250k ENTROPY ANNEALING
## 12/11
ape_models_task2 = [
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

control_models_task2 = [
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


effs_sorted_train = [0, 0.25,   0.75,  1.0]
effs_sorted_test = [0.125, 0.375, 0.5,0.785, 0.875]
effs = effs_sorted = np.arange(0, 1.01, 0.125)

test_start = len(effs_sorted_train)

include_sleep = True

#exp_date = '6-19'
#exp_date = '518-525-619-706'
#exp_date = '12-11'

exp_date = '24-01-22-29'
day = 'days123'

#group = 'groupB'
group = None

save_df_file = os.path.join('results', 'behavior', '%s_behavior_diff_effs_%s_%s' %(get_timestamp(), exp_date, day))

analysis_folder = os.path.join('analysis', 'traj_diff_efficacy', 'across_days', exp_date, day)

day1_test_mask_cutoff = 10
#day1_test_mask_cutoff = None

if group is not None:
    analysis_folder = os.path.join(analysis_folder, group)

if day1_test_mask_cutoff is not None:
    analysis_folder = os.path.join(analysis_folder, 'mask_test_%d' %day1_test_mask_cutoff)

os.makedirs(analysis_folder, exist_ok=True)

# %% LOAD DATA

df_day1, effs_train_day1, effs_test_day1, test_start_day1 = get_clean_data(day=1, exp_date=exp_date, day1_test_mask_cutoff=day1_test_mask_cutoff, group= group)
#df_day1 = df_day1.dropna()

#df_day2 = pd.read_pickle(df_day2_file)
#df_day3 = pd.read_pickle(df_day3_file)

df_day2, effs_train_day2, effs_test_day2, test_start_day2 = get_clean_data(day=2, exp_date=exp_date, day1_test_mask_cutoff=day1_test_mask_cutoff, group=group)
#df_day2 = df_day2.dropna()

# mask_day2 = get_mask_behav(df_day2)
# df_day2 = df_day2[mask_day2]

df_day3, effs_train_day3, effs_test_day3, test_start_day3 = get_clean_data(day=3, exp_date=exp_date, day1_test_mask_cutoff=day1_test_mask_cutoff, group=group)
#df_day3 = df_day3.dropna()

# mask_day3 = get_mask_behav(df_day3)
# df_day3 = df_day3[mask_day3]

# %% CREATE DATA - ACCURACY ESTIMATES - SKIP

def calculate_mse(list1, list2):
    if isinstance(list1, list):
        return mean_squared_error(list1, list2)
    else:
        return np.nan
    
df_day2['estimates_mse'] = df_day2.apply(lambda row: calculate_mse(row['efficacy_estimates'], row['effs']), axis=1)
df_day2['neg_estimates_mse'] = - df_day2['estimates_mse']

#df_day2['dev_mse'] = df_day2.apply(lambda row: np.array(row['efficacy_estimates']) - np.array(row['effs']) if isinstance(row['efficacy_estimates'], list) else np.nan, axis=1)
df_day2['dev_mse'] = df_day2.apply(lambda row: np.array(row['efficacy_estimates']) - np.array(row['effs']), axis=1)
#df_day2['total_dev_mse'] = df_day2['dev_mse'].apply(lambda row: sum(row) if isinstance(row, list) else [np.nan]*len(effs_sorted))
df_day2['total_dev_mse'] = df_day2['dev_mse'].apply(sum)

dmtr, dmte = sort_train_test(df_day2['dev_mse'], df_day2['effs'], test_start)
df_day2['dev_mse_train'] = dmtr.tolist()
df_day2['dev_mse_test'] = dmte.tolist()

## same thing for df_day1

def simplify_nan_estimates(x):
    if isinstance(x, list):
        if not all(np.isnan(x)):
            return np.nan
        else:
            return x
    else:
        return np.nan
df_day1['efficacy_estimates'] = df_day1['efficacy_estimates'].apply(simplify_nan_estimates)
df_day1['estimates_mse'] = df_day1.apply(lambda row: calculate_mse(row['efficacy_estimates'], row['effs']), axis=1)
df_day1['neg_estimates_mse'] = - df_day1['estimates_mse']
#df_day1['dev_mse'] = df_day1.apply(lambda row: np.array(row['efficacy_estimates']) - np.array(row['effs']) if isinstance(row['efficacy_estimates'], list) else np.nan, axis=1)
df_day1['dev_mse'] = df_day1.apply(lambda row: print(row), axis=1)#'np.array(row['efficacy_estimates']) - np.array(row['effs']), axis=1)

#df_day1['dev_mse'] = df_day1.apply(lambda row: np.array(row['efficacy_estimates']) - np.array(row['effs']), axis=1)

df_day1['total_dev_mse'] = df_day1['dev_mse'].apply(lambda row: sum(row) if isinstance(row, list) else [np.nan]*len(effs_sorted))

dmtr, dmte = sort_train_test(df_day1['dev_mse'], df_day1['effs'], test_start)
df_day1['dev_mse_train'] = dmtr.tolist()
df_day1['dev_mse_test'] = dmte.tolist()

# %% CREATE DATA - AVERAGE PREDICTIVENESS OF EACH PARTICIPANT FOR MODELS

i = 0

for df, ape_models in zip([df_day1, df_day2, df_day3], [ape_models_task1, ape_models_task1, ape_models_task2]):
    print(i)
    df['max_ll_ape_train'], df['max_ll_ape_test'] = get_max_ll(df, ape_models, test_start)
    i+=1

for df, control_models in zip([df_day1, df_day2, df_day3], [control_models_task1, control_models_task1, control_models_task2]):

    df['max_ll_control_train'], df['max_ll_control_test'] = get_max_ll(df, control_models, test_start)

    df['llr_max_train'] = df['max_ll_ape_train'] - df['max_ll_control_train']
    df['llr_max_test'] = df['max_ll_ape_test'] - df['max_ll_control_test']

# %% JOIN DATA FRAMES

#df = df_day2.join(df_day3, lsuffix='_day2', rsuffix='_day3')
#df = df_day1.join(df_day2, how='outer', lsuffix='_day1', rsuffix='_day2')
#df = df.join(df_day3.rename(columns=lambda x: f"{x}_day3"), how='outer')

## WITH INNER JOIN SO THAT ONLY ROWS FROM ALL THREE DAYS ARE KEPT
df = df_day1.join(df_day2, how='outer', lsuffix='_day1', rsuffix='_day2')

# Then join the result with df3 with suffix "day3"
df = df.join(df_day3.rename(columns=lambda x: f"{x}_day3"), how='inner')


# %% DROP DATA POINTS WHERE N_OBSERVES == 0 OR N_SLEEPS == 0 -- IF APPLICABLE

# df = df.dropna(subset=['n_observes_day2', 'n_sleeps'])
# df = df[df['n_sleeps'].apply(sum) != 0]
# df = df[df['n_observes_day2'].apply(sum) != 0]

# %% CREATE CORRELATION MATRIX
corr_matrix = df[['neg_estimates_mse', 
                  'max_ll_ape_train_day2', 'max_ll_ape_train_day3', 'max_ll_ape_test_day2', 'max_ll_ape_test_day3',
                   # 'max_ll_control_train_day2', 'max_ll_control_train_day3', 'max_ll_control_test_day2', 'max_ll_control_test_day3'
                    ]].corr()

fig = plt.figure(dpi=300, figsize=(10,8)) # size of the plot

#mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
#plt.show()

fig.savefig(os.path.join(analysis_folder, '%s_correlation_matrix.png' %get_timestamp()), dpi=300)
fig.savefig(os.path.join(analysis_folder, '%s_correlation_matrix.svg' %get_timestamp()), dpi=300)

# %% 

columns = ['neg_estimates_mse', 'dev_mse_train', 'dev_mse_test', 'n_observes_day2', 'n_observes_day3', 'n_sleeps', 'mean_ll_ape_train_day2', 'mean_ll_ape_test_day2', 'mean_ll_ape_train_day3', 'mean_ll_ape_test_day3',
           'rewards_tallies_day2', 'rewards_tallies_day3',
    'mean_odl_ape_train_day2', 'mean_odl_ape_test_day2',
       'mean_odl_control_train_day2', 'mean_odl_control_test_day2',
     'mean_odl_ape_train_day3',
       'mean_odl_ape_test_day3', 'mean_odl_control_train_day3',
       'mean_odl_control_test_day3',
       'mean_sdl_ape_train', 'mean_sdl_ape_test',
       'mean_sdl_control_train', 'mean_sdl_control_test',

       ]

## take the sum of all of the elements in columns separately
total_columns = ['total_' + col for col in columns]

for col in columns:
    # sum the elements in each list in the column
    df['total_' + col] = df[col].apply(np.sum)

# %% 

corr_matrix = df[total_columns].corr()

fig = plt.figure(figsize=(16, 13), dpi=300)

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

fig.savefig(os.path.join(analysis_folder, '%s_correlation_matrix_all.png' %get_timestamp()), dpi=300)
fig.savefig(os.path.join(analysis_folder, '%s_correlation_matrix_all.svg' %get_timestamp()), dpi=300)


# %% ONLY DIRECTIONAL MEASURES

columns = [#'dev_mse_train', 
    'dev_mse_test', #'n_observes_day2', 'n_observes_day3', 'n_sleeps',
    #'mean_odl_ape_train_day2', 
    'mean_odl_ape_test_day2',
       #'mean_odl_control_train_day2', 'mean_odl_control_test_day2',
    # 'mean_odl_ape_train_day3',
      # 'mean_odl_ape_test_day3', #'mean_odl_control_train_day3',
       #'mean_odl_control_test_day3',
       #'mean_sdl_ape_train', 
       'mean_sdl_ape_test',
      # 'mean_sdl_control_train', 'mean_sdl_control_test',

       ]

total_columns = ['total_' + col for col in columns]

corr_matrix = df[total_columns].corr()

fig = plt.figure(figsize=(16, 13), dpi=300)

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

# %% 

df = df_day2.join(df_day3, lsuffix='_day2', rsuffix='_day3')
corr_matrix = df[['neg_estimates_mse', 
                  'llr_max_train_day2', 'llr_max_train_day3', 'llr_max_test_day2', 'llr_max_test_day3',
                   # 'max_ll_control_train_day2', 'max_ll_control_train_day3', 'max_ll_control_test_day2', 'max_ll_control_test_day3'
                    ]].corr()

fig = plt.figure(dpi=300, figsize=(10,8)) # size of the plot

#mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
#plt.show()

fig.savefig(os.path.join(analysis_folder, '%s_correlation_matrix_directional.png' %get_timestamp()), dpi=300)
fig.savefig(os.path.join(analysis_folder, '%s_correlation_matrix_directional.svg' %get_timestamp()), dpi=300)

# %%

df.to_pickle(save_df_file + '.pkl')

# %% CALCULATE PERFORMANCE DAY 2 DAY 3 REDUCED CORRELATION MATRIX

df['abs_dev_obs_train_day2'], df['abs_dev_obs_test_day2'] = calc_dev_behavior(df['n_observes_day2'], df['effs_day2'], use_abs = True)
df['abs_dev_obs_train_day3'], df['abs_dev_obs_test_day3'] = calc_dev_behavior(df['n_observes_day3'], df['effs_day3'], use_abs = True)
df['abs_dev_sleeps_train'], df['abs_dev_sleeps_test'] = calc_dev_behavior(df['n_sleeps'], df['effs_day3'], use_abs = True)


# %%

columns = ['neg_estimates_mse', 
           'total_rewards_tallies_day2', 'total_rewards_tallies_day3',
           'abs_dev_obs_train_day2', 'abs_dev_obs_train_day3',
           'abs_dev_sleeps_train', 'abs_dev_sleeps_test'
           ]

corr_matrix = df[columns].corr()

fig = plt.figure(dpi=300, figsize=(10,8)) # size of the plot

#mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
#plt.show()

fig.savefig(os.path.join(analysis_folder, '%s_correlation_matrix_performance.png' %get_timestamp()), dpi=300)
fig.savefig(os.path.join(analysis_folder, '%s_correlation_matrix_performance.svg' %get_timestamp()), dpi=300)

# %%

df['signed_dev_obs_train_day2'], df['signed_dev_obs_test_day2'] = calc_dev_behavior(df['n_observes_day2'], df['effs_day2'], use_abs = False)
df['signed_dev_obs_train_day3'], df['signed_dev_obs_test_day3'] = calc_dev_behavior(df['n_observes_day3'], df['effs_day3'], use_abs = False)
df['signed_dev_sleeps_train'], df['signed_dev_sleeps_test'] = calc_dev_behavior(df['n_sleeps'], df['effs_day3'], use_abs = False)
df['total_efficacy_estimates'] = np.stack(df['efficacy_estimates'].values).sum(axis=1)

# %%

columns = ['total_efficacy_estimates', 
           'total_n_observes_day2', 'total_n_observes_day3',
           'total_n_sleeps'
           ]

corr_matrix = df[columns].corr()

fig = plt.figure(dpi=300, figsize=(10,8)) # size of the plot

#mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
#plt.show()

fig.savefig(os.path.join(analysis_folder, '%s_correlation_matrix_signed_directional.png' %get_timestamp()), dpi=300)
fig.savefig(os.path.join(analysis_folder, '%s_correlation_matrix_signed_directional.svg' %get_timestamp()), dpi=300)

# %% SCATTER PLOT FOR OBSERVED PERFORMANCE EFFECTS 

n_tr, n_te = sort_train_test(df['rewards_tallies_day2'].values, df['effs_day2'].values, test_start_day2)
df['total_rewards_day2_train'], df['total_rewards_day2_test']  = n_tr.mean(axis=1), n_te.mean(axis=1)

n_tr, n_te = sort_train_test(df['rewards_tallies_day3'].values, df['effs_day3'].values, test_start_day2)
df['total_rewards_day3_train'], df['total_rewards_day3_test']  = n_tr.mean(axis=1), n_te.mean(axis=1)

n_tr, n_te = sort_train_test(df['n_observes_day2'].values, df['effs_day2'].values, test_start_day2)
df['total_n_observes_day2_train'], df['total_n_observes_day2_test']  = n_tr.mean(axis=1), n_te.mean(axis=1)

n_tr, n_te = sort_train_test(df['n_observes_day3'].values, df['effs_day3'].values, test_start_day2)
df['total_n_observes_day3_train'], df['total_n_observes_day3_test']  = n_tr.mean(axis=1), n_te.mean(axis=1)

n_tr, n_te = sort_train_test(df['n_sleeps'].values, df['effs_day3'].values, test_start_day2)
df['total_n_sleeps_train'], df['total_n_sleeps_test']  = n_tr.mean(axis=1), n_te.mean(axis=1)


# %% 

slider_train, slider_test = sort_train_test(df['efficacy_estimates'], df['effs_day2'], test_start_day2)

slider_mse_test = ((slider_test - effs_test_day2)**2).mean(axis=1)
slider_mse_train = ((slider_train - effs_train_day2)**2).mean(axis=1)

df['mse_estimates_train'], df['mse_estimates_test'] = slider_mse_train, slider_mse_test
df['total_efficacy_estimates_train'], df['total_efficacy_estimates_test'] = slider_train.sum(axis=1), slider_test.sum(axis=1)

# n_tr, n_te = sort_train_test(df['efficacy_estimates'].values, df['effs_day2'].values, test_start)
# df['total_neg_mse_day2_train'], df['total_neg_mse_day2_test']  = n_tr.sum(axis=1), n_te.sum(axis=1)

# %% SCATTER PLOT

fig = plot_single_train_test_td(df, 'mse_estimates', 'total_rewards_day2', y_label = 'Rewards', x_label = 'MSE Estimates', train_test_td = True)

fig.savefig(os.path.join(analysis_folder, '%s_scatterplot_mse_rewards.png' %get_timestamp()), dpi=300)
fig.savefig(os.path.join(analysis_folder, '%s_scatterplot_mse_rewards.svg' %get_timestamp()), dpi=300)
# %%
from scipy.stats import pearsonr

rews_train, rews_test = sort_train_test(df['rewards_tallies_day2'], df['effs_day2'], test_start_day2)

rews_train = rews_train.sum(axis=1)
rews_test = rews_test.sum(axis=1)

print("Correlation slider and rews test", pearsonr(slider_mse_test, rews_test))
print("Correlation slider and rews train", pearsonr(slider_mse_train, rews_train))

# %% CORRELATION DAYS 2 AND 3

fig = plot_single_train_test_td(df, 'total_rewards_day2', 'total_rewards_day3', y_label = 'Rewards Day 3', x_label = 'Rewards Day 2', train_test_td = True)

fig.savefig(os.path.join(analysis_folder, '%s_scatterplot_rewards_day2_day3.png' %get_timestamp()), dpi=300)
fig.savefig(os.path.join(analysis_folder, '%s_scatterplot_rewards_day2_day3.svg' %get_timestamp()), dpi=300)

# %%

fig = plot_single_train_test_td(df, 'total_rewards_day2', 'total_rewards_day3', y_label = 'Rewards Day 3', x_label = 'Rewards Day 2', train_test_td = True)

fig.savefig(os.path.join(analysis_folder, '%s_scatterplot_rewards_day2_day3.png' %get_timestamp()), dpi=300)
fig.savefig(os.path.join(analysis_folder, '%s_scatterplot_rewards_day2_day3.svg' %get_timestamp()), dpi=300)

# %% 

fig = plot_single_train_test_td(df, 'total_efficacy_estimates', 'total_n_observes_day2', y_label = 'Number of Observes', x_label = 'Efficacy Estimates', train_test_td = True)

fig.savefig(os.path.join(analysis_folder, '%s_scatterplot_estimates_observes.png' %get_timestamp()), dpi=300)
fig.savefig(os.path.join(analysis_folder, '%s_scatterplot_estimates_observes.svg' %get_timestamp()), dpi=300)

# %%

fig = plot_single_train_test_td(df, 'total_n_observes_day2', 'total_n_sleeps', y_label = 'Number of Sleeps Day 2', x_label = 'Number of Observes Day 1', train_test_td = True)

fig.savefig(os.path.join(analysis_folder, '%s_scatterplot_observes_sleeps.png' %get_timestamp()), dpi=300)
fig.savefig(os.path.join(analysis_folder, '%s_scatterplot_observes_sleeps.svg' %get_timestamp()), dpi=300)

# %% BUILD EFFICACY-WIDE HEATMAPS

columns_to_efficacy_correlate = (
    ['rewards_tallies_day%d' % (i + 1 ) for i in range(3)] +
    # ['efficacy_estimates_day%d' % (i + 1) for i in range(2)] +
    ['n_observes_day%d' % (i + 1) for i in range(3)] +
    ['n_sleeps_day3']
)

# def sort_array_by_another(arr, sort_indices):
#     return arr[np.argsort(sort_indices))

### for all pairwise combinations of columns_to_efficacy_correlate

for i in range(len(columns_to_efficacy_correlate)):
    for j in range(len(columns_to_efficacy_correlate)):
        print(columns_to_efficacy_correlate[i],columns_to_efficacy_correlate[j])
        corr_fig, pvs_fig = compute_2D_correlation(df[columns_to_efficacy_correlate[i]], df[columns_to_efficacy_correlate[j]], df['effs' + columns_to_efficacy_correlate[i][-5:]], df['effs' + columns_to_efficacy_correlate[j][-5:]], columns_to_efficacy_correlate[i], columns_to_efficacy_correlate[j], resize_colorbar=True)
        corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s.png' %(get_timestamp(), columns_to_efficacy_correlate[i], columns_to_efficacy_correlate[j])), dpi=300)
        pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s.png' %(get_timestamp(), columns_to_efficacy_correlate[i], columns_to_efficacy_correlate[j])), dpi=300)

plt.close('all')


# %% CPOMBINING GROUPS INTELLIGENTLY

columns_to_efficacy_correlate = (
    ['rewards_tallies_day%d' % (i + 1 ) for i in range(3)] +
    # ['efficacy_estimates_day%d' % (i + 1) for i in range(2)] +
    ['n_observes_day%d' % (i + 1) for i in range(3)] +
    ['n_sleeps_day3']
)

# def sort_array_by_another(arr, sort_indices):
#     return arr[np.argsort(sort_indices))

### for all pairwise combinations of columns_to_efficacy_correlate

for i in range(len(columns_to_efficacy_correlate)):
    for j in range(len(columns_to_efficacy_correlate)):
        print(columns_to_efficacy_correlate[i],columns_to_efficacy_correlate[j])
        corr_fig, pvs_fig = compute_2D_correlation(df[columns_to_efficacy_correlate[i]], df[columns_to_efficacy_correlate[j]], df['effs' + columns_to_efficacy_correlate[i][-5:]], df['effs' + columns_to_efficacy_correlate[j][-5:]], columns_to_efficacy_correlate[i], columns_to_efficacy_correlate[j], annot=False, groups=df['group_day1'], resize_colorbar=True,)
        corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s.png' %(get_timestamp(), columns_to_efficacy_correlate[i], columns_to_efficacy_correlate[j])), dpi=300)
        corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s.svg' %(get_timestamp(), columns_to_efficacy_correlate[i], columns_to_efficacy_correlate[j])), dpi=300)
        pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s.png' %(get_timestamp(), columns_to_efficacy_correlate[i], columns_to_efficacy_correlate[j])), dpi=300)
        pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s.svg' %(get_timestamp(), columns_to_efficacy_correlate[i], columns_to_efficacy_correlate[j])), dpi=300)

plt.close('all')

# %% CALCULATE "PARTIALED OUT" CORRELATION BETWEEN 

corr_fig, pvs_fig = compute_partial_2D_correlation(df['n_observes_day2'], df['n_sleeps_day3'], df['n_observes_day3'], df['effs_day2'], df['effs_day3'], df['effs_day3'], semi=False, col1name = 'n_observes_day2', col2name = 'n_sleeps_day3', annot=False, resize_colorbar=True)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed.png' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed.svg' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed.png' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed.svg' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)

# %% CALCULATE "PARTIALED OUT" CORRELATION BETWEEN  -- COMBINING INTELLIGENTLY OVER GROUPS

corr_fig, pvs_fig = compute_partial_2D_correlation(df['n_observes_day2'], df['n_sleeps_day3'], df['n_observes_day3'], df['effs_day2'], df['effs_day3'], df['effs_day3'], semi=False, col1name = 'n_observes_day2', col2name = 'n_sleeps_day3', groups=df['group_day1'], annot=False)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed.png' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed.svg' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed.png' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed.svg' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)

# %% CALCULATE "SEMI-PARTIALED OUT" CORRELATION BETWEEN 

corr_fig, pvs_fig = compute_partial_2D_correlation(df['n_observes_day2'], df['n_sleeps_day3'], df['n_observes_day3'], df['effs_day2'], df['effs_day3'], df['effs_day3'], semi=True, annot=False, resize_colorbar=True)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_semipartialed.png' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_semipartialed.svg' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_semipartialed.png' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_semipartialed.svg' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3')), dpi=300)

# %% CALCULATE "SEMI-PARTIALED OUT" CORRELATION BETWEEN 

corr_fig, pvs_fig = compute_partial_2D_correlation(df['n_observes_day2'], df['n_sleeps_day3'], df['mean_ll_ape_day3'], df['effs_day2'], df['effs_day3'], df['sorted_effs'], semi=False, annot=False, resize_colorbar=True)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.png' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3', 'll')), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.svg' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3', 'll')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.png' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3', 'll')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.svg' %(get_timestamp(), 'n_observes_day2', 'n_sleeps_day3', 'll')), dpi=300)

# %% PARTIAL BY NETWORK ACCURACY

df['sorted_effs'] = [effs.copy() for _ in range(len(df))]
df['mean_ll_ape_day3'] = df.apply(lambda x: combine_train_test(x['mean_ll_ape_train_day3'], x['mean_ll_ape_test_day3'], effs_sorted_train, effs_sorted_test), axis=1)

# %% 

corr_matrix, pvs_matrix = compute_partial_2D_correlation(df['n_observes_day2'], df['n_observes_day3'], df['mean_ll_ape_day3'], df['effs_day2'], df['effs_day3'], df['sorted_effs'], semi=False)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.png' %(get_timestamp(), 'n_observes_day2', 'n_observes_day3', 'mean_ll_ape_day3')), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.svg' %(get_timestamp(), 'n_observes_day2', 'n_observes_day3', 'mean_ll_ape_day3')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.png' %(get_timestamp(), 'n_observes_day2', 'n_observes_day3', 'mean_ll_ape_day3')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.svg' %(get_timestamp(), 'n_observes_day2', 'n_observes_day3', 'mean_ll_ape_day3')), dpi=300)

# %% COMPETITIVE REGRESSION

model = competitive_corr_regression(np.stack(df['n_observes_day3'].values), (np.stack(df['n_observes_day2'].values), np.stack(df['mean_ll_ape_day3'].values)), fisher_transform=False)

#          OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.599
# Model:                            OLS   Adj. R-squared:                  0.598
# Method:                 Least Squares   F-statistic:                     549.8
# Date:                Mon, 06 Nov 2023   Prob (F-statistic):          1.03e-146
# Time:                        11:06:04   Log-Likelihood:                -1680.1
# No. Observations:                 738   AIC:                             3366.
# Df Residuals:                     735   BIC:                             3380.
# Df Model:                           2                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          6.4617      0.232     27.875      0.000       6.007       6.917
# x1             0.3737      0.021     17.557      0.000       0.332       0.415
# x2             0.0239      0.001     20.275      0.000       0.022       0.026
# ==============================================================================
# Omnibus:                       36.862   Durbin-Watson:                   1.239
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               75.582
# Skew:                           0.305   Prob(JB):                     3.87e-17
# Kurtosis:                       4.445   Cond. No.                         419.
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# %%

model = competitive_lasso_corr_regression(np.stack(df['n_observes_day3'].values), (np.stack(df['n_observes_day2'].values), np.stack(df['mean_ll_ape_day3'].values)), fisher_transform=False)

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.174
# Model:                            OLS   Adj. R-squared:                  0.171
# Method:                 Least Squares   F-statistic:                     77.15
# Date:                Mon, 06 Nov 2023   Prob (F-statistic):           3.86e-31
# Time:                        11:20:35   Log-Likelihood:                -1947.3
# No. Observations:                 738   AIC:                             3901.
# Df Residuals:                     735   BIC:                             3914.
# Df Model:                           2                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const               0      0.333          0      1.000      -0.654       0.654
# x1             0.7343      0.031     24.020      0.000       0.674       0.794
# x2            -0.0043      0.002     -2.518      0.012      -0.008      -0.001
# ==============================================================================
# Omnibus:                      209.069   Durbin-Watson:                   0.696
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1442.968
# Skew:                          -1.087   Prob(JB):                         0.00
# Kurtosis:                       9.496   Cond. No.                         419.
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# %% READ IN PERTURBATION

df_nns_perturbations_day2 = pd.read_pickle('results/perturbation_only_NN/day2/518-525-619-706/20231023013412_perturbation_only_nns_df_lr05.pkl')
df_nns_perturbations_day3 = pd.read_pickle('results/perturbation_only_NN/day3/518-525-619-706/20231027024138_perturbation_only_nns_df_lr05.pkl')

### join the two dataframes keeping only rows that are in both, and appending '_day2' and '_day3' to the column names
df_nns_perturbations = df_nns_perturbations_day2.join(df_nns_perturbations_day3, lsuffix='_day2', rsuffix='_day3', how='inner')

perturbations_day2 = np.stack(df_nns_perturbations['perturbation_day2'].values).squeeze()
perturbations_day3 = np.stack(df_nns_perturbations['perturbation_day3'].values).squeeze()

nns_pertubations_analysis_folder = os.path.join(analysis_folder, 'perturbation_only_NN')
os.makedirs(nns_pertubations_analysis_folder, exist_ok=True)

# %% PLOT

corr_fig, pvs_fig = compute_2D_correlation(perturbations_day2, perturbations_day3, effs, effs, col1name = 'perturbations task 1', col2name = 'perturbations task 2')
corr_fig.savefig(os.path.join(nns_pertubations_analysis_folder, '%s_2D_correlation_perturbations.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(nns_pertubations_analysis_folder, '%s_2D_correlation_pvs_perturbations.svg' %(get_timestamp())), dpi=300)

# %% PARTIALED CORRELATION BETWEEN PERTURBATIONS DAY 2 AND PERTURBATIONS DAY 3

df_nns_perturbations['sorted_effs'] = [effs.copy() for _ in range(len(df_nns_perturbations))]
df_nns_perturbations['mean_ll_ape_day3'] = df_nns_perturbations.apply(lambda x: combine_train_test(x['mean_ll_ape_train_day3'], x['mean_ll_ape_test_day3'], effs_sorted_train, effs_sorted_test), axis=1)

# %%

corr_fig, pvs_fig = compute_partial_2D_correlation(df_nns_perturbations['perturbation_day2'], df_nns_perturbations['perturbation_day3'], df_nns_perturbations['mean_ll_ape_day3'], df_nns_perturbations['sorted_effs'], df_nns_perturbations['sorted_effs'], df_nns_perturbations['sorted_effs'], semi=False)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.png' %(get_timestamp(), 'perturbation', 'perturbation', 'mean_ll_ape_day3')), dpi=300)
corr_fig.savefig(os.path.join(analysis_folder, '%s_2D_correlation_%s_%s_partialed_%s.svg' %(get_timestamp(), 'perturbation', 'perturbation', 'mean_ll_ape_day3')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.png' %(get_timestamp(), 'perturbation', 'perturbation', 'mean_ll_ape_day3')), dpi=300)
pvs_fig.savefig(os.path.join(analysis_folder, '%s_2D_pvs_%s_%s_partialed_%s.svg' %(get_timestamp(), 'perturbation', 'perturbation', 'mean_ll_ape_day3')), dpi=300)

# %% ANALYZE SIMULATED PARTICIPANTS CORRELATION MATRICES FROM TASK 1 TO TASK 2 - IMPORT TASK 1

simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/pepe/sim'
#modelname = '20230922111413'
modelname = '20230923060013'
#sim_timestamp = '20231007230234'
#sim_timestamp = '20231008220417'
#sim_timestamp = '20231008220818'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
sim_timestamp = '20231015161128'

sim_rewss_t1, sim_obss_t1, _ = load_simulated_participants_across_models(simulated_participants_folder, ape_models_task1, sim_timestamp)
sim_rewss_t1 = sim_rewss_t1.mean(axis=0)
sim_obss_t1 = sim_obss_t1.mean(axis=0)


sim_parts_analysis_folder = os.path.join(analysis_folder, 'simulated_participants', 'across_models')
os.makedirs(sim_parts_analysis_folder, exist_ok=True)

# %% NN SIMULATED PARTICIPANTS W/ NO STRUCTURE

simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/pepe/nostruc'
#modelname = '20230922111413'
modelname = '20230923060013'
#sim_timestamp = '20231008220307'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
sim_timestamp = '20231015161128'

nostruc_rewss_t1, nostruc_obss_t1, _ = load_simulated_participants_across_models(simulated_participants_folder, ape_models_task1, sim_timestamp)

nostruc_rewss_t1 = nostruc_rewss_t1.mean(axis=0)
nostruc_obss_t1 = nostruc_obss_t1.mean(axis=0)

# %%

simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/pepe/random'
#modelname = '20230922111413'
modelname = '20230923060013'
#sim_timestamp = '20231008220307'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
sim_timestamp = '20231015161128'

random_rewss_t1, random_obss_t1, _ = load_simulated_participants_across_models(simulated_participants_folder, ape_models_task1, sim_timestamp)

random_rewss_t1 = random_rewss_t1.mean(axis=0)
random_obss_t1 = random_obss_t1.mean(axis=0)

# %% IMPORT TASK 2

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
sim_timestamp = '20240225232718'


sim_rewss_t2, sim_obss_t2, sim_sleepss_t2 = load_simulated_participants_across_models(simulated_participants_folder, ape_models_task2, sim_timestamp, include_sleep=True)
sim_rewss_t2 = sim_rewss_t2.mean(axis=0)
sim_obss_t2 = sim_obss_t2.mean(axis=0)
sim_sleepss_t2 = sim_sleepss_t2.mean(axis=0)

# %% NN SIMULATED PARTICIPANTS W/ NO STRUCTURE

simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/nostruc/mag10'
#/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/sim/mag1/20231017233738/20231218104238_perturbed_control_errs_taus_ape.pkl
#modelname = '20230922111413'
#modelname = '20230923060013'

## WITH VOLATILITY 0.1 
modelname = ape_models_task2[0]

#sim_timestamp = '20231007230234'
#sim_timestamp = '20231008220417'
#sim_timestamp = '20231008220818'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
#sim_timestamp = '20231016001832'
#sim_timestamp = '20231218104238'
sim_timestamp = '20231218172807'


nostruc_rewss_t2, nostruc_obss_t2, nostruc_sleepss_t2 = load_simulated_participants_across_models(simulated_participants_folder, ape_models_task2, sim_timestamp, include_sleep=True)

nostruc_rewss_t2 = nostruc_rewss_t2.mean(axis=0)
nostruc_obss_t2 = nostruc_obss_t2.mean(axis=0)
nostruc_sleepss_t2 = nostruc_sleepss_t2.mean(axis=0)

# %%

#simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/nostruc/mag10'
simulated_participants_folder = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/nostruc/mag10'
#/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/sim/mag1/20231017233738/20231218104238_perturbed_control_errs_taus_ape.pkl
#modelname = '20230922111413'
#modelname = '20230923060013'

## WITH VOLATILITY 0.1 
modelname = ape_models_task2[0]

#sim_timestamp = '20231007230234'
#sim_timestamp = '20231008220417'
#sim_timestamp = '20231008220818'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
#sim_timestamp = '20231016001832'
#sim_timestamp = '20231218104238'
sim_timestamp = '20231218172807'

random_rewss_t2, random_obss_t2, random_sleepss_t2 = load_simulated_participants_across_models(simulated_participants_folder, ape_models_task2, sim_timestamp, include_sleep=True)

random_rewss_t2 = random_rewss_t2.mean(axis=0)
random_obss_t2 = random_obss_t2.mean(axis=0)
random_sleepss_t2 = random_sleepss_t2.mean(axis=0)

# %% ANALYZE CORRELATION MATRICES FROM TASK 1 TO TASK 2

corr_fig, pvs_fig = compute_2D_correlation(sim_obss_t1.T, sim_obss_t2.T, effs_sorted, effs_sorted, "task 1 simulated observations", "task 2 simulated observations", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_obs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_obs.svg' %(get_timestamp())), dpi=300)

# %% ANALYZE CORRELATION MATRICES FROM TASK 1 TO TASK 2

corr_fig, pvs_fig = compute_2D_correlation(nostruc_obss_t1.T, nostruc_obss_t2.T, effs_sorted, effs_sorted, "task 1 simulated observations", "task 2 simulated observations", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_correlation_obs.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_correlation_obs.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_pvs_obs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_pvs_obs.svg' %(get_timestamp())), dpi=300)

# %% ANALYZE CORRELATION MATRICES FROM TASK 1 TO TASK 2

corr_fig, pvs_fig = compute_2D_correlation(random_obss_t1.T, random_obss_t2.T, effs_sorted, effs_sorted, "task 1 simulated observations", "task 2 simulated observations", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_correlation_obs.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_correlation_obs.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_pvs_obs.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_pvs_obs.svg' %(get_timestamp())), dpi=300)

# %% SIMULATED REWARDS

corr_fig, pvs_fig = compute_2D_correlation(sim_rewss_t1.T, sim_rewss_t2.T, effs_sorted, effs_sorted, "task 1 simulated rewards", "task 2 simulated rewards", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_sim_correlation_rews.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_sim_correlation_rews.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_sim_pvs_rews.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_sim_pvs_rews.svg' %(get_timestamp())), dpi=300)


# %% SIMULATED REWARDS

corr_fig, pvs_fig = compute_2D_correlation(nostruc_rewss_t1.T, nostruc_rewss_t2.T, effs_sorted, effs_sorted, "task 1 simulated rewards", "task 2 simulated rewards", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_correlation_rews.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_correlation_rews.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_pvs_rews.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_pvs_rews.svg' %(get_timestamp())), dpi=300)


# %% SIMULATED REWARDS

corr_fig, pvs_fig = compute_2D_correlation(random_rewss_t1.T, random_rewss_t2.T, effs_sorted, effs_sorted, "task 1 simulated rewards", "task 2 simulated rewards", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_correlation_rews.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_correlation_rews.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_pvs_rews.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_pvs_rews.svg' %(get_timestamp())), dpi=300)


# %% SIMULATED SLEEPS

corr_fig, pvs_fig = compute_2D_correlation(sim_obss_t1.T, sim_sleepss_t2.T, effs_sorted, effs_sorted, "task 1 simulated observations", "task 2 simulated sleeps", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs_sleeps.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs_sleeps.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_obs_sleeps.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_obs_sleeps.svg' %(get_timestamp())), dpi=300)

# %%

corr_fig, pvs_fig = compute_2D_correlation(nostruc_obss_t1.T, nostruc_sleepss_t2.T, effs_sorted, effs_sorted, "task 1 simulated observations", "task 2 simulated sleeps", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_correlation_obs_sleeps.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_correlation_obs_sleeps.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_pvs_obs_sleeps.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_pvs_obs_sleeps.svg' %(get_timestamp())), dpi=300)

# %% 

corr_fig, pvs_fig = compute_2D_correlation(random_obss_t1.T, random_sleepss_t2.T, effs_sorted, effs_sorted, "task 1 simulated observations", "task 2 simulated sleeps", annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_correlation_obs_sleeps.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_correlation_obs_sleeps.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_pvs_obs_sleeps.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_pvs_obs_sleeps.svg' %(get_timestamp())), dpi=300)

# %% SIMULATED SLEEPS - PARTIALED

corr_fig, pvs_fig = compute_partial_2D_correlation(sim_obss_t1.T, sim_sleepss_t2.T, sim_obss_t2.T, effs_sorted, effs_sorted, effs_sorted, semi=False, col1name = 'n_observes_day2', col2name = 'n_sleeps_day3', annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs_sleeps_partialed.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_correlation_obs_sleeps_partialed.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_obs_sleeps_partialed.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_pvs_obs_sleeps_partialed.svg' %(get_timestamp())), dpi=300)

# %%

corr_fig, pvs_fig = compute_partial_2D_correlation(nostruc_obss_t1.T, nostruc_sleepss_t2.T, nostruc_obss_t2.T, effs_sorted, effs_sorted, effs_sorted, semi=False, col1name = 'n_observes_day2', col2name = 'n_sleeps_day3', annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_correlation_obs_sleeps_partialed.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_correlation_obs_sleeps_partialed.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_pvs_obs_sleeps_partialed.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_nostruc_pvs_obs_sleeps_partialed.svg' %(get_timestamp())), dpi=300)

# %%

corr_fig, pvs_fig = compute_partial_2D_correlation(random_obss_t1.T, random_sleepss_t2.T, random_obss_t2.T, effs_sorted, effs_sorted, effs_sorted, semi=False, col1name = 'n_observes_day2', col2name = 'n_sleeps_day3', annot=False)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_correlation_obs_sleeps_partialed.png' %(get_timestamp())), dpi=300)
corr_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_correlation_obs_sleeps_partialed.svg' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_pvs_obs_sleeps_partialed.png' %(get_timestamp())), dpi=300)
pvs_fig.savefig(os.path.join(sim_parts_analysis_folder, '%s_2D_random_pvs_obs_sleeps_partialed.svg' %(get_timestamp())), dpi=300)

# %% COMPUTE LIN REG COEFFICIENTS - STATISTICAL TESTS

if group is not None:
    corr_obs_data = np.corrcoef(np.stack(df['n_observes'].values).T)
    corr_sleeps_data = np.corrcoef(np.stack(df['n_sleeps'].values).T)
    corr_rews_data = np.corrcoef(np.stack(df['rewards_tallies'].values).T)
else:

    ## day 2
    data_obs_corr_g1, data_obs_pvs_g1 = compute_2D_correlation_matrices(np.stack(df[~df['group_day2']]['n_observes_day2'].values), np.stack(df[~df['group_day2']]['n_observes_day3'].values), np.stack(df[~df['group_day2']]['effs_day2'].values), np.stack(df[~df['group_day3']]['effs_day3'].values),)
    data_obs_corr_g2, data_obs_pvs_g2 = compute_2D_correlation_matrices(np.stack(df[df['group_day2']]['n_observes_day2'].values), np.stack(df[df['group_day2']]['n_observes_day3'].values), np.stack(df[df['group_day2']]['effs_day2'].values), np.stack(df[df['group_day3']]['effs_day3'].values),)

    data_rews_corr_g1, data_rews_pvs_g1 = compute_2D_correlation_matrices(np.stack(df[~df['group_day2']]['rewards_tallies_day2'].values), np.stack(df[~df['group_day2']]['rewards_tallies_day3'].values), np.stack(df[~df['group_day2']]['effs_day2'].values), np.stack(df[~df['group_day3']]['effs_day3'].values),)
    data_rews_corr_g2, data_rews_pvs_g2 = compute_2D_correlation_matrices(np.stack(df[df['group_day2']]['rewards_tallies_day2'].values), np.stack(df[df['group_day2']]['rewards_tallies_day3'].values), np.stack(df[df['group_day2']]['effs_day2'].values), np.stack(df[df['group_day3']]['effs_day3'].values),)

    data_obs_corr = (~df['group_day2']).sum() / len(df) * data_obs_corr_g1 + (df['group_day2']).sum() / len(df) * data_obs_corr_g2
    data_obs_pvs = (~df['group_day2']).sum() / len(df) * data_obs_pvs_g1 + (df['group_day2']).sum() / len(df) * data_obs_pvs_g2

    data_rews_corr = (~df['group_day2']).sum() / len(df) * data_rews_corr_g1 + (df['group_day2']).sum() / len(df) * data_rews_corr_g2
    data_rews_pvs = (~df['group_day2']).sum() / len(df) * data_rews_pvs_g1 + (df['group_day2']).sum() / len(df) * data_rews_pvs_g2

    data_sleeps_corr_g1, data_sleeps_pvs_g1 = compute_partial_2D_correlation_matrices(np.stack(df[~df['group_day2']]['n_observes_day2'].values), np.stack(df[~df['group_day2']]['n_sleeps_day3'].values), np.stack(df[~df['group_day2']]['n_observes_day3'].values), np.stack(df[~df['group_day2']]['effs_day2'].values), np.stack(df[~df['group_day2']]['effs_day3'].values), np.stack(df[~df['group_day2']]['effs_day3'].values),)
    data_sleeps_corr_g2, data_sleeps_pvs_g2 = compute_partial_2D_correlation_matrices(np.stack(df[df['group_day2']]['n_observes_day2'].values), np.stack(df[df['group_day2']]['n_sleeps_day3'].values), np.stack(df[df['group_day2']]['n_observes_day3'].values), np.stack(df[df['group_day2']]['effs_day2'].values), np.stack(df[df['group_day2']]['effs_day3'].values),np.stack(df[df['group_day2']]['effs_day3'].values), )

    data_sleeps_corr = (~df['group_day2']).sum() / len(df) * data_sleeps_corr_g1 + (df['group_day2']).sum() / len(df) * data_sleeps_corr_g2
    data_sleeps_pvs = (~df['group_day2']).sum() / len(df) * data_sleeps_pvs_g1 + (df['group_day2']).sum() / len(df) * data_sleeps_pvs_g2

sim_obs_corr, sim_obs_pvs = compute_2D_correlation_matrices(sim_obss_t1.T, sim_obss_t2.T, effs, effs,)
sim_rews_corr, sim_rews_pvs = compute_2D_correlation_matrices(sim_rewss_t1.T, sim_rewss_t2.T, effs, effs,)
sim_sleeps_corr, sim_sleeps_pvs = compute_partial_2D_correlation_matrices(sim_obss_t1.T, sim_sleepss_t2.T, sim_obss_t2.T, effs_sorted, effs_sorted, effs_sorted, semi=False,)
nostruc_obs_corr, nostruc_obs_pvs = compute_2D_correlation_matrices(nostruc_obss_t1.T, nostruc_obss_t2.T, effs, effs,)
nostruc_rews_corr, nostruc_rews_pvs = compute_2D_correlation_matrices(nostruc_rewss_t1.T, nostruc_rewss_t2.T, effs, effs,)
nostruc_sleeps_corr, nostruc_sleeps_pvs = compute_partial_2D_correlation_matrices(nostruc_obss_t1.T, nostruc_sleepss_t2.T, nostruc_obss_t2.T, effs_sorted, effs_sorted, effs_sorted, semi=False,)
random_obs_corr, random_obs_pvs = compute_2D_correlation_matrices(random_obss_t1.T, random_obss_t2.T, effs, effs,)
random_rews_corr, random_rews_pvs = compute_2D_correlation_matrices(random_rewss_t1.T, random_rewss_t2.T, effs, effs,)
random_sleeps_corr, random_sleeps_pvs = compute_partial_2D_correlation_matrices(random_obss_t1.T, random_sleepss_t2.T, random_obss_t2.T, effs_sorted, effs_sorted, effs_sorted, semi=False,)
null_obs_corr, null_obs_pvs = np.eye(len(effs)), np.eye(len(effs))
null_rews_corr, null_rews_pvs = np.eye(len(effs)), np.eye(len(effs))
null_sleeps_corr, null_sleeps_pvs = np.eye(len(effs)), np.eye(len(effs))

# %% COMPUTE LIN REG COEFFICIENTS - STATISTICAL TESTS


competitive_corr_regression((data_obs_corr), [(sim_obs_corr), (nostruc_obs_corr), (random_obs_corr)], do_fisher_transform=True)

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
competitive_corr_regression((data_rews_corr), [(sim_rews_corr), (nostruc_rews_corr), (random_obs_corr)])

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
competitive_corr_regression((data_sleeps_corr), [(sim_sleeps_corr), (nostruc_sleeps_corr), (random_sleeps_corr)])

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



# %% ANALYZING DATA AT THE LEVEL OF OVERALL FACTORS

data_obss_t1 = np.stack(df['n_observes_day2'].values).T
data_rewss_t1 = np.stack(df['rewards_tallies_day2'].values).T

data_obss_t2 = np.stack(df['n_observes_day3'].dropna().values).T
data_rewss_t2 = np.stack(df['rewards_tallies_day3'].dropna().values).T
data_sleepss_t2 = np.stack(df['n_sleeps_day3'].dropna().values).T

data_obss = np.concatenate([data_obss_t1, data_sleepss_t2, data_obss_t2], axis=1)
data_rewss = np.concatenate([data_rewss_t1, data_rewss_t2], axis=1)
#data_sleepss = np.concatenate([data_sleepss_t2], axis=1)

# %%

sim_obss_t1 = np.flip(sim_obss_t1, axis=1)
sim_rewss_t1 = np.flip(sim_rewss_t1, axis=1)

nostruc_obss_t1 = np.flip(nostruc_obss_t1, axis=1)
nostruc_rewss_t1 = np.flip(nostruc_rewss_t1, axis=1)

random_obss_t1 = np.flip(random_obss_t1, axis=1)
random_rewss_t1 = np.flip(random_rewss_t1, axis=1)

sim_obss_t2 = np.flip(sim_obss_t2, axis=1)
sim_rewss_t2 = np.flip(sim_rewss_t2, axis=1)
sim_sleepss_t2 = np.flip(sim_sleepss_t2, axis=1)

nostruc_obss_t2 = np.flip(nostruc_obss_t2, axis=1)
nostruc_rewss_t2 = np.flip(nostruc_rewss_t2, axis=1)
nostruc_sleepss_t2 = np.flip(nostruc_sleepss_t2, axis=1)

random_obss_t2 = np.flip(random_obss_t2, axis=1)
random_rewss_t2 = np.flip(random_rewss_t2, axis=1)
random_sleepss_t2 = np.flip(random_sleepss_t2, axis=1)

sim_obss = np.concatenate([sim_obss_t1, sim_sleepss_t2, sim_obss_t2], axis=1)
sim_rewss = np.concatenate([sim_rewss_t1, sim_rewss_t2], axis=1)

nostruc_obss = np.concatenate([nostruc_obss_t1, nostruc_sleepss_t2, nostruc_obss_t2], axis=1)
nostruc_rewss = np.concatenate([nostruc_rewss_t1, nostruc_rewss_t2], axis=1)

random_obss = np.concatenate([random_obss_t1, random_sleepss_t2, random_obss_t2], axis=1)
random_rewss = np.concatenate([random_rewss_t1, random_rewss_t2], axis=1)

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

# Data Info for data_obss: Number of Factors: 3 Variance Explained: 0.7426089755527101
# Data Info for sim_obss: Number of Factors: 3 Variance Explained: 0.9980966759094801
# Data Info for nostruc_obss: Number of Factors: 3 Variance Explained: 0.9985678827107767
# Data Info for random_obss: Number of Factors: 3 Variance Explained: 0.3368417016748841


# Similarity with sim_obss: 0.8414183557631946 Confidence Interval: [0.69804311 0.89871962]
# Similarity with nostruc_obss: 0.3420733475569781 Confidence Interval: [0.26067715 0.48204154]
# Similarity with random_obss: 0.4553708601998167 Confidence Interval: [0.26625896 0.59526009]

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

# Data Info for data_obss: Number of Factors: 3 Variance Explained: 0.32531375595872614
# Data Info for sim_obss: Number of Factors: 3 Variance Explained: 0.9966992292569907
# Data Info for nostruc_obss: Number of Factors: 3 Variance Explained: 0.815667576147506
# Data Info for random_obss: Number of Factors: 3 Variance Explained: 0.3229644151435964

# Similarity with sim_obss: 0.350918152315611 Confidence Interval: [0.287664   0.58109178]
# Similarity with nostruc_obss: 0.19077841179906155 Confidence Interval: [-0.12703499  0.31023362]
# Similarity with random_obss: 0.13813992080206502 Confidence Interval: [-0.01641413  0.47674508]

# sim_obss is the most similar to the ground truth.

# %% ANALYZE SURVEY RESPONSES

surveys = df_day3['survey_responses'].to_list()
#surveys = [survey[5] for survey in surveys if (survey != [] and 'leb.1' not in survey[-1])]
surveys= [survey[-1] for survey in surveys if (survey != [] and 'leb.1' not in survey[-1])]
surveys

# %% CORRELATION MATRICES AVERAGING ACROSS GROUPS


