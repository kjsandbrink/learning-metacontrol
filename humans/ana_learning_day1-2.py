# Kai Sandbrink
# 2023-07-30
# This script compares data from Day 1 and Day 2 to see if any learning took place.

# %% LIBRARY IMPORT

import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind

from human_utils_project import sort_train_test, get_clean_data

from human_utils_behavioral_analysis import mean_slope_train_test, get_evs_wanted, compute_2D_correlation

# %% LOAD DATA

exp_date = '24-01-22-29'
group = "groupB"
day1_test_mask_cutoff=10

df_day1, effs_train_day1, effs_test_day1, test_start_day1 = get_clean_data(day = 1, exp_date=exp_date, group=group, day1_test_mask_cutoff=day1_test_mask_cutoff)
df_day2, effs_train_day2, effs_test_day2, test_start_day2 = get_clean_data(day = 2, exp_date=exp_date, group=group, day1_test_mask_cutoff=day1_test_mask_cutoff)

effs = np.arange(0, 1.125, 0.125)

analysis_folder = os.path.join('analysis', 'traj_diff_efficacy', 'across_day1_to_day2', exp_date, group)
os.makedirs(analysis_folder, exist_ok=True)

# %% PREPROCESS COMPARISON BOTH DAYS TO EV

## SARSOP
#evs = np.array([25, 25, 25, 25, 25.7681, 26.8289, 29.3261, 30.6419, 32.0193])

## NNs
# evs = np.array([30.88555556, 29.53222222, 28.07888889, 26.79333333, 25.82777778, 25.20222222,
#  24.45111111, 24.56222222, 24.32666667])
# eobs = np.array([7.10666667, 6.76555556, 6.22444444, 5.22,       3.95444444, 2.76777778,
#  1.89555556, 1.27333333, 1.10666667])

rews_train_day1, rews_test_day1 = sort_train_test(df_day1['rewards_tallies'], df_day1['effs'], test_start_day1)
rews_train_day2, rews_test_day2 = sort_train_test(df_day2['rewards_tallies'], df_day2['effs'], test_start_day2)

evs_train_day1 = get_evs_wanted(evs, effs, effs_train_day1)
evs_test_day1 = get_evs_wanted(evs, effs, effs_test_day1)

evs_train_day2 = get_evs_wanted(evs, effs, effs_train_day2)
evs_test_day2 = get_evs_wanted(evs, effs, effs_test_day2)

# %% REWS

print('Day 1 Rews Train:', np.mean(rews_train_day1), '+/-', np.std(rews_train_day1)/np.sqrt(len(rews_train_day1)))
print('Day 1 Rews Test:', np.mean(rews_test_day1), '+/-', np.std(rews_test_day1)/np.sqrt(len(rews_test_day1)))

print('Day 2 Rews Train:', np.mean(rews_train_day2), '+/-', np.std(rews_train_day2)/np.sqrt(len(rews_train_day2)))
print('Day 2 Rews Test:', np.mean(rews_test_day2), '+/-', np.std(rews_test_day2)/np.sqrt(len(rews_test_day2)))

## RESULTS FOR RUN FEATURING DATA FROM 5-18, 5-25, 6-19, 7-06

# Day 1 Rews Train: 25.333892617449663 +/- 0.465277324998038
# Day 1 Rews Test: 25.36510067114094 +/- 0.4021998494076815
# Day 2 Rews Train: 25.736912751677853 +/- 0.41463731333660536
# Day 2 Rews Test: 26.29530201342282 +/- 0.4724044088196562


mrtrd1 = np.mean(rews_train_day1, axis=1)
mrted1 = np.mean(rews_test_day1, axis=1)
mrtrd2 = np.mean(rews_train_day2, axis=1)
mrted2 = np.mean(rews_test_day2, axis=1)

#res1 = ttest_ind(mrtrd1, mrtrd2)

print("Two-sided t-test train: ", ttest_rel(mrtrd1, mrtrd2))
print("Two-sided t-test test: ", ttest_rel(mrted1, mrted2))
print("DoF: ", len(df_day1) - 1)

# %% OBS

nobs_train_day1, nobs_test_day1 = sort_train_test(df_day1['n_observes'], df_day1['effs'], test_start_day1)
nobs_train_day2, nobs_test_day2 = sort_train_test(df_day2['n_observes'], df_day2['effs'], test_start_day2)

print('Day 1 Nobs Train:', np.mean(nobs_train_day1), '+/-', np.std(nobs_train_day1)/np.sqrt(len(nobs_train_day1)))
print('Day 1 Nobs Test:', np.mean(nobs_test_day1), '+/-', np.std(nobs_test_day1)/np.sqrt(len(nobs_test_day1)))

print('Day 2 Nobs Train:', np.mean(nobs_train_day2), '+/-', np.std(nobs_train_day2)/np.sqrt(len(nobs_train_day2)))
print('Day 2 Nobs Test:', np.mean(nobs_test_day2), '+/-', np.std(nobs_test_day2)/np.sqrt(len(nobs_test_day2)))

## t-test
mrtrd1 = np.mean(nobs_train_day1, axis=1)
mrted1 = np.mean(nobs_test_day1, axis=1)
mrtrd2 = np.mean(nobs_train_day2, axis=1)
mrted2 = np.mean(nobs_test_day2, axis=1)

#res1 = ttest_ind(mrtrd1, mrtrd2)

print("Two-sided t-test train: ", ttest_rel(mrtrd1, mrtrd2))
print("Two-sided t-test test: ", ttest_rel(mrted1, mrted2))
print("DoF: ", len(df_day1) - 1)


# %% COMPARISON REWARDS

rews_diffs_train_day1 = np.array(rews_train_day1) - evs_train_day1
rews_diffs_test_day1 = np.array(rews_test_day1) - evs_test_day1

rews_diffs_train_day2 = np.array(rews_train_day2) - evs_train_day2
rews_diffs_test_day2 = np.array(rews_test_day2) - evs_test_day2

# %%

print('Day 1 Rews Train:', np.mean(rews_diffs_train_day1), '+/-', np.std(rews_diffs_train_day1)/np.sqrt(len(rews_diffs_train_day1)))
print('Day 1 Rews Test:', np.mean(rews_diffs_test_day1), '+/-', np.std(rews_diffs_test_day1)/np.sqrt(len(rews_diffs_test_day1)))

print('Day 2 Rews Train:', np.mean(rews_diffs_train_day2), '+/-', np.std(rews_diffs_train_day2)/np.sqrt(len(rews_diffs_train_day2)))
print('Day 2 Rews Test:', np.mean(rews_diffs_test_day2), '+/-', np.std(rews_diffs_test_day2)/np.sqrt(len(rews_diffs_test_day2)))

### RESULTS FOR RUN FEATURING DATA FROM 5-18, 5-25, 6-19, 7-06

## Compared with NN Models
# Day 1 Rews Train: -1.6016629400503364 +/- 0.5649751453065064
# Day 1 Rews Test: -1.0184548828590607 +/- 0.46557313463377165
# Day 2 Rews Train: -0.6466428023221477 +/- 0.470695860026369
# Day 2 Rews Test: -0.640253544077182 +/- 0.5905907317259009

# %% CHECK T-TEST

mrtrd1 = np.mean(rews_diffs_train_day1, axis=1)
mrted1 = np.mean(rews_diffs_test_day1, axis=1)
mrtrd2 = np.mean(rews_diffs_train_day2, axis=1)
mrted2 = np.mean(rews_diffs_test_day2, axis=1)

#res1 = ttest_ind(mrtrd1, mrtrd2)

print("Two-sided t-test train: ", ttest_rel(mrtrd1, mrtrd2))
print("Two-sided t-test test: ", ttest_rel(mrted1, mrted2))
print("DoF: ", len(df_day1) - 1)

### RESULTS FOR RUN FEATURING DATA FROM 5-18, 5-25, 6-19, 7-06

# Two-sided t-test train:  Ttest_relResult(statistic=-3.4083973076705956, pvalue=0.000842000333671661)
# Two-sided t-test test:  Ttest_relResult(statistic=-1.3223754480659238, pvalue=0.18808308358836767)
# DoF:  148

# %% COMPARE LEARNING SLOPS

msod1 = mean_slope_train_test(df_day1['n_observes'], df_day1['effs'], effs_train_day1, effs_test_day1)
msod2 = mean_slope_train_test(df_day2['n_observes'], df_day2['effs'], effs_train_day2, effs_test_day2)

# %%

print('Day 1 Slopes Train:', np.mean(msod1[0]), '+/-', np.std(msod1[0])/np.sqrt(len(msod1[0])))
print('Day 1 Rews Test:', np.mean(msod1[1]), '+/-', np.std(msod1[1])/np.sqrt(len(msod1[1])))

print('Day 2 Slopes Train:', np.mean(msod2[0]), '+/-', np.std(msod2[0])/np.sqrt(len(msod2[0])))
print('Day 2 Rews Test:', np.mean(msod2[1]), '+/-', np.std(msod2[1])/np.sqrt(len(msod2[1])))

print("Two-sided t-test train: ", ttest_rel(msod1[0], msod2[0]))
print("Two-sided t-test test: ", ttest_rel(msod1[1], msod2[1]))
print("DoF: ", len(df_day1) - 1)

# Day 1 Slopes Train: 5.422818791946308 +/- 0.6920720476052695
# Day 1 Rews Test: 2.7248322147651005 +/- 0.46224600765838525
# Day 2 Slopes Train: 3.9865771812080535 +/- 0.43349539083912186
# Day 2 Rews Test: 2.152125279642058 +/- 0.3317075681674441
# Two-sided t-test train:  TtestResult(statistic=2.080505974147818, pvalue=0.03920272333490683, df=148)
# Two-sided t-test test:  TtestResult(statistic=1.166954500701751, pvalue=0.24510553595017848, df=148)
# DoF:  148

# %% COMPARE NUMBER OF OBSERVES TO NN MODEL BASELINES

eobs_train_day1 = get_evs_wanted(eobs, effs, effs_train_day1)
eobs_test_day1 = get_evs_wanted(eobs, effs, effs_test_day1)

eobs_train_day2 = get_evs_wanted(eobs, effs, effs_train_day2)
eobs_test_day2 = get_evs_wanted(eobs, effs, effs_test_day2)

nobs_train_day1, nobs_test_day1 = sort_train_test(df_day1['n_observes'], df_day1['effs'], test_start_day1)
nobs_train_day2, nobs_test_day2 = sort_train_test(df_day2['n_observes'], df_day2['effs'], test_start_day2)

dev_obs_train_day1 = np.abs(np.array(eobs_train_day1) - np.array(nobs_train_day1))
dev_obs_test_day1 = np.abs(np.array(eobs_test_day1) - np.array(nobs_test_day1))
dev_obs_train_day2 = np.abs(np.array(eobs_train_day2) - np.array(nobs_train_day2))
dev_obs_test_day2 = np.abs(np.array(eobs_test_day2) - np.array(nobs_test_day2))

# %% 

print('Day 1 Dev Obs Train:', np.mean(dev_obs_train_day1), '+/-', np.std(dev_obs_train_day1)/np.sqrt(len(dev_obs_train_day1)))
print('Day 1 Dev Obs Test:', np.mean(dev_obs_test_day1), '+/-', np.std(dev_obs_test_day1)/np.sqrt(len(dev_obs_test_day1)))

print('Day 2 Dev Obs Train:', np.mean(dev_obs_train_day2), '+/-', np.std(dev_obs_train_day2)/np.sqrt(len(dev_obs_train_day2)))
print('Day 2 Dev Obs Test:', np.mean(dev_obs_test_day2), '+/-', np.std(dev_obs_test_day2)/np.sqrt(len(dev_obs_test_day2)))

## RESULTS FOR RUN FEATURING DATA FROM 5-18, 5-25, 6-19, 7-06
# Day 1 Dev Obs Train: 5.512177478557048 +/- 0.33980818170041577
# Day 1 Dev Obs Test: 4.267413871073825 +/- 0.27072221006605574
# Day 2 Dev Obs Train: 4.111067860765101 +/- 0.23623183362682118
# Day 2 Dev Obs Test: 4.464981356543624 +/- 0.23572295956837053

mrtrd1 = np.mean(dev_obs_train_day1, axis=1)
mrted1 = np.mean(dev_obs_test_day1, axis=1)
mrtrd2 = np.mean(dev_obs_train_day2, axis=1)
mrted2 = np.mean(dev_obs_test_day2, axis=1)

#res1 = ttest_ind(mrtrd1, mrtrd2)

print("Two-sided t-test train: ", ttest_rel(mrtrd1, mrtrd2))
print("Two-sided t-test test: ", ttest_rel(mrted1, mrted2))
print("DoF: ", len(df_day1) - 1)

# %% LOOK AT UNSORTED EPISODES AND SEE IF THERE IS A TREND

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

erews = np.flip(np.array([30.137, 29.282, 28.337, 26.76, 25.597, 25.078, 24.64, 24.693, 24.553]))
stdrews = np.flip(np.array([0.55636328, 0.4134847, 0.32582219, 0.20409802, 0.12408908, 0.13327265, 0.11044456, 0.08972235, 0.16780971]))
eobs = np.flip(np.array([6.561, 6.334, 5.741, 4.829, 3.641, 2.531, 1.8, 1.324, 1.089]))
stdeobs = np.flip(np.array([0.80767004, 0.78759025, 0.73035122, 0.65029909, 0.56589124, 0.38641545, 0.2786862, 0.2229852, 0.13387644]))
ecorr = np.flip(np.array([0.68109297, 0.68260722, 0.67462006, 0.64070059, 0.60635619, 0.58156844, 0.5455282, 0.54777943, 0.53220974]))
stdecorr = np.flip(np.array([0.02063236, 0.0176135, 0.01929194, 0.01756656, 0.0139558, 0.01069981, 0.00754262, 0.00725149, 0.0067924]))
effs = np.arange(0, 1.01, 0.125)

def bring_into_human_order(data, human_effs, effs = np.arange(0, 1.01, 0.125)):
    ordered = np.zeros_like(data)

    human_effs = list(human_effs)
    #print(human_effs, effs)
    for e, d in zip(effs, data):
        #print(human_effs.index(e))
        ordered[human_effs.index(e)] = d

    return ordered

df_day1['n_observes_exp_nn'] = df_day1['effs'].apply(lambda x:bring_into_human_order(eobs, x))
df_day1['dev_obs'] = df_day1['n_observes'] - df_day1['n_observes_exp_nn']

df_day2['n_observes_exp_nn'] = df_day2['effs'].apply(lambda x:bring_into_human_order(eobs, x))
df_day2['dev_obs'] = df_day2['n_observes'] - df_day2['n_observes_exp_nn']

# %% MAKE PLOT LOOKING AT DEVIATION OVER TIME

import matplotlib.pyplot as plt
from utils import format_axis

def make_deviation_over_time_plot(devs_obs):

    mean_devs = devs_obs.mean(axis=0)
    stderr_devs = devs_obs.std(axis=0)/np.sqrt(len(devs_obs))

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    x = np.arange(1, len(mean_devs)+1)
    ax.plot(x, mean_devs, color='C2')
    ax.fill_between(x, mean_devs - stderr_devs, mean_devs + stderr_devs, alpha=0.2, color='C2')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Deviation from Neural Network')

    format_axis(ax)

    return fig


# %%

fig = make_deviation_over_time_plot(np.stack(df_day1['dev_obs'].values))
# %%

fig.savefig(os.path.join(analysis_folder, 'dev_obs_day1.png'))
fig.savefig(os.path.join(analysis_folder, 'dev_obs_day1.svg'))


# %%

fig = make_deviation_over_time_plot(np.stack(df_day2['dev_obs']))

# %%

fig.savefig(os.path.join(analysis_folder, 'dev_obs_day2.png'))
fig.savefig(os.path.join(analysis_folder, 'dev_obs_day2.svg'))

# %%
