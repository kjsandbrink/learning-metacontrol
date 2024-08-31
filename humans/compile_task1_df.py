# Kai Sandbrink
# 2023-04-27
# This script analyzes Prolific Participants with different efficacy

# %% LIBRARY IMPORTS

import numpy as np
import pandas as pd
import os, ast, glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from utils import format_axis, get_timestamp
from human_utils_project import plot_reward_tally, plot_n_observes, plot_prob_intended_correct_takes, plot_train_test_comp, plot_evidence_ratios, plot_prob_action_over_ep
from human_utils_project import extract_participant_trajs, combine_partial_datas, print_turkers, calculate_bonus_payouts
from human_utils_project import get_participant_data, get_p_datass, sort_overall, sort_train_test, load_config_files

# %% PARAMETERS

participant_ids = ['xyz']

contains_slider = True
contains_group = True

day = 'day2B'
exp_date = '24-01-29'

#### GENERAL PARAMETERS

effs_sorted = np.arange(0,1.01, 0.125)
#test_start = len(effs_sorted_train)
n_effs = len(effs_sorted)

n_steps = 50
save_df_file = os.path.join('results', 'behavior', '%s_behavior_diff_effs_%s_%s' %(get_timestamp(), exp_date, day))

#n_trials = 10
analysis_folder = os.path.join('analysis', 'traj_diff_efficacy', day, exp_date)
data_folder = os.path.join('data',day, 'data')

cmap_train = mpl.colormaps['Greens']
cmap_test = mpl.colormaps['Blues']

cmaps = {
    'train': cmap_train,
    'test': cmap_test,
}

# %% CONVERT LIST OF PARTICIPANT IDS TO FILENAMES AND PARTIAL_FILENAMES

datas, complete_pids, df = get_p_datass(participant_ids, data_folder, n_effs, contains_slider=contains_slider, contains_group = contains_group)

# %% PAYOUTS

print("Bonus payments:")
calculate_bonus_payouts(datas)

print("\nApproved participants:")
print_turkers(datas)

# %% PRINT PARTICIPANT IDS IN GROUP A/B IF APPLICABLE

print('\nwithout quotation marks')

if contains_group:
    
    print("Group A:")

    indices = df.index[~df['group']]
    
    print(' ,\n'.join(indices))

print('\nwith quotation marks')

if contains_group:
    
    indices = df.index[~df['group']]
    
    print("Group A:\n\'" + '\' ,\n\''.join(indices) + '\'')
    

# %% 
if contains_group:
    print("\nGroup B:")
    indices = df.index[df['group']]
    
    print(' ,\n'.join(indices))

    print('\nwith quotation marks')
    print("Group B:\n\'" + '\' ,\n\''.join(indices) + '\'')

# %% DATA SAVE

df.to_csv(save_df_file + '.csv')
df.to_pickle(save_df_file + '.pkl')

# %% INITIALIZATIONS FOR SOME BASIC "OVERVIEW" ANALYSES

n_participants = len(df)
os.makedirs(analysis_folder, exist_ok=True)

group = ~df['group'].values
#group  = [True]*n_participants

# %% PLOT REWARDS TALLIES VS. EFFICACIES -- DATA

rewardss_tallies_sorted = sort_overall(df['rewards_tallies'][group], df['effs'][group].values)

fig = plot_reward_tally(effs_sorted, rewardss_tallies_sorted)
fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.png' %get_timestamp()))
#fig

# %% SEPARATE INTO TRAINING V TESTING

rewardss_tallies_sorted_train, rewardss_tallies_sorted_test = sort_train_test(df['rewards_tallies'], df['effs'].values, test_start)

fig = plot_reward_tally(effs_sorted_train, rewardss_tallies_sorted_train)
fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy_train.png' %get_timestamp()))
#fig

# %% TEST

fig = plot_reward_tally(effs_sorted_test, rewardss_tallies_sorted_test)
fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy_test.png' %get_timestamp()))
#fig

# %% PLOT
rewardss_tallies_sorted_train, rewardss_tallies_sorted_test = sort_train_test(df['rewards_tallies'], df['effs'].values, test_start)
fig = plot_train_test_comp(effs_sorted_train, rewardss_tallies_sorted_train, effs_sorted_test, rewardss_tallies_sorted_test)
fig.savefig(os.path.join(analysis_folder, '%s_rewards_tr_te_comp.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_rewards_tr_te_comp.svg' %get_timestamp()))

# %% N_OBSERVES vs. EFFICACY

n_observes_sorted = sort_overall(df['n_observes'][group].values, df['effs'][group].values)

fig = plot_n_observes(effs_sorted, n_observes_sorted)

plt.ylim(0, 25)

fig.savefig(os.path.join(analysis_folder, '%s_n_observes_efficacy.png' %get_timestamp()))
#fig

# %% TRAIN

n_observes_sorted_train, n_observes_sorted_test = sort_train_test(df['n_observes'].values, df['effs'].values, test_start)

fig = plot_n_observes(effs_sorted_train, n_observes_sorted_train)
fig.savefig(os.path.join(analysis_folder, '%s_n_observes_efficacy_train.png' %get_timestamp()))
#fig

# %% TEST

fig = plot_n_observes(effs_sorted_test, n_observes_sorted_test)
fig.savefig(os.path.join(analysis_folder, '%s_n_observes_efficacy_test.png' %get_timestamp()))

# %% PLOT

obs_sorted_train, obs_sorted_test = sort_train_test(df['n_observes'], df['effs'].values, test_start)
fig = plot_train_test_comp(effs_sorted_train, obs_sorted_train, effs_sorted_test, obs_sorted_test, y_label="Number of Observes")
fig.savefig(os.path.join(analysis_folder, '%s_n_observes_efficacy_trte.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_n_observes_efficacy_trte.svg' %get_timestamp()))


# # %% PROB TAKE CORRECT (OUTCOME) - DATA

# bets = np.where(transitionss_ep_rightwrong == 0.5, np.nan, transitions_ep_rightwrong)

# prob_take_correct = np.nanmean(bets, axis=2)
# prob_take_correct_sorted = np.array([row[order[i]] for i, row in enumerate(prob_take_correct)])

# mean_prob = np.mean(prob_take_correct_sorted, axis=0)
# std_prob = np.mean(prob_take_correct_sorted, axis=0)/np.sqrt(n_participants)

# # %%

# fig = plt.figure(dpi=600)
# ax = fig.add_subplot(111)

# for i in range(n_participants):
#     ax.plot(effs_sorted, prob_take_correct_sorted[i], label=filenames[i][-10:], color='C%d' %i, alpha=0.5)

# ax.legend()
# format_axis(ax)

# ax.plot(effs_sorted, mean_prob, color='black', label='Mean', linewidth=3.5)
# ax.fill_between(effs_sorted, mean_prob - std_prob, mean_prob + std_prob, color='black', alpha=0.2)

# ### format axis
# ax.set_xlabel('Efficacy')
# ax.set_ylabel('Probability of Taking Correct Arm')

# plt.tight_layout()

# fig.savefig(os.path.join(analysis_folder, '%s_p_take_corr_efficacy.png' %get_timestamp()))
# fig

# %% PROB TAKE CORRECT (INTENDED) - DATA

intended_correct_takes_ep_sorted = sort_overall(df['intended_correct'][group].values, df['effs'][group].values)

fig = plot_prob_intended_correct_takes(effs_sorted, intended_correct_takes_ep_sorted)

fig.savefig(os.path.join(analysis_folder, '%s_p_intended_take_corr_efficacy.png' %get_timestamp()))
#fig

# %% TRAIN AND TEST SPLIT

intended_correct_takes_ep_sorted_train, intended_correct_takes_ep_sorted_test = sort_train_test(df['intended_correct'].values, df['effs'].values, test_start)

fig = plot_prob_intended_correct_takes(effs_sorted_train, intended_correct_takes_ep_sorted_train)
fig.savefig(os.path.join(analysis_folder, '%s_p_intended_take_corr_efficacy_train.png' %get_timestamp()))
#fig

# %% TEST

fig = plot_prob_intended_correct_takes(effs_sorted_test, intended_correct_takes_ep_sorted_test)
fig.savefig(os.path.join(analysis_folder, '%s_p_intended_take_corr_efficacy_test.png' %get_timestamp()))
#fig

# %% PLOT COMPARISON

ic_sorted_train, ic_sorted_test = sort_train_test(df['intended_correct'], df['effs'].values, test_start)
fig = plot_train_test_comp(effs_sorted_train, ic_sorted_train, effs_sorted_test, ic_sorted_test, y_label="Probability of Intended Correct Bet")
fig.savefig(os.path.join(analysis_folder, '%s_ic_tr_te_comp.png' %get_timestamp()))

fig.savefig(os.path.join(analysis_folder, '%s_ic_tr_te_comp.svg' %get_timestamp()))

# %% 

from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = ListedColormap(['blue', 'black', 'red'])
norm = BoundaryNorm([0, 0.5, 1, 2], cmap.N)

## write a function that takes transitions_ep for a participant and returns a figure that plots the matrix
def create_transitions_heatmap(row):
    p_analysis_folder = os.path.join(analysis_folder, "transitions_ep", row.name)
    os.makedirs(p_analysis_folder, exist_ok=True)

    for ie, pt in enumerate(row.transitions_ep):

        # Create the heatmap using seaborn
        plt.figure(dpi=300, figsize=(15,3))
        sns.heatmap(np.array(pt).T, cmap=cmap, norm=norm, cbar=False)
        
        # Format the axes
        plt.xlabel('Step')
        
        plt.yticks(ticks=np.arange(0.5,2.6,1), labels=['Correct', 'Chosen', 'Taken'])
        plt.title('Participant %s, Episode %s, Eff %.3f, Rew %d' %(row.name, ie, row.effs[ie], row.rewards_tallies[ie]))
        plt.tight_layout()
                
        # Save the figure to the analysis_folder with the index as the filename
        plt.savefig(os.path.join(p_analysis_folder, 'transitions_heatmap_ep%d_eff%d.png' %(ie, row.effs[ie]*1000)))
        plt.close()

# Apply the function to each row of the DataFrame
df.apply(create_transitions_heatmap, axis=1)

# %% SHOW AGGREGATED CHOICES OVER ONE EPISODE

trans_rw_tr, trans_rw_te = sort_train_test(df['transitions_ep_rightwrong'].values, df['effs'].values, test_start)

for trans, effs, cond in zip((trans_rw_tr, trans_rw_te), (effs_sorted_train, effs_sorted_test), ('train', 'test')):

    for action, actionname, shortname in zip([0.5, 1], ['Observing', 'Correct Bet'], ['obs', 'cb']):
        
        fig = plot_prob_action_over_ep(effs, trans, action, cmaps[cond], ylabel='Probability of %s' %actionname)
        fig.savefig(os.path.join(analysis_folder, '%s_prob_%s_%s_efficacy.png' %(get_timestamp(), shortname, cond)))
        fig.savefig(os.path.join(analysis_folder, '%s_prob_%s_%s_efficacy.svg' %(get_timestamp(), shortname, cond)))


# %% FACTORING IN TIME SINCE LAST TAKE

def calculate_freq_observed_choice_per_t(row):
    """
    Calculate the fraction of correct choices at the time they observed (0.5).

    """

    fractions = []

    for ep in row:

        counter_since_observes = np.zeros(n_steps,)
        counter_correct_choices = np.zeros(n_steps,)
        
        counter_since_observe = 0
        current_choice = None
        # Iterate over the sublists in each row
        for sublist in ep:
            # Check if intended choice is "observed"
            if sublist[1] == 0.5:
                counter_since_observe = 0
                current_choice = sublist[0]
            else:
                counter_since_observe += 1
                counter_since_observes[counter_since_observe - 1] += 1
            
                if current_choice == sublist[2]:
                    counter_correct_choices[counter_since_observe - 1] += 1
                        
        # Calculate fraction
        fraction = [correct / total if total > 0 else np.nan for correct, total in zip(counter_correct_choices, counter_since_observes)]
        fractions.append(fraction)
        
    return fractions

# Calculate fractions for each row
df['frac_takes_obs_t_since'] = df['transitions_ep'].apply(calculate_freq_observed_choice_per_t)

# %%

def calculate_freq_observes_per_t(row):
    """
    Calculate the fraction of correct choices at the time they observed (0.5).

    """

    fractions = []

    for ep in row:

        counter_since_observes = np.zeros(n_steps,)
        counter_correct_choices = np.zeros(n_steps,)
        
        counter_since_observe = 0
        current_choice = None
        # Iterate over the sublists in each row
        for sublist in ep:
            # Check if intended choice is "observed"
            counter_since_observes[counter_since_observe] += 1
            if sublist[1] == 0.5:
                counter_correct_choices[counter_since_observe] += 1
                counter_since_observe = 0
            counter_since_observe += 1
 
        # Calculate fraction
        fraction = [correct / total if total > 0 else np.nan for correct, total in zip(counter_correct_choices, counter_since_observes)]
        fractions.append(fraction)
        
    return fractions

# Calculate fractions for each row
df['frac_observes_t_since'] = df['transitions_ep'].apply(calculate_freq_observes_per_t)

# %%

def calculate_freq_correct_choice_per_t(row):
    """
    Calculate the fraction of correct choices

    """

    fractions = []

    for ep in row:

        counter_since_observes = np.zeros(n_steps,)
        counter_correct_choices = np.zeros(n_steps,)
        
        counter_since_observe = 0
        # Iterate over the sublists in each row
        for sublist in ep:
            # Check if intended choice is "observed"
            if sublist[1] == 0.5:
                counter_since_observe = 0
            else:
                counter_since_observe += 1
                counter_since_observes[counter_since_observe - 1] += 1
            
                if sublist[0] == sublist[2]:
                    counter_correct_choices[counter_since_observe - 1] += 1
                        
        # Calculate fraction
        fraction = [correct / total if total > 0 else np.nan for correct, total in zip(counter_correct_choices, counter_since_observes)]
        fractions.append(fraction)
        
    return fractions

# Calculate fractions for each row
df['frac_corr_takes_t_since'] = df['transitions_ep'].apply(calculate_freq_correct_choice_per_t)

# %% PLOT

from matplotlib.colors import ListedColormap, BoundaryNorm
cmap = mpl.colormaps['Reds']

## write a function that takes transitions_ep for a participant and returns a figure that plots the matrix
def create_evidence_ratio_plot(row, metric):
    p_analysis_folder = os.path.join(analysis_folder, "evidence ratio", "participants", row.name)
    os.makedirs(p_analysis_folder, exist_ok=True)
    plt.figure()

    for ie, pm in enumerate(row[metric]):
        
        plt.plot(range(n_steps), pm, label='Eff %.3f' %row.effs[ie], c=cmap(row.effs[ie]))
        # Format the axes
    plt.xlabel('Time since last observe')
    plt.ylabel(metric)
            
    plt.legend()
    plt.tight_layout()
            
    # Save the figure to the analysis_folder with the index as the filename
    plt.savefig(os.path.join(p_analysis_folder, 'evidence_ratio_%s.png' %(metric)))
    plt.close()

# Apply the function to each row of the DataFrame
# for metric in ['frac_takes_obs_t_since', 'frac_observes_t_since', 'frac_corr_takes_t_since']:
#     df.apply(create_evidence_ratio_plot, args=(metric,), axis=1)

# %% EVIDENCE TALLY PLOTS AGGREGATED OVER PARTICIPANTS

os.makedirs(os.path.join(analysis_folder, 'evidence_ratio'), exist_ok=True)

for metric in ['frac_takes_obs_t_since', 'frac_observes_t_since', 'frac_corr_takes_t_since']:
    er_sorted_train, er_sorted_test = sort_train_test(df[metric].values, df['effs'].values, test_start)

    fig = plot_evidence_ratios(er_sorted_train, effs_sorted_train, cmap_train, metric, jitter=True)
    fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_train.png' %(get_timestamp(), metric)))
    fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_train.svg' %(get_timestamp(), metric)))

    fig = plot_evidence_ratios(er_sorted_test, effs_sorted_test, cmap_test, metric, jitter=True)
    fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_test.png' %(get_timestamp(), metric)))
    fig.savefig(os.path.join(analysis_folder, 'evidence_ratio', '%s_evidence_ratio_%s_test.svg' %(get_timestamp(), metric)))

# %% ANALYZE SLIDER VALUES

estimates_train, estimates_test = sort_train_test(df['efficacy_estimates'].values, df['effs'].values, test_start)

fig = plot_train_test_comp(effs_sorted_train, estimates_train, effs_sorted_test, estimates_test, y_label='Efficacy estimates')

fig.savefig(os.path.join(analysis_folder, '%s_efficacy_estimates_trte.png' %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, '%s_efficacy_estimates_trte.svg' %get_timestamp()))

# %%

