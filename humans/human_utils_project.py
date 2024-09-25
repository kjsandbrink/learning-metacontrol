# Kai Sandbrink
# 2023-05-09
# This script shows

# %% LIBRARY IMPORT

import numpy as np
import ast, glob
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from utils import format_axis
import pandas as pd
import os
from scipy import stats

from utils import Config

# %% PLOTTING FUNCTIONS

def plot_train_test_comp(effs_train, metric_train, effs_test, metric_test, x_label = "Efficacy", y_label="Rewards", ylim=None):
        
    n_participants = len(metric_train)
    print(n_participants)
        
    mean_rews_sorted_tr = np.nanmean(metric_train, axis=0)
    stderr_rews_sorted_tr = np.nanstd(metric_train, axis=0)/np.sqrt(n_participants)

    mean_rews_sorted_te = np.nanmean(metric_test, axis=0)
    stderr_rews_sorted_te = np.nanstd(metric_test, axis=0)/np.sqrt(n_participants)

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    # for i in range(n_participants):
    #     ax.plot(effs_sorted, rewardss_tallies_sorted_train[i], color='C2', alpha=0.5)

    #ax.legend()
    format_axis(ax)

    ax.plot(effs_train, mean_rews_sorted_tr, color='C8', label='Train', linewidth=3.5)
    ax.fill_between(effs_train, mean_rews_sorted_tr - stderr_rews_sorted_tr, mean_rews_sorted_tr + stderr_rews_sorted_tr, color='C8', alpha=0.2)

    ax.plot(effs_test, mean_rews_sorted_te, color='C9', label='Test', linewidth=3.5)
    ax.fill_between(effs_test, mean_rews_sorted_te - stderr_rews_sorted_te, mean_rews_sorted_te + stderr_rews_sorted_te, color='C9', alpha=0.2)

    ### format axis
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title(title)
    plt.legend()

    if ylim is not None:
        ax.set_ylim(ylim)

    #plt.tight_layout()

    return fig

def plot_reward_tally(effs_sorted, rewardss_tallies_sorted, ylim=None):
    
    n_participants = len(rewardss_tallies_sorted)
    #print(n_participants)
        
    mean_rews_sorted = np.nanmean(rewardss_tallies_sorted, axis=0)
    stderr_rews_sorted = np.nanstd(rewardss_tallies_sorted, axis=0)/np.sqrt(n_participants)

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    for i in range(n_participants):
        ax.plot(effs_sorted, rewardss_tallies_sorted[i], color='C%d' %i, alpha=0.5)

    #ax.legend()
    format_axis(ax)

    ax.plot(effs_sorted, mean_rews_sorted, color='black', label='Mean', linewidth=3.5)
    ax.fill_between(effs_sorted, mean_rews_sorted - stderr_rews_sorted, mean_rews_sorted + stderr_rews_sorted, color='black', alpha=0.2)

    ### format axis
    ax.set_xlabel('Efficacy')
    ax.set_ylabel('Rewards')
    #ax.set_title(title)

    plt.tight_layout()

    return fig

# %% 

def plot_n_observes(effs_sorted, n_observes_sorted, ylim=None):

    n_participants = len(n_observes_sorted)

    mean_n_observes_sorted = n_observes_sorted.mean(axis=0)
    stderr_n_observes_sorted = n_observes_sorted.std(axis=0)/np.sqrt(n_participants)

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    for i in range(n_participants):
        ax.plot(effs_sorted, n_observes_sorted[i], color='C%d' %i, alpha=0.5)

    #ax.legend()
    format_axis(ax)

    ax.plot(effs_sorted, mean_n_observes_sorted, color='black', label='Mean', linewidth=3.5)
    ax.fill_between(effs_sorted, mean_n_observes_sorted - stderr_n_observes_sorted, mean_n_observes_sorted + stderr_n_observes_sorted, color='black', alpha=0.2)

    ### format axis
    ax.set_xlabel('Efficacy')
    ax.set_ylabel('Number of Observes')
    
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    return fig

# %% 

def plot_n_actions(effs_sorted, n_actions_sorted, ylim=None, ylabel='Number of Actions'):

    n_participants = len(n_actions_sorted)

    mean_n_observes_sorted = n_actions_sorted.mean(axis=0)
    stderr_n_observes_sorted = n_actions_sorted.std(axis=0)/np.sqrt(n_participants)

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    for i in range(n_participants):
        ax.plot(effs_sorted, n_actions_sorted[i], color='C%d' %i, alpha=0.5)

    #ax.legend()
    format_axis(ax)

    ax.plot(effs_sorted, mean_n_observes_sorted, color='black', label='Mean', linewidth=3.5)
    ax.fill_between(effs_sorted, mean_n_observes_sorted - stderr_n_observes_sorted, mean_n_observes_sorted + stderr_n_observes_sorted, color='black', alpha=0.2)

    ### format axis
    ax.set_xlabel('Efficacy')
    ax.set_ylabel(ylabel)
    
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    return fig


# %% EVIDENCE RATIOS PLOT

def plot_evidence_ratios(evidence_ratios, effs_sorted, cmap, ylabel, xlabel ='Time since last observe', jitter = False, ylim=None,):

    n_participants = len(evidence_ratios)

    mean_sorted = np.nanmean(evidence_ratios,axis=0)
    stderr_sorted = np.nanstd(evidence_ratios,axis=0)/np.sqrt(n_participants)

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    # for i in range(n_participants):
    #     for j, eff in enumerate(effs_sorted):
    #         ax.plot(range(evidence_ratios[i,j]), evidence_ratios[i,j], color=cmap(eff), alpha=0.5)

    #ax.legend()
    format_axis(ax)

    for j, eff in enumerate(effs_sorted):
        ax.plot(range(mean_sorted.shape[1]), mean_sorted[j], color=cmap((effs_sorted[j]+0.2)/1.2), label='Eff ' + str(eff), linewidth=3.5)
        ax.fill_between(range(mean_sorted.shape[1]), mean_sorted[j] - stderr_sorted[j], mean_sorted[j] + stderr_sorted[j], color=cmap((effs_sorted[j]+0.2)/1.2), alpha=0.2)

    if jitter:
        print("Jitter still needs to be implemented")
        # jitter_strength = 0.02
        # # Adding individual data points as a scatter plot
        # x_jitter = np.array(effs_sorted) + np.random.uniform(-jitter, jitter, size=evidence_ratios.shape)

        # ax.scatter(x_jitter, evidence_ratios, color='black', alpha=0.3, s=20)

    ### format axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()
    
    return fig

# %% CREATE PLOT FOR AVERAGE PROBABILITY OF TAKING A SPECIFIC ACTION OVER COURSE OF ONE EPISODE

def plot_prob_action_over_ep(effs_sorted, transitions_rw_sorted, action, cmap, ylabel='Probability of Action', ylim=None):

    n_participants = len(transitions_rw_sorted)

    bool_transitions = transitions_rw_sorted == action

    mean_prob_action = np.mean(bool_transitions,axis=0)
    stderr_prob_action = np.std(bool_transitions, axis=0)/np.sqrt(n_participants)

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    for j, eff in enumerate(effs_sorted):
        ax.plot(range(mean_prob_action.shape[1]), mean_prob_action[j], color=cmap((effs_sorted[j]+0.2)/1.2), label='Eff ' + str(eff), linewidth=3.5)
        ax.fill_between(range(mean_prob_action.shape[1]), mean_prob_action[j] - stderr_prob_action[j], mean_prob_action[j] + stderr_prob_action[j], color=cmap((effs_sorted[j]+0.2)/1.2), alpha=0.2)

    ### format axis
    ax.set_xlabel('Time in episode')
    ax.set_ylabel(ylabel)

    ax.legend()

    if ylim is not None:
        ax.set_ylim(ylim)

    return fig

# %%

def plot_prob_intended_correct_takes(effs_sorted, intended_correct_takes_ep_sorted, ylim=None):

    n_participants = len(intended_correct_takes_ep_sorted)

    mean_ict = np.nanmean(intended_correct_takes_ep_sorted,axis=0)
    stderr_ict = np.nanstd(intended_correct_takes_ep_sorted,axis=0)/np.sqrt(n_participants)

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    for i in range(n_participants):
        ax.plot(effs_sorted, intended_correct_takes_ep_sorted[i], color='C%d' %i, alpha=0.5)

    #ax.legend()
    format_axis(ax)

    ax.plot(effs_sorted, mean_ict, color='black', label='Mean', linewidth=3.5)
    ax.fill_between(effs_sorted, mean_ict - stderr_ict, mean_ict + stderr_ict, color='black', alpha=0.2)

    ### format axis
    ax.set_xlabel('Efficacy')
    ax.set_ylabel('Prob of Intending Correct Bet')

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    return fig



# %% PLOT SCATTER LINE FIT

def plot_scatter_linefit(td_values, beh_train_values, beh_test_values, y_label=None, x_label=None, ylim=None, xlim=None, title=None, group=None):

    def format_with_one_decimal(value, tick_number):
        return f"{value:.1f}"

    # Create a custom formatter
    formatter = FuncFormatter(format_with_one_decimal)

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    plt.scatter(td_values, beh_train_values, color='C8', label='Train', alpha=0.7)
    plt.scatter(td_values, beh_test_values, color='C9', label='Test', alpha=0.7)

    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)
    if xlim is not None:
        plt.xlim(xlim)

    ax.xaxis.set_major_formatter(formatter)

    # format_axis(ax)

    plt.legend()

    if group is not None:
        group_A = np.where(group == False)
        group_B = np.where(group == True)

        # Calculate the line of best fit for group A
        slope_A, intercept_A, r_value_A, p_value_A, std_err_A = stats.linregress(td_values[group_A], beh_train_values[group_A])
        # Calculate the line of best fit for group B
        slope_B, intercept_B, r_value_B, p_value_B, std_err_B = stats.linregress(td_values[group_B], beh_train_values[group_B])

        # Weighted average of slopes and intercepts
        n_A = len(group_A[0])
        n_B = len(group_B[0])
        slope = (slope_A * n_A + slope_B * n_B) / (n_A + n_B)
        intercept = (intercept_A * n_A + intercept_B * n_B) / (n_A + n_B)
    else:
        # Calculate the line of best fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(td_values, beh_train_values)

    # Generate the x-values for the line of best fit
    x = np.linspace(td_values.min(),td_values.max(), 100)
    # Generate the y-values for the line of best fit
    y = slope * x + intercept
    # Plot the line of best fit
    plt.plot(x, y, color='C8' , label=f'Line of best fit', linewidth=4)

    title = "Train " + f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}, p-value={p_value:.3f}' + "\n"
    
    # Calculate the line of best fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(td_values, beh_test_values)
    # Generate the x-values for the line of best fit
    x = np.linspace(td_values.min(),td_values.max(), 100)
    # Generate the y-values for the line of best fit
    y = slope * x + intercept
    # Plot the line of best fit
    plt.plot(x, y, color='C9', label=f'Line of best fit', linewidth=4)

    title += "Test " + f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}, p-value={p_value:.3f}'
    
    plt.title(title)

    return fig

# %% GET PARTICIPANT SCORES

def convert_to_continuum_representation(traj, includes_sleep=False):
    traj = np.array(traj)
    traj = np.where(traj == 2, 0.5, traj)

    if includes_sleep:
        traj = np.where(traj == 3, -1, traj)
    return traj

def convert_to_right_wrong_representation(traj):
    new_traj = []
    for transition in traj:
        if transition[1] == 0.5:
            new_traj.append(0.5)
        elif transition[1] == -1:
            new_traj.append(-1)
        elif transition[2] == transition[0]:
            new_traj.append(1)
        else:
            new_traj.append(0)
    return new_traj

def extract_participant_trajs (participant_data, includes_sleep=False):
    ''' extracts trajectories and rewards_tallies from a participant's data frame '''
        
    transitions_ep = participant_data['transitions_ep'].dropna()
    rewards_tallies = participant_data['rewards_tally'].dropna()
    #transitions_ep = np.array(transitions_ep)
    #transitions_ep = ast.literal_eval(transitions_ep)
    transitions_ep = transitions_ep.apply(ast.literal_eval)
    #transitions_ep = np.array(transitions_ep

    transitions_ep = transitions_ep.apply(convert_to_continuum_representation, args=(includes_sleep,))

    transitions_ep_rightwrong = transitions_ep.apply(convert_to_right_wrong_representation)

    transitions_ep = transitions_ep.apply(np.array)

    transitions_ep = np.stack(transitions_ep.values)

    transitions_ep_rightwrong = transitions_ep_rightwrong.apply(np.array)
    transitions_ep_rightwrong = np.stack(transitions_ep_rightwrong)

    ps = participant_data['ps_ep'].dropna().apply(ast.literal_eval).apply(np.array).values
    ps = np.stack(ps)
    ps = ps.reshape((-1,50,2))

    n_observes = np.where(transitions_ep_rightwrong == 0.5, 1, 0).sum(axis=1)

    if 'response' in participant_data.columns:
        responses = participant_data['response'].dropna().values
        vlen = np.vectorize(len)
        mask = vlen(responses) >= 2
        survey_responses = responses[mask]
    else:
        survey_responses = []

    correct_choice = np.argmax(ps, axis=-1, )
    intended_choice = transitions_ep[:,:,1]
    intended_correct_takes = np.where(correct_choice == intended_choice, 1, np.where(intended_choice == 0.5, np.nan, 0))
    intended_correct_takes_ep = np.nanmean(intended_correct_takes, axis=-1)

    return transitions_ep, transitions_ep_rightwrong, rewards_tallies.values, ps, participant_data['eff_ep'].dropna().values, n_observes, intended_correct_takes_ep, survey_responses

def extract_participant_trajs_task2(participant_data):
    transitions_ep, transitions_ep_rightwrong, rewards_tallies, ps, peffs, n_observes, intended_correct_takes_ep, survey_responses = extract_participant_trajs(participant_data, includes_sleep=True)

    n_sleeps = np.where(transitions_ep_rightwrong == -1, 1, 0).sum(axis=1)

    return transitions_ep, transitions_ep_rightwrong, rewards_tallies, ps, peffs, n_observes, intended_correct_takes_ep, survey_responses, n_sleeps


def extract_participant_trajs_slider(participant_data):
    transitions_ep, transitions_ep_rightwrong, rewards_tallies, ps, peffs, n_observes, intended_correct_takes_ep, survey_responses = extract_participant_trajs(participant_data, includes_sleep=True)

    efficacy_estimates = participant_data['efficacy_estimate'].dropna().values

    ### PREPROCESS VALUES TO BE ON THE USUAL SCALE

    efficacy_estimates = (efficacy_estimates - 50) * 2 / 100

    return transitions_ep, transitions_ep_rightwrong, rewards_tallies, ps, peffs, n_observes, intended_correct_takes_ep, survey_responses, efficacy_estimates


def combine_partial_datas (file_base, data_folder):
    #folder_path = os.path.join('data', day_string, 'data')
    #folder_path = os.path.join('data', day_string)
    folder_path = data_folder

    participant_partials = []

    participant_files = glob.glob(folder_path + "/*_" + file_base + "_*")
    participant_files.sort()

    for file_path in participant_files:
        if 'params' not in file_path and 'survey' not in file_path:
            participant_partials.append(pd.read_csv(file_path))

    participant_data = pd.concat(participant_partials)

    return participant_data, participant_partials

def calculate_bonus_payouts(datass):
    for turker, d in datass.items():
        total_rewards_tally = d['rewards_tally'].sum()

        print(turker, ',', '%.2f' %(total_rewards_tally/100))

def print_turkers(datass):
    for turker, d in datass.items():
        #turker = d['turker'].dropna().values[0]

        print(turker, ',')

# def print_group_turkers(datass, group):
#     for turker, d in datass.items():
#         #turker = d['turker'].dropna().values[0]

#         if datass[turker]['group'][0] == group:
#             print(turker, ',')

# %%

def get_participant_data(pid, data_folder, n_episodes, includes_sleep = False):
    ''' Returns the data associated with a participant found in data_folder, preferentially sampling from complete dataset but alternately from partial one 
    
    Arguments
    ---------
    pid : str, prolific ID
    data_folder : str, data folder
    n_episodes : int, number of episodes for it to be counted complete

    Returns
    -------
    participant_data :  pd.DataFrame containing the raw data
    is_complete : bool, True if we have data on n_episodes different episodes for this participant
    
    '''

    folder_path = data_folder
    participant_files = glob.glob(folder_path + "/*_" + pid + "_*")
    if len(participant_files) > 0:
        participant_files.sort()
        #print(participant_files)
        last_participant_file = participant_files[-1]
        is_complete = None

        if '_survey' in last_participant_file:
            last_participant_file = participant_files[-2]
        elif includes_sleep:
            ## if we are on day3, then we need a survey file, else the data is not complete
            is_complete = False

        if '_ep' in last_participant_file:
            pdata, partials = combine_partial_datas(pid, data_folder)
            if(len(partials) >= n_episodes):
                if(len(partials) == n_episodes) and is_complete is None:
                    is_complete = True
                else:
                    #assert False, 'unexpected number of episodes for ' + pid
                    is_complete = False
                    print('too many episodes for particiapnt', pid, 'marking as incomplete')
            else:
                is_complete = False

        elif '_params' not in last_participant_file:
            pdata = pd.read_csv(os.path.join(last_participant_file))

            is_complete=True
        else:
            pdata = None
            is_complete = False
    
    else:
        pdata = None
        is_complete = False

    return pdata, is_complete

def get_p_datass(participant_ids, data_folder, n_episodes, includes_sleep=False, contains_slider = False, contains_group = False):
    ''' Gets dataframes associated participant_ids
    
    Arguments
    ---------
    participant_ids : list of participant turker ids
    data_folder : str, data folder where to look for data
    
    '''
    
    columns = [ 'transitions_ep', 'transitions_ep_rightwrong', 'rewards_tallies', 'ps', 'effs', 'n_observes', 'intended_correct', 'survey_responses']

    if includes_sleep:
        columns += ['n_sleeps']

    if contains_slider:
        columns += ['efficacy_estimates']

    if contains_group:
        columns += ['group']

    complete_pids = []
    datas = {}
    trajs = []

    ## exclude participants that don't have survey files

    for participant_id in participant_ids:
        #print('processing', participant_id)
        pdata, is_complete = get_participant_data(participant_id, data_folder, n_episodes = n_episodes, includes_sleep=includes_sleep)
        #print('got participant data')

        #print(pdata)

        if pdata is not None:
            datas[participant_id] = pdata
        else:
            print("no data", participant_id)
        if is_complete:
            complete_pids.append(participant_id)
        
            if not includes_sleep:

                if not contains_slider:
                    traj = extract_participant_trajs(pdata)
                else:
                    traj = extract_participant_trajs_slider(pdata)
            else:
                traj = extract_participant_trajs_task2(pdata)

            if contains_group:
                traj += (pdata['group'].dropna().values,)

            trajs.append( {columns[i] : traj[i] for i, _ in enumerate(traj)})
        else:
            print("incomplete data for ", participant_id)

    df = pd.DataFrame.from_records(trajs, index = complete_pids)

    if contains_group:
        df['group'] = df['group'].apply(lambda x : x[0])

    return datas, complete_pids, df

def sort_overall(metric, sort_by):
    '''' Sorts a metric in same order as sort_by (i.e. same order that puts sort_by in increasing order)
    
    Arguments
    ---------
    metric : iterable [n_participants,] of arrays [n_episodes,] containing metric for each participant
    sort_by : iterable [n_participants] of arrays [n_episodes,] containing metric to sort by
    
    Returns
    -------
    sorted : np.array [n_participants, n_episodes], sorted array

    '''

    if (len(sort_by) == len(metric)):
        sort_by = np.stack(sort_by)
    else: #assume that it needs to be broadcast
        sort_by = np.expand_dims(sort_by, 0).repeat(27, axis=0)

    order = np.argsort(sort_by, axis=1)
    sorted = np.array([row[order[i]] for i, row in enumerate(metric)])

    return sorted

def sort_train_test(metric, sort_by, test_start, group = None):
    '''' Sorts a metric in same order as sort_by (i.e. same order that puts sort_by in increasing order)
    
    Arguments
    ---------
    metric : iterable [n_participants,] of arrays [n_episodes,] containing metric for each participant
    sort_by : iterable [n_participants] of arrays [n_episodes,] containing metric to sort by
    
    Returns
    -------
    sorted_train : np.array [n_participants, test_start]
    sorted_test : np.array [n_participants, n_episodes - test_start]
    
    '''

    if group is None:
        sort_by = np.stack(sort_by)
        metric = np.stack(metric)
        order_train = np.argsort(sort_by[:,:test_start], axis=1)
        order_test = np.argsort(sort_by[:,test_start:], axis=1)

        metric_train = metric[:,:test_start]
        metric_test = metric[:,test_start:]

        sorted_train = np.array([row[order_train[i]] for i, row in enumerate(metric_train)])
        sorted_test = np.array([row[order_test[i]] for i, row in enumerate(metric_test)])

        return sorted_train, sorted_test
    
    else:
        sorted_train_groupA, sorted_test_groupA = sort_train_test(metric[~group], sort_by[~group], test_start[0])
        sorted_train_groupB, sorted_test_groupB = sort_train_test(metric[group], sort_by[group], test_start[1])

        return (sorted_train_groupA, sorted_train_groupB), (sorted_test_groupA, sorted_test_groupB)


# %% NN UTILS

def load_config_files(modelname, base_model_folder):

    model_folder = os.path.join(base_model_folder, str(modelname))

    config = Config({})
    config.load_config_file(os.path.join(model_folder, 'config.yaml'))

    task_options = Config({})
    task_options.load_config_file(os.path.join(model_folder, 'task_options.yaml'))

    nn_options = Config({})
    nn_options.load_config_file(os.path.join(model_folder, 'nn_options.yaml'))

    return config, task_options, nn_options

# %%

def get_max_ll(df, models, test_start, aggregate_efficacies = True):
    lls_train = []
    lls_test = []
    for model in models:
        ll_train, ll_test = sort_train_test(df['ll_' + str(model)].values, df['effs'].values, test_start)

        if aggregate_efficacies:
            ll_train = ll_train.sum(axis=1)
            ll_test = ll_test.sum(axis=1)

        lls_train.append(ll_train)
        lls_test.append(ll_test)        

    lls_train = np.stack(lls_train)
    lls_test = np.stack(lls_test)

    return np.max(lls_train, axis=0), np.max(lls_test, axis=0)

# %%

def get_mean_ll(df, models, test_start, prefix = 'll', aggregate_efficacies = True):
    ''' Returns mean log likelihood of models for each participant
    
    Arguments
    ---------
    df : dataframe, containing log likelihoods for each model
    models : list of model names
    test_start : int, episode number where test starts
    prefix : str, prefix of log likelihood columns
    aggregate_efficacies: bool, if True, aggregate over efficacy settings, i.e. return joint probability across all efficacy settings

    Returns
    -------
    lls_train : list [n_participants,] or list of lists [n_participants, n_train_settings], mean log likelihood for each participant on training data
    lls_test : list [n_participants,] or list of lists [n_participants, n_test_settings], mean log likelihood for each participant on test data

    '''
    
    lls_train = []
    lls_test = []
    for model in models:
        ll_train, ll_test = sort_train_test(df[prefix + '_' + str(model)].values, df['effs'].values, test_start)

        if aggregate_efficacies:
            ll_train = ll_train.sum(axis=1)
            ll_test = ll_test.sum(axis=1)

        lls_train.append(ll_train)
        lls_test.append(ll_test)        

    lls_train = np.stack(lls_train)
    lls_test = np.stack(lls_test)

    lls_train = np.mean(lls_train, axis=0)
    lls_test = np.mean(lls_test, axis=0)

    return lls_train.tolist(), lls_test.tolist()

# %%

def calculate_freq_observed_choice_per_t(row, n_steps):
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

# %%

def calculate_nns_freq_observed_choice_per_t(row, n_steps, ape_models, include_sleep=False):

    probss = []

    for i, ep in enumerate(row['transitions_ep']):
        probs = [[] for _ in range(n_steps)]
        
        counter_since_observe = 0
        current_choice = None

        # Iterate over the sublists in each row
        for j, step in enumerate(ep):
            # Check if intended choice is "observed"
            if step[1] == 0.5:
                counter_since_observe = 0
                current_choice = step[0]
            else:

                if current_choice is not None:
                    for modelname in ape_models:
                        print(current_choice, counter_since_observe)
                        probs[counter_since_observe].append(row['step_l_' + str(modelname)][i][j][1 + int(include_sleep) + int(current_choice)])

                counter_since_observe += 1

        # Calculate fraction
        probs = [np.mean(ps) if ps != [] else np.nan for ps in probs]
        probss.append(probs)
        
    return probss

# %%

def calculate_freq_observes_per_t(row, n_steps):
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


# %%

def calculate_nns_freq_observes_per_t(row, n_steps, ape_models):
    """
    Calculate the frequency of observed choices per time step for each episode in a given row in a dataset.

    Args:
        row (pd.Series): A row of data containing the transitions.
        n_steps (int): The number of time steps.
        ape_models (list): A list of APE model names.

    Returns:
        list: A list of lists representing the frequency of observed choices per time step for each row.

    Raises:
        None
    """

    probss = []

    for i, ep in enumerate(row['transitions_ep']):
        probs = [[] for _ in range(n_steps)]

        counter_since_observe = 0
        current_choice = None

        # Iterate over the sublists in each row
        for j, step in enumerate(ep):
            # Check if intended choice is "observed"
            if step[1] == 0.5:
                counter_since_observe = 0
                current_choice = step[0]
            else:

                if current_choice is not None:
                    for modelname in ape_models:
                        print(current_choice, counter_since_observe)
                        probs[counter_since_observe].append(row[f'step_l_{str(modelname)}'][i][j][0])

                counter_since_observe += 1

        # Calculate fraction
        probs = [np.mean(ps) if ps != [] else np.nan for ps in probs]
        probss.append(probs)

    return probss

def calculate_freq_correct_choice_per_t(row, n_steps):
    """
    Calculate the fraction of correct choices at the time they observed (0.5).

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

def calculate_freq_sleeps(row, n_steps):
    '''
    Calculates frequency of sleep actions since the begininng of the episode
    
    '''

    sleepss = []

    for ep in row:

        sleeps = np.where(np.array(ep)[:,1] == -1, 1, 0)
    
        sleepss.append(sleeps)

    return sleepss

# %% 

def calculate_nns_freq_sleeps_per_t(row, n_steps, ape_models):
    """
    Calculate the frequency of observed choices per time step for each episode in a given row in a dataset.

    Args:
        row (pd.Series): A row of data containing the transitions.
        n_steps (int): The number of time steps.
        ape_models (list): A list of APE model names.

    Returns:
        list: A list of lists representing the frequency of observed choices per time step for each row.

    Raises:
        None
    """

    probss = []

    for i, ep in enumerate(row['transitions_ep']):
        probs = [[] for _ in range(n_steps)]

        counter_since_observe = 0
        current_choice = None

        # Iterate over the sublists in each row
        for j, step in enumerate(ep):
            # Check if intended choice is "observed"
            for modelname in ape_models:
                probs[j].append(row[f'step_l_{str(modelname)}'][i][j][1])
        
        # Calculate fraction
        probs = [np.mean(ps) if ps != [] else np.nan for ps in probs]
        probss.append(probs)

    return probss

# %%

def plot_train_test_td(df, metric_name, y_label='Deviation from Mean Observes per Efficacy', ylim=None, group=None):
    
    def format_with_one_decimal(value, tick_number):
        return f"{value:.1f}"

    # Create a custom formatter
    formatter = FuncFormatter(format_with_one_decimal)

    # Create scatter plots
    fig = plt.figure(figsize=(15, 5))

    tdlabels = ['Anxiety-Depression Score','Compulsivity','Social Withdrawal']

    for i, tdname in enumerate(['AD', 'Compul', 'SW']):

        print(tdname)

        plt.subplot(1,3,i+1)
        plt.scatter(df[tdname], df[metric_name + '_train'], color='C8', label='Train', alpha=0.7)
        plt.scatter(df[tdname], df[metric_name + '_test'], color='C9', label='Test', alpha=0.7)
        plt.ylabel(y_label)
        plt.xlabel(tdlabels[i])

        plt.gca().xaxis.set_major_formatter(formatter)

        if ylim is not None:
            plt.ylim(ylim)

        format_axis(plt.gca())

        plt.legend()

        if group is not None:
            group_A = np.where(group == False)
            group_B = np.where(group == True)

            # Calculate the line of best fit for group A
            slope_A, intercept_A, r_value_A, p_value_A, std_err_A = stats.linregress(df[tdname][group_A], df[metric_name + '_train'][group_A])
            # Calculate the line of best fit for group B
            slope_B, intercept_B, r_value_B, p_value_B, std_err_B = stats.linregress(df[tdname][group_B], df[metric_name + '_train'][group_B])

            # Weighted average of slopes and intercepts
            n_A = len(group_A[0])
            n_B = len(group_B[0])
            slope = (slope_A * n_A + slope_B * n_B) / (n_A + n_B)
            intercept = (intercept_A * n_A + intercept_B * n_B) / (n_A + n_B)
        else:
            # Calculate the line of best fit
            slope, intercept, r_value, p_value, std_err = stats.linregress(df[tdname], df[metric_name + '_train'])

        # Generate the x-values for the line of best fit
        x = np.linspace(df[tdname].min(),df[tdname].max(), 100)
        # Generate the y-values for the line of best fit
        y = slope * x + intercept
        # Plot the line of best fit
        plt.plot(x, y, color='C8' , label=f'Line of best fit', linewidth=4)

        print("Train", f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}, p-value={p_value:.3f}')        
        # Calculate the line of best fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[tdname], df[metric_name + '_test'])
        # Generate the x-values for the line of best fit
        x = np.linspace(df[tdname].min(),df[tdname].max(), 100)
        # Generate the y-values for the line of best fit
        y = slope * x + intercept
        # Plot the line of best fit
        plt.plot(x, y, color='C9', label=f'Line of best fit', linewidth=4)

        print("Test", f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}, p-value={p_value:.3f}')
        plt.tight_layout()

    return fig

# %%

def plot_single_train_test_td(df, tdname, metric_name, y_label, x_label, ylim=None, train_test_td = False):

    def format_with_one_decimal(value, tick_number):
        return f"{value:.1f}"

    # Create a custom formatter
    formatter = FuncFormatter(format_with_one_decimal)

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
        
    if not train_test_td:
        plt.scatter(df[tdname], df[metric_name + '_train'], color='C8', label='Train', alpha=0.7)
        plt.scatter(df[tdname], df[metric_name + '_test'], color='C9', label='Test', alpha=0.7)
    else:
        plt.scatter(df[tdname + '_train'], df[metric_name + '_train'], color='C8', label='Train', alpha=0.7)
        plt.scatter(df[tdname + '_test'], df[metric_name + '_test'], color='C9', label='Test', alpha=0.7)

    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.gca().xaxis.set_major_formatter(formatter)

    if ylim is not None:
        plt.ylim(ylim)

    format_axis(plt.gca())

    plt.legend()

    # Calculate the line of best fit

    if not train_test_td:

        slope, intercept, r_value, p_value, std_err = stats.linregress(df[tdname], df[metric_name + '_train'])
        # Generate the x-values for the line of best fit
        x = np.linspace(df[tdname].min(),df[tdname].max(), 100)
        # Generate the y-values for the line of best fit
        y = slope * x + intercept
        # Plot the line of best fit
        #plt.plot(x, y, color='C8' label=f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}')

        plt.plot(x, y, color='C8' , label=f'Line of best fit', linewidth=4)

        print("Train", f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}, p-value={p_value:.3f}')        
        # Calculate the line of best fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[tdname], df[metric_name + '_test'])
        # Generate the x-values for the line of best fit
        x = np.linspace(df[tdname].min(),df[tdname].max(), 100)
        # Generate the y-values for the line of best fit
        y = slope * x + intercept
        # Plot the line of best fit
        #plt.plot(x, y, color='C8' label=f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}')

        plt.plot(x, y, color='C9', label=f'Line of best fit', linewidth=4)

        print("Test", f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}, p-value={p_value:.3f}')

    else:
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[tdname + '_train'], df[metric_name + '_train'])
        # Generate the x-values for the line of best fit
        x = np.linspace(df[tdname + '_train'].min(),df[tdname  + '_train'].max(), 100)
        # Generate the y-values for the line of best fit
        y = slope * x + intercept
        # Plot the line of best fit
        #plt.plot(x, y, color='C8' label=f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}')

        plt.plot(x, y, color='C8' , label=f'Line of best fit', linewidth=4)

        print("Train", f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}, p-value={p_value:.3f}')        
        # Calculate the line of best fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[tdname + '_test'], df[metric_name + '_test'])
        # Generate the x-values for the line of best fit
        x = np.linspace(df[tdname + '_test'].min(),df[tdname + '_test'].max(), 100)
        # Generate the y-values for the line of best fit
        y = slope * x + intercept
        # Plot the line of best fit
        #plt.plot(x, y, color='C8' label=f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}')

        plt.plot(x, y, color='C9', label=f'Line of best fit', linewidth=4)

        print("Test", f'Line of best: y={slope:.2f}x+{intercept:.2f}, $r^2$={r_value**2:.2f}, p-value={p_value:.3f}')
        
    plt.tight_layout()
    
    format_axis(ax)

    return fig
# %%

def get_clean_data(day = 1, exp_date = '12-11', day1_mask_cutoff = None, day1_test_mask_cutoff = None, group = None, file_base = '', exclude_based_on_day3 = True, fascimile=True):
    ''' This function loads the clean dataset for the specified day.
    
    Parameters
    ----------
    day : int
        The day for which the data should be loaded. Valid values are 1, 2, and 3.
    exp_date : str
        The date of the experiment.
    day1_mask_cutoff : int
        The cutoff (based on number of observes) for the mask for day 1.
    day1_test_mask_cutoff : int
        The cutoff (based on number of observes on the test set) for the test mask for day 1.
    group : str
        The group for which the data should be loaded. Valid values are None (for both), 'groupA' and 'groupB'.
    
    Returns
    -------
    df : pandas.DataFrame
        The clean dataset for the specified day.
    effs_train : list
        The list of training efficacy values.
    effs_test : list
        The list of test efficacy values.
    test_start : int
        The index of the first test efficacy value.
    '''
    
    if exp_date == '24-01-22-29':
        files_day1 = [
            'data/behavior_24-01-22_day1.pkl',
            'data/behavior_24-01-29_day1.pkl',
        ]

        files_day2 = [
            'data/behavior_24-01-22_day2.pkl',
            'data/behavior_24-01-22_day2B.pkl',            
            'data/behavior_24-01-29_day2.pkl',
            'data/behavior_24-01-29_day2B.pkl',
        ]

        files_day3 = [
            'data/behavior_24-01-22_day3.pkl',
            'data/behavior_24-01-22_day3B.pkl',
            'data/behavior_24-01-29_day3.pkl',
            'data/behavior_24-01-29_day3B.pkl',
        ]

    else:
        assert False, 'Invalid experiment date specified.'

    df_day1 = pd.DataFrame()
    df_day2 = pd.DataFrame()
    df_day3 = pd.DataFrame()

    for file in files_day1:
        df_day1 = pd.concat([df_day1, pd.read_pickle(os.path.join(file_base, file))])

    for file in files_day2:
        df_day2 = pd.concat([df_day2, pd.read_pickle(os.path.join(file_base, file))])

    for file in files_day3:
        df_day3 = pd.concat([df_day3, pd.read_pickle(os.path.join(file_base, file))])

    ### ADD GROUP COLUMN IF NOT ALREADY INCLUDED (i.e. for old data)
    if 'group' not in df_day1.columns:
        df_day1['group'] = False
    if 'group' not in df_day2.columns:
        df_day2['group'] = False
    if 'group' not in df_day3.columns:
        df_day3['group'] = False

    ## PERFORM CUTOFF BASED ON MASKS
    # if day1_mask_cutoff is not None:
    #     df_day1 = df_day1[df_day1['n_observes'].apply(lambda x : np.sum(x)) > day1_mask_cutoff]

    if day1_test_mask_cutoff is not None and type(day1_test_mask_cutoff) == int:
        n_obs_train, n_obs_test = sort_train_test(df_day1['n_observes'].values, df_day1['effs'].values, 5 if group == "groupA" else 4)
        df_day1 = df_day1[n_obs_test.sum(axis=1) > day1_test_mask_cutoff]
    elif day1_test_mask_cutoff is not None and type(day1_test_mask_cutoff) == dict:
        for g in ["groupA", "groupB"] if group is None else [group]:
            n_obs_train, n_obs_test = sort_train_test(df_day1['n_observes'].values, df_day1['effs'].values, 5 if g == "groupA" else 4)
            #df_day1 = df_day1[((np.logical_not(np.logical_xor(group == g, df_day1["group"]))) & (n_obs_test.sum(axis=1) > day1_test_mask_cutoff[g]['lower'])) | (np.logical_xor(group != g, df_day1["group"])) ]
            df_day1 = df_day1[((df_day1["group"] == (g == "groupB") ) & (n_obs_test.sum(axis=1) >= day1_test_mask_cutoff[g]['lower'])) | (df_day1["group"] != (g == "groupB") ) ]
            n_obs_train, n_obs_test = sort_train_test(df_day1['n_observes'].values, df_day1['effs'].values, 5 if g == "groupA" else 4)
            df_day1 = df_day1[((df_day1["group"] == (g == "groupB") ) & (n_obs_test.sum(axis=1) <= day1_test_mask_cutoff[g]['upper'])) | (df_day1["group"] != (g == "groupB") ) ]
            #df_day1 = df_day1[((np.logical_not(np.logical_xor(group == g, df_day1["group"]))) & (n_obs_test.sum(axis=1) < day1_test_mask_cutoff[g]['upper'])) | (np.logical_xor(group != g, df_day1["group"])) ]

    ### read in and concatenate pandas dataframes stored in above files
        
    #df_day1, df_day2 = df_day1.align(df_day2, join='inner')

    # Align columns with 'outer' join
    df_day1_aligned, df_day2_aligned = df_day1.align(df_day2, axis=1, join='outer')

    # Align rows with 'inner' join
    df_day1, df_day2 = df_day1_aligned.align(df_day2_aligned, axis=0, join='inner')

    ## keep those rows of df_day3 that are in df_day1 and df_day2
    ## align day 3 with day 1 and day 2 (keep all rows for days 1 and 2)

    df_day3 = df_day3[df_day3.index.isin(df_day1.index)]
    common_indices = df_day2.index.intersection(df_day3.index)
    df_day3 = df_day3.reindex(common_indices)
    
    ## drop those participants from days 1 and 2 that are not in day 3
    if exclude_based_on_day3:
        df_day1 = df_day1[df_day1.index.isin(df_day3.index)]
        df_day2 = df_day2[df_day2.index.isin(df_day3.index)]

    if day == 1:
        df = df_day1
        effs_test = [0.125, 0.375, 0.5, 0.625, 0.875]
        effs_train = [0, 0.25,   0.75,  1.0]
        
    elif day == 2:
        df = df_day2
        effs_train = [0.125, 0.375, 0.5, 0.625, 0.875]
        effs_test = [0, 0.25,   0.75,  1.0]
    
    elif day == 3:
        df = df_day3
        effs_train = [0.125, 0.375, 0.5, 0.625, 0.875]
        effs_test = [0, 0.25,   0.75,  1.0]

    else:
        assert False, 'Invalid day specified.'

    if 'group' not in df.columns:
        test_start = len(effs_train)
    elif group == 'groupA':
        df = df[~df['group']]
        test_start = len(effs_train)
    elif group == 'groupB':
        df = df[df['group']]

        ## switch effs_train and effs_test assignment
        effs_train, effs_test = effs_test, effs_train
        test_start = len(effs_train)
    elif group is None:
        effs_train, effs_test = ((effs_train, effs_test), (effs_test, effs_train))
        test_start = (len(effs_train[0]), len(effs_train[1]))
    else:
        assert False, 'Invalid group specified.'

    return df, effs_train, effs_test, test_start

# %%

def get_effs(group, day):
    ''' This function returns the efficacy values for the specified group and day.
    
    Parameters
    ----------
    group : str
        The group for which the efficacy values should be returned. Valid values are 'groupA' and 'groupB'.
    day : int
        The day for which the efficacy values should be returned. Valid values are 1, 2, and 3.
    
    Returns
    -------
    effs : list
        The efficacy values for the specified group and day.
    '''
    
    if (group == 'groupA' and (day == 2 or day == 3)) or (group == 'groupB' and day == 1):
        effs_train = [0.125, 0.375, 0.5, 0.625, 0.875]
        effs_test = [0, 0.25,   0.75,  1.0]
    else:
        effs_test = [0.125, 0.375, 0.5, 0.625, 0.875]
        effs_train = [0, 0.25,   0.75,  1.0]
    
    return effs_train, effs_test