# Kai Sandbrink
# 2023-04-27
# This script analyzes Prolific Participants with different efficacy using neural networks

# %% LIBRARY IMPORTS

import numpy as np
import pandas as pd
import os, ast, glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import torch
from ObserveBetEfficacyTask import ObserveBetEfficacyTask
from PeekTakeTorchRNN import PeekTakeTorchAPERNN, PeekTakeTorchRNN
from torch.distributions import Categorical

from utils import format_axis, get_timestamp
from human_utils_project import plot_reward_tally, plot_n_observes, plot_prob_intended_correct_takes, plot_train_test_comp, plot_evidence_ratios, plot_prob_action_over_ep
from human_utils_project import extract_participant_trajs, combine_partial_datas, print_turkers, calculate_bonus_payouts, get_effs
from human_utils_project import get_participant_data, get_p_datass, sort_overall, sort_train_test, load_config_files, get_max_ll, get_mean_ll
from utils_nn import load_env_and_model, convert_action_representation, simulate_step_in_env, get_likelihood_trajectory, get_logits_trajectory

# %% PARAMETERS

##### DICT WITH DIFFERENT RUNS

experiment_params = [
    {
        'df_file': 'results/behavior/20240129124114_behavior_diff_effs_24-01-22_day1.pkl',
        'exp_date': '24-01-22',
        'day': 'day1',
        'include_sleep': False
    },
    {
        'df_file': 'results/behavior/20240129130556_behavior_diff_effs_24-01-22_day2.pkl',
        'exp_date': '24-01-22',
        'day': 'day2',
        'include_sleep': False
    },
    {
        'df_file': 'results/behavior/20240129132922_behavior_diff_effs_24-01-22_day3.pkl',
        'exp_date': '24-01-22',
        'day': 'day3',
        'include_sleep': True,
    },
    {
        'df_file': 'results/behavior/20240129131053_behavior_diff_effs_24-01-22_day2B.pkl',
        'exp_date': '24-01-22',
        'day': 'day2B',
        'include_sleep': False,
    },
    {
        'df_file': 'results/behavior/20240129133603_behavior_diff_effs_24-01-22_day3B.pkl',
        'exp_date': '24-01-22',
        'day': 'day3B',
        'include_sleep': True,
    },
    {
        'df_file': 'results/behavior/20240205190303_behavior_diff_effs_24-01-29_day1.pkl',
        'exp_date': '24-01-29',
        'day': 'day1',
        'include_sleep': False,
    },
    {
        'df_file': 'results/behavior/20240205190355_behavior_diff_effs_24-01-29_day2.pkl',
        'exp_date': '24-01-29',
        'day': 'day2',
        'include_sleep': False,
    },
    {
        'df_file': 'results/behavior/20240205190457_behavior_diff_effs_24-01-29_day3.pkl',
        'exp_date': '24-01-29',
        'day': 'day3',
        'include_sleep': True,
    },
    {
        'df_file': 'results/behavior/20240205190422_behavior_diff_effs_24-01-29_day2B.pkl',
        'exp_date': '24-01-29',
        'day': 'day2B',
        'include_sleep': False,
    },
    {
        'df_file': 'results/behavior/20240205190527_behavior_diff_effs_24-01-29_day3B.pkl',
        'exp_date': '24-01-29',
        'day': 'day3B',
        'include_sleep': True,
    }
]


#### GENERAL PARAMETERS
notes = ''
cmap_train = mpl.colormaps['Greens']
cmap_test = mpl.colormaps['Blues']

cmaps = {
    'train': cmap_train,
    'test': cmap_test,
}

n_steps = 50
#model_folder = 'models/'
model_folder = '/home/kai/Documents/Projects/meta-peek-take/models'
model_file = 'model.pt'
device = 'cpu'
effs_sorted = np.arange(0,1.01, 0.125)

# %% ANALYSIS FUNCTIONS

def get_log_likelihood_trajectory(trajectory, model, env, n_steps_to_reward=50, device='cpu'):
    ''' Computes the log likelihood of a given trajectory in a model'''

    env.start_new_episode()
    lstm_hidden = None
    state = env.get_state()

    saved_logits = []

    for j in range(n_steps_to_reward):

        logits, lstm_hidden, value, control = model(torch.tensor(state).to(device).float(), lstm_hidden)

        correct = int(trajectory[j][0])
        selected_action = torch.tensor(convert_action_representation(trajectory[j][1], include_sleep)).to(device)
        action = torch.tensor(convert_action_representation(trajectory[j][2], include_sleep)).to(device)

        sampler = Categorical(logits=logits)
        saved_logits.append(sampler.log_prob(selected_action).item())

        state = simulate_step_in_env(env, correct, action, selected_action)

    return np.array(saved_logits).sum()

def get_step_likelihood_trajectory(trajectory, model, env, n_steps_to_reward=50, device='cpu'):
    ''' Computes the likelihoods of THE SELECTED actions along a given trajectory in a model'''

    env.start_new_episode()
    lstm_hidden = None
    state = env.get_state()

    probs = []

    for j in range(n_steps_to_reward):

        logits, lstm_hidden, value, control = model(torch.tensor(state).to(device).float(), lstm_hidden)

        correct = int(trajectory[j][0])
        selected_action = torch.tensor(convert_action_representation(trajectory[j][1], include_sleep)).to(device)
        action = torch.tensor(convert_action_representation(trajectory[j][2], include_sleep)).to(device)

        sampler = Categorical(logits=logits)
        #probs.append(sampler.probs.detach().numpy())
        probs.append(sampler.probs.detach().numpy()[selected_action.item()])

        state = simulate_step_in_env(env, correct, action, selected_action)

    return np.stack(probs)

def get_observe_deviation_likelihood_trajectories(row, modelname, model_folder, device='cpu'):
    ''' Computes the deviation in observe actions from that given by the model 

    Arguments
    ---------
    row : pandas.Series containing the trajectories of a single participant

    Returns
    -------
    observe_deviation_lik : list of floats
    
    '''

    env, model = load_env_and_model(modelname, model_folder, device=device)

    observe_deviation_lik = []

    for trajectory in row['transitions_ep']:
        _, odl, _ = get_likelihood_trajectory(trajectory, model, env)
        observe_deviation_lik.append(odl)

    return observe_deviation_lik

def get_mean_likelihood_trajectories(row, modelname, model_folder, device='cpu'):
    ''' Computes the deviation in observe actions from that given by the model 

    Arguments
    ---------
    row : pandas.Series containing the trajectories of a single participant

    Returns
    -------
    observe_deviation_lik : list of floats
    
    '''

    env, model = load_env_and_model(modelname, model_folder, device=device)

    mls = []

    for trajectory in row['transitions_ep']:
        ml, odl, _ = get_likelihood_trajectory(trajectory, model, env)
        mls.append(ml)

    return mls

def get_logits_trajectories(row, modelname, model_folder, device='cpu'):

    env, model = load_env_and_model(modelname, model_folder, device=device)

    logitss, controlss = [], []

    for trajectory in row['transitions_ep']:
        l, c = get_logits_trajectory(trajectory, model, env)
        logitss.append(l)
        controlss.append(c)
    
    return logitss, controlss

def get_sleep_deviation_likelihood_trajectories(row, modelname, model_folder, device='cpu'):
    ''' Computes the deviation in observe actions from that given by the model 

    Arguments
    ---------
    row : pandas.Series containing the trajectories of a single participant

    Returns
    -------
    observe_deviation_lik : list of floats
    
    '''

    env, model = load_env_and_model(modelname, model_folder, device=device)

    deviation_lik = []

    for trajectory in row['transitions_ep']:
        _, _, sdl = get_likelihood_trajectory(trajectory, model, env, include_sleep=True)
        deviation_lik.append(sdl)

    return deviation_lik

def get_log_likelihood_trajectories(row, modelname, model_folder, device):

    env, model = load_env_and_model(modelname, model_folder, device=device)

    taken_logits = []

    for trajectory in row['transitions_ep']:
        tl = get_log_likelihood_trajectory(trajectory, model, env)
        taken_logits.append(tl)

    return taken_logits

def get_step_likelihood_trajectories(row, modelname, model_folder, device):

    env, model = load_env_and_model(modelname, model_folder, device=device)

    probs = []

    for trajectory in row['transitions_ep']:
        tl = get_step_likelihood_trajectory(trajectory, model, env)
        probs.append(tl)

    return probs

def get_step_max_ll_trajectories(row, modelnames, model_folder, device):
    ''' Computes the stepwise maximum likelihood of a given trajectory by across all models 
    
    Arguments
    ---------
    row : pandas.DataFrame row, row of the dataframe containing the trajectories
    models : list of str, list of model names
    model_folder : str, path to the folder containing the models
    device : str, device to run the models on

    Returns
    -------
    step_max_lls : list of floats, list of stepwise-maximized likelihoods for each efficacy level
    '''

    models = []
    step_max_lls = []
    mean_stepmax_ls = []

    for modelname in modelnames:
        env, model = load_env_and_model(str(modelname), model_folder, device=device)
        models.append(model)

    env.setup()

    for trajectory in row['transitions_ep']:

        ep_max_lls = []
        ep_max_ls = []

        env.start_new_episode()
        #lstm_hidden = None
        state = env.get_state()

        lstm_hiddens = [None for _ in models]

        for step in trajectory:

            model_lls = []

            for im, model in enumerate(models):

                logits, lstm_hidden, value, control = model(torch.tensor(state).to(device).float(), lstm_hiddens[im])

                lstm_hiddens[im] = lstm_hidden

                correct = int(step[0])
                selected_action = torch.tensor(convert_action_representation(step[1])).to(device)
                action = torch.tensor(convert_action_representation(step[2])).to(device)

                sampler = Categorical(logits=logits)
                model_lls.append(sampler.log_prob(selected_action).item())
            
            ## max_logit = np.argmax(np.array(model_lls))
            max_logit = np.max(np.array(model_lls))
            ep_max_lls.append(max_logit)
            ep_max_ls.append(np.exp(max_logit))
            state = simulate_step_in_env(env, correct, action, selected_action)            

        step_max_lls.append(np.sum(np.array(ep_max_lls)))
        mean_stepmax_ls.append(np.mean(np.array(ep_max_ls)))

    return np.stack(step_max_lls), np.stack(mean_stepmax_ls)


# %% START MAIN LOOP

for pars in experiment_params:

    ## extract parameters from dict
    df_file = pars['df_file']
    exp_date = pars['exp_date']
    day = pars['day']
    include_sleep = pars['include_sleep']

    ## generate run specific parameters
    save_df_file = os.path.join(
        'results',
        'behavior',
        f'{get_timestamp()}_behavior_diff_effs_{exp_date}_{day}_with_nets{notes}',
    )
    analysis_folder = os.path.join('analysis', 'traj_diff_efficacy', day, exp_date)
    data_folder = os.path.join('data', day, 'data')
    
    df = pd.read_pickle(df_file)

    if 'group' in df and day[-1] == '1':
    #if(True):
        dfA = df[~df['group']]
        dfB = df[df['group']]
        dfs = [dfA, dfB]
        groups = ['groupA', 'groupB']
    elif 'group' in df and day[-1] != 'B':
        dfA = df[~df['group']]
        dfs = [dfA]
        groups = ['groupA']
    elif 'group' in df:
        dfB = df[df['group']]
        dfs = [dfB]
        groups = ['groupB']
    else:
        dfs = [df]
        groups = ['groupA']

    print('detected the following groups for file %s: %s' %(df_file, groups))


    for df, group in zip(dfs, groups):

        effs_sorted_train, effs_sorted_test = get_effs(group, day[-1] if group == 'groupA' else day[-2])

        #n_trials = 10
        test_start = len(effs_sorted_train)

        ### PEPE

        if not include_sleep:

            ## VOLATILITY 0.1, MANUAL SEED
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

            ## VOLATILITY 0.2
            # ape_models = [
            #  20230704231542,
            #  20230704231540,
            #  20230704231539,
            #  20230704231537,
            #  20230704231535,
            #  20230704231525,
            #  20230704231524,
            #  20230704231522,
            #  20230704231521,
            #  20230704231519
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


            #### WITH BIAS 0.5, VOLATILITY 0.2, AND NO HELDOUT TEST REGION
            #### 10/06

            # ape_models = [
            #     20230923060019,
            #     20230923060017,
            #     20230923060016,
            #     20230923060014,
            #     20230923060013,
            #     20230922111420,
            #     20230922111418,
            #     20230922111417,
            #     20230922111415,
            #     20230922111413,
            # ]

            # control_models = [
            #     20230923023710,
            #     20230923023709,
            #     20230923023707,
            #     20230923023706,
            #     20230923023704,
            #     20230922110530,
            #     20230922110528,
            #     20230922110527,
            #     20230922110525,
            #     20230922110524,
            # ]


            ### 

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
                ## FC
                20240405185747,
                20240405185746,
                20240405185744,
                20240405185742,
                20240405185741,
                20240405185739,
                20240405185738,
                20240405185736,
                20240405185735,
                20240405185733,
            ]


        ## EVC

        else:

            ## VOLATILITY 0.1  
            # ape_models = [
            #     20230519234556,
            #     20230519234555,
            #     20230519234553,
            #     20230519234552,
            #     20230519234550,

            #     20230519234540,
            #     20230519234538,
            #     20230519234536,
            #     20230519234534,
            #     20230519234533,
            # ]

            # control_models = [
            #     20230519234603,
            #     20230519234602,
            #     20230519234601,
            #     20230519234559,
            #     20230519234558,

            #     20230519234549,
            #     20230519234547,
            #     20230519234546,
            #     20230519234545,
            #     20230519234543,
            # ]

            # ## VOLATILITY 0.2, NO MANUAL SEED

            # ape_models = [
            #     #20230711231709, #unconverged
            #     20230711231707,
            #     20230711231706,
            #     #20230711231705, #semi-converged
            #     #20230711231702, #unconverged
            #     20230711231701,
            #     20230711231659,
            #     20230711231658,
            #     20230711231657,
            #     #20230711231655, #unconverged    
            #     ]

            # control_models = [
            #     20230710230317,
            #     20230710230315,
            #     20230710230314,
            #     20230710230314,
            #     20230710230311,
            #     20230710230302,
            #     20230710230300,
            #     20230710230258,
            #     20230710230257,
            #     20230710230255
            # ]

            ## NO HOLDOUT BIAS 0.5, VOL 0.2, HIGHER ENTROPY
            # 10/13

            # ape_models = [
            #     20231013081402,
            #     20231013081400,
            #     20231013081359,
            #     20231013081357,
            #     20231013081356,
            #     20231013081346,
            #     20231013081344,
            #     20231013081343,
            #     20231013081341,
            #     20231013081339,
            # ]

            # control_models = [
            #     20231013081411,
            #     20231013081409,
            #     20231013081407,
            #     20231013081406,
            #     20231013081404,
            #     20231013081354,
            #     20231013081353,
            #     20231013081351,
            #     20231013081349,
            #     20231013081348
            # ]

            ape_models = [
                20240305173412,
                20240305173411,
                20240305173409,
                20240305173407,
                20240305173406,
                20240305173403,
                20240305173402,
                20240305173400,
                20240305173359,
                20240305173357,
            ]

            control_models = [
                20240406130255,
                20240406130254,
                20240406130252,
                20240406130251,
                20240406130249,
                20240405190151,
                20240405190150,
                20240405190148,
                20240405190147,
                20240405190145,
            ]

        modelname = ape_models[0]
        # %% CREATE STEP

        for modelname in ape_models + control_models:
        #for modelname in control_models:
            modelname = str(modelname)
            print(modelname)
            step_l = df.apply(
                get_step_likelihood_trajectories,
                args=(modelname, model_folder, device),
                axis=1,
            )
            print(step_l)
            df[f'step_l_{modelname}'] = step_l

        print("Done with step likelihoods")

        df.to_csv(save_df_file + '_%s.csv' %group)
        df.to_pickle(save_df_file + '_%s.pkl' %group)

        continue

        # df.to_csv(save_df_file + '.csv')
        # df.to_pickle(save_df_file + '.pkl')

        # assert False, "early termination because rest is not needed"
        
        # %% CREATE LOGIT TRAJECTORIES FOR APE MODELS

        for modelname in ape_models:
            modelname = str(modelname)
            print(modelname)
            result_series = df.apply(lambda row: get_logits_trajectories(row, modelname, model_folder, device), axis=1)

            # Now, split this Series of tuples into two separate Series and assign them to the DataFrame
            df[f'logits_{modelname}'] = result_series.apply(lambda x: x[0])
            df[f'control_{modelname}'] = result_series.apply(lambda x: x[1])
            # df[f'logits_{modelname}'], df[f'control_{modelname}'] = df.apply(
            #     get_logits_trajectories,
            #     args=(modelname, model_folder, device),
            #     axis=1,
            # )


        # %% CREATE LOG LIKEHOOD TRAJECTORIES COLS

        for modelname in ape_models + control_models:
        #for modelname in control_models:
            modelname = str(modelname)
            print(modelname)
            df[f'll_{modelname}'] = df.apply(
                get_log_likelihood_trajectories,
                args=(modelname, model_folder, device),
                axis=1,
            )
            

        # %% COMPUTE LIKELIHOOD RATIO BETWEEN APE AND CONTROL MODELS

        mean_ll_ape = np.stack(
            [df[f'll_{str(model)}'].values.mean() for model in ape_models]
        ).mean()
        mean_ll_control = np.stack(
            [df[f'll_{str(model)}'].values.mean() for model in control_models]
        ).mean()

        print('log likelihood ratio', mean_ll_ape - mean_ll_control)

        # %%

        df['mean_ll_ape_train'], df['mean_ll_ape_test'] = get_mean_ll(df, ape_models, test_start, aggregate_efficacies=False)
        df['mean_ll_control_train'], df['mean_ll_control_test'] = get_mean_ll(df, control_models, test_start, aggregate_efficacies=False)

        # %%

        df['max_ll_ape_train'], df['max_ll_ape_test'] = get_max_ll(df, ape_models, test_start)
        df['max_ll_control_train'], df['max_ll_control_test'] = get_max_ll(df, control_models, test_start)

        # %% AGGREGATE AT LEVEL OF STEP

        #sml_ape, mean_sml_ape = df.apply(get_step_max_ll_trajectories, args=(ape_models, model_folder, device), axis=1)
        #df['step_max_ll_ape'], df['mean_stepmax_l_ape'] = np.stack(sml_ape), np.stack(mean_sml_ape)
        stepmax_res_df = df.apply(get_step_max_ll_trajectories, args=(ape_models, model_folder, device), axis=1).apply(lambda x: pd.Series([x[0], x[1]]))
        df['step_max_ll_ape'], df['mean_stepmax_l_ape'] = stepmax_res_df[0], stepmax_res_df[1]

        stepmax_res_df = df.apply(get_step_max_ll_trajectories, args=(control_models, model_folder, device), axis=1).apply(lambda x: pd.Series([x[0], x[1]]))
        df['step_max_ll_control'], df['mean_stepmax_l_control'] = stepmax_res_df[0], stepmax_res_df[1]

        # %% GET MEAN LIKELIHOOD TRAJECTORIES

        for modelname in ape_models + control_models:
        #for modelname in control_models:
            modelname = str(modelname)
            df[f'ml_{modelname}'] = df.apply(
                get_mean_likelihood_trajectories,
                args=(modelname, model_folder, device),
                axis=1,
            )

        # %% 

        df['mean_lik_ape_train'], df['mean_lik_ape_test'] = get_mean_ll(df, ape_models, test_start, prefix='ml', aggregate_efficacies=False)
        df['mean_lik_control_train'], df['mean_lik_control_test'] = get_mean_ll(df, control_models, test_start, prefix='ml', aggregate_efficacies=False)


        # %% GET OBSERVE DEVIATION TRAJECTORIES

        for modelname in ape_models + control_models:
        #for modelname in control_models:
            modelname = str(modelname)
            df[f'odl_{modelname}'] = df.apply(
                get_observe_deviation_likelihood_trajectories,
                args=(modelname, model_folder, device),
                axis=1,
            )

        # %% COMPUTE MEAN OBSERVE DEVIATION LIKELIHOOD

        df['mean_odl_ape_train'],df['mean_odl_ape_test'] = get_mean_ll(df, ape_models, test_start, 'odl', aggregate_efficacies=False)
        df['mean_odl_control_train'],df['mean_odl_control_test'] = get_mean_ll(df, control_models, test_start, 'odl', aggregate_efficacies=False)

        # %% SLEEP

        if include_sleep:
            for modelname in ape_models + control_models:
            #for modelname in control_models:
                modelname = str(modelname)
                df[f'sdl_{modelname}'] = df.apply(
                    get_sleep_deviation_likelihood_trajectories,
                    args=(modelname, model_folder, device),
                    axis=1,
                )

        # %% COMPUTE MEAN OBSERVE DEVIATION LIKELIHOOD
        if include_sleep:
            df['mean_sdl_ape_train'],df['mean_sdl_ape_test'] = get_mean_ll(df, ape_models, test_start, 'sdl', aggregate_efficacies=False)
            df['mean_sdl_control_train'],df['mean_sdl_control_test'] = get_mean_ll(df, control_models, test_start, 'sdl', aggregate_efficacies=False)
