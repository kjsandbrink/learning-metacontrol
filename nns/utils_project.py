# Kai Sandbrink
# 2023-02-01
# This script has project-specific util functions (unlike utils.py which is meant to be global)

# %% LIBRARY IMPORT

import numpy as np
import yaml, os, copy, math, pickle

from utils import Config

# %% PARAMETERS

base_model_folder = 'models'

# %% DATA READ IN

def load_config_files(modelname):

    model_folder = os.path.join(base_model_folder, str(modelname))

    config = Config({})
    config.load_config_file(os.path.join(model_folder, 'config.yaml'))

    task_options = Config({})
    task_options.load_config_file(os.path.join(model_folder, 'task_options.yaml'))

    nn_options = Config({})
    nn_options.load_config_file(os.path.join(model_folder, 'nn_options.yaml'))

    return config, task_options, nn_options

# %% 

def load_modelrun_files(models, traj_timestamp , ape = False, traj_base = os.path.join('data', 'eval', 'pepe', ), includes_sleep=False):
        
    rewss_taus_ape = [] #list of lists that will store rews for diff tau values for APE
    counterss_peeks_taus_ape = []

    trajss_actions_taus_ape = []
    trajss_logits_taus_ape = []
    trajss_ps_taus_ape = []

    if ape:
        control_errss_taus_ape = []
    else:
        control_errss_taus_ape = None

    if includes_sleep:
        counterss_sleeps_taus_ape = []


    for model in models:

        orig_traj_folder = os.path.join(traj_base, str(model))
        
        counters_peeks_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_counters_peeks_taus.pkl'), 'rb'))
        rewss_taus = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_rewss_taus.pkl'), 'rb'))

        trajs_actions_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_actionss_taus.pkl'), 'rb'))
        trajs_logits_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_logitss_taus.pkl'), 'rb'))
        trajs_ps_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_traj_pss_taus.pkl'), 'rb'))

        counterss_peeks_taus_ape.append(counters_peeks_taus)
        rewss_taus_ape.append(rewss_taus)

        trajss_actions_taus_ape.append(trajs_actions_taus_ape)
        trajss_logits_taus_ape.append(trajs_logits_taus_ape)
        trajss_ps_taus_ape.append(trajs_ps_taus_ape)

        if ape:
            control_errs_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_control_errs_taus_ape.pkl'), 'rb'))
            control_errss_taus_ape.append(control_errs_taus_ape)

        if includes_sleep:
            counters_sleeps_taus_ape = pickle.load(open(os.path.join(orig_traj_folder, traj_timestamp + '_sleep_errs_taus_ape.pkl'), 'rb'))
            counterss_sleeps_taus_ape.append(counters_sleeps_taus_ape)

    control_errss_taus_ape = np.array(control_errss_taus_ape).T
    counterss_peeks_taus_ape = np.array(counterss_peeks_taus_ape).T
    rewss_taus_ape = np.array(rewss_taus_ape).T

    if includes_sleep:
        counterss_sleeps_taus_ape = np.array(counterss_sleeps_taus_ape).T

    trajss_actions_taus_ape = np.array(trajss_actions_taus_ape)
    trajss_logits_taus_ape = np.array(trajss_logits_taus_ape)
    trajss_ps_taus_ape = np.array(trajss_ps_taus_ape)

    if not includes_sleep:
        return rewss_taus_ape, counterss_peeks_taus_ape, control_errss_taus_ape, trajss_actions_taus_ape, trajss_logits_taus_ape, trajss_ps_taus_ape
    else:
        return rewss_taus_ape, counterss_peeks_taus_ape, control_errss_taus_ape, counterss_sleeps_taus_ape, trajss_actions_taus_ape, trajss_logits_taus_ape, trajss_ps_taus_ape

# %% EVIDENCE RATIO CALCULATIONS

def calculate_freq_observed_choice_per_t(actions, logits, pss, include_sleep = False):
    """
    Calculate the fraction of correct choices at the time they observed (0.5).

    Arguments
    ---------
    actions : np.array [n_episodes, n_steps,], actions taken in episode
    logits : np.array [n_episodes, n_steps, n_actions,], actions taken in episode
    pss : np.array [n_episodes, n_steps, 2], probability values of payout arms
    include_sleep : bool, whether to include sleep action in calculation

    Returns
    -------
    fractions : np.array [n_episodes, n_steps,], sum of logits of people picking the arm they observed as a function of time since last observation
    """

    fractions = []
    n_steps = actions.shape[1]

    for ep, ep_logits, ep_ps in zip(actions, logits, pss):

        counter_since_observes = np.zeros(n_steps,)
        counter_correct_choices = np.zeros(n_steps,)
        
        counter_since_observe = 0
        current_choice = None
        # Iterate over the sublists in each row
        for action, ls, ps in zip(ep, ep_logits, ep_ps):
            # Check if intended choice is "observed"
            if action == 0:
                counter_since_observe = 0
                current_choice = np.argmax(ps)
            else:
                counter_since_observe += 1
                counter_since_observes[counter_since_observe - 1] += 1
                    
                if current_choice is not None:
                   counter_correct_choices[counter_since_observe - 1] += np.exp(ls[1 + int(include_sleep) + current_choice])
                            
        # Calculate fraction
        fraction = [correct / total if total > 1 else np.nan for correct, total in zip(counter_correct_choices, counter_since_observes)]
        fractions.append(fraction)
        
    return fractions

def calculate_freq_observes_per_t(actions, logits, pss, include_sleep = False):
    """
    Calculate the fraction of time that an observe action was chosen as a function of time since last observation.

    Arguments
    ---------
    actions : np.array [n_episodes, n_steps,], actions taken in episode
    logits : np.array [n_episodes, n_steps, n_actions,], actions taken in episode
    pss : np.array [n_episodes, n_steps, 2], probability values of payout arms

    Returns
    -------
    fractions : np.array [n_episodes, n_steps,], fraction of time people observed as a function of time since last observation
    """

    fractions = []
    n_steps = actions.shape[1]

    for ep, ep_ls, ep_ps in zip(actions, logits, pss):

        counter_since_observes = np.zeros(n_steps,)
        counter_observes = np.zeros(n_steps,)
        
        counter_since_observe = 0
        # Iterate over the actions and probability values in each episode
        for action, ls, ps in zip(ep, ep_ls, ep_ps):
            counter_since_observes[counter_since_observe] += 1
            
            if action == 0:
                counter_observes[counter_since_observe] += np.exp(ls[0])
                counter_since_observe = 0
            else:
                counter_since_observe += 1

        # Calculate fraction
        fraction = [observe / total if total > 1 else np.nan for observe, total in zip(counter_observes, counter_since_observes)]
        fractions.append(fraction)
        
    return fractions


def calculate_freq_correct_choice_per_t(actions, logits, pss, include_sleep = False):
    """
    Calculate the fraction of time that an the correct arm was chosen as a function of time since last observation.

    Arguments
    ---------
    actions : np.array [n_episodes, n_steps,], actions taken in episode
    logits : np.array [n_episodes, n_steps, n_actions,], actions taken in episode
    pss : np.array [n_episodes, n_steps, 2], probability values of payout arms
    include_sleep : bool, whether to include sleep action in calculation

    Returns
    -------
    fractions : np.array [n_episodes, n_steps,], fraction of time people observed as a function of time since last observation
    """

    fractions = []
    n_steps = actions.shape[1]

    for ep, ep_ls, ep_ps in zip(actions, logits, pss):

        counter_since_observes = np.zeros(n_steps,)
        counter_correct_choices = np.zeros(n_steps,)
        
        counter_since_observe = 0
        # Iterate over the sublists in each row
        for action, ls, ps in zip(ep, ep_ls, ep_ps):
            # Check if intended choice is "observed"
            if action == 0:
                counter_since_observe = 0
            else:
                counter_since_observe += 1
                counter_since_observes[counter_since_observe - 1] += 1
                counter_correct_choices[counter_since_observe - 1] += np.exp(ls[np.argmax(ps) + 1 + int(include_sleep)])

        # Calculate fraction
        fraction = [correct / total if total > 1 else np.nan for correct, total in zip(counter_correct_choices, counter_since_observes)]
        fractions.append(fraction)
        
    return fractions

# %%

def calculate_freq_correct_take(logits_taus, ps_taus, include_sleep = False):
    ''' Calculates frequency of correct take (conditional on taking) given ps and choices 
    
    Arguments
    ---------
    logits_taus : np.array [n_models, n_taus, n_episodes, n_steps, n_actions,], actions taken in episode
    ps_taus : np.array [n_models, n_taus, n_episodes, n_steps, 2], probability values of payout arms

    Returns
    -------
    fractions : np.array [n_models, n_taus, n_episodes, n_steps,], probability of taking correct arm
    '''
    
    correct_arm = np.argmax(ps_taus, axis = -1)
    
    ### select probability based on the correct arm
    i, j, k, l = np.ogrid[:correct_arm.shape[0], :correct_arm.shape[1], :correct_arm.shape[2], :correct_arm.shape[3]]
    correct_take = logits_taus[i, j, k, l, 1 + int(include_sleep) + correct_arm]
    other_take = logits_taus[i, j, k, l, 1 + int(include_sleep) + 1 - correct_arm]

    ### calculate frequency of correct take
    freq_correct_take = np.exp(correct_take) / (np.exp(correct_take) + np.exp(other_take))

    return freq_correct_take
    
# %%
