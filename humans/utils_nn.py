# Kai Sandbrink
# 2023-09-28
# This script contains utility functions for the neural network scrips

# %% LIBRARY IMPORT

import numpy as np
import pandas as pd
import os, ast, glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import torch
from ObserveBetEfficacyTask import ObserveBetEfficacyTask
from PeekTakeTorchRNN import PeekTakeTorchAPERNN, PeekTakeTorchRNN, PeekTakeTorchPerturbedAPERNN
from torch.distributions import Categorical

from human_utils_project import load_config_files

# %%

def load_env_and_model(modelname, model_folder, model_file = 'model.pt', device='cpu', perturbed = False):

    config, task_options, nn_options = load_config_files(modelname, model_folder)

    env = ObserveBetEfficacyTask(**task_options)
    env.setup()

    if nn_options.ape_loss_coeff is not None and not perturbed:
        model = PeekTakeTorchAPERNN(env.actions, env.encoding_size, **nn_options)
    elif not perturbed:
        model = PeekTakeTorchRNN(env.actions, env.encoding_size, **nn_options)
    else:
        model = PeekTakeTorchPerturbedAPERNN(env.actions, env.encoding_size, **nn_options)

    model.load_state_dict(torch.load(os.path.join(model_folder, modelname, model_file),map_location=device))

    return env, model

# %% 

def convert_action_representation(human_action, include_sleep = False):
    ''' Converts the human action representation to the one used in the model'''

    if human_action == 0:
        return 1 + int(include_sleep)
    elif human_action == 1:
        return 2 + int(include_sleep)
    elif human_action == 0.5:
        return 0
    elif human_action == -1:
        return 1
    else:
        raise ValueError('Human action not recognized.')

# %% 

def simulate_step_in_env(env, correct, action, selected_action):
    env.action = env.actions[action]
    env.selected_action = env.actions[selected_action]

    if action != selected_action:
        env.action_failed = {'peek': False, 'take': False}
    else:
        env.action_failed = {'peek': False, 'take': True}

    rewards_step = (env.action[0] == 'take' and env.action[1] == correct)
    env.rewards_tally += rewards_step

    env.feedback = [0, 0]
    if env.action[0] == 'peek':
        env.feedback_given = True
        env.feedback[correct] = 1

    env.steps += 1
    env.reveal_rewards_tally = False
    env.tally_feedback = 0

    return env.get_state()

# %%

def get_likelihood_trajectory(trajectory, model, env, n_steps_to_reward=50, device='cpu', include_sleep = False):
    ''' Computes the likelihood of a given trajectory in a model
    
    Arguments
    ---------
    trajectory : array
        Trajectory of a single episode
    model : torch.nn.Module
        Model to compute the likelihood in
    env : ObserveBetEfficacyTask
        Environment to simulate the trajectory in
    n_steps_to_reward : int
        Number of steps to simulate
    device : str    
        Device to run the model on
    include_sleep : bool
        Whether the model includes the sleep action

    Returns
    -------
    taken_logits : float
        Likelihood of the taken actionlogits
    observe_deviation_logits : float
        Likelihood of the observe deviation
    sleep_deviation_logits : float
        Likelihood of the sleep deviation   
    '''

    env.setup()
    env.start_new_episode()
    lstm_hidden = None
    state = env.get_state()

    saved_taken_logits = []
    saved_observe_deviation_logits = []

    if include_sleep:
        saved_sleep_deviation_logits = []
    else:
        saved_sleep_deviation_logits = None

    for j in range(n_steps_to_reward):

        logits, lstm_hidden, value, control = model(torch.tensor(state).to(device).float(), lstm_hidden)

        correct = int(trajectory[j][0])
        selected_action = torch.tensor(convert_action_representation(trajectory[j][1], include_sleep)).to(device)
        action = torch.tensor(convert_action_representation(trajectory[j][2], include_sleep)).to(device)

        sampler = Categorical(logits=logits)
        saved_taken_logits.append(np.exp(sampler.log_prob(selected_action).item()))

        if trajectory[j][1] == 0.5:
            saved_observe_deviation_logits.append(1 - np.exp(logits[0].item()))
        else:
            saved_observe_deviation_logits.append(- np.exp(logits[0].item()))

        if include_sleep:
            if trajectory[j][1] == -1:
                saved_sleep_deviation_logits.append(1 - np.exp(logits[1].item()))
            else:
                saved_sleep_deviation_logits.append(- np.exp(logits[1].item()))

        ### IMITATE BEHAVIOR OF OBSERVE-BET-EFFICACY TASK STEP
        state = simulate_step_in_env(env, correct, action, selected_action)

    if include_sleep:
        saved_sleep_deviation_logits = np.array(saved_sleep_deviation_logits).mean()

    return np.array(saved_taken_logits).mean(), np.array(saved_observe_deviation_logits).mean(), saved_sleep_deviation_logits

# %%

def get_logits_trajectory(trajectory, model, env, n_steps_to_reward=50, device='cpu', include_sleep = False):
    ''' Computes the logits of a given trajectory in a model
    
    Arguments
    ---------
    trajectory : array
        Trajectory of a single episode
    model : torch.nn.Module
        Model to compute the likelihood in
    env : ObserveBetEfficacyTask
        Environment to simulate the trajectory in
    n_steps_to_reward : int
        Number of steps to simulate
    device : str    
        Device to run the model on
    include_sleep : bool
        Whether the model includes the sleep action

    Returns
    -------
    logitss : array
        Logits of actor policy
    controlss : array
        control estimated by the network
    '''

    env.setup()
    env.start_new_episode()
    lstm_hidden = None
    state = env.get_state()

    logitss = []
    controlss = []

    for j in range(n_steps_to_reward):

        logits, lstm_hidden, value, control = model(torch.tensor(state).to(device).float(), lstm_hidden)
        logitss.append(logits.detach().numpy())
        controlss.append(control.detach().numpy())

        correct = int(trajectory[j][0])
        selected_action = torch.tensor(convert_action_representation(trajectory[j][1], include_sleep)).to(device)
        action = torch.tensor(convert_action_representation(trajectory[j][2], include_sleep)).to(device)

        ### IMITATE BEHAVIOR OF OBSERVE-BET-EFFICACY TASK STEP
        state = simulate_step_in_env(env, correct, action, selected_action)

    return logitss, controlss