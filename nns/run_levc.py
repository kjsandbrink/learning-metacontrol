# Kai Sandbrink
# 2022-10-18
# Script that loads configuration options and runs training loop

# %% LIBRARY IMPORTS

import torch
import numpy as np
import torch.multiprocessing as mp
import copy, time

from utils import Config

from trainers import train

# %% PARAMETERS

n_runs = 5

# %% CONFIG OPTIONS

config = Config({
    'env': 'ObserveBetEfficacyTask',
    'recurrent_type': 'LSTMCell',
    'n_runs' : 1,
    'n_episodes': 500000,
    'n_steps_to_reward': 50,
    'note': '',
    'tags': ['LEVC_w APE', 'human', 'no manual seed'], #wandb tag
    #'tags': None, #to use if None
    'baseline': '1/n_actions',
    'baseline_type': 'all', #one of ['all', 'all']
    'entropy_reg_coeff': 5, #only used in geometric annealing or if entropy is not annealed
    'entropy_final_reg_coeff': 0.001,
    'discount': 1,
    'log_every_k_episodes': 100,
    'anneal_entropy': 'geom', #one of ['geom', 'lin', False]
    'n_anneal_entropy_episodes': 150000,
    'batchsize': 50,
    'trialsize': 1,
    'save_model': True, #only saves final model (and one 'just in case' at 95% of training)
    'save_rewards': True,
    'training_checkpoints': 100, # CAN BE LIST OR INT, if list they are spaced uniformly
    #'training_checkpoints': [i/100 for i in range(100)], #which training checkpoints to save
        #will only activate if save_model is True, if empty, only final model runs saved
    'harcode_prob_fail_signal': None,
    'weight_decay': 0,
    'ape_readout_l2_reg': 0,
        # if this is not None, then action_failed_prob signal is overwritten (with failure rate given by its value)
    'device': 'cuda',
    'task': 'levc',
})

if type(config.training_checkpoints) == int:
    config.training_checkpoints =  [i/config.training_checkpoints for i in range(config.training_checkpoints)]

nn_options = Config({
    'lstm_hidden_size': 48, #value used in Wang (2016)
    'value_loss_coeff': 0,
    #'ape_loss_coeff': None, #5
    'ape_loss_coeff': 100, #5
    'hardcode_efficacy': False, ### This is only "on" if ape_loss_coeff != 0; however, if it's on, it doesn't matter what ape_loss_coeff is
    #'hidden_size': None, #None for no hidden layer (not 0!)
    'hidden_size': 24,
    #'hidden_size': None,
    #'value_loss_coeff': 0.1, #uses A2C
        #if 0 defaults to using hardcoded critic (REINFORCE w baseline)
        #TODO: make generalized
})

## POLY TYPE
# task_options = Config({
#     'n_arms': 2,
#     'tiredness_form': {'peek': 'poly', 'take': 'poly'},
#     'alphas': {'peek': 1, 'take': 1}, # tau^alpha calculates failure prob if poly or cst
#         # if exp, prob of choosing selected arm is given by (1/n_arms)+(1-1/(n_arms-1))*e^(-alpha*tau)
#     #'alphas': {'peek': 0.5, 'take': 0.5},
#     'increase_taus_factor': {'peek': 0, 'take': 0}, ## this is the factor by which taus are increased every step an action is taken
#     'include_sleep_actions': {'peek': False, 'take': True},
#     #'sleep_factors': {'peek': 0, 'take': 0},
#     'sleep_factors': {'peek': 0, 'take': 0.1},
#     #'sleep_factors': {'peek': 'max', 'take': 'max'}, #can be int or 'max'
#     'max_tiredness_reached_after': {'peek': 1, 'take': 1},
#         #only applied in poly case
#     'starting_taus': "uniform_held-out-middle",
#     'reset_taus': True,
#     'fail_action' : {'peek': 'switch', 'take': 'switch'}, #one of ['switch', 'fail']
#         #also only applied in poly case
#     'stationary': True,
#     'p_dist': 'opposites',
#     'n_steps_to_reward': config.n_steps_to_reward,
#     'reset_every_k_steps': 0.1, #0.05, #25, #if < 1, taken to be an expected frequency / per-step prob
#     #'reset_every_k_steps': None,
#         # 0 special value where switch only occurs at the beginning of the episoded
#     'reset_taus': True,
#     'noise_type': 'switch', #one of ['gaussian', 'middle']
#     'payout_type': 'prob', #one of ['prob', 'mag']
#     'actions_encoding_type': 'both_actions', #one of ['failed_flag', 'both_actions', 'intended_only']
#     'bias': 0.49,
# })

### HUMAN-LIKE
task_options = Config({
    'n_arms': 2,
    'tiredness_form': {'peek': 'poly', 'take': 'poly'},
    'alphas': {'peek': 1, 'take': 1}, # tau^alpha calculates failure prob if poly or cst
        # if exp, prob of choosing selected arm is given by (1/n_arms)+(1-1/(n_arms-1))*e^(-alpha*tau)
    #'alphas': {'peek': 0.5, 'take': 0.5},
    'increase_taus_factor': {'peek': 0, 'take': 0},
    'include_sleep_actions': {'peek': False, 'take': True},
    #'sleep_factors': {'peek': 0, 'take': 0},
    'sleep_factors': {'peek': 0, 'take': 0.1}, ## REMEMBER TO SWITCH TOGETHER WITH TIREDNESS_FORM
    #'sleep_factors': {'peek': 'max', 'take': 'max'}, #can be int or 'max'
    'max_tiredness_reached_after': {'peek': 1, 'take': 1},
        #only applied in poly case
    #'starting_taus': "uniform_held-out-middle", ## REMEMBER TO SWITCH TOGETHER WITH TIREDNESS_FORM
    'starting_taus': "uniform_no-holdout",
    'reset_taus': True,
    'fail_action' : {'peek': 'switch', 'take': 'switch'}, #one of ['switch', 'fail']
        #also only applied in poly case
    'stationary': True,
    'p_dist': 'opposites',
    'n_steps_to_reward': config.n_steps_to_reward,
    'reset_every_k_steps': 0.1, #0.05, #25, #if < 1, taken to be an expected frequency / per-step prob
    #'reset_every_k_steps': None,
        # 0 special value where switch only occurs at the beginning of the episoded
    'reset_taus': True,
    'noise_type': 'switch', #one of ['gaussian', 'middle']
    'payout_type': 'prob', #one of ['prob', 'mag']
    'actions_encoding_type': 'both_actions', #one of ['failed_flag', 'both_actions', 'intended_only']
    'bias': 0.5,
})

# %% RUN

if __name__ == '__main__':

    devices = ['cpu', 'cpu', 'cpu']
    tags = ['LEVC_w APE', 'LEVC_w APE', 'LEVC_no APE']
    ape_loss_coeffs = [100, 100, 0]

    # devices = ['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    # #devices = ['cpu'] * 4
    # tags = ['LEVC_w APE', 'LEVC_no APE', 'LEVC_w APE', 'LEVC_no APE']
    # ape_loss_coeffs = [100, 0, 100, 0]

    #starting_taus = np.arange(0, 1.01, 0.25)

    ## SET UP MULTIPROCESSING
    mp.set_start_method('spawn')
    processes = []

    for device, tag, ape_loss_coeff in zip(devices, tags, ape_loss_coeffs):

        ## LOAD DEVICE
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        #device = config.device if torch.cuda.is_available() else "cpu"
        device = device if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")

        for i in range(n_runs):
            current_config = Config(copy.deepcopy(config.__dict__))
            current_config.tags[0] = tag

            current_nn_options = Config(copy.deepcopy(nn_options.__dict__))
            current_nn_options.ape_loss_coeff = ape_loss_coeff
            #current_nn_options = nn_options
            
            p = mp.Process(target=train, args=(current_config, current_nn_options, task_options, device))
            p.start()
            processes.append(p)

            time.sleep(1.5)
    
        for p in processes:
            p.join()

    print('done with all models')