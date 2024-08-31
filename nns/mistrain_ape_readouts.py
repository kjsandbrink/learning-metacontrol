# Kai Sandbrink
# 2023-02-01
# This script takes a converged network with APE layer and mistrains the efficacy signal

# %% LIBRARY IMPORTS

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os, random
import time

import wandb
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from PeekTakeTorchTask import PeekTakeTorchTask
from ObserveBetEfficacyTask import ObserveBetEfficacyTask
from PeekTakeTorchRNN import PeekTakeTorchRNN, PeekTakeTorchAPERNN
import multiprocessing as mp

from utils import Config, plot_learning_curve, format_axis, flatten, get_timestamp

from utils_project import load_config_files

# %% MODEL RUNS AND OTHER PARAMETERS

#### 5/23 WITH BIAS 0.5, VOL 0.1

pepe_models =  [
    20230427201627,
    20230427201629,
    20230427201630,
    20230427201632,
    20230427201633,
    
    20230427201644,
    20230427201646,
    20230427201647,
    20230427201648,
    20230427201649
]


### NO HOLDOUT BIAS 0.5, VOL 0.2, HIGHER ENTROPY
## 10/13

levc_models = [
    20231013081402,
    20231013081400,
    20231013081359,
    20231013081357,
    20231013081356,
    20231013081346,
    20231013081344,
    20231013081343,
    20231013081341,
    20231013081339,
]

ape_models = pepe_models + levc_models
ape_models = pepe_models
target_taus = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
n_episodes = 100000
n_checkpoints = 100

# %% CREATE MISTRAINING FUNCTION

def mistrain(ape_model, target_tau, n_episodes, n_checkpoints=100, timestamp=None):

    if timestamp is None:
        timestamp = get_timestamp()

    checkpoint = ''
    ape_model_folder = os.path.join('models', str(ape_model))
    results_folder = os.path.join('data', 'mistrained_models', str(ape_model), timestamp + '_mistrained_tau%d' %int(target_tau*100))

    # %% LOAD CONFIG FILES

    config, task_options, nn_options = load_config_files(ape_model)

    config.results_folder = results_folder
    config.n_episodes = n_episodes
    config.training_checkpoints = n_checkpoints

    if type(config.training_checkpoints) == int:
        config.training_checkpoints =  [i/config.training_checkpoints for i in range(config.training_checkpoints)]

    # %% INITIALIZATIONS

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using {device} device")

    env = ObserveBetEfficacyTask(**task_options)
    env.setup()

    model = PeekTakeTorchAPERNN(env.actions, env.encoding_size, **nn_options)

    if checkpoint == '':
        checkpoint_name = 'model.pt'
    else:
        checkpoint_name = 'model_%s.pt' %str(checkpoint)

    model.load_state_dict(torch.load(os.path.join(ape_model_folder, checkpoint_name), map_location=device))
    model.to(device)

    #TRAINING FUNCTIONS
    optimizer = torch.optim.Adam(model.parameters())

    n_steps = config.n_steps_to_reward*config.n_episodes

    prob_fail = target_tau/2

    # %% FREEZE ALL LAYERS AFTER READOUT

    model_pars = list(model.parameters())
    #model_pars_to_freeze = model_pars[:2] + model_pars[4:]
    model_pars_to_freeze = model_pars[4:]

    for param in model_pars_to_freeze:
        param.requires_grad = False

    # %% TRAINING LOOP

    print("Starting timer")
    start_time = datetime.now()

    n_episode = 0 #actually this is tracking number of log steps, quarantined for deletion

    total_returns = []

    #lstm_hidden = model.init_hidden(device)
    lstm_hidden = None

    ### SET UP BATCH COLLECTION
    batch_pseudo_losses = []
    batch_peek_steps = []
    batch_correct_take_steps = []
    batch_incorrect_take_steps = []
    batch_sleep_peek_steps = []
    batch_sleep_take_steps = []
    batch_log = {}

    #while env.steps < n_steps:
    for i_e in range(config.n_episodes):

        if i_e % 1000 == 0:
            print('on episode', i_e)

        if config.trialsize != 0 and i_e % config.trialsize == 0:
            env.draw_trial_pars()

        env.start_new_episode()

        states = []
        actions = []
        logitss = []
        valuess = []
        controls = []
        tallies_rewards = []
        rewards = []
        saved_log_probs = []
        peek_steps = []
        correct_take_steps = []
        incorrect_take_steps = []
        sleep_peek_steps = []
        sleep_take_steps = []
        clarity_steps = []
        taus_at_clarity = []
        efficacy_steps = []
        taus_at_efficacy = []
        action_descs = []
        actions_failed_take = []

        state = env.get_state()

        pss = []

        counter_correct_takes = 0
        counter_total_takes = 0
        counter_peeks = 0
        counter_sleep_peeks = 0
        counter_sleep_takes = 0
        counter_intended_correct_takes = 0

        #while not reveal_rewards:
        for i_s in range(config.n_steps_to_reward):

            if nn_options.ape_loss_coeff == 0 or not nn_options.hardcode_efficacy:
                logits, lstm_hidden, values, control = model(torch.tensor(state).to(device).float(), lstm_hidden)
            elif nn_options.hardcode_efficacy:
                logits, lstm_hidden, values, control = model(torch.tensor(state).to(device).float(), lstm_hidden, torch.tensor([env.taus['take']]).to(device))

            sampler = Categorical(logits=logits)
            action = sampler.sample()
            saved_log_probs.append(sampler.log_prob(action))

            new_state, reveal_rewards, tally_rewards, rewards_step, selected_action, action_failed, (steps, ps, taus) = env.step(action.item())

            action_desc = env.actions[action]
            if action_desc[0] == 'peek':
                if action_desc[1] != -1:
                    counter_peeks += 1
                    peek_steps.append(i_s)
                else:
                    counter_sleep_peeks += 1
                    clarity_steps.append(i_s)
                    taus_at_clarity.append(taus[action_desc[0]])
                    sleep_peek_steps.append(i_s)

            else:
                if action_desc[1] != -1: #excluding sleep
                    correct_arm = np.argmax(np.array(ps))
                    if correct_arm == selected_action[1]:
                        counter_correct_takes += 1
                    if correct_arm == action_desc[1]:
                        counter_intended_correct_takes += 1
                        correct_take_steps.append(i_s)
                    else:
                        incorrect_take_steps.append(i_s)
                    counter_total_takes += 1
                else:
                    counter_sleep_takes += 1
                    efficacy_steps.append(i_s)
                    taus_at_efficacy.append(taus[action_desc[0]])
                    sleep_take_steps.append(i_s)

            if steps <= 1:
                print(logits)

            assert len(logits.shape) == 1, 'unexpected shape for logits %s' %logits.shape

            if steps < 200 or (steps > n_steps / 2 and steps < n_steps / 2 + 200) or (steps > n_steps - 200):
                print('step %d: s: %s, a: %s / %s, sel: %s, failed: %s, fb: %s, rew: %s, ps: %s, taus: %s, alphas: %s' %(steps, state, action.item(), env.actions[action], selected_action, env.action_failed, env.feedback, rewards_step, ps, env.taus, env.alphas))

            states.append(state)
            actions.append(action)
            logitss.append(logits)
            valuess.append(values)
            controls.append(control)
            tallies_rewards.append(tally_rewards)
            pss.append(ps)
            rewards.append(rewards_step)
            action_descs.append(action_desc)
            actions_failed_take.append(int(action_failed['take']))

            state = new_state

        if steps < 200:
            print('total peeks:', counter_peeks)
            print('correct takes:', counter_correct_takes)
            print('intended correct takes:', counter_intended_correct_takes)
            print('counter_total_takes', counter_total_takes)
        
        ### perform update step

        ### CONSTRUCT LOSS FUNCTION

        ## REWARD COMPUTATIONS FOR STATS
        tally_rewards = np.array(tally_rewards) #baseline offset
        if type(config.baseline) == int or type(config.baseline) == float:
            overall_baseline = config.baseline
        elif config.baseline == 'mean':
            overall_baseline = np.mean(np.array(pss))
        elif config.baseline == '1/n_actions':
            overall_baseline = 1/env.n_actions
        else:
            assert False, "invalid baseline type in overall_baseline specification"
        #tally_rewards_offset = tally_rewards - overall_baseline*counter_total_takes #baseline of 0.5
        tally_rewards_offset = tally_rewards - overall_baseline*config.n_steps_to_reward #baseline of 0.5
        tally_rewards_offset_groundtruth = tally_rewards - np.concatenate(pss).mean()*config.n_steps_to_reward

        ## preprocess states and actions
        states = torch.tensor(np.concatenate(states, axis=0)).to(device).float()
        #print(action.shape)
        actions = torch.stack(actions, axis=0).to(device)
        logitss = torch.stack(logitss, axis=0).to(device)
        saved_log_probs = torch.stack(saved_log_probs, axis=0).to(device)


        ## OLD METHOD BASED ON SIMPLE REINFORCE / HARDCODED CRITIC
        # based mostly on https://github.com/pytorch/examples/blob/main/distributed/rpc/batch/reinforce.py
        if nn_options.value_loss_coeff == 0:

            rewards = np.array(rewards)
            action_descs = np.array(action_descs) ## NOTE: This converts the second entry to string which affects the data type we need to check below

            if nn_options.ape_loss_coeff != 0:
                controls = torch.stack(controls, axis=0).to(device)
            
            ape_returns = []

            for i in reversed(range(len(rewards))):
                a = action_descs[i]
                c = controls[i]

                ##OPTION OF HARDCODED SIGNAL FOR HELPLESSNESS TRIALS 
                if a[0] == 'take':

                    ## HERE THE RECORDED APE SIGNAL IS OVERWRITTEN AND REPLACED WITH A RANDOMLY GENERATED ONE
                    ape = int(random.random() < prob_fail)
                    ape_returns.append(0.5*((c - ape).pow(2)))
                
                else:
                    ape_returns.append(0)

            pseudo_loss = sum(ape_returns)

        ### BATCH COLLECTION
        batch_pseudo_losses.append(pseudo_loss)
        batch_peek_steps.append(peek_steps)
        batch_correct_take_steps.append(correct_take_steps)
        batch_incorrect_take_steps.append(incorrect_take_steps)
        batch_sleep_peek_steps.append(sleep_peek_steps)
        batch_sleep_take_steps.append(sleep_take_steps)

        #### PERFORM GRADIENT UPDATE IF AT END OF BATCH
        if i_e > 0 and i_e % config.batchsize == 0:
            #print('list of batch_pseudo_losses', batch_pseudo_losses)

            ## PERFORM BATCH NORMALIZATION IF NECESSARY
            #batch_pseudo_losses = torch.stack(batch_pseudo_losses)
            #batch_pseudo_losses = (batch_pseudo_losses - torch.mean(batch_pseudo_losses))/torch.std(batch_pseudo_losses)

            #batch_pseudo_losses = torch.sum(batch_pseudo_losses)

            ## ELSE
            batch_pseudo_losses = torch.mean(torch.stack(batch_pseudo_losses))

            ## update policy weights
            optimizer.zero_grad()
            batch_pseudo_losses.backward()
            optimizer.step()

            batch_pseudo_losses = []
        
        #### OTHER RESETS
        if config.trialsize == 0 or (i_e != 0 and i_e % config.trialsize == 0):
            lstm_hidden = None

        #### LOGGING AND SAVING
        if i_e % config.log_every_k_episodes == 0 or i_e == config.n_episodes - 1:

            total_returns.append(tally_rewards)
            if n_episode % 1000 == 0:
                print("Episode: {:6d}\t Return: {:6.2f}\t Time: {:s}".format(n_episode*config.log_every_k_episodes, total_returns[-1], str(datetime.now() - start_time)))
            n_episode += 1 ## actually this is tracking number of log steps

        ## SET UP BATCH-SPEC LOGGING
            if i_e > 0 and i_e % config.batchsize == 0:
                batch_peek_steps = []
                batch_correct_take_steps = []
                batch_incorrect_take_steps = []
                batch_sleep_peek_steps = []
                batch_sleep_take_steps = []

        #### SAVE TOTAL REWARDS IF OPTION IS ENABLED
        if i_e == config.n_episodes - 1 and config.save_rewards:
            os.makedirs(results_folder, exist_ok=True)
            np.save(os.path.join(results_folder, 'logged_returns.npy'), np.array(total_returns))

        #### SAVE MODEL IF NECESSARY
        if config.save_model:
            if i_e == int(config.n_episodes*0.95):
                os.makedirs(results_folder, exist_ok=True)
                config.save_config_file(os.path.join(results_folder, 'config.yaml'))
                nn_options.save_config_file(os.path.join(results_folder, 'nn_options.yaml'))
                task_options.save_config_file(os.path.join(results_folder, 'task_options.yaml'))
                torch.save(model.state_dict(), os.path.join(results_folder, 'model_95.pt'))
                print('model saved at 95%')

            elif i_e == config.n_episodes - 1:
                torch.save(model.state_dict(), os.path.join(results_folder, 'model.pt'))
                print('model saved at 100%')

            if any([i_e == int(config.n_episodes*training_checkpoint) for training_checkpoint in config.training_checkpoints]):
                os.makedirs(results_folder, exist_ok=True)
                percent_complete = int(i_e/config.n_episodes*100)
                torch.save(model.state_dict(), os.path.join(results_folder, 'model_%d.pt' %percent_complete))
                print('model saved at %d percent'%percent_complete)

# %% RUN ALL TOGETHER

if __name__ == "__main__":

    ## SET UP MULTIPROCESSING
    mp.set_start_method('spawn')
    processes = []

    timestamp = get_timestamp()

    for ape_model in ape_models:

        for target_tau in target_taus:
            p = mp.Process(target=mistrain, args=(ape_model, target_tau, n_episodes, n_checkpoints, timestamp))
            p.start()
            processes.append(p)
            time.sleep(1)
            print("started", ape_model, target_tau)
            
        for p in processes:
            p.join()