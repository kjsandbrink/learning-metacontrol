# Kai Sandbrink
# 2022-10-27
# Script that tests OvB cases

# %% LIBRARY IMPORTS

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os, random
import itertools
import json

import wandb
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from PeekTakeTorchTask import PeekTakeTorchTask
from ObserveBetEfficacyTask import ObserveBetEfficacyTask
from PeekTakeTorchRNN import PeekTakeTorchRNN, PeekTakeTorchAPERNN, PeekTakeTorchPerturbedAPERNN

from utils import Config, plot_learning_curve, format_axis, flatten

# %% UTILITY FUNCTIONS

def compute_losses(rewards_steps, taken_logits, baseline = 0, controls= None, actions_failed = None, discount=1):
    ''' computes test loss assuming a constant baseline
    
    '''

    returns = []
    ape_returns = []
    G = 0

    for i in reversed(range(len(rewards_steps))):
    #for a, r, ps, c, ape in zip(action_descs[::-1], rewards[::-1], pss[::-1], torch.flip(controls, (0,)), apes[::-1]):
        r = rewards_steps[i]

        G = r - baseline + discount*G
        returns.append(G)

        if controls is not None and controls[i] is not None and actions_failed is not None:
            c = controls[i]
            ape = actions_failed[i]
            ape_returns.append(0.5*((c - ape) ** 2))
                #no need to cumulate because i want efficacy to be a stable read-out across episode
        else:
            ape_returns.append(0)

    returns.reverse()
    returns = np.array(returns)
    taken_logits = np.array(taken_logits)
    test_loss = sum( - taken_logits * returns)

    ape_loss = sum(ape_returns)
            
    return test_loss, ape_loss

def convert_control_to_prob_random(c):
    ''' converts control readout (between 0 and 1) to prediction of probability fail signal '''

    #prob_random = (c - 0.5)*2
    prob_random = 2*c
    return prob_random

# %% TEST CASES

def test(config, nn_options, task_options, device, checkpoint = '', model_folder = None, n_repeats_case = None, verbose=False):
    ''' 
    Evaluates a trained model's performance on a variety of metrics
    over n_repeats_case episodes

    Arguments
    ---------
    model_folder : str or None, if None assumed that the folder is saved in config
    
    Returns
    -------
    trajectories: 
        logitss : np.array of floats [n_repeats_case, n_steps_until_reward, n_actions] giving logits outputted by policy network
        actionss : np.array of ints [n_repeats_case, n_steps_until_reward], gives chosen action sampled by logits in trial
        valuess : np.array of floats [n_repeats_case, n_steps_until_reward], gives value estimate given by network critic if applicable, else 0s
        controlss : np.array of floats [n_repeats_case, n_steps_until_reward],
        pss : np.array of floats [n_repeats_case, n_steps_until_reward, n_arms]
        squared control_errss : squared errors in efficacy prediction [n_repeats_case, n_steps_until_reward],

    stats: tuple giving summary statistics
        starting_taus : float, starting tau value
        efficacy : float, starting efficacy (1 - starting_taus)
        rewards : float, average rewards per episode
        counter_frac_correct_takes_mean : float, average fraction correct takes
        counter_peeks_mean : float, average number of peeks per episode
        counter_sleeps_mean : float, average number of sleeps per episode
    '''
    
    ## DEFINE ENVIRONMENT
    if config.env == 'PeekTakeTorchTask':
        env = PeekTakeTorchTask(**task_options)
    elif config.env == 'ObserveBetEfficacyTask':
        env = ObserveBetEfficacyTask(**task_options)
    else:
        assert False, "invalid envt"

    env.setup()

    if nn_options.ape_loss_coeff is None:
        model = PeekTakeTorchRNN(env.actions, env.encoding_size, **nn_options)
        
        if checkpoint == '':
            modelname = 'model.pt'
        else:
            modelname = 'model_%s.pt' %str(checkpoint)

        if model_folder is None:
            model_folder = config.model_folder

        model.load_state_dict(torch.load(os.path.join(model_folder, modelname),map_location=device))
    else:
        try:
            model = PeekTakeTorchAPERNN(env.actions, env.encoding_size, **nn_options)

            if checkpoint == '':
                modelname = 'model.pt'
            else:
                modelname = 'model_%s.pt' %str(checkpoint)

            if model_folder is None:
                model_folder = config.model_folder

            model.load_state_dict(torch.load(os.path.join(model_folder, modelname),map_location=device))
        except RuntimeError:
            model = PeekTakeTorchRNN(env.actions, env.encoding_size, **nn_options)
                        
            if checkpoint == '':
                modelname = 'model.pt'
            else:
                modelname = 'model_%s.pt' %str(checkpoint)

            if model_folder is None:
                model_folder = config.model_folder

            model.load_state_dict(torch.load(os.path.join(model_folder, modelname),map_location=device))


    if checkpoint == '':
        modelname = 'model.pt'
    else:
        modelname = 'model_%s.pt' %str(checkpoint)

    if model_folder is None:
        model_folder = config.model_folder

    if n_repeats_case is None:
        n_repeats_case = config.n_repeats_case

    #print(model)
    model.to(device)
    model.eval()

    rewardss = []
    counter_sleepss = []
    saved_logitss = []
    valuess = []
    actionss = []
    controlss = []
    control_errss = []
    pss = []

    lossess_returns = []
    lossess_apes = []

    counter_peekss = []
    counter_frac_correct_takess = []

    lstm_hidden = None

    ## GET DATA
    for i in range(n_repeats_case):
        env.start_new_episode()
        state = env.get_state()

        rewards_steps = []
        saved_logits = []
        saved_taken_logits = []
        values = []
        actions = []
        controls = []
        control_errs = []
        ps_case = []
        actions_failed = []

        counter_sleeps = 0
        counter_peeks = 0
        counter_correct_takes = 0
        counter_total_takes = 0

        for j in range(config.n_steps_to_reward):

            if nn_options.ape_loss_coeff == 0 or not nn_options.hardcode_efficacy:
                logits, lstm_hidden, value, control = model(torch.tensor(state).to(device).float(), lstm_hidden)
            elif nn_options.hardcode_efficacy:
                logits, lstm_hidden, value, control = model(torch.tensor(state).to(device).float(), lstm_hidden, torch.tensor([env.taus['take']]).float().to(device))

            saved_logits.append(logits.cpu().detach().tolist())
            
            if nn_options.value_loss_coeff != 0:
                values.append(value.item())
            else:
                values.append(0)

            sampler = Categorical(logits=logits)
            action = sampler.sample()
            saved_taken_logits.append(sampler.log_prob(action).item())

            actions.append(action.item())

            new_state, reveal_rewards, tally_rewards, rewards_step, selected_action, action_failed, (steps, ps, taus) = env.step(action.item())
            
            ps_case.append(ps)
            rewards_steps.append(rewards_step)
            actions_failed.append(action_failed['take'])

            if nn_options.ape_loss_coeff != 0 and nn_options.ape_loss_coeff is not None:
                controls.append(control.item())
                control_errs.append(convert_control_to_prob_random(control.item()) - taus['take'])
            else:
                controls.append(None)
                control_errs.append(None)

            if i == 0 and j<10 and verbose:
                print('step %d: s: %s, a: %s / %s, sel: %s, failed: %s, fb: %s, rew: %s, ps: %s, taus: %s' %(j, state, action.item(), env.actions[action], selected_action, env.action_failed, env.feedback, rewards_step, ps, env.taus))

            action_desc = env.actions[action]
            if action_desc[0] == 'take' and action_desc[1] == -1:
                counter_sleeps += 1
            
            if action_desc[0] == 'peek':
                counter_peeks += 1

            if action_desc[1] != -1: #excluding sleep
                correct_arm = np.argmax(np.array(ps))
                if correct_arm == action_desc[1]:
                    counter_correct_takes += 1
                counter_total_takes += 1
            if reveal_rewards:
                assert j == config.n_steps_to_reward - 1, "revealing rewards at wrong step! step %d" %j
                rewardss.append(tally_rewards)

            state = new_state
        
        counter_sleepss.append(counter_sleeps)
        saved_logitss.append(saved_logits)
        valuess.append(values)
        actionss.append(actions)
        controlss.append(controls)
        control_errss.append(control_errs)
        pss.append(ps_case)

        counter_peekss.append(counter_peeks)

        if counter_total_takes != 0:
            counter_frac_correct_takess.append(counter_correct_takes / counter_total_takes)
        else:
            counter_frac_correct_takess.append(0)

        if "trialsize" in config and i % config.trialsize == 0:
            lstm_hidden = None

        loss_return, loss_ape = compute_losses(rewards_steps, saved_taken_logits, 1/3, controls, actions_failed)

        lossess_returns.append(loss_return)
        lossess_apes.append(loss_ape)

    rewardss = np.array(rewardss)
    counter_sleepss = np.array(counter_sleepss)
    saved_logitss = np.array(saved_logitss)
    valuess = np.array(valuess)
    actionss = np.array(actionss)
    controlss = np.array(controlss)
    pss = np.array(pss)
    lossess_returns = np.array(lossess_returns)
    lossess_apes = np.array(lossess_apes)
    control_errss = np.array(control_errss)

    counter_peekss = np.array(counter_peekss)
    counter_frac_correct_takess = np.array(counter_frac_correct_takess)

    if nn_options.ape_loss_coeff != 0 and nn_options.ape_loss_coeff is not None:
        control_errs = np.power(control_errss, 2)
        control_mse = np.mean(control_errs)
    else:
        control_mse = None

    #print(rewardss, counter_sleepss)

    #print('for bias %.2f and decay %.1f over the first 10 episodes the agent received reward %f and slept %f times' %(task_options.bias, task_options.alphas['take'], rewardss[:10].mean(), counter_sleepss[:10].mean()))
    #print('for bias %.2f and decay %.1f over the first 10 episodes the agent received reward %f and slept %f times' %(task_options.bias, task_options.alphas['take'], rewardss[:10].mean(), counter_sleepss[:10].mean()))

    #print('taus, efficacy, rewards, fraction correct takes, peeks, sleeps')
    #print(task_options.starting_taus['take'], 1 - task_options.starting_taus['take'], task_options.alphas['take'], rewardss.mean(), counter_frac_correct_takess.mean(), counter_peekss.mean(), counter_sleepss.mean())

    return (actionss, saved_logitss, actionss, valuess, controlss, pss, control_errs), (rewardss.mean(), counter_frac_correct_takess.mean(), counter_peekss.mean(), counter_sleepss.mean(), lossess_returns.mean(), lossess_apes.mean(), control_mse )

# %% TEST AND RETURN HIDDEN REPS

def test_return_hidden(config, nn_options, task_options, device, checkpoint = '', model_folder = None):
    ''' 
    Evaluates a trained model's performance on a variety of metrics
    over n_repeats_case episodes

    Arguments
    ---------
    model_folder : str or None, if None assumed that the folder is saved in config
    
    Returns
    -------
    trajectories: tuple of np.arrays giving information about the policy's trajectories
        statess : np.array of ints [n_repeats_case, n_steps_until_reward, n_features] giving state encoding
        logitss : np.array of floats [n_repeats_case, n_steps_until_reward, n_actions] giving logits outputted by policy network
        actionss : np.array of ints [n_repeats_case, n_steps_until_reward], gives chosen action sampled by logits in trial
        valuess : np.array of floats [n_repeats_case, n_steps_until_reward], gives value estimate given by network critic if applicable, else 0s
        controlss : np.array of floats [n_repeats_case, n_steps_until_reward],
        pss : np.array of floats[n_repeats_case, n_steps_until_reward, n_arms]
        hidden_statess : np.array of floats [n_repeats_case, n_steps_until_reward, n_hidden]

    stats: tuple giving summary statistics
        starting_taus : float, starting tau value
        efficacy : float, starting efficacy (1 - starting_taus)
        rewards : float, average rewards per episode
        counter_frac_correct_takes_mean : float, average fraction correct takes
        counter_peeks_mean : float, average number of peeks per episode
        counter_sleeps_mean : float, average number of sleeps per episode

    '''
    
    ## DEFINE ENVIRONMENT
    if config.env == 'PeekTakeTorchTask':
        env = PeekTakeTorchTask(**task_options)
    elif config.env == 'ObserveBetEfficacyTask':
        env = ObserveBetEfficacyTask(**task_options)
    else:
        assert False, "invalid envt"

    env.setup()

    if nn_options.ape_loss_coeff is None:
        model = PeekTakeTorchRNN(env.actions, env.encoding_size, **nn_options)
        
        if checkpoint == '':
            modelname = 'model.pt'
        else:
            modelname = 'model_%s.pt' %str(checkpoint)

        if model_folder is None:
            model_folder = config.model_folder

        model.load_state_dict(torch.load(os.path.join(model_folder, modelname),map_location=device))
    else:
        try:
            model = PeekTakeTorchAPERNN(env.actions, env.encoding_size, **nn_options)

            if checkpoint == '':
                modelname = 'model.pt'
            else:
                modelname = 'model_%s.pt' %str(checkpoint)

            if model_folder is None:
                model_folder = config.model_folder

            model.load_state_dict(torch.load(os.path.join(model_folder, modelname),map_location=device))
        except RuntimeError:
            model = PeekTakeTorchRNN(env.actions, env.encoding_size, **nn_options)
                        
            if checkpoint == '':
                modelname = 'model.pt'
            else:
                modelname = 'model_%s.pt' %str(checkpoint)

            if model_folder is None:
                model_folder = config.model_folder

            model.load_state_dict(torch.load(os.path.join(model_folder, modelname),map_location=device))

    #print(model)
    model.to(device)
    model.eval()

    rewardss = []
    counter_sleepss = []
    saved_logitss = []
    valuess = []
    actionss = []
    controlss = []
    pss = []
    hidden_statess = []
    cell_statess = []
    outss = []

    counter_peekss = []
    counter_frac_correct_takess = []

    lstm_hidden = None

    ## GET DATA
    for i in range(config.n_repeats_case):
        env.start_new_episode()
        state = env.get_state()

        saved_logits = []
        values = []
        actions = []
        controls = []
        ps_case = []
        hidden_states = []
        cell_states = []
        outs = []

        counter_sleeps = 0
        counter_peeks = 0
        counter_correct_takes = 0
        counter_total_takes = 0

        for j in range(config.n_steps_to_reward):

            if nn_options.ape_loss_coeff == 0 or not nn_options.hardcode_efficacy:
                logits, lstm_hidden, value, control, out = model.forward_return_all_hidden(torch.tensor(state).to(device).float(), lstm_hidden) ## Changed from forward
            elif nn_options.hardcode_efficacy:
                logits, lstm_hidden, value, control, out = model.forward_return_all_hidden(torch.tensor(state).to(device).float(), lstm_hidden, torch.tensor([env.taus['take']]).float().to(device)) ## Changed from forward

            saved_logits.append(logits.cpu().detach().tolist())
            hidden_states.append(lstm_hidden[0].cpu().detach().tolist())
            cell_states.append(lstm_hidden[1].cpu().detach().tolist())
            outs.append(out.cpu().detach().tolist())
            
            if nn_options.value_loss_coeff != 0:
                values.append(value.item())
            else:
                values.append(0)

            if nn_options.ape_loss_coeff != 0 and nn_options.ape_loss_coeff is not None:
                controls.append(control.item())
            else:
                controls.append(0)

            sampler = Categorical(logits=logits)
            action = sampler.sample()

            actions.append(action.item())

            new_state, reveal_rewards, tally_rewards, rewards_step, selected_action, action_failed, (steps, ps, taus) = env.step(action.item())
            
            ps_case.append(ps)

            if i == 0 and j<10:
                print('step %d: s: %s, a: %s / %s, sel: %s, failed: %s, fb: %s, rew: %s, ps: %s, taus: %s' %(j, state, action.item(), env.actions[action], selected_action, env.action_failed, env.feedback, rewards_step, ps, env.taus))

            action_desc = env.actions[action]
            if action_desc[0] == 'take' and action_desc[1] == -1:
                counter_sleeps += 1
            
            if action_desc[0] == 'peek':
                counter_peeks += 1

            if action_desc[1] != -1: #excluding sleep
                correct_arm = np.argmax(np.array(ps))
                if correct_arm == action_desc[1]:
                    counter_correct_takes += 1
                counter_total_takes += 1
            if reveal_rewards:
                assert j == config.n_steps_to_reward - 1, "revealing rewards at wrong step! step %d" %j
                rewardss.append(tally_rewards)

            state = new_state
        
        counter_sleepss.append(counter_sleeps)
        saved_logitss.append(saved_logits)
        valuess.append(values)
        actionss.append(actions)
        controlss.append(controls)
        pss.append(ps_case)
        hidden_statess.append(hidden_states)
        cell_statess.append(cell_states)
        outss.append(outs)

        counter_peekss.append(counter_peeks)

        if counter_total_takes != 0:
            counter_frac_correct_takess.append(counter_correct_takes / counter_total_takes)
        else:
            counter_frac_correct_takess.append(0)

        if "trialsize" in config and i % config.trialsize == 0:
            lstm_hidden = None

    rewardss = np.array(rewardss)
    counter_sleepss = np.array(counter_sleepss)
    saved_logitss = np.array(saved_logitss)
    valuess = np.array(valuess)
    actionss = np.array(actionss)
    controlss = np.array(controlss)
    pss = np.array(pss)

    hidden_statess = np.array(hidden_statess)
    cell_statess = np.array(cell_statess)
    outss = np.array(outss)

    counter_peekss = np.array(counter_peekss)
    counter_frac_correct_takess = np.array(counter_frac_correct_takess)

    #print(rewardss, counter_sleepss)

    #print('for bias %.2f and decay %.1f over the first 10 episodes the agent received reward %f and slept %f times' %(task_options.bias, task_options.alphas['take'], rewardss[:10].mean(), counter_sleepss[:10].mean()))
    #print('for bias %.2f and decay %.1f over the first 10 episodes the agent received reward %f and slept %f times' %(task_options.bias, task_options.alphas['take'], rewardss[:10].mean(), counter_sleepss[:10].mean()))

    #print('taus, efficacy, rewards, fraction correct takes, peeks, sleeps')
    print(task_options.starting_taus['take'], 1 - task_options.starting_taus['take'], task_options.alphas['take'], rewardss.mean(), counter_frac_correct_takess.mean(), counter_peekss.mean(), counter_sleepss.mean())

    return (None, saved_logitss, actionss, valuess, controlss, pss, hidden_statess, cell_statess, outss), (rewardss.mean(), counter_frac_correct_takess.mean(), counter_peekss.mean(), counter_sleepss.mean())

# %% HELPLESSNESS EVALUATION

def test_helplessness(config, nn_options, task_options, device, checkpoint = '', verbose=True, model_folder = None, taus_clamp = None, n_repeats_case=100):
    
    
    ## DEFINE ENVIRONMENT
    if config.env == 'PeekTakeTorchTask':
        env = PeekTakeTorchTask(**task_options)
    elif config.env == 'ObserveBetEfficacyTask':
        env = ObserveBetEfficacyTask(**task_options)
    else:
        assert False, "invalid envt"

    env.setup()

    if nn_options.ape_loss_coeff is None:
        model = PeekTakeTorchRNN(env.actions, env.encoding_size, **nn_options)
    else:
        model = PeekTakeTorchAPERNN(env.actions, env.encoding_size, **nn_options)

    if checkpoint == '':
        modelname = 'model.pt'
    else:
        modelname = 'model_%s.pt' %str(checkpoint)

    if model_folder is None:
        model_folder = config.model_folder

    model.load_state_dict(torch.load(os.path.join(model_folder, modelname),map_location=device))

    #print(model)
    model.to(device)
    model.eval()

    rewardss = []
    counter_sleepss = []
    saved_logitss = []
    valuess = []
    actionss = []
    controlss = []
    control_errss = []
    pss = []

    lossess_returns = []
    lossess_apes = []

    counter_peekss = []
    counter_frac_correct_takess = []

    lstm_hidden = None

    ## GET DATA
    for i in range(n_repeats_case):
        env.start_new_episode()
        state = env.get_state()

        rewards_steps = []
        saved_logits = []
        saved_taken_logits = []
        values = []
        actions = []
        controls = []
        control_errs = []
        ps_case = []
        actions_failed = []

        counter_sleeps = 0
        counter_peeks = 0
        counter_correct_takes = 0
        counter_total_takes = 0

        for j in range(config.n_steps_to_reward):

            if taus_clamp == 'random':
                clamped_tau = random.random()
            elif type(taus_clamp) == list:
                clamped_tau = random.choice(taus_clamp)
            else:
                clamped_tau = taus_clamp

            logits, lstm_hidden, value, control = model(torch.tensor(state).to(device).float(), lstm_hidden, torch.tensor([clamped_tau]).float().to(device))

            saved_logits.append(logits.cpu().detach().tolist())
            
            if nn_options.value_loss_coeff != 0:
                values.append(value.item())
            else:
                values.append(0)

            sampler = Categorical(logits=logits)
            action = sampler.sample()
            saved_taken_logits.append(sampler.log_prob(action).item())

            actions.append(action.item())

            new_state, reveal_rewards, tally_rewards, rewards_step, selected_action, action_failed, (steps, ps, taus) = env.step(action.item())
            
            ps_case.append(ps)
            rewards_steps.append(rewards_step)
            actions_failed.append(action_failed['take'])

            if nn_options.ape_loss_coeff != 0:
                controls.append(control.item())
                control_errs.append(convert_control_to_prob_random(control.item()) - taus['take'])
            else:
                controls.append(None)
                control_errs.append(None)

            if i == 0 and j<10 and verbose:
                print('step %d: s: %s, a: %s / %s, sel: %s, failed: %s, fb: %s, rew: %s, ps: %s, taus: %s' %(j, state, action.item(), env.actions[action], selected_action, env.action_failed, env.feedback, rewards_step, ps, env.taus))

            action_desc = env.actions[action]
            if action_desc[0] == 'take' and action_desc[1] == -1:
                counter_sleeps += 1
            
            if action_desc[0] == 'peek':
                counter_peeks += 1

            if action_desc[1] != -1: #excluding sleep
                correct_arm = np.argmax(np.array(ps))
                if correct_arm == action_desc[1]:
                    counter_correct_takes += 1
                counter_total_takes += 1
            if reveal_rewards:
                assert j == config.n_steps_to_reward - 1, "revealing rewards at wrong step! step %d" %j
                rewardss.append(tally_rewards)

            state = new_state
        
        counter_sleepss.append(counter_sleeps)
        saved_logitss.append(saved_logits)
        valuess.append(values)
        actionss.append(actions)
        controlss.append(controls)
        control_errss.append(control_errs)
        pss.append(ps_case)

        counter_peekss.append(counter_peeks)

        if counter_total_takes != 0:
            counter_frac_correct_takess.append(counter_correct_takes / counter_total_takes)
        else:
            counter_frac_correct_takess.append(0)

        if "trialsize" in config and i % config.trialsize == 0:
            lstm_hidden = None

        loss_return, loss_ape = compute_losses(rewards_steps, saved_taken_logits, 1/3, controls, actions_failed)

        lossess_returns.append(loss_return)
        lossess_apes.append(loss_ape)

    rewardss = np.array(rewardss)
    counter_sleepss = np.array(counter_sleepss)
    saved_logitss = np.array(saved_logitss)
    valuess = np.array(valuess)
    actionss = np.array(actionss)
    controlss = np.array(controlss)
    pss = np.array(pss)
    lossess_returns = np.array(lossess_returns)
    lossess_apes = np.array(lossess_apes)
    control_errss = np.array(control_errss)

    counter_peekss = np.array(counter_peekss)
    counter_frac_correct_takess = np.array(counter_frac_correct_takess)

    if nn_options.ape_loss_coeff != 0:
        control_errs = np.power(control_errss, 2)
        control_mse = np.mean(control_errs)
    else:
        control_mse = None

    #print(rewardss, counter_sleepss)

    #print('for bias %.2f and decay %.1f over the first 10 episodes the agent received reward %f and slept %f times' %(task_options.bias, task_options.alphas['take'], rewardss[:10].mean(), counter_sleepss[:10].mean()))
    #print('for bias %.2f and decay %.1f over the first 10 episodes the agent received reward %f and slept %f times' %(task_options.bias, task_options.alphas['take'], rewardss[:10].mean(), counter_sleepss[:10].mean()))

    #print('taus, efficacy, rewards, fraction correct takes, peeks, sleeps')
    #print(task_options.starting_taus['take'], 1 - task_options.starting_taus['take'], task_options.alphas['take'], rewardss.mean(), counter_frac_correct_takess.mean(), counter_peekss.mean(), counter_sleepss.mean())

    return (None, saved_logitss, actionss, valuess, controlss, pss, control_errs), (rewardss.mean(), counter_frac_correct_takess.mean(), counter_peekss.mean(), counter_sleepss.mean(), lossess_returns.mean(), lossess_apes.mean(), control_mse )

# %% TEST LOSS

def compute_test_loss():
    pass

# %% PERTURBED TEST

def perturbed_test(config, nn_options, task_options, device, checkpoint = '', model_folder = None, n_repeats_case = None, target_tau=1):
    ''' 
    Evaluates a trained model's performance on a variety of metrics
    over n_repeats_case episodes

    Arguments
    ---------
    model_folder : str or None, if None assumed that the folder is saved in config
    
    Returns
    -------
    trajectories: tuple of np.arrays giving information about the policy's trajectories
        statess : np.array of ints [n_repeats_case, n_steps_until_reward, n_features] giving state encoding
        logitss : np.array of floats [n_repeats_case, n_steps_until_reward, n_actions] giving logits outputted by policy network
        actionss : np.array of ints [n_repeats_case, n_steps_until_reward], gives chosen action sampled by logits in trial
        valuess : np.array of floats [n_repeats_case, n_steps_until_reward], gives value estimate given by network critic if applicable, else 0s
        controlss : np.array of floats [n_repeats_case, n_steps_until_reward],
        pss : np.array of floats [n_repeats_case, n_steps_until_reward, n_arms]
        squared control_errss : squared errors in efficacy prediction [n_repeats_case, n_steps_until_reward],
    
    stats: tuple giving summary statistics
        starting_taus : float, starting tau value
        efficacy : float, starting efficacy (1 - starting_taus)
        rewards : float, average rewards per episode
        counter_frac_correct_takes_mean : float, average fraction correct takes
        counter_peeks_mean : float, average number of peeks per episode
        counter_sleeps_mean : float, average number of sleeps per episode

    '''
    
    ## DEFINE ENVIRONMENT
    if config.env == 'PeekTakeTorchTask':
        env = PeekTakeTorchTask(**task_options)
    elif config.env == 'ObserveBetEfficacyTask':
        env = ObserveBetEfficacyTask(**task_options)
    else:
        assert False, "invalid envt"

    env.setup()

    model = PeekTakeTorchPerturbedAPERNN(env.actions, env.encoding_size, **nn_options)

    if checkpoint == '':
        modelname = 'model.pt'
    else:
        modelname = 'model_%s.pt' %str(checkpoint)

    if model_folder is None:
        model_folder = config.model_folder

    if n_repeats_case is None:
        n_repeats_case = config.n_repeats_case

    model.load_state_dict(torch.load(os.path.join(model_folder, modelname), map_location=device))
    #print(model)
    model.to(device)
    model.eval()

    rewardss = []
    counter_sleepss = []
    saved_logitss = []
    valuess = []
    actionss = []
    controlss = []
    control_errss = []
    pss = []

    lossess_returns = []
    lossess_apes = []

    counter_peekss = []
    counter_frac_correct_takess = []

    lstm_hidden = None

    ## GET DATA
    for i in range(n_repeats_case):
        env.start_new_episode()
        state = env.get_state()

        rewards_steps = []
        saved_logits = []
        saved_taken_logits = []
        values = []
        actions = []
        controls = []
        control_errs = []
        ps_case = []
        actions_failed = []

        counter_sleeps = 0
        counter_peeks = 0
        counter_correct_takes = 0
        counter_total_takes = 0

        for j in range(config.n_steps_to_reward):

            logits, lstm_hidden, value, control = model(torch.tensor(state).to(device).float(), lstm_hidden, torch.tensor([target_tau]).float().to(device))

            saved_logits.append(logits.cpu().detach().tolist())
            
            if nn_options.value_loss_coeff != 0:
                values.append(value.item())
            else:
                values.append(0)

            sampler = Categorical(logits=logits)
            action = sampler.sample()
            saved_taken_logits.append(sampler.log_prob(action).item())

            actions.append(action.item())

            new_state, reveal_rewards, tally_rewards, rewards_step, selected_action, action_failed, (steps, ps, taus) = env.step(action.item())
            
            ps_case.append(ps)
            rewards_steps.append(rewards_step)
            actions_failed.append(action_failed['take'])

            if nn_options.ape_loss_coeff != 0:
                controls.append(control.item())
                control_errs.append(control.item() - taus['take'])
            else:
                controls.append(None)
                control_errs.append(None)

            if i == 0 and j<10:
                print('step %d: s: %s, a: %s / %s, sel: %s, failed: %s, fb: %s, rew: %s, ps: %s, taus: %s' %(j, state, action.item(), env.actions[action], selected_action, env.action_failed, env.feedback, rewards_step, ps, env.taus))

            action_desc = env.actions[action]
            if action_desc[0] == 'take' and action_desc[1] == -1:
                counter_sleeps += 1
            
            if action_desc[0] == 'peek':
                counter_peeks += 1

            if action_desc[1] != -1: #excluding sleep
                correct_arm = np.argmax(np.array(ps))
                if correct_arm == action_desc[1]:
                    counter_correct_takes += 1
                counter_total_takes += 1
            if reveal_rewards:
                assert j == config.n_steps_to_reward - 1, "revealing rewards at wrong step! step %d" %j
                rewardss.append(tally_rewards)

            state = new_state
        
        counter_sleepss.append(counter_sleeps)
        saved_logitss.append(saved_logits)
        valuess.append(values)
        actionss.append(actions)
        controlss.append(controls)
        control_errss.append(control_errs)
        pss.append(ps_case)

        counter_peekss.append(counter_peeks)

        if counter_total_takes != 0:
            counter_frac_correct_takess.append(counter_correct_takes / counter_total_takes)
        else:
            counter_frac_correct_takess.append(0)
            
        if "trialsize" in config and i % config.trialsize == 0:
            lstm_hidden = None

        loss_return, loss_ape = compute_losses(rewards_steps, saved_taken_logits, 1/3, controls, actions_failed)

        lossess_returns.append(loss_return)
        lossess_apes.append(loss_ape)

    rewardss = np.array(rewardss)
    counter_sleepss = np.array(counter_sleepss)
    saved_logitss = np.array(saved_logitss)
    valuess = np.array(valuess)
    actionss = np.array(actionss)
    controlss = np.array(controlss)
    pss = np.array(pss)
    lossess_returns = np.array(lossess_returns)
    lossess_apes = np.array(lossess_apes)
    control_errss = np.array(control_errss)

    counter_peekss = np.array(counter_peekss)
    counter_frac_correct_takess = np.array(counter_frac_correct_takess)

    if nn_options.ape_loss_coeff != 0:
        control_errs = np.power(control_errss, 2)
        control_mse = np.mean(control_errs)
    else:
        control_mse = None

    #print(rewardss, counter_sleepss)

    #print('for bias %.2f and decay %.1f over the first 10 episodes the agent received reward %f and slept %f times' %(task_options.bias, task_options.alphas['take'], rewardss[:10].mean(), counter_sleepss[:10].mean()))
    #print('for bias %.2f and decay %.1f over the first 10 episodes the agent received reward %f and slept %f times' %(task_options.bias, task_options.alphas['take'], rewardss[:10].mean(), counter_sleepss[:10].mean()))

    #print('taus, efficacy, rewards, fraction correct takes, peeks, sleeps')
    #print(task_options.starting_taus['take'], 1 - task_options.starting_taus['take'], task_options.alphas['take'], rewardss.mean(), counter_frac_correct_takess.mean(), counter_peekss.mean(), counter_sleepss.mean())

    return (None, saved_logitss, actionss, valuess, controlss, pss, control_errs), (rewardss.mean(), counter_frac_correct_takess.mean(), counter_peekss.mean(), counter_sleepss.mean(), lossess_returns.mean(), lossess_apes.mean(), control_mse )

# %%
