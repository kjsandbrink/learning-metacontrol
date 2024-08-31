# Kai Sandbrink
# 2022-10-18
# Script that contains training loop for given specifications (takes config file)

# %% LIBRARY IMPORTS

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os, random
import itertools

import wandb
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from PeekTakeTorchTask import PeekTakeTorchTask
from ObserveBetEfficacyTask import ObserveBetEfficacyTask
from PeekTakeTorchRNN import PeekTakeTorchRNN, PeekTakeTorchAPERNN

from utils import Config, plot_learning_curve, format_axis, flatten

# %% TRAINING LOOP

def train(config, nn_options, task_options, device):

    ## DEFINE ENVIRONMENT
    if config.env == 'PeekTakeTorchTask':
        env = PeekTakeTorchTask(**task_options)
    elif config.env == 'ObserveBetEfficacyTask':
        env = ObserveBetEfficacyTask(**task_options)
    else:
        assert False, "invalid envt"

    n_steps = config.n_steps_to_reward*config.n_episodes
    print('total number of steps: %d ...' %n_steps)

    if config.anneal_entropy == 'geom':
        n_batches = config.n_episodes // config.batchsize
        
        n_anneal = config.n_anneal_entropy_episodes // config.batchsize
        if n_anneal > n_batches:
            n_anneal = n_batches
            print("Warning: Re-adjusting n_anneal to be n_batches since n_batches > n_anneal")
        entropy_space = np.flip(np.geomspace(config.entropy_final_reg_coeff, config.entropy_reg_coeff, num=n_anneal,endpoint=True))
        entropy_space = np.concatenate((entropy_space, np.zeros(n_batches - n_anneal)))
        print('entropy space', entropy_space)

    elif config.anneal_entropy == 'lin':
        n_batches = config.n_episodes // config.batchsize
        n_anneal = n_batches*4//5
        #entropy_space = np.flip(np.linspace(0, 1.0, num= config.n_episodes// config.batchsize, endpoint=True))
        entropy_space = np.flip(np.linspace(0, 1.0, num= n_anneal, endpoint=True))
        entropy_space = np.concatenate((entropy_space, np.zeros(n_batches - n_anneal)))
        print('entropy space', entropy_space)

    else:
        entropy_reg_coeff = config.entropy_reg_coeff

    for i_r in range(config.n_runs):

        # %% FOR EACH RUN - INSTANTIATE MODEL AND TRAIN
        env.setup()

        if nn_options.ape_loss_coeff is None:
            model = PeekTakeTorchRNN(env.actions, env.encoding_size, **nn_options)
        else:
            model = PeekTakeTorchAPERNN(env.actions, env.encoding_size, **nn_options)

        model = model.to(device)
        print(model)

        ## project name to be used for both WANDB and the local folder
        project_name = datetime.now().strftime('%Y%m%d%H%M%S')
        results_folder = os.path.join('models', project_name)
        config.results_folder = results_folder

        print(dict(list(config.items()) + list(nn_options.items()) + list(task_options.items())))

        wandb.init(project="20220728_PeekTakeTorch", 
            name=project_name,
            config = dict(list(config.items()) + list(nn_options.items()) + list(task_options.items())),
            tags = config.tags,
            #mode='disabled'
            )

        #wandb.watch(model, log='all')
        wandb.watch(model)

        #TRAINING FUNCTIONS
        optimizer = torch.optim.Adam(model.parameters(), weight_decay = config.weight_decay)

        # LEARNING

        print("Starting timer")
        start_time = datetime.now()

        n_episode = 0 #actually this is tracking number of log steps, quarantined for deletion

        total_returns = []

        #lstm_hidden = model.init_hidden(device)
        lstm_hidden = None

        peek_steps_table = wandb.Table(columns=["episode","peek_steps", "clarity_steps", "taus_at_clarity", "efficacy_steps", "taus_at_efficacy"])

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
                apes = np.array(actions_failed_take)

                if nn_options.ape_loss_coeff is not None:
                    controls = torch.stack(controls, axis=0).to(device)
                pss = np.array(pss)
                #print(pss)
                #print(controls.reverse())
                #print(apes)
                returns = []
                ape_returns = []
                G = 0

                for i in reversed(range(len(rewards))):
                #for a, r, ps, c, ape in zip(action_descs[::-1], rewards[::-1], pss[::-1], torch.flip(controls, (0,)), apes[::-1]):
                    a = action_descs[i]
                    r = rewards[i]
                    ps = pss[i]
                    
                    if task_options.p_dist == 'opposites' or task_options.p_dist == 'uniform':
                        if (a[0] == 'peek' or a[1] == '-1') and config.baseline_type == 'take-only':
                            baseline = 0
                        elif type(config.baseline) == int or type(config.baseline) == float:
                            #baseline = 0.5
                            baseline = config.baseline
                        elif config.baseline == 'mean':
                            #print('computing mean baseline from %s : %f' %(str(ps), np.mean(ps)))
                            baseline = np.mean(ps)
                        elif config.baseline == '1/n_actions':
                            baseline = 1/env.n_actions
                        else:
                            assert False, "invalid baseline!"
                    else:
                        baseline = 0

                    #print("using baseline %f " %baseline)
                    #print(r, baseline, r-baseline)
                    G = r - baseline + config.discount*G
                    #G = r + config.discount*G
                    returns.append(G)

                    if nn_options.ape_loss_coeff is not None and not nn_options.hardcode_efficacy:
                        c = controls[i]

                        ##OPTION OF HARDCODED SIGNAL FOR HELPLESSNESS TRIALS 
                        if a[0] == 'take':
                            if config.harcode_prob_fail_signal is not None:
                                ape = int(random.random() < config.harcode_prob_fail_signal)
                            else:
                                ape = apes[i]

                            ape_returns.append(0.5*((c - ape).pow(2)))
                        
                        else:
                            ape_returns.append(0)

                returns.reverse()
                returns = torch.tensor(returns).to(device)

                if nn_options.ape_loss_coeff is not None and not nn_options.hardcode_efficacy:
                    ape_loss = sum(ape_returns)
                pseudo_loss = torch.sum( - saved_log_probs * returns)


            ## NEW METHOD WITH A2C
            # ref: https://github.com/BKHMSI/Meta-RL-Harlow/blob/65210fe28109cc4e2cdba41f9ea1e506208846cc/Harlow_1D/train.py#L302
            else:
                pseudo_loss = 0
                value_loss = 0
                ape_loss = 0
                    #this method uses iterative summing of rewards
                R = 0
                A = 0 #advantage estimate (total n-step)
                for i in reversed(range(len(rewards))):
                    R = rewards[i] + config.discount*R
                    advantage = R - valuess[i]

                    value_loss = value_loss + 0.5 * advantage.pow(2)

                    ## calculate n-step returns
                    #A = advantage + config.discount*A

                    pseudo_loss = pseudo_loss - saved_log_probs[i]*advantage

                    ape_loss = ape_loss + 0.5 * (actions_failed_take[i] - controls[i]).pow(2)

            log_losses = {'losses/returns_loss': pseudo_loss.cpu().detach().float()}

            ## ADD CRITIC LOSS TERM IF NECESSARY
            if nn_options.value_loss_coeff != 0:
                log_losses['losses/value_loss'] = value_loss.cpu().detach().float()

                pseudo_loss += nn_options.value_loss_coeff * value_loss

            if nn_options.ape_loss_coeff is not None and not nn_options.hardcode_efficacy:
                log_losses['losses/ape_loss'] = ape_loss.cpu().detach().float()
                pseudo_loss += nn_options.ape_loss_coeff * ape_loss[0]

                if config.ape_readout_l2_reg != 0:
                    #print("adding regularization to layer", list(model.parameters())[1])
                    log_losses['losses/ape_l2'] = (torch.linalg.norm(list(model.parameters())[2].flatten(), 2) + torch.linalg.norm(list(model.parameters())[3].flatten(), 2).cpu().detach().float())
                    pseudo_loss += config.ape_readout_l2_reg * (torch.linalg.norm(list(model.parameters())[2].flatten(), 2) + torch.linalg.norm(list(model.parameters())[3].flatten(), 2))
                        #weights and biases of ape_readout

            entropy = - (logitss * torch.exp(logitss)).sum()
            if config.anneal_entropy == 'geom':
                if i_e // config.batchsize < len(entropy_space):
                    entropy_reg_coeff = entropy_space[i_e // config.batchsize]
                else:
                    entropy_reg_coeff = 0
                pseudo_loss -= entropy_reg_coeff * entropy
            else:
                pseudo_loss -= entropy_reg_coeff * entropy #sign bc grad descent and not ascent

            log_losses['losses/entropy'] = entropy.cpu().detach().float()
            log_losses['losses/total_loss'] = pseudo_loss.cpu().detach().float()

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

                # compute average probs of actions
                policy = np.exp(np.squeeze(logitss.cpu().detach().numpy()))

                # compute average prob values over last episode
                pss = np.vstack(pss)

                ## compute dicts with metrics
                metrics = {"episodes/rewards": tally_rewards,
                    "episodes/rewards_baseline_offset": tally_rewards_offset,
                    "episodes/rewards_groundtruth_offset": tally_rewards_offset_groundtruth,
                    "episodes/groundtruth_baseline": np.concatenate(pss).mean()}

                action_probs = {'policy/'+str(a) : p for a, p in zip(env.actions, policy.mean(axis=0))}

                action_counters = {'actions/counter_peeks' : counter_peeks,
                    'actions/frac_correct_takes' : counter_correct_takes/counter_total_takes if counter_total_takes != 0 else 1,
                    'actions/frac_intended_correct_takes': counter_intended_correct_takes/counter_total_takes if counter_total_takes != 0 else 1,
                    'actions/counter_sleep_peeks' : counter_sleep_peeks,
                    'actions/counter_sleep_takes': counter_sleep_takes,
                    'actions/actions_failed_take': sum(actions_failed_take),
                    #'actions/peek_steps': np.array(peek_steps),
                    } 

                env_vars = {'environment/arm%d' %i : p for i , p in enumerate(pss.mean(axis=0))}
                if config.anneal_entropy == 'geom' or config.anneal_entropy == 'lin':
                    env_vars['environment/entropy_reg_coeff'] = entropy_reg_coeff

                ## SET UP BATCH-SPEC LOGGING
                if i_e > 0 and i_e % config.batchsize == 0:
                    batch_log['batch/peek_steps'] = wandb.Histogram(np_histogram=np.histogram(flatten(batch_peek_steps), bins = range(config.n_steps_to_reward+1)))
                    batch_log['batch/correct_take_steps'] = wandb.Histogram(np_histogram=np.histogram(flatten(batch_correct_take_steps), bins = range(config.n_steps_to_reward+1)))
                    batch_log['batch/incorrect_take_steps'] = wandb.Histogram(np_histogram=np.histogram(flatten(batch_incorrect_take_steps), bins = range(config.n_steps_to_reward+1)))
                    batch_log['batch/sleep_peek_steps'] = wandb.Histogram(np_histogram=np.histogram(flatten(batch_sleep_peek_steps), bins = range(config.n_steps_to_reward+1)))
                    batch_log['batch/sleep_take_steps'] = wandb.Histogram(np_histogram=np.histogram(flatten(batch_sleep_take_steps), bins = range(config.n_steps_to_reward+1)))
                    batch_peek_steps = []
                    batch_correct_take_steps = []
                    batch_incorrect_take_steps = []
                    batch_sleep_peek_steps = []
                    batch_sleep_take_steps = []

                ## wandb watch
                wandb.log({'episode': i_e} | metrics | action_probs | action_counters | env_vars | log_losses | batch_log)

                peek_steps_table.add_data(i_e, str(peek_steps), str(clarity_steps), str(taus_at_clarity), str(efficacy_steps), str(taus_at_efficacy))

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

        wandb.log({'peek_steps':peek_steps_table})

        print("Finished all episodes. Total time: %s" %(str(datetime.now() - start_time)))

        wandb.finish()