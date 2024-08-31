# Kai Sandbrink
# 26.06.2022
# Implements a bandit task with k regular arms but stochastic transitions

import numpy as np
import random

from PeekTakeTorchTask import PeekTakeTorchTask

class ObserveBetEfficacyTask(PeekTakeTorchTask):

    def __init__(self, **kwargs):

        #print("Setting bias to 0")
        self.bias = kwargs.pop('bias')
        if self.bias is not None:
            self.reset_bias = False
        else:
            self.reset_bias = True
            self.bias = 0.5
        
        super().__init__(**kwargs)
        self.payout_type = 'prob'

        if self.actions_encoding_type == 'failed_flag':
            self.encoding_size = self.n_actions + self.n_arms + 4
            assert False, 'need to double check encoding size, might be 1 less for failed_flag'
        elif self.actions_encoding_type == 'efficacy':
            self.encoding_size = self.n_actions + self.n_arms + 4
        elif self.actions_encoding_type == 'both_actions+efficacy':
            self.encoding_size = self.n_actions*2 + self.n_arms + 4
        else:
            self.encoding_size = self.n_actions*2 + self.n_arms + 3
        
        if type(self.alphas) == list or type(self.alphas) == str or self.alphas['take'] is None:
            self.reset_alpha = True
            if type(self.alphas) == list or type(self.alphas) == str:
                self.alphas_options = self.alphas
            else:
                self.alphas_options = None
            self.alphas = {'peek': 0, 'take': 0} #initialize
        else:
            self.reset_alpha = False

        if type(self.starting_taus) == list or self.starting_taus == "uniform_held-out-middle" or self.starting_taus is None or self.starting_taus == "inv_uniform_held-out-middle" or self.starting_taus == 'uniform_no-holdout':
            self.reset_starting_taus = True
            self.starting_taus_options = self.starting_taus
            self.starting_taus = {'peek': 0, 'take': 0} #initialize
        else:
            self.reset_starting_taus = False

    def draw_trial_pars(self):
        if self.reset_alpha:
            if self.alphas_options is None:
                self.alphas['take'] = random.random()
            elif self.alphas_options == 'uniform_held-out-middle':
                alpha = random.random()

                ## for total range between 0 and 1.5
                '''
                if alpha >= 0.5:
                    alpha = alpha + 1
                '''

                ## for total range between 0 and 1
                if alpha < 0.5:
                    alpha = alpha * 2/3
                else:
                    alpha = (alpha - 0.5) * 2/3 + 2/3

                self.alphas['take'] = alpha

            else:
                self.alphas = random.choice(self.alphas_options)
        if self.reset_bias:
            self.bias = random.random()/2
        if self.reset_starting_taus:
            if self.starting_taus_options is None:
                self.starting_taus = {'peek': 0, 'take': random.random()}
            elif type(self.starting_taus_options) == list:
                self.starting_taus = random.choice(self.starting_taus_options)
            elif self.starting_taus_options == "uniform_held-out-middle":
                starting_tau = random.random()
                if starting_tau < 0.5:
                    starting_tau = starting_tau * 2/3
                else:
                    starting_tau = (starting_tau - 0.5) * 2/3 + 2/3
                self.starting_taus = {'peek': 0, 'take': starting_tau}
            elif self.starting_taus_options == "inv_uniform_held-out-middle":
                starting_p = random.random()

                if starting_p < 0.5:
                    starting_p = starting_p * 2/3
                else:
                    starting_p = (starting_p - 0.5) * 2/3 + 2/3
                
                starting_tau = -1/self.alphas['take'] * np.log(1 - starting_p)

                self.starting_taus = {'peek': 0, 'take': starting_tau}
            elif self.starting_taus_options == 'uniform_no-holdout':
                self.starting_taus = {'peek': 0, 'take': random.random()}
                print('no holdout, resampled tau from uniform,', self.starting_taus)
            else:
                assert False, "invalid starting_taus_options"
        self.draw_ps()

    def draw_ps(self):
        bias_multiple = np.random.choice([+1, -1])
        self.ps = [0.5+bias_multiple*self.bias,  0.5-bias_multiple*self.bias]

    def define_actions(self):
        ''' action given by tuple 
        '''
        self.modes = ['peek', 'take']
        self.n_modes = len(self.modes)
        if self.include_sleep_actions['take']:
            self.arms = np.arange(-1, self.n_arms) #-1 signalizes sleep, other actions signalize pulling a given arm
        else:
            self.arms = np.arange(self.n_arms)
        self.actions = [('peek', 0)] + [('take', y) for y in self.arms]
        self.n_actions = len(self.actions)

    def encode_action(self, action):
        ''' encodes action in one hot format'''
        encoding = np.zeros((self.n_actions,))
        if action is not None and action[0] == 'take':
            encoding[(2)*(action[0] == 'take') - int(not self.include_sleep_actions['take']) + action[1]] = 1
        elif action is not None:
            encoding[0] = 1

        return encoding
        
    def get_state(self):
        ''' return current state 
        
        Returns
        -------
        encoding : np.array of floats [n_arms*3 + 2 + 1 + n_arms + 1] :
            first n_arms*3 : one-hot of chosen action on previous move
            next two : boolean if the action was successful, first for peek then for take
            next 1 : boolean if feedback was given
            next n_arms : feedback + corresponding arm
            final 1 : boolean, did episode switch or
        '''
        encoding = np.zeros((self.encoding_size,))
        encoding[0:self.n_actions] = self.encode_action(self.action)

        ### for version with action_failed
        if self.actions_encoding_type == 'failed_flag':
            if self.action is not None and self.action[0] == 'take':
                encoding[self.n_actions] = self.action_failed['take']
            actions_encoding_length = self.n_actions
        # elif self.actions_encoding_type == 'both_actions':
        #     encoding[self.n_actions:self.n_actions*2] = self.encode_action(self.selected_action)
        #     actions_encoding_length = self.n_actions*2
        # elif self.actions_encoding_type == 'intended_only':
        #     actions_encoding_length = self.n_actions
        elif self.actions_encoding_type == 'both_actions' or self.actions_encoding_type == 'both_actions+efficacy':
            if self.action is not None:
                #encoding[self.n_actions:self.n_actions*2] = self.encode_action((self.action[0], self.selected_arm))
                encoding[self.n_actions:self.n_actions*2] = self.encode_action(self.selected_action)
            actions_encoding_length = self.n_actions*2 - 1
            
            if self.actions_encoding_type == 'both_actions+efficacy':
                encoding[actions_encoding_length + 1] = self.taus['take']
                actions_encoding_length += 1
        elif self.actions_encoding_type == 'efficacy':
            encoding[self.n_actions] = self.taus['take']
            actions_encoding_length = self.n_actions
        else:
            assert False, 'invalid encoding type %s' %self.actions_encoding_type
        
        if not self.encode_time:
            encoding[actions_encoding_length + 1] = self.feedback_given
        else:
            encoding[actions_encoding_length + 1] = (self.n_steps_to_reward - (self.steps % self.n_steps_to_reward))/self.n_steps_to_reward
        if self.action is not None and self.action[1] != -1:
            encoding[actions_encoding_length + 2:actions_encoding_length + 4] = self.feedback
        encoding[-2] = self.reveal_rewards_tally
        encoding[-1] = self.tally_feedback
        
        return encoding

    def step(self, action):
        """Implements an OpenAI gym type function

        Arguments
        ---------
        action : tuple [2] or integer index; if tuple
            action[0] : str, one of ['peek', 'take']
            action[1] : int, chosen arm or -1 for sleep

        Returns
        -------
        state : np.array of ints [1, 1, enc_length], encoding of state
        reveal_rewards_tally : bool, True if reward revealed this turn
        tally_feedback : float, rewards tally if revealed or 0 else
        rewards_step : float, instantaneous rewards revealed by the system
        info : tuple, output of get_info() with various information for debugging
        """

        ## SAVE MOVE INFO TO BE RETURNED BEFORE ACTION IS TAKEN AND VALUES ARE CHANGED 
        info = self.get_info()

        #print(type(action))
        if type(action) == int:
            action = self.actions[action]

        self.action = action
        self.selected_action = action

        ## re-initialize feedback values
        self.feedback = [0, 0]
        rewards_step = 0
        self.feedback_given = False
        self.action_failed = {'peek': False, 'take': False}

        # if an arm is chosen
        if action[1] != -1:

            ## determine if bandit was taken successfully; if not, action fails
            #if random.random() > self.calc_failure_prob(action[0]): #i.e. action successful

            ### EXECUTE ACTION
            if action[0] == 'take':
                self.selected_arm, self.action_failed[action[0]] = self.select_arm(*action)
                rewards_step = self.pull_arm(self.selected_arm)
                self.selected_action = (self.selected_action[0], self.selected_arm)
                self.rewards_tally += rewards_step

            else:
                self.feedback_given = True
                successful_arm = 1 - int(random.random() < self.ps[0])

                self.feedback[successful_arm] = 1
                rewards_step = 0

            if self.tiredness_form[action[0]] != 'cst':
                self.taus[action[0]] = self.taus[action[0]] + self.increase_taus_factor[action[0]]
                if self.tiredness_form[action[0]] == 'poly' or self.tiredness_form[action[0]] == 'poly_limited_09':
                    self.taus[action[0]] = min(self.taus[action[0]], self.max_tiredness_reached_after[action[0]])

        elif self.tiredness_form[action[0]] != 'cst':
            if self.sleep_factors[action[0]] == 'max':
                self.taus[action[0]] = 0
            else:
                self.taus[action[0]] -= self.sleep_factors[action[0]]
                self.taus[action[0]] = max(self.taus[action[0]], 0)
            #self.action_failed[action[0]] = False

        ## perform action, including check to see if reward is obtained
        self.steps += 1

        ## if end of episode, reveal reward
        if self.steps % self.n_steps_to_reward == 0:
            self.reveal_rewards_tally = True
            self.tally_feedback = self.rewards_tally
            self.rewards_tally = 0
        else:
            self.reveal_rewards_tally = False
            self.tally_feedback = 0

        ### END OF MOVE ACTIONS (PREPARATION FOR NEXT STEP)

        ## reset probabilities if necessary
        if self.reveal_rewards_tally or (self.reset_every_k_steps != 0 and ((self.reset_every_k_steps >= 1 and self.steps % self.reset_every_k_steps == 0) or (self.reset_every_k_steps < 1 and random.random() < self.reset_every_k_steps))):
            #print('re-drawing probabilities after %d steps' %self.steps)
            self.draw_ps()

        ## if process is not stationary, drift probabilities
        if not self.stationary:
            self.ps = [self.drift(p) for p in self.ps]

        return self.get_state(), self.reveal_rewards_tally, self.tally_feedback, rewards_step, self.selected_action, self.action_failed, info