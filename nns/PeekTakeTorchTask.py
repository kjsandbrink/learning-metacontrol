# Kai Sandbrink
# 26.06.2022
# Implements a bandit task with k regular arms but stochastic transitions

from xml.etree.ElementInclude import include
import numpy as np
import random

from utils import sigmoid

class PeekTakeTorchTask():

    def __init__(self, n_arms = 3, alphas = {'peek': 0, 'take': 0},
            tiredness_form = {'peek': 'exp', 'take': 'exp'},
            include_sleep_actions= {'peek': True, 'take': True},
            sleep_factors = {'peek': 4, 'take': 4},
            increase_taus_factor = {'peek': 0, 'take': 0},
            max_tiredness_reached_after = {'peek': 4, 'take': 4}, 
            starting_taus = {'peek': 0, 'take': 0}, 
            fail_action = {'peek': 'switch', 'take': 'switch'}, #one of ['switch', 'obs_noise', 'fail]
            stationary=True, 
            drift_std = 0.1, n_steps_to_reward = 20, p_dist = 'uniform',
            reset_every_k_steps = 20, obs_noise = 0, noise_type = 'gaussian',
            payout_type = 'mag', actions_encoding_type = 'failed_flag', 
            reset_taus = False, encode_time = True, p_options=None,
            harcode_prob_fail_signal = None,
            ):# trialsize = 0):
        
        #set up dynamics of environment
        self.stationary = stationary
        self.tiredness_form = tiredness_form
        self.include_sleep_actions = include_sleep_actions
        self.alphas = alphas
        self.max_tiredness_reached_after = max_tiredness_reached_after
        self.increase_taus_factor = increase_taus_factor
        self.starting_taus = starting_taus
        self.fail_action = fail_action
        self.drift_std = drift_std
        self.n_steps_to_reward = n_steps_to_reward
        self.p_dist = p_dist
        self.sleep_factors = sleep_factors
        self.reset_every_k_steps = reset_every_k_steps
        self.obs_noise = obs_noise
        self.noise_type = noise_type
        self.payout_type = payout_type
        self.actions_encoding_type = actions_encoding_type
        self.reset_taus = reset_taus
        self.encode_time = encode_time
        self.p_options = p_options
        #self.trialsize = trialsize

        #define action space
        self.n_arms = n_arms
        self.define_actions()

        #encoding size
        ## if peek check fails, feedback is separated across other two arms => this is a noisy signal
        #self.include_trial_start = self.trialsize != 0 
        self.encoding_size = self.n_actions + self.n_arms + 3 + self.encode_time #common elements
            ## switch to +3 if I am including feedback_given into encoding

        if self.actions_encoding_type == 'failed_flag':
            self.encoding_size += 2
                # one_hot encoding of actions [n_actions]
                # plus encoding of action_failed for each arm [2]
                # plus encoding of if feedback given
                # plus encoding of feedback one per arm
                # plus end of episode flag
        elif self.actions_encoding_type == 'both_actions':
            self.encoding_size += self.n_actions
        elif self.actions_encoding_type == 'intended_only':
            pass
        elif self.actions_encoding_type == 'efficacy':
            self.encoding_size += 1
        elif self.actions_encoding_type == 'both_actions+efficacy':
            self.encoding_size += self.n_actions+1
        else:
            assert False, 'invalid encoding type %s' %self.actions_encoding_type

        ### TRIAL PARS
        '''
        if self.alphas is not None:
            self.reset_alphas = False
        else:
            self.alphas = {'peek': 0, 'take': 0}
            self.reset_alphas = True
        '''
        if self.reset_every_k_steps is None:
            self.reset_resets = True
        else:
            self.reset_resets = False
        if self.p_options is None and self.p_dist == 'options':
            self.p_options = [0.5, 0.5, 0.5]
            self.reset_p_options = True
        else:
            self.reset_p_options = False

    def setup(self):
        ''' to be initialized only at the beginning of experimental run '''
        #self.taus = {mode : 0 for mode in self.modes}
        self.steps = 0
        self.rewards_tally = 0
        self.tally_feedback = 0
        self.reveal_rewards_tally = False  

        #self.draw_ps()
        ## Setting up trial & episode options - this part can probably safely be deleted (as is rerun at start of trial/episode anyway)
        # kept for legacy reasons so that it matches with no-trial configuration
        self.new_trial = True
        self.draw_trial_pars()
        self.taus = self.starting_taus.copy()
            #done here also in case self.reset_taus is False
        
        self.start_new_episode()

    def draw_trial_pars(self):
        self.new_trial = True
        if self.reset_alphas:
            self.alphas['peek'] = random.random()*1
            self.alphas['take'] = random.random()*1
        if self.reset_resets:
            self.reset_every_k_steps = random.random()*1.0
        if self.reset_p_options:
            bias1 = random.choice([+1, -1])*(random.random()*0.5+0.5) - 0.5
            bias2 = random.random()*min(abs(1 - bias1), abs(bias1))*2

            p1 = 0.5 + bias1
            p2 = 0.5 - bias1/2 + bias2
            p3 = 0.5 - bias1/2 - bias2
            
            self.p_options = [p1, p2, p3]
            
        self.draw_ps()

    def start_new_episode(self):
        ''' to be repeated at the start of every episode '''
        #self.draw_ps() ##drawn based on 'self.reveal_rewards_tally' at start of episode automatically
        #self.taus = {mode : 0 for mode in self.modes}
        if self.reset_taus:
            self.taus = self.starting_taus.copy()
        
        ## if revealed previous reward, reset to 0
        self.rewards_tally = 0

        ## RESET OPTIONS THAT ARE USUALLY PART OF ENCODING
        self.action = None
        self.selected_action = None
        self.action_failed = None
        self.feedback_given = False
        self.action_failed = {'peek': False, 'take': False}
        self.action = None

    def draw_ps(self):
        if self.p_dist == 'uniform':
            self.ps = [random.random() for _ in range(self.n_arms)]
        elif self.p_dist == 'opposites':
            correct_arm = np.random.randint(self.n_arms)
            self.ps = [0.1] * self.n_arms
            self.ps[correct_arm] = 0.9
        elif self.p_dist == '+/-1':
            correct_arm = np.random.randint(self.n_arms)
            self.ps = [-1] * self.n_arms
            self.ps[correct_arm] = 1
        elif self.p_dist == 'options':
            self.ps = np.random.permutation(self.p_options)
        else:
            assert False, 'unknown p_dist'

    def define_actions(self):
        ''' action given by tuple 
        '''

        assert self.include_sleep_actions == {'peek': True, 'take': True}, 'not including slep actions is not implemented yet!'

        self.modes = ['peek', 'take']
        self.n_modes = len(self.modes)
        self.arms = np.arange(-1, self.n_arms) #-1 signalizes sleep, other actions signalize pulling a given arm
        self.actions =[(x, y) for x in self.modes for y in self.arms]
        self.n_actions = len(self.actions)

    def calc_failure_prob(self, mode):
        ''' for poly fail types '''
        #print('hi')
        #return self.alphas[mode] * np.exp(- self.taus[mode])
        if self.alphas[mode] != 0 and self.max_tiredness_reached_after[mode] != 0:
            if self.tiredness_form[mode] == 'poly':
                return (self.taus[mode]/self.max_tiredness_reached_after[mode]) ** (self.alphas[mode])
            elif self.tiredness_form[mode] == 'poly_limited_09':
                return min((self.taus[mode]/self.max_tiredness_reached_after[mode]) ** (self.alphas[mode]), 0.9)
            assert False, "invalid tiredness form"
        else:
            return 0

    def get_info(self):
        return self.steps, self.ps, self.taus.copy() #, self.action_failed

    def drift(self, p):
        '''Drift payout probabilities according to bounded reflective Brownian motion'''
        drift = np.random.normal(scale=self.drift_std)
        p += drift

        if p > 1:
            p = 2 - p
        elif p < 0:
            p = -p
        
        return p

    def pull_arm(self, arm):
        ''' return reward if it is achieved based on action and reward type.

        Arguments
        ---------
        arm : int or None, the action taken by the agent, 
            where None signifies that no arm has been chosen (i.e. an action failed)

        Returns
        -------
        reward : float, reward achieved by the agent

        '''
        #print(action)

        if arm is not None:
            p = self.ps[arm]
            if self.payout_type == 'mag':
                return p
            elif self.payout_type == 'prob':
                #print('returning reward with prob p')
                return int(random.random() < p)
            else:
                assert False, 'invalid payout type'
        else:
            return 0

    def select_arm(self, mode, intended_arm):
        ''' returns the selected arm intention, i.e. checks for and applies fail 
        
        Arguments
        ---------

        Returns
        -------
        selected_arm: int, one of 0:n_arms, gives index of selected arm
        action_failed: bool, was the chosen arrm successful
        
        '''
        ### NEW VERSION: FAILURE ACTIONS CALCULATED SEPARATELY FOR BOTH MODES
        ### CHECK TO SEE IF ACTION FAILS

        action_failed = False

        if self.tiredness_form[mode] == 'poly' or self.tiredness_form[mode] == 'cst' or self.tiredness_form[mode] == 'poly_limited_09':

            if random.random() < self.calc_failure_prob(mode):
                #action_failed = True
                if self.fail_action[mode] == 'fail':
                    selected_arm = None
                elif self.fail_action[mode] == 'switch':
                    selected_arm = np.random.randint(self.n_arms)
                else:
                    assert False, 'fail action not implemented'
            else:
                selected_arm = intended_arm

        elif self.tiredness_form[mode] == 'sig':

            base_prob = 1/self.n_arms
            p_int = base_prob + (1-base_prob)*sigmoid(self.alphas[mode]*self.taus[mode]-np.log(1/0.95-1))
                ### for probability of success of 97.5% for tau=0
            p_other = (1-p_int)/(self.n_arms-1)

            ps_arms = [p_other]*self.n_arms
            ps_arms[intended_arm] = p_int

            selected_arm = np.random.choice(self.n_arms, p=ps_arms)

        
        elif self.tiredness_form[mode] == 'exp':
            base_prob = 1/self.n_arms
            p_int = base_prob + (1-base_prob)*np.exp(-self.alphas[mode]*self.taus[mode])
            p_other = (1-p_int)/(self.n_arms-1)

            ps_arms = [p_other]*self.n_arms
            ps_arms[intended_arm] = p_int

            selected_arm = np.random.choice(self.n_arms, p=ps_arms)

        elif self.tiredness_form[mode] == '1-exp' or self.tiredness_form[mode] == '1-exp_limited':
            if self.tiredness_form[mode] == '1-exp_limited':
                A = 0.9
            else:
                A = 1

            base_prob = 1/self.n_arms
            p_int = base_prob + (1-base_prob)*A*(1-1*np.exp(-self.alphas[mode]*self.taus[mode]))
            p_other = (1-p_int)/(self.n_arms-1)

            ps_arms = [p_other]*self.n_arms
            ps_arms[intended_arm] = p_int

            selected_arm = np.random.choice(self.n_arms, p=ps_arms)

            #print(self.n_arms, base_prob, p_int, p_other, ps_arms, selected_arm)

        else:
            assert False, 'incorrect tiredness type'

        ## additional standard fail cases
        if selected_arm is None or selected_arm != intended_arm:
            action_failed = True

        return selected_arm, action_failed

    def encode_action(self, action):
        ''' encodes action in one hot format'''
        encoding = np.zeros((self.n_actions,))
        if action is not None:
            encoding[(self.n_arms + 1 )*(action[0] == 'take') + action[1] + 1] = 1

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
        #encoding[self.n_actions] = int(self.action_successful)
        #if self.action is not None and self.action[1] != -1:
        #    encoding[self.n_actions + 1 + self.action[1]] = self.feedback
        #encoding[self.n_actions + 2] = self.reveal_rewards_tally
        #encoding[self.n_actions + 3] = self.tally_feedback

        #if self.action is not None and self.action[0] == 'peek':
        #    encoding[self.n_actions + 1: ] = self.ps
        
        ## always on
        #encoding[self.n_actions + 1: ] = self.ps

        #return encoding.reshape((1,1,-1))

        ### for version with action_failed
        if self.actions_encoding_type == 'failed_flag':
            encoding[self.n_actions] = self.action_failed['peek']
            encoding[self.n_actions + 1] = self.action_failed['take']
            current_encoding_length = self.n_actions + 2
        elif self.actions_encoding_type == 'both_actions':
            encoding[self.n_actions:self.n_actions*2] = self.encode_action(self.selected_action)
            current_encoding_length = self.n_actions*2
        elif self.actions_encoding_type == 'intended_only':
            current_encoding_length = self.n_actions
        else:
            assert False, 'invalid encoding type %s' %self.actions_encoding_type

        ## encode time
        if self.encode_time:
            encoding[current_encoding_length] = (self.n_steps_to_reward - (self.steps % self.n_steps_to_reward))/self.n_steps_to_reward
            current_encoding_length += 1

        ## encode remaining common elements
        #encoding[current_encoding_length] = self.feedback_given
        if self.action is not None and self.action[1] != -1:
            #encoding[current_encoding_length + 1:current_encoding_length + 1 + self.n_arms] = self.feedback
            encoding[current_encoding_length:current_encoding_length + self.n_arms] = self.feedback
        #encoding[-2 - self.include_trial_start] = self.reveal_rewards_tally
        #encoding[-1 - self.include_trial_start] = self.tally_feedback      
        encoding[-3] = self.reveal_rewards_tally
        encoding[-2] = self.tally_feedback        
        encoding[-1] = self.new_trial
        
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
        state : np.array of ints [1,1,enc_length], encoding of state
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
        self.feedback = [0]*self.n_arms
        rewards_step = 0
        self.feedback_given = False
        self.action_failed = {'peek': False, 'take': False}

        # if an arm is chosen
        if action[1] != -1:

            ## determine if bandit was taken successfully; if not, action fails
            #if random.random() > self.calc_failure_prob(action[0]): #i.e. action successful

            selected_arm, self.action_failed[action[0]] = self.select_arm(*action)
            rewards_step = self.pull_arm(selected_arm)

            ### EXECUTE ACTION
            if action[0] == 'take':
                self.rewards_tally += rewards_step

            else:
                self.feedback_given = True

                ### IF NECESSARY, SWITCH ARM
                '''
                ### add failure prob due to sleeping ##TODO: combine with above obs noise
                if self.payout_type == 'mag' and self.alphas['peek'] > 0:
                    if self.p_dist == 'opposites':
                        if random.random() < self.calc_failure_prob('peek'):
                            self.action_failed['peek'] = True
                            if self.fail_action['peek'] == 'obs_noise':
                                rewards_step = feedback = 0.5
                            elif self.fail_action['peek'] == 'switch':
                                random_arm = np.random.randint(self.n_arms)
                                self.feedback = self.pull_arm(random_arm)
                            else:
                                assert False, 'fail action %s not implemented' %self.fail_action['peek']
                            #print('action failed due to clarity')
                '''

                if not self.action_failed[action[0]]:
                    self.feedback[selected_arm] = rewards_step
                else:

                    ## this is to signify that the received signal could be in the other two arms
                    self.feedback = np.ones(self.n_arms,)*rewards_step

                    ## in the polynomial tiredness form, the signal in failed setting could come from any of the 3 arms
                    ## in the exponential form, it could only come from one of the other two (bc decay down to 0 prob)
                    if self.tiredness_form[action[0]] == 'exp':
                        self.feedback[action[1]] = 0

                rewards_step = 0

                ### add obs noise if necessary, of a type depending on the distribution
                if self.payout_type == 'mag' and self.obs_noise != 0:
                    if self.p_dist == 'uniform':
                        self.feedback += np.random.normal(scale = self.obs_noise)
                        self.feedback = max(min(self.feedback, 1),0)

                    elif self.p_dist == 'opposites':
                        if random.random() < self.obs_noise:
                            self.feedback = 0.5

                    elif self.p_dist == '+/-1' and self.noise_type == 'gaussian':
                        if self.obs_noise > 0:
                            self.feedback += np.random.normal(scale = self.obs_noise)

                    else:
                        assert False, 'invalid p_dist and/or noise_type'

            self.taus[action[0]] = self.taus[action[0]] + self.increase_taus_factor[action[0]]
            if self.tiredness_form[action[0]] == 'poly':
                self.taus[action[0]] = min(self.taus[action[0]], self.max_tiredness_reached_after[action[0]])

        else:
            if self.sleep_factors[action[0]] == 'max':
                self.taus[action[0]] = 0
            else:
                self.taus[action[0]] -= self.sleep_factors[action[0]]
                self.taus[action[0]] = max(self.taus[action[0]], 0)
            #self.action_failed[action[0]] = False

        ## perform action, including check to see if reward is obtained
        self.steps += 1
        self.new_trial = False

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