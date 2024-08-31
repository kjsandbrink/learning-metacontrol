# Kai Sandbrink
# 2022-08-23
# Baseline models for the torch version of the PeekTake task

# %% LIBRARY IMPORTS

import numpy as np
import random

# %% INITIALIZATIONS

modes = ['peek', 'take']

# %% RANDOM BASELINE

class RandomBaseline():

    def __init__(self, n_arms = 2):
        self.n_arms = n_arms
        self.n_actions = 3*n_arms

    def select_action(self, state):
        return np.random.randint(self.n_actions)

# %% REASONABLE BASELINE

class ReasonableBaseline():
    def __init__(self, n_arms = 2, sleep_freq = 0.5, peek_freq = 0.5,):
        self.n_arms = n_arms
        self.n_actions = 3*n_arms
        self.sleep_freq = sleep_freq
        self.peek_freq = peek_freq

        self.qs = np.zeros((self.n_arms,))
        self.counters_samples = np.zeros((self.n_arms,)) #count how often the arm has been sampled

    def select_action(self, state):
        
        ## incorporate feedback from previous action

        state = state.flatten()
        feedback = state[-2:]

        for i, f in enumerate(feedback):
            if f != 0:
                self.qs[i] = f
                self.counters_samples[i] += 1

        mode = modes[int(random.random() < 1 - self.peek_freq)] # 0 if peek 1 else

        ## choose new action
        if random.random() < self.sleep_freq: # i.e. sleep chosen
            arm = -1
        else:
            if mode == 'peek':
                arm = np.argmin(self.counters_samples)
                self.counters_samples[arm] += 1
            else:
                arm = np.argmax(self.qs)

        return (mode, arm)

# %% OPTIMAL AGENT FOR STATIC ENVIRONMENT

class OptimalAgentStatic():

    def __init__(self, n_arms = 3):
        self.n_arms = n_arms
        self.n_actions = 2 + 2*self.n_arms

        self.q_values = np.array([None]*self.n_arms)
    
    def select_action(self, state):

        feedback = state[-3-self.n_arms : -3]

        peek_failed = state[self.n_actions]

        ### WORKS UNDER ASSUMPTION OF MAGNITUDE-BASED PAYOFFS
        if any(feedback != 0) and not peek_failed:
            #print(state, feedback, peek_failed)
            assert sum(feedback != 0) == 1, "too many arms gave feedback"

            peeked_arm = np.where(feedback != 0)[0]
            self.q_values[peeked_arm] = feedback[peeked_arm]

        ## IF ALREADY FOUND CORRECT ARM SELECT IT OTHERWISE PEEK AGAIN
        # 0.99 assumed to be highest value
        if any(self.q_values == 0.99):
            arm_to_take = np.where(self.q_values == 0.99)[0]
            action = ('take', arm_to_take)
        else:
            arms_to_peek = [x for x in range(self.n_arms) if self.q_values[x] is None]
            arm_to_peek = random.choice(arms_to_peek)
            action = ('peek', arm_to_peek)

        return action
