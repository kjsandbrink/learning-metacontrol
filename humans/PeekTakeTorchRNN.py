# Kai Sandbrink
# 2022-07-26
# PyTorch implementation of meta-learning RNN a la Wang et al (2016)

# Helpful reference from papers with code: https://github.com/BKHMSI/Meta-RL-Harlow/tree/master/models

# %% LIBRARY IMPORT

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# %% BASIC RNN

class PeekTakeTorchRNN(nn.Module):
    ### quarantined for deletion

    def __init__(self, action_space, input_size, lstm_hidden_size=32, hidden_size = None,
            value_loss_coeff = False, ape_loss_coeff = None, hardcode_efficacy = None, **kwargs
            ):
        super().__init__()
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.value_loss_coeff = value_loss_coeff
        self.hidden_size = hidden_size
        
        ## encode NN
        #torch.manual_seed(1)
        self.initialize_model()

    def initialize_model(self):
        #self.lstm = nn.LSTM(self.input_size, self.lstm_hidden_size, batch_first = True)
        self.lstm = nn.LSTMCell(self.input_size, self.lstm_hidden_size)

        if self.hidden_size is not None:
            self.fc = nn.Linear(self.lstm_hidden_size, self.hidden_size)
            self.nonlin = torch.nn.ReLU()
            self.out = nn.Linear(self.hidden_size, self.n_actions)
        else:
            self.out = nn.Linear(self.lstm_hidden_size, self.n_actions)
    
        self.logits = nn.LogSoftmax(dim=0)
            #Note: Version from PWC does not perform logsoftmax here

        if self.value_loss_coeff != 0:
            self.critic = nn.Linear(self.lstm_hidden_size, 1)


    def forward(self, x, lstm_hidden = None):
        if lstm_hidden is None:
            lstm_h, lstm_c = self.lstm(x)
        else:
            lstm_h, lstm_c = self.lstm(x, lstm_hidden)

        if self.hidden_size is not None:
            fc = self.fc(lstm_h)
            nonlin = self.nonlin(fc)
            out = self.out(nonlin)
        else:
            out = self.out(lstm_h)
        logits = self.logits(out)

        if self.value_loss_coeff != 0:
            values = self.critic(lstm_h)
        else:
            values = None

        return logits, (lstm_h, lstm_c), values, None
    
    def forward_return_all_hidden(self, x, lstm_hidden = None):
        if lstm_hidden is None:
            lstm_h, lstm_c = self.lstm(x)
        else:
            lstm_h, lstm_c = self.lstm(x, lstm_hidden)

        if self.hidden_size is not None:
            fc = self.fc(lstm_h)
            nonlin = self.nonlin(fc)
            out = self.out(nonlin)
        else:
            out = self.out(lstm_h)
            fc = None
        
        logits = self.logits(out)

        if self.value_loss_coeff != 0:
            values = self.critic(lstm_h)
        else:
            values = None

        return logits, (lstm_h, lstm_c), values, None, fc

# %% RNN with APE-MODULE

class PeekTakeTorchAPERNN(nn.Module):

    def __init__(self, action_space, input_size, lstm_hidden_size=32,
            value_loss_coeff = False, ape_loss_coeff = True, hidden_size = None, 
            hardcode_efficacy = False, harcode_prob_fail_signal = None
            ):
        super().__init__()
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.value_loss_coeff = value_loss_coeff
        self.ape_loss_coeff = ape_loss_coeff

        self.hidden_size = hidden_size

        self.hardcode_efficacy = hardcode_efficacy
        
        ## encode NN
        #torch.manual_seed(1)
        self.initialize_model()

    def initialize_model(self):
        #self.lstm = nn.LSTM(self.input_size, self.lstm_hidden_size, batch_first = True)
        self.lstm = nn.LSTMCell(self.input_size, self.lstm_hidden_size)

        if not self.hardcode_efficacy:
            self.control = nn.Linear(self.lstm_hidden_size, 1)

        ### IF FEEDFORWARD LAYER IS NEEDED CONNECT HERE
        if self.hidden_size is not None:
            self.fc = nn.Linear(self.lstm_hidden_size + 1, self.hidden_size)
            self.nonlin = torch.nn.ReLU()
            self.out = nn.Linear(self.hidden_size, self.n_actions)
        else:
            self.out = nn.Linear(self.lstm_hidden_size + 1, self.n_actions)
        self.logits = nn.LogSoftmax(dim=0)
            #Note: Version from PWC does not perform logsoftmax here

        if self.value_loss_coeff != 0:
            assert self.ape_loss_coeff == 0, "Critic not implemented for version with Efficacy yet"
            self.critic = nn.Linear(self.lstm_hidden_size, 1)

    def forward(self, x, lstm_hidden = None, in_efficacy = None):
        if lstm_hidden is None:
            lstm_h, lstm_c = self.lstm(x)
        else:
            lstm_h, lstm_c = self.lstm(x, lstm_hidden)

        #if not self.hardcode_efficacy or in_efficacy is not:
        if in_efficacy is None:
            control = self.control(lstm_h)
        else:
            control = in_efficacy
    
        hidden = torch.cat((lstm_h, control), dim=0)
        
        if self.hidden_size is not None:
            fc = self.fc(hidden)
            nonlin = self.nonlin(fc)
            out = self.out(nonlin)
        else:
            out = self.out(hidden)

        logits = self.logits(out)

        if self.value_loss_coeff != 0:
            values = self.critic(lstm_h)
        else:
            values = None

        return logits, (lstm_h, lstm_c), values, control
    
    def forward_return_all_hidden(self, x, lstm_hidden = None, in_efficacy = None):
        if lstm_hidden is None:
            lstm_h, lstm_c = self.lstm(x)
        else:
            lstm_h, lstm_c = self.lstm(x, lstm_hidden)

        #if not self.hardcode_efficacy or in_efficacy is not:
        if in_efficacy is None:
            control = self.control(lstm_h)
        else:
            control = in_efficacy
    
        hidden = torch.cat((lstm_h, control), dim=0)

        if self.hidden_size is not None:
            fc = self.fc(hidden)
            nonlin = self.nonlin(fc)
            out = self.out(nonlin)
        else:
            out = self.out(hidden)
            fc = None

        logits = self.logits(out)

        if self.value_loss_coeff != 0:
            values = self.critic(lstm_h)
        else:
            values = None

        return logits, (lstm_h, lstm_c), values, control, fc

# %% PERTURBED APERNN

class PeekTakeTorchPerturbedAPERNN(nn.Module):

    def __init__(self, action_space, input_size, lstm_hidden_size=32,
            value_loss_coeff = False, ape_loss_coeff = True, hidden_size = None, 
            hardcode_efficacy = False, harcode_prob_fail_signal = None
            ):
        super().__init__()
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.value_loss_coeff = value_loss_coeff
        self.ape_loss_coeff = ape_loss_coeff

        self.hidden_size = hidden_size

        self.hardcode_efficacy = hardcode_efficacy
        
        ## encode NN
        #torch.manual_seed(1)
        self.initialize_model()

    def initialize_model(self):
        #self.lstm = nn.LSTM(self.input_size, self.lstm_hidden_size, batch_first = True)
        self.lstm = nn.LSTMCell(self.input_size, self.lstm_hidden_size)

        if not self.hardcode_efficacy:
            self.control = nn.Linear(self.lstm_hidden_size, 1)

        ### IF FEEDFORWARD LAYER IS NEEDED CONNECT HERE
        if self.hidden_size is not None:
            self.fc = nn.Linear(self.lstm_hidden_size + 1, self.hidden_size)
            self.nonlin = torch.nn.ReLU()
            self.out = nn.Linear(self.hidden_size, self.n_actions)
        else:
            self.out = nn.Linear(self.lstm_hidden_size + 1, self.n_actions)
        self.logits = nn.LogSoftmax(dim=0)
            #Note: Version from PWC does not perform logsoftmax here

        if self.value_loss_coeff != 0:
            assert self.ape_loss_coeff == 0, "Critic not implemented for version with Efficacy yet"
            self.critic = nn.Linear(self.lstm_hidden_size, 1)

    def forward(self, x, lstm_hidden = None, target_tau = None):
        if lstm_hidden is None:
            lstm_h, lstm_c = self.lstm(x)
        else:
            lstm_h, lstm_c = self.lstm(x, lstm_hidden)

        control = self.control(lstm_h)
        
        #control_weights = self.control.weight

        lstm_h = lstm_h + self.control.weight * (target_tau-control)
        lstm_h = torch.flatten(lstm_h)
    
        hidden = torch.cat((lstm_h, target_tau), dim=0)

        ### IF FEEDFORWARD LAYER IS NEEDED CONNECT HERE
        ## TODO: Feedforward layer
        #bypass for now
        if self.hidden_size is not None:
            fc = self.fc(hidden)
            nonlin = self.nonlin(fc)
            out = self.out(nonlin)
        else:
            out = self.out(hidden)

        logits = self.logits(out)

        if self.value_loss_coeff != 0:
            values = self.critic(lstm_h)
        else:
            values = None

        return logits, (lstm_h, lstm_c), values, control