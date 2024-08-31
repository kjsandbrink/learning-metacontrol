# Kai Sandbrink
# 2022-01-10
# This script saves the hidden layer representations of models for sample trajectories, to be used in e.g. a decoding analysis

# %% LIBRARY IMPORTS

import torch
import os, copy
import numpy as np

from utils import Config, get_timestamp
from utils_project import load_config_files

from test_case import test_return_hidden

from ObserveBetEfficacyTask import ObserveBetEfficacyTask

from PeekTakeTorchRNN import PeekTakeTorchAPERNN

# %% GET HIDDEN MODEL REPRESENTATIONS

def get_hidden_model_reps(modelname, test_taus, device, checkpoint='', n_repeats_case = 100):
    '''' gets hidden model representations and efficacies for given modelname, to be used in decoding analyses etc 
    
    Arguments
    ---------
    
    Returns
    -------

    '''

    
    ## TORCH
    base_results_folder = os.path.join('models')

    # nn_options = Config({})
    # nn_options.load_config_file(os.path.join(base_results_folder, modelname, 'nn_options.yaml'))

    config, task_options, nn_options = load_config_files(modelname)
    config.n_repeats_case = n_repeats_case

    hidden_statess = []
    cell_statess = []
    efficaciess = []
    outss = []

    for tau in test_taus:

        starting_taus = {'peek': 0, 'take': tau}
        current_task_options = Config(copy.deepcopy(task_options.__dict__))
        current_task_options.starting_taus = starting_taus

        (_, _, _, _, tau_outs, _, hidden_states, cell_states, outs), _ = test_return_hidden(config, nn_options, current_task_options, device, checkpoint=checkpoint, model_folder = os.path.join(base_results_folder , str(modelname)))

        hidden_statess.append(hidden_states)
        cell_statess.append(cell_states)
        outss.append(outs)

        efficaciess.extend([1- starting_taus['take']]*len(hidden_states))

        print('tau', tau, 'hidden_states', hidden_states.shape, 'cell_states', cell_states.shape, 'outs', outs.shape)

    hidden_statess = np.concatenate(hidden_statess)
    cell_statess = np.concatenate(cell_statess)
    efficaciess = np.array(efficaciess)
    outss = np.concatenate(outss)
    
    return hidden_statess, cell_statess, efficaciess, outss

# %% CALL FUNCTION

if __name__ == '__main__':
        
    # %% PARAMETERS AND INITIALIZATIONS
    # from settings_anal import pepe_nn_ape_models as ape_models, pepe_nn_control_models as control_models
    # models = ape_models + control_models
    from nns.settings_ana import pepe_nn_efficacy_at_input_models as models
    test_taus = np.arange(0, 1.01, 0.125)
    n_repeats_case = 100
    base_reps_folder = 'data/reps'
    n_checkpoints = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    timestamp = get_timestamp()

    for modelname in models:
    #for modelname in [20230107213639]:

        modelname = str(modelname)
            
        ### SAVE MODEL REPRESENTATIONS
        reps_folder = os.path.join(base_reps_folder, modelname, '%s_%dcases' %(timestamp, n_repeats_case))
        os.makedirs(reps_folder, exist_ok = True)
        print('folder', reps_folder)

        #for checkpoint in [str(int(i/n_checkpoints*100)) for i in range(n_checkpoints)] + ['']:
        for checkpoint in ['']:

            print('saving representations for checkpoint %s' %checkpoint)

            hidden_statess, cell_statess, efficaciess, fcss = get_hidden_model_reps(modelname, test_taus, device, checkpoint=checkpoint)

            if checkpoint != '':
                checkpoint_str = '_' + checkpoint
            else:
                checkpoint_str = ''

            np.save(os.path.join(reps_folder, 'hidden_states%s.npy' %checkpoint_str), hidden_statess)
            np.save(os.path.join(reps_folder, 'cell_states%s.npy' %checkpoint_str), cell_statess)
            np.save(os.path.join(reps_folder, 'efficacies%s.npy' %checkpoint_str), efficaciess)
            np.save(os.path.join(reps_folder, 'fcss%s.npy' %checkpoint_str), fcss)

# %%
