# Kai Sandbrink
# 2022-10-01
# This script runs the decoding analyses for the Explore-Exploit Setting

# %% LIBRARY IMPORTS

import torch
import os, copy
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from test_analyses import plot_comparison_curves_several_runs

from save_model_representations import get_hidden_model_reps

from utils import get_timestamp

# %% PARAMETERS

# control_models = [20230107113326, 
#     20230107213512, 
#     20230107213627, 
#     20230107213634,
#     20230107213639
# ]

# ape_models = [ 20230107113249,
#     20230107214023,
#     20230107214150,
#     20230107214157,
#     20230107214201
# ]

# ape_models = [
#     20230323155549,
#     20230322225814,
#     20230322133845,
#     20230322133843,
#     20230322133841,
# ]

# # ape_models = []

# control_models = [
#     20230323155550,
#     20230322133850,
#     20230322133849,
#     20230322133848,
#     20230322133847,
# ]


# ape_models =  [
#     20230427201627,
#     20230427201629,
#     20230427201630,
#     20230427201632,
#     20230427201633,
#     20230427201644,
#     20230427201646,
#     20230427201647,
#     20230427201648,
#     20230427201649
# ]

# control_models = [
#     20230427201636,
#     20230427201637,
#     20230427201639,
#     20230427201640,
#     20230427201642,
#     20230427201657,
#     20230427201656,
#     20230427201655,
#     20230427201653,
#     20230427201652
# ]

from nns.settings_ana import pepe_nn_ape_models as ape_models, pepe_nn_control_models as control_models
#from settings_anal import pepe_human_control_models as human_control_models
#models = ape_models + control_models
from nns.settings_ana import pepe_nn_efficacy_at_input_models as efficacy_at_input_models
models = ape_models + control_models + efficacy_at_input_models

task = 'pepe'

timestamp = get_timestamp()
#base_data_folder = 'data/reps'
base_data_folder = None

#reps_folder_suffix = '20240311204516_1000cases'
reps_folder_suffix = None

# reps_folder_suffix = '20230110194453_100cases'
# reps_folder_suffix = '20230524191709_100cases'
# reps_folder_suffix = '20230524191845_100cases'

taus_train = [0, 0.125, 0.25, 0.75, 0.875, 1]
taus_test = [0.375, 0.5, 0.625]
taus = np.arange(0, 1.01, 0.125)

n_repeats_case = 100

n_checkpoints = 100

# %% FUNCTION TO PERFORM MODEL FITTING

def resize_neural_reps_efficacies(X, Y, decoding_start_time=None):
    ''' fits and evaluatess a decoding Ridge regression model to neural representations and efficacy values

    Arguments
    ---------
    X : np.array, shape (n_samples, n_timesteps, n_features), neural representations
    Y : np.array, shape (n_samples,), efficacy values
    decoding_start_time: int, start time for decoding. if None, assumed to be X_train.shape[1] // 2


    Returns
    -------
    X : np.array, shape (n_samples * n_timesteps, n_features), neural representations
    Y : np.array, shape (n_samples * n_timesteps,), efficacy values
    '''

    if decoding_start_time is None:
        decoding_start_time = X.shape[1] // 2

    X = X[:,decoding_start_time:,:]
    Y = np.repeat(Y[:, np.newaxis], X.shape[1], axis=1)

    X = X.reshape((-1, X.shape[2]))

    Y = Y.reshape((-1,))
    
    return X, Y

def fiteval_lm(X_train, X_test, Y_train, Y_test):
    ''' fits and evaluatess a decoding Ridge regression model to neural representations and efficacy values

    Arguments
    ---------
    X : np.array, shape (n_samples, n_timesteps, n_features), neural representations
    Y : np.array, shape (n_samples,), efficacy values
    decoding_start_time: int, start time for decoding. if None, assumed to be X_train.shape[1] // 2


    Returns
    -------
    tuple of (r2, mse) : metrics on test set
        r2 : fraction explained variance
        mse : mean squared error

    tuple of (train_r2, train_mse) : metrics on training set
        r2 : fraction explained variance
        mse : mean squared error
    '''

    lm = Ridge(alpha=1)
    lm.fit(X_train, Y_train)

    Y_pred = lm.predict(X_test)

    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)

    Y_train_pred = lm.predict(X_train)

    train_r2 = r2_score(Y_train, Y_train_pred)
    train_mse = mean_squared_error(Y_train, Y_train_pred)

    return (r2, mse), (train_r2, train_mse)

def fiteval_encoding(X, Y, decoding_start_time=None):
    ''' fits and evaluatess a decoding Ridge regression model to neural representations and efficacy values

    Arguments
    ---------
    X : np.array, shape (n_samples, n_timesteps, n_features), neural representations
    Y : np.array, shape (n_samples,), efficacy values
    decoding_start_time: int, start time for decoding. if None, assumed to be X_train.shape[1] // 2


    Returns
    -------
    tuple of (r2, mse) : metrics on test set
        r2 : fraction explained variance
        mse : mean squared error

    tuple of (train_r2, train_mse) : metrics on training set
        r2 : fraction explained variance
        mse : mean squared error
    '''

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    
    X_train, Y_train = resize_neural_reps_efficacies(X_train, Y_train, decoding_start_time=decoding_start_time)
    X_test, Y_test = resize_neural_reps_efficacies(X_test, Y_test, decoding_start_time=decoding_start_time)

    print("Encoding shapes")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    ### resize a second time to make 1D so that it is ready to switch X and Y
    Y_train = np.repeat(Y_train[:, np.newaxis], X_train.shape[1], axis=1)
    Y_test = np.repeat(Y_test[:, np.newaxis], X_test.shape[1], axis=1)

    print("added dims y")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    
    X_train = X_train.reshape((-1,))
    X_test = X_test.reshape((-1,))
    Y_train = Y_train.reshape((-1,1)) ## Reshaping to match for features
    Y_test = Y_test.reshape((-1,1))

    print('final reshape')
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    ## run LM with switched X and Y
    return fiteval_lm(Y_train, Y_test, X_train, X_test)
    
def fiteval_decoding(X, Y, decoding_start_time=None):
    ''' fits and evaluatess a decoding Ridge regression model to neural representations and efficacy values

    Arguments
    ---------
    X : np.array, shape (n_samples, n_timesteps, n_features), neural representations
    Y : np.array, shape (n_samples,), efficacy values
    decoding_start_time: int, start time for decoding. if None, assumed to be X_train.shape[1] // 2

    Returns
    -------
    tuple of (r2, mse) : metrics on test set
        r2 : fraction explained variance
        mse : mean squared error

    tuple of (train_r2, train_mse) : metrics on training set
        r2 : fraction explained variance
        mse : mean squared error
    '''

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    
    X_train, Y_train = resize_neural_reps_efficacies(X_train, Y_train, decoding_start_time=decoding_start_time)
    X_test, Y_test = resize_neural_reps_efficacies(X_test, Y_test, decoding_start_time=decoding_start_time)

    return fiteval_lm(X_train, X_test, Y_train, Y_test)

def fiteval_efficacy_decoding(modelname, checkpoint_str=''):
    ''' fits and evaluatess a decoding Ridge regression model to neural representations and efficacy values for given model

    Arguments
    ---------
    modelname : str, ID of model to run for

    Returns
    -------
    tuple of (r2, mse) : metrics on test set
        r2 : fraction explained variance
        mse : mean squared error

    tuple of (train_r2, train_mse) : metrics on training set
        r2 : fraction explained variance
        mse : mean squared error
    '''

    if base_data_folder is not None and reps_folder_suffix is not None:
        ### TODO: Update so that it works with subdivisions of data (or delete this option)
        # READ IN DATA
        model_data_folder = os.path.join(base_data_folder, modelname, reps_folder_suffix)

        cell_states = np.load(os.path.join(model_data_folder, 'cell_states%s.npy' %checkpoint_str))
        hidden_states = np.load(os.path.join(model_data_folder, 'hidden_states%s.npy' %checkpoint_str))

        efficacies = np.load(os.path.join(model_data_folder, 'efficacies%s.npy' %checkpoint_str)).flatten()
            
        # SELECT RELEVANT CHUNK OF DATA

        X = np.concatenate((cell_states, hidden_states), axis=2)

    else:

        # X_train = np.concatenate((X_train_taus_train, X_train_taus_test), axis=0)
        # X_test = np.concatenate((X_test_taus_train, X_test_taus_test), axis=0)

        # Y_train = np.concatenate((Y_train_taus_train, Y_train_taus_test))
        # Y_test = np.concatenate((Y_test_taus_train, Y_test_taus_test))

        cell_states, hidden_states, efficacies, outs = get_hidden_model_reps(modelname, taus, device='cpu', checkpoint=checkpoint_str, n_repeats_case=n_repeats_case)
        X = np.concatenate((cell_states, hidden_states), axis=2)

        cell_states_train, hidden_states_train, efficacies_train, outs_train = get_hidden_model_reps(modelname, taus_train, device='cpu', checkpoint=checkpoint_str, n_repeats_case=n_repeats_case)
        cell_states_test, hidden_states_test, efficacies_test, outs_test = get_hidden_model_reps(modelname, taus_test, device='cpu', checkpoint=checkpoint_str, n_repeats_case=n_repeats_case)

        X_taus_train = np.concatenate((cell_states_train, hidden_states_train), axis=2)
        X_taus_test = np.concatenate((cell_states_test, hidden_states_test), axis=2)
        
    ### DECODE EFFICACY FROM NEURAL REC REPS
    (r2, mse), (train_r2, train_mse) = fiteval_decoding(X, efficacies)
    (r2_test_taus_test, mse_test_taus_test), (r2_train_taus_test, mse_train_taus_test) = fiteval_decoding(X_taus_test, efficacies_test)
    (r2_test_taus_train, mse_test_taus_train), (r2_train_taus_train, mse_train_taus_train) = fiteval_decoding(X_taus_train, efficacies_train)

    ### DECODE EFFICACY FROM NEURAL FC REPS
    print("X",X.shape, efficacies.shape, X_taus_test.shape, efficacies_test.shape, X_taus_train.shape, efficacies_train.shape)
    print("OUTS",outs.shape, efficacies.shape, outs_test.shape, efficacies_test.shape, outs_train.shape, efficacies_train.shape)
    (r2_mse_fc, mse_mse_fc), (train_r2_mse_fc, train_mse_mse_fc) = fiteval_decoding(outs, efficacies)
    (r2_test_taus_test_fc, mse_test_taus_test_fc), (r2_train_taus_test_fc, mse_train_taus_test_fc) = fiteval_decoding(outs_test, efficacies_test)
    (r2_test_taus_train_fc, mse_test_taus_train_fc), (r2_train_taus_train_fc, mse_train_taus_train_fc) = fiteval_decoding(outs_train, efficacies_train)

    ### ENCODE EFFICACY IN NEURAL REC REPS
    (enc_r2, enc_mse), (enc_train_r2, enc_train_mse) = fiteval_encoding(X, efficacies)
    (enc_r2_test_taus_test, enc_mse_test_taus_test), (enc_r2_train_taus_test, enc_mse_train_taus_test) = fiteval_encoding(X_taus_test, efficacies_test)
    (enc_r2_test_taus_train, enc_mse_test_taus_train), (enc_r2_train_taus_train, enc_mse_train_taus_train) = fiteval_encoding(X_taus_train, efficacies_train)

    ### ENCODE EFFICACY IN NEURAL FC REPS
    (enc_r2_mse_fc, enc_mse_mse_fc), (enc_train_r2_mse_fc, enc_train_mse_mse_fc) = fiteval_encoding(outs, efficacies)
    (enc_r2_test_taus_test_fc, enc_mse_test_taus_test_fc), (enc_r2_train_taus_test_fc, enc_mse_train_taus_test_fc) = fiteval_encoding(outs_test, efficacies_test)
    (enc_r2_test_taus_train_fc, enc_mse_test_taus_train_fc), (enc_r2_train_taus_train_fc, enc_mse_train_taus_train_fc) = fiteval_encoding(outs_train, efficacies_train)

    return ((r2, mse), (train_r2, train_mse), (r2_test_taus_train, mse_test_taus_train), (r2_train_taus_train, mse_train_taus_train), (r2_test_taus_test, mse_test_taus_test), (r2_train_taus_test, mse_train_taus_test)), ((r2_mse_fc, mse_mse_fc), (train_r2_mse_fc, train_mse_mse_fc), (r2_test_taus_train_fc, mse_test_taus_train_fc), (r2_train_taus_train_fc, mse_train_taus_train_fc), (r2_test_taus_test_fc, mse_test_taus_test_fc), (r2_train_taus_test_fc, mse_train_taus_test_fc)), ((enc_r2, enc_mse), (enc_train_r2, enc_train_mse), (enc_r2_test_taus_train, enc_mse_test_taus_train), (enc_r2_train_taus_train, enc_mse_train_taus_train), (enc_r2_test_taus_test, enc_mse_test_taus_test), (enc_r2_train_taus_test, enc_mse_train_taus_test)), ((enc_r2_mse_fc, enc_mse_mse_fc), (enc_train_r2_mse_fc, enc_train_mse_mse_fc), (enc_r2_test_taus_train_fc, enc_mse_test_taus_train_fc), (enc_r2_train_taus_train_fc, enc_mse_train_taus_train_fc), (enc_r2_test_taus_test_fc, enc_mse_test_taus_test_fc), (enc_r2_train_taus_test_fc, enc_mse_train_taus_test_fc))

# %% PERFORM FOR ALL MODELS

def perform_fiteval_decoding_across_checkpoints(model, n_checkpoints):

    ape_model_r2scores = []
    ape_model_mses = []
    ape_model_train_r2scores = []
    ape_model_train_mses = []

    ape_model_train_taus_train_r2scores = []
    ape_model_train_taus_train_mses = []
    ape_model_test_taus_train_r2scores = []
    ape_model_test_taus_train_mses = []

    ape_model_train_taus_test_r2scores = []
    ape_model_train_taus_test_mses = []
    ape_model_test_taus_test_r2scores = []
    ape_model_test_taus_test_mses = []

    ape_model_r2scores_fc = []
    ape_model_mses_fc = []
    ape_model_train_r2scores_fc = []
    ape_model_train_mses_fc = []

    ape_model_train_taus_train_r2scores_fc = []
    ape_model_train_taus_train_mses_fc = []
    ape_model_test_taus_train_r2scores_fc = []
    ape_model_test_taus_train_mses_fc = []

    ape_model_train_taus_test_r2scores_fc = []
    ape_model_train_taus_test_mses_fc = []
    ape_model_test_taus_test_r2scores_fc = []
    ape_model_test_taus_test_mses_fc = []

    ape_model_enc_r2scores = []
    ape_model_enc_mses = []
    ape_model_train_enc_r2scores = []
    ape_model_train_enc_mses = []

    ape_model_train_taus_train_enc_r2scores = []
    ape_model_train_taus_train_enc_mses = []
    ape_model_test_taus_train_enc_r2scores = []
    ape_model_test_taus_train_enc_mses = []

    ape_model_train_taus_test_enc_r2scores = []
    ape_model_train_taus_test_enc_mses = []
    ape_model_test_taus_test_enc_r2scores = []
    ape_model_test_taus_test_enc_mses = []

    ape_model_enc_r2scores_fc = []
    ape_model_enc_mses_fc = []
    ape_model_train_enc_r2scores_fc = []
    ape_model_train_enc_mses_fc = []

    ape_model_train_taus_train_enc_r2scores_fc = []
    ape_model_train_taus_train_enc_mses_fc = []
    ape_model_test_taus_train_enc_r2scores_fc = []
    ape_model_test_taus_train_enc_mses_fc = []

    ape_model_train_taus_test_enc_r2scores_fc = []
    ape_model_train_taus_test_enc_mses_fc = []
    ape_model_test_taus_test_enc_r2scores_fc = []
    ape_model_test_taus_test_enc_mses_fc = []

    #for checkpoint_str in ['_' + str(int(i/n_checkpoints*100)) for i in range(n_checkpoints)] + ['']:
    for checkpoint_str in [str(int(i/n_checkpoints*100)) for i in range(n_checkpoints)] + ['']:

        #(r2, mse), (train_r2, train_mse), (r2_test_taus_train, mse_test_taus_train), (r2_train_taus_train, mse_train_taus_train), (r2_test_taus_test, mse_test_taus_test), (r2_train_taus_test, mse_train_taus_test)  = fiteval_efficacy_decoding(str(model), checkpoint_str=checkpoint_str)
        #(r2, mse), (train_r2, train_mse) = fiteval_efficacy_decoding(str(model), checkpoint_str=checkpoint_str)
        # ape_model_r2scores[-1].append(r2)
        # ape_model_mses[-1].append(mse)
        # ape_model_train_r2scores[-1].append(train_r2)
        # ape_model_train_mses[-1].append(train_mse)

        # ape_model_train_taus_train_r2scores[-1].append(r2_train_taus_train)
        # ape_model_train_taus_train_mses[-1].append(mse_train_taus_train)
        # ape_model_test_taus_train_r2scores[-1].append(r2_test_taus_train)
        # ape_model_test_taus_train_mses[-1].append(mse_test_taus_train)

        # ape_model_train_taus_test_r2scores[-1].append(r2_train_taus_test)
        # ape_model_train_taus_test_mses[-1].append(mse_train_taus_test)
        # ape_model_test_taus_test_r2scores[-1].append(r2_test_taus_test)
        # ape_model_test_taus_test_mses[-1].append(mse_test_taus_test)

        #((r2, mse), (train_r2, train_mse), (r2_test_taus_train, mse_test_taus_train), (r2_train_taus_train, mse_train_taus_train), (r2_test_taus_test, mse_test_taus_test), (r2_train_taus_test, mse_train_taus_test)), ((r2_mse_fc, mse_mse_fc), (train_r2_mse_fc, train_mse_mse_fc), (r2_test_taus_train_fc, mse_test_taus_train_fc), (r2_train_taus_train_fc, mse_train_taus_train_fc), (r2_test_taus_test_fc, mse_test_taus_test_fc), (r2_train_taus_test_fc, mse_train_taus_test_fc)) = fiteval_efficacy_decoding(str(model), checkpoint_str=checkpoint_str)
        ((r2, mse), (train_r2, train_mse), (r2_test_taus_train, mse_test_taus_train), (r2_train_taus_train, mse_train_taus_train), (r2_test_taus_test, mse_test_taus_test), (r2_train_taus_test, mse_train_taus_test)), ((r2_mse_fc, mse_mse_fc), (train_r2_mse_fc, train_mse_mse_fc), (r2_test_taus_train_fc, mse_test_taus_train_fc), (r2_train_taus_train_fc, mse_train_taus_train_fc), (r2_test_taus_test_fc, mse_test_taus_test_fc), (r2_train_taus_test_fc, mse_train_taus_test_fc)), ((r2_enc, mse_enc), (train_r2_enc, train_mse_enc), (r2_test_taus_train_enc, mse_test_taus_train_enc), (r2_train_taus_train_enc, mse_train_taus_train_enc), (r2_test_taus_test_enc, mse_test_taus_test_enc), (r2_train_taus_test_enc, mse_train_taus_test_enc)), ((r2_mse_fc_enc, mse_mse_fc_enc), (train_r2_mse_fc_enc, train_mse_mse_fc_enc), (r2_test_taus_train_fc_enc, mse_test_taus_train_fc_enc), (r2_train_taus_train_fc_enc, mse_train_taus_train_fc_enc), (r2_test_taus_test_fc_enc, mse_test_taus_test_fc_enc), (r2_train_taus_test_fc_enc, mse_train_taus_test_fc_enc)) = fiteval_efficacy_decoding(str(model), checkpoint_str=checkpoint_str)

        ape_model_r2scores.append(r2)
        ape_model_mses.append(mse)
        ape_model_train_r2scores.append(train_r2)
        ape_model_train_mses.append(train_mse)

        ape_model_train_taus_train_r2scores.append(r2_train_taus_train)
        ape_model_train_taus_train_mses.append(mse_train_taus_train)
        ape_model_test_taus_train_r2scores.append(r2_test_taus_train)
        ape_model_test_taus_train_mses.append(mse_test_taus_train)

        ape_model_train_taus_test_r2scores.append(r2_train_taus_test)
        ape_model_train_taus_test_mses.append(mse_train_taus_test)
        ape_model_test_taus_test_r2scores.append(r2_test_taus_test)
        ape_model_test_taus_test_mses.append(mse_test_taus_test)

        ape_model_r2scores_fc.append(r2_mse_fc)
        ape_model_mses_fc.append(mse_mse_fc)
        ape_model_train_r2scores_fc.append(train_r2_mse_fc)
        ape_model_train_mses_fc.append(train_mse_mse_fc)

        ape_model_train_taus_train_r2scores_fc.append(r2_train_taus_train_fc)
        ape_model_train_taus_train_mses_fc.append(mse_train_taus_train_fc)
        ape_model_test_taus_train_r2scores_fc.append(r2_test_taus_train_fc)
        ape_model_test_taus_train_mses_fc.append(mse_test_taus_train_fc)

        ape_model_train_taus_test_r2scores_fc.append(r2_train_taus_test_fc)
        ape_model_train_taus_test_mses_fc.append(mse_train_taus_test_fc)
        ape_model_test_taus_test_r2scores_fc.append(r2_test_taus_test_fc)
        ape_model_test_taus_test_mses_fc.append(mse_test_taus_test_fc)

        ape_model_enc_r2scores.append(r2_enc)
        ape_model_enc_mses.append(mse_enc)
        ape_model_train_enc_r2scores.append(train_r2_enc)
        ape_model_train_enc_mses.append(train_mse_enc)

        ape_model_train_taus_train_enc_r2scores.append(r2_train_taus_train_enc)
        ape_model_train_taus_train_enc_mses.append(mse_train_taus_train_enc)
        ape_model_test_taus_train_enc_r2scores.append(r2_test_taus_train_enc)
        ape_model_test_taus_train_enc_mses.append(mse_test_taus_train_enc)

        ape_model_train_taus_test_enc_r2scores.append(r2_train_taus_test_enc)
        ape_model_train_taus_test_enc_mses.append(mse_train_taus_test_enc)
        ape_model_test_taus_test_enc_r2scores.append(r2_test_taus_test_enc)
        ape_model_test_taus_test_enc_mses.append(mse_test_taus_test_enc)

        ape_model_enc_r2scores_fc.append(r2_mse_fc_enc)
        ape_model_enc_mses_fc.append(mse_mse_fc_enc)
        ape_model_train_enc_r2scores_fc.append(train_r2_mse_fc_enc)
        ape_model_train_enc_mses_fc.append(train_mse_mse_fc_enc)

        ape_model_train_taus_train_enc_r2scores_fc.append(r2_train_taus_train_fc_enc)
        ape_model_train_taus_train_enc_mses_fc.append(mse_train_taus_train_fc_enc)
        ape_model_test_taus_train_enc_r2scores_fc.append(r2_test_taus_train_fc_enc)
        ape_model_test_taus_train_enc_mses_fc.append(mse_test_taus_train_fc_enc)

        ape_model_train_taus_test_enc_r2scores_fc.append(r2_train_taus_test_fc_enc)
        ape_model_train_taus_test_enc_mses_fc.append(mse_train_taus_test_fc_enc)
        ape_model_test_taus_test_enc_r2scores_fc.append(r2_test_taus_test_fc_enc)
        ape_model_test_taus_test_enc_mses_fc.append(mse_test_taus_test_fc_enc)

    # %% SAVE

    results_folder = os.path.join('results', 'decoding', str(model))

    os.makedirs(results_folder, exist_ok=True)

    np.save(os.path.join(results_folder, '%s_ape_model_r2scores.npy' %timestamp), ape_model_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_mses.npy' %timestamp), ape_model_mses)
    np.save(os.path.join(results_folder, '%s_ape_model_train_r2scores.npy' %timestamp), ape_model_train_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_train_mses.npy' %timestamp), ape_model_train_mses)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_train_r2scores.npy' %timestamp), ape_model_train_taus_train_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_train_mses.npy' %timestamp), ape_model_train_taus_train_mses)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_train_r2scores.npy' %timestamp), ape_model_test_taus_train_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_train_mses.npy' %timestamp), ape_model_test_taus_train_mses)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_test_r2scores.npy' %timestamp), ape_model_train_taus_test_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_test_mses.npy' %timestamp), ape_model_train_taus_test_mses)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_test_r2scores.npy' %timestamp), ape_model_test_taus_test_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_test_mses.npy' %timestamp), ape_model_test_taus_test_mses)

    np.save(os.path.join(results_folder, '%s_ape_model_r2scores_fc.npy' %timestamp), ape_model_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_mses_fc.npy' %timestamp), ape_model_mses_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_r2scores_fc.npy' %timestamp), ape_model_train_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_mses_fc.npy' %timestamp), ape_model_train_mses_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_train_r2scores_fc.npy' %timestamp), ape_model_train_taus_train_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_train_mses_fc.npy' %timestamp), ape_model_train_taus_train_mses_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_train_r2scores_fc.npy' %timestamp), ape_model_test_taus_train_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_train_mses_fc.npy' %timestamp), ape_model_test_taus_train_mses_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_test_r2scores_fc.npy' %timestamp), ape_model_train_taus_test_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_test_mses_fc.npy' %timestamp), ape_model_train_taus_test_mses_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_test_r2scores_fc.npy' %timestamp), ape_model_test_taus_test_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_test_mses_fc.npy' %timestamp), ape_model_test_taus_test_mses_fc)

    np.save(os.path.join(results_folder, '%s_ape_model_enc_r2scores.npy' %timestamp), ape_model_enc_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_enc_mses.npy' %timestamp), ape_model_enc_mses)
    np.save(os.path.join(results_folder, '%s_ape_model_train_enc_r2scores.npy' %timestamp), ape_model_train_enc_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_train_enc_mses.npy' %timestamp), ape_model_train_enc_mses)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_train_enc_r2scores.npy' %timestamp), ape_model_train_taus_train_enc_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_train_enc_mses.npy' %timestamp), ape_model_train_taus_train_enc_mses)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_train_enc_r2scores.npy' %timestamp), ape_model_test_taus_train_enc_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_train_enc_mses.npy' %timestamp), ape_model_test_taus_train_enc_mses)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_test_enc_r2scores.npy' %timestamp), ape_model_train_taus_test_enc_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_test_enc_mses.npy' %timestamp), ape_model_train_taus_test_enc_mses)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_test_enc_r2scores.npy' %timestamp), ape_model_test_taus_test_enc_r2scores)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_test_enc_mses.npy' %timestamp), ape_model_test_taus_test_enc_mses)

    np.save(os.path.join(results_folder, '%s_ape_model_enc_r2scores_fc.npy' %timestamp), ape_model_enc_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_enc_mses_fc.npy' %timestamp), ape_model_enc_mses_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_enc_r2scores_fc.npy' %timestamp), ape_model_train_enc_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_enc_mses_fc.npy' %timestamp), ape_model_train_enc_mses_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_train_enc_r2scores_fc.npy' %timestamp), ape_model_train_taus_train_enc_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_train_enc_mses_fc.npy' %timestamp), ape_model_train_taus_train_enc_mses_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_train_enc_r2scores_fc.npy' %timestamp), ape_model_test_taus_train_enc_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_train_enc_mses_fc.npy' %timestamp), ape_model_test_taus_train_enc_mses_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_test_enc_r2scores_fc.npy' %timestamp), ape_model_train_taus_test_enc_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_train_taus_test_enc_mses_fc.npy' %timestamp), ape_model_train_taus_test_enc_mses_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_test_enc_r2scores_fc.npy' %timestamp), ape_model_test_taus_test_enc_r2scores_fc)
    np.save(os.path.join(results_folder, '%s_ape_model_test_taus_test_enc_mses_fc.npy' %timestamp), ape_model_test_taus_test_enc_mses_fc)
    

# %% START WITH MULTIPROCESSING
    
if __name__ == '__main__':
    from multiprocessing import Process

    processes = []

    i = 0

    for model in models:
        p = Process(target=perform_fiteval_decoding_across_checkpoints, args=(model, n_checkpoints))
        p.start()
        processes.append(p)

        i += 1

        if i % 5 == 0 or i == len(models):
            for p in processes:
                p.join()

# %% OLD


'''
for model in ape_models:
    ape_model_r2scores.append([])
    ape_model_mses.append([])
    ape_model_train_r2scores.append([])
    ape_model_train_mses.append([])

    ape_model_train_taus_train_r2scores.append([])
    ape_model_train_taus_train_mses.append([])
    ape_model_test_taus_train_r2scores.append([])
    ape_model_test_taus_train_mses.append([])

    ape_model_train_taus_test_r2scores.append([])
    ape_model_train_taus_test_mses.append([])
    ape_model_test_taus_test_r2scores.append([])
    ape_model_test_taus_test_mses.append([])
    for checkpoint_str in ['_' + str(int(i/n_checkpoints*100)) for i in range(n_checkpoints)] + ['']:

        (r2, mse), (train_r2, train_mse), (r2_test_taus_train, mse_test_taus_train), (r2_train_taus_train, mse_train_taus_train), (r2_test_taus_test, mse_test_taus_test), (r2_train_taus_test, mse_train_taus_test)  = fiteval_efficacy_decoding(str(model), checkpoint_str=checkpoint_str)
        ape_model_r2scores[-1].append(r2)
        ape_model_mses[-1].append(mse)
        ape_model_train_r2scores[-1].append(train_r2)
        ape_model_train_mses[-1].append(train_mse)

ape_model_r2scores = np.array(ape_model_r2scores)
ape_model_mses = np.array(ape_model_mses)
ape_model_train_r2scores = np.array(ape_model_train_r2scores)
ape_model_train_mses = np.array(ape_model_train_mses)

control_model_r2scores = []
control_model_mses = []
control_model_train_r2scores = []
control_model_train_mses = []
for model in control_models:
    control_model_r2scores.append([])
    control_model_mses.append([])
    control_model_train_r2scores.append([])
    control_model_train_mses.append([])
    for checkpoint_str in ['_' + str(int(i/n_checkpoints*100)) for i in range(n_checkpoints)] + ['']:

        (r2, mse), (train_r2, train_mse) = fiteval_efficacy_decoding(str(model), checkpoint_str=checkpoint_str)
        control_model_r2scores[-1].append(r2)
        control_model_mses[-1].append(mse)
        control_model_train_r2scores[-1].append(train_r2)
        control_model_train_mses[-1].append(train_mse)

control_model_r2scores = np.array(control_model_r2scores)
control_model_mses = np.array(control_model_mses)
control_model_train_r2scores = np.array(control_model_train_r2scores)
control_model_train_mses = np.array(control_model_train_mses)

# %% SAVE OUTPUT
        
results_folder = os.path.join('results', task, 'decoding')
os.makedirs(results_folder, exist_ok=True)

np.save(os.path.join(results_folder, '%s_ape_model_r2scores.npy' %timestamp), ape_model_r2scores)
np.save(os.path.join(results_folder, '%s_ape_model_mses.npy' %timestamp), ape_model_mses)
np.save(os.path.join(results_folder, '%s_ape_model_train_r2scores.npy' %timestamp), ape_model_train_r2scores)
np.save(os.path.join(results_folder, '%s_ape_model_train_mses.npy' %timestamp), ape_model_train_mses)

np.save(os.path.join(results_folder, '%s_control_model_r2scores.npy' %timestamp), control_model_r2scores)
np.save(os.path.join(results_folder, '%s_control_model_mses.npy' %timestamp), control_model_mses)
np.save(os.path.join(results_folder, '%s_control_model_train_r2scores.npy' %timestamp), control_model_train_r2scores)
np.save(os.path.join(results_folder, '%s_control_model_train_mses.npy' %timestamp), control_model_train_mses)

# %% PLOT OUTPUT

n_episodes = 500000

test_episodes = [n_episodes * i / n_checkpoints for i in range(n_checkpoints)] + [n_episodes]

fig = plot_comparison_curves_several_runs(test_episodes , np.array(ape_model_mses).T, test_episodes, np.array(control_model_mses).T, title='Decoding Loss', axis_xlabel='Training Episodes', axis_ylabel='MSE', label_exp='APE-trained', label_control="no APE")
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

#fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.png' %get_timestamp()))

analysis_folder = os.path.join('analysis', 'explore-exploit')
os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, '%s_decoding_loss_mse.png' %timestamp))
fig.savefig(os.path.join(analysis_folder, '%s_decoding_loss_mse.svg' %timestamp))

# %% PLOT OUTPUT

n_episodes = 500000

test_episodes = [n_episodes * i / n_checkpoints for i in range(n_checkpoints)] + [n_episodes]

fig = plot_comparison_curves_several_runs(test_episodes , np.array(ape_model_r2scores).T, test_episodes, np.array(control_model_r2scores).T, title='Decoding Loss', axis_xlabel='Training Episodes', axis_ylabel='r2', label_exp='APE-trained', label_control="no APE")
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

#fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.png' %get_timestamp()))

analysis_folder = os.path.join('analysis', 'explore-exploit')
os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, '%s_decoding_loss_r2.png' %timestamp))
fig.savefig(os.path.join(analysis_folder, '%s_decoding_loss_r2.svg' %timestamp))

# %%
'''