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

from utils import get_timestamp

# %% PARAMETERS

ape_models = [
    20230317215542,
    20230317215541,
    20230317215540,
    20230317215539,
    20230317215538,
]

control_models = [
    20230317215720,
    20230317215719,
    20230317215718,
    20230317215717,
    20230317215715,
]

base_data_folder = 'data/reps'

reps_folder_suffix = '20230416183157_100cases'

timestamp = get_timestamp()

# %% FUNCTION TO PERFORM MODEL FITTING

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

    # READ IN DATA
    model_data_folder = os.path.join(base_data_folder, modelname, reps_folder_suffix)

    cell_states = np.load(os.path.join(model_data_folder, 'cell_states%s.npy' %checkpoint_str))
    hidden_states = np.load(os.path.join(model_data_folder, 'hidden_states%s.npy' %checkpoint_str))

    efficacies = np.load(os.path.join(model_data_folder, 'efficacies%s.npy' %checkpoint_str)).flatten()

    # SELECT RELEVANT CHUNK OF DATA

    X = np.concatenate((cell_states, hidden_states), axis=2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, efficacies, test_size=0.20, random_state=42)

    n_decoding_timepoints = X_train.shape[1] // 2
    X_train = X_train[:,n_decoding_timepoints:,:]
    X_test = X_test[:,n_decoding_timepoints:,:]

    Y_train = np.repeat(Y_train[:, np.newaxis], X_train.shape[1], axis=1)
    Y_test = np.repeat(Y_test[:, np.newaxis], X_train.shape[1], axis=1)

    X_train = X_train.reshape((-1, X_train.shape[2]))
    X_test = X_test.reshape((-1, X_test.shape[2]))

    Y_train = Y_train.reshape((-1,))
    Y_test = Y_test.reshape((-1,))

    # PERFORM LINEAR REGRESSION

    lm = Ridge(alpha=1)
    lm.fit(X_train, Y_train)

    Y_pred = lm.predict(X_test)

    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)

    Y_train_pred = lm.predict(X_train)

    train_r2 = r2_score(Y_train, Y_train_pred)
    train_mse = mean_squared_error(Y_train, Y_train_pred)

    return (r2, mse), (train_r2, train_mse)

# %% PERFORM FOR ALL MODELS

ape_model_r2scores = []
ape_model_mses = []
ape_model_train_r2scores = []
ape_model_train_mses = []

for model in ape_models:
    ape_model_r2scores.append([])
    ape_model_mses.append([])
    ape_model_train_r2scores.append([])
    ape_model_train_mses.append([])
    for checkpoint_str in ['_' + str(int(i/10*100)) for i in range(10)] + ['']:

        (r2, mse), (train_r2, train_mse) = fiteval_efficacy_decoding(str(model), checkpoint_str=checkpoint_str)
        ape_model_r2scores[-1].append(r2)
        ape_model_mses[-1].append(mse)
        ape_model_train_r2scores[-1].append(train_r2)
        ape_model_train_mses[-1].append(train_mse)

control_model_r2scores = []
control_model_mses = []
control_model_train_r2scores = []
control_model_train_mses = []
for model in control_models:
    control_model_r2scores.append([])
    control_model_mses.append([])
    control_model_train_r2scores.append([])
    control_model_train_mses.append([])
    for checkpoint_str in ['_' + str(int(i/10*100)) for i in range(10)] + ['']:

        (r2, mse), (train_r2, train_mse) = fiteval_efficacy_decoding(str(model), checkpoint_str=checkpoint_str)
        control_model_r2scores[-1].append(r2)
        control_model_mses[-1].append(mse)
        control_model_train_r2scores[-1].append(train_r2)
        control_model_train_mses[-1].append(train_mse)

# %% PLOT OUTPUT

n_episodes = 500000

test_episodes = [n_episodes * i / 10 for i in range(10)] + [n_episodes]

fig = plot_comparison_curves_several_runs(test_episodes , np.array(ape_model_mses).T, test_episodes, np.array(control_model_mses).T, title='Decoding Loss', axis_xlabel='Training Episodes', axis_ylabel='MSE', label_exp='APE-trained', label_control="no APE")
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

#fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.png' %get_timestamp()))

analysis_folder = os.path.join('analysis', 'levc')
os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, '%s_decoding_loss_mse.png' %timestamp))
fig.savefig(os.path.join(analysis_folder, '%s_decoding_loss_mse.svg' %timestamp))

# %% PLOT OUTPUT

n_episodes = 500000

test_episodes = [n_episodes * i / 10 for i in range(10)] + [n_episodes]

fig = plot_comparison_curves_several_runs(test_episodes, np.array(ape_model_r2scores).T, test_episodes, np.array(control_model_r2scores).T, title='Decoding Loss', axis_xlabel='Training Episodes', axis_ylabel='r2', label_exp='APE-trained', label_control="no APE")
    # using reversed because outputting efficacy values -- assumes symmetric tau values lists

#fig.savefig(os.path.join(analysis_folder, '%s_rewards_efficacy.png' %get_timestamp()))

analysis_folder = os.path.join('analysis', 'levc')
os.makedirs(analysis_folder, exist_ok=True)
fig.savefig(os.path.join(analysis_folder, '%s_decoding_loss_r2.png' %timestamp))
fig.savefig(os.path.join(analysis_folder, '%s_decoding_loss_r2.svg' %timestamp))

# %%
