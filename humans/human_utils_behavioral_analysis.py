# Kai Sandbrink
# 2023-07-31
# This script provides common utils for the analysis of the learning experiment.

# %% LIBRARY IMPORT

import numpy as np
import pandas as pd
import os, pickle

from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import Ridge, Lasso

from matplotlib.ticker import FuncFormatter

import seaborn as sns
from matplotlib import pyplot as plt

from human_utils_project import sort_train_test

from factor_analyzer import FactorAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
from statsmodels.tools.tools import pinv_extended

import statsmodels.formula.api as smf

from pingouin import partial_corr

# %% FUNCTIONS FOR DATA LOADING

def get_mask_behav(df, signed_dev_cutoff_lower = -3, signed_dev_cutoff_higher = 10,
            effs_train = [0.125, 0.375, 0.5, 0.625, 0.875], effs_test = [0, 0.25, 0.75, 1]):
    ''' Return a mask based on the following criteria: Mean deviation from the average number of observations 
    on the test set is greater than signed_dev_cutoff_lower and less than signed_dev_cutoff_higher,
    where the cutoffs are visually determined to be distribution changes in histogram
    
    Arguments
    ---------
    df : pandas.DataFrame
        The dataframe containing the data.
    signed_dev_cutoff_lower : float
        The lower cutoff for the mean deviation from the average number of observations on the test set.
    signed_dev_cutoff_higher : float
        The higher cutoff for the mean deviation from the average number of observations on the test set.
    effs_train : list
        The list of training efficacy values.
    effs_test : list
        The list of test efficacy values.

    Returns
    -------
    mask : numpy.ndarray of bools
        The mask for the dataframe, where True indicates the value is within the normal range of observations.
    '''
    
    eobs = np.flip(np.array([7.10666667, 6.76555556, 6.22444444, 5.22,       3.95444444, 2.76777778,
        1.89555556, 1.27333333, 1.10666667]))
    
    effs = np.sort(np.array(list(effs_train) + list(effs_test)))

    test_start = len(effs_train)

    nobs_train, nobs_test = sort_train_test(df['n_observes'], df['effs'], test_start)
        
    eobs_train = get_evs_wanted(eobs, effs, effs_train)
    eobs_test = get_evs_wanted(eobs, effs, effs_test)

    signed_dev_obs_train = np.mean(nobs_train - eobs_train, axis=1)
    signed_dev_obs_test = np.mean(nobs_test - eobs_test, axis=1)
        
    mask_group_low = signed_dev_obs_test < signed_dev_cutoff_lower
    mask_group_high = signed_dev_obs_test > signed_dev_cutoff_higher

    mask = ~mask_group_low & ~mask_group_high

    print("Size of low group: %i" %np.sum(mask_group_low))
    print("Size of high group: %i" %np.sum(mask_group_high))

    return mask

    
# %% FUNCTIONS FOR DATA ANALYSIS

def mean_slope_train_test(metric, effs, effs_sorted_train, effs_sorted_test):
    test_start = len(effs_sorted_train)

    obs_sorted_train, obs_sorted_test = sort_train_test(metric, effs, test_start)
    #mean_slope_ostr, mean_slope_oste = np.diff(obs_sorted_train, axis=1).mean(axis=1), np.diff(obs_sorted_test, axis=1).mean(axis=1)

    delta_effs_train = np.diff(effs_sorted_train)
    delta_effs_test = np.diff(effs_sorted_test)

    delta_ostr = np.diff(obs_sorted_train, axis=1)
    delta_oste = np.diff(obs_sorted_test, axis=1)

    mean_slope_ostr, mean_slope_oste = (delta_ostr / delta_effs_train).mean(axis=1), (delta_oste / delta_effs_test).mean(axis=1)

    return mean_slope_ostr, mean_slope_oste
# %%

def get_evs_wanted(evs, effs_all, effs_wanted):
    evs_wanted = []
    for i, e in enumerate(effs_all):
        #print(e, effs_wanted, e == effs_wanted)
        if e in np.array(effs_wanted):
            evs_wanted.append(evs[i])
    return np.array(evs_wanted)

# %% 

def calc_dev_behavior(series, series_effs, ebeh, aggregate_efficacies = False, use_abs = False, effs_train = [0.125, 0.375, 0.5, 0.625, 0.875], effs_test = [0, 0.25, 0.75, 1]):
    ''' Calculate deviation of participant in behavioral metric from average behavioral metric for each efficacy value passed in by parameter (usually NN)

    Arguments
    ---------
    series : pd.Series
        The dataframe containing the observation data.
    series_effs : pd.Series
        The dataframe containing the efficacy data.
    ebeh : numpy.ndarray
        The average behavioral metric for each efficacy value.
    aggregate_efficacies : bool
        Whether to aggregate the efficacy values. 
    use_abs : bool
        Whether to use the absolute deviation or the signed deviation.
    effs_train : list
        The list of training efficacy values.
    effs_test : list
        The list of test efficacy values.

    Returns
    -------
    dev_obs_train : numpy.ndarray
        The deviation of the behavioral metric on the training set. (If signed or abs depends on input.)
    dev_obs_test : numpy.ndarray
        The deviation of the behavioral metric on the test set. (If signed or abs depends on input.)
    
    '''
    
    test_start = len(effs_train)
    effs = np.sort(np.array(list(effs_train) + list(effs_test)))

    nobs_train, nobs_test = sort_train_test(series, series_effs, test_start)

    eobs_train = get_evs_wanted(ebeh, effs, effs_train)
    eobs_test = get_evs_wanted(ebeh, effs, effs_test)

    if aggregate_efficacies:
        if not use_abs:
            dev_obs_train = np.mean(nobs_train - eobs_train, axis=1)
            dev_obs_test = np.mean(nobs_test - eobs_test, axis=1)

        else:
            dev_obs_train = np.mean(np.abs(nobs_train - eobs_train), axis=1)
            dev_obs_test = np.mean(np.abs(nobs_test - eobs_test), axis=1)
    else:
        if not use_abs:
            dev_obs_train = nobs_train - eobs_train
            dev_obs_test = nobs_test - eobs_test

        else:
            dev_obs_train = np.abs(nobs_train - eobs_train)
            dev_obs_test = np.abs(nobs_test - eobs_test)

    return dev_obs_train.tolist(), dev_obs_test.tolist()

# %%

def compute_2D_correlation_matrices(col1, col2, effs_col1, effs_col2,):
    ''' Computes 2D Correlation Matrices and associated pv values 
    
    Arguments
    ---------
    col1 : pd.Series or np.ndarray
        The first column of data.
        Note the assumption that either col1, col2, effs_col1, effs_col2 are all pd.Series or all np.ndarray
    col2 : pd.Series or np.ndarray
        The second column of data.
    effs_col1 : pd.Series or np.ndarray
        The efficacies associated with the first column of data.
    effs_col2 : pd.Series or np.ndarray
        The efficacies associated with the second column of data.

    Returns
    -------
    corr_matrix : np.ndarray
        The correlation matrix.
    pvs_matrix : np.ndarray
        The p-values matrix.   
    
    '''

    if isinstance(col1, pd.DataFrame) or isinstance(col1, pd.Series):

        ### only keep those rows where both A and B are not np.nan
        mask = col1.apply(lambda x: type(x) == np.ndarray or type(x) == list) & col2.apply(lambda x: type(x) == np.ndarray or type(x) == list)
        col1 = col1[mask]
        col2 = col2[mask]
        effs_col1 = effs_col1[mask]
        effs_col2 = effs_col2[mask]

        ### sort the col1 and col2 by the effs_col1 and effs_col2
        # col1 = col1.apply(lambda x, idx: sort_array_by_another(x, effs_col1.loc[idx]), args=(effs_col1.index,))
        # col2 = col2.apply(lambda x, idx: sort_array_by_another(x, effs_col2.loc[idx]), args=(effs_col2.index,))
        sorted_col1 = pd.Series(dtype=object)
        sorted_col2 = pd.Series(dtype=object)

        for idx in col1.index:
            sorted_col1.at[idx] = col1.loc[idx][np.argsort(effs_col1.loc[idx])]
            sorted_col2.at[idx] = col2.loc[idx][np.argsort(effs_col2.loc[idx])]

        col1 = sorted_col1
        col2 = sorted_col2

        A = np.stack(col1.values)
        B = np.stack(col2.values)

    else:
        ### sort the col1 and col2 by the effs_col1 and effs_col2
        ## we want to sort the columns by the efficacies on a row-by-row basis

        ## tile the efficacies to match the shape of the columns if necessary
        if len(effs_col1.shape) == 1:
            effs_col1 = np.tile(effs_col1, (col1.shape[0], 1))
            effs_col2 = np.tile(effs_col2, (col2.shape[0], 1))

        sorted_col1 = []
        sorted_col2 = []

        print(effs_col1.shape)

        for i in range(len(col1)):
            sorted_col1.append(col1[i][np.argsort(effs_col1[i])])
            sorted_col2.append(col2[i][np.argsort(effs_col2[i])])

        A = np.stack(sorted_col1)
        B = np.stack(sorted_col2)

    mask = ~np.isnan(A).any(axis=1) & ~np.isnan(B).any(axis=1)
    A = A[mask]
    B = B[mask]

    n_efficacies = A.shape[1]
    
    corr_matrix = np.zeros((n_efficacies, n_efficacies))
    pvs_matrix = np.zeros((n_efficacies, n_efficacies))
    
    for i in range(n_efficacies):
        for j in range(n_efficacies):
            corr, pv = pearsonr(A[:,i], B[:,j])
            if not np.isnan(corr):
                corr_matrix[i, j] = corr
            else:
                corr_matrix[i, j] = 0
            if not np.isnan(pv):
                pvs_matrix[i, j] = pv
            else:
                pvs_matrix[i, j] = 0

    return corr_matrix, pvs_matrix

#def compute_2D_correlation(col1, col2, effs_col1, effs_col2, col1name="Column 1", col2name = "Column 2",  effs_sorted = np.arange(0, 1.01, 0.125), annot=True, groups = None, resize_colorbar=False, square=False):
def compute_2D_correlation(col2, col1, effs_col2, effs_col1, col2name="Column 1", col1name = "Column 2",  effs_sorted = np.arange(0, 1.01, 0.125), annot=True, groups = None, resize_colorbar=False, square=False, axes_off=False, font_size_multiplier = 1):

    if groups is None:
        corr_matrix, pvs_matrix = compute_2D_correlation_matrices(col1, col2, effs_col1, effs_col2,)
    else:
        corr_matrix_g1, pvs_matrix_g1 = compute_2D_correlation_matrices(col1[~groups], col2[~groups], effs_col1[~groups], effs_col2[~groups])
        corr_matrix_g2, pvs_matrix_g2 = compute_2D_correlation_matrices(col1[groups], col2[groups], effs_col1[groups], effs_col2[groups])

        corr_matrix = sum(~groups) / len(groups) * corr_matrix_g1 + sum(groups) / len(groups) * corr_matrix_g2
        pvs_matrix = sum(~groups) / len(groups) * pvs_matrix_g1 + sum(groups) / len(groups) * pvs_matrix_g2

    # Plot the 2D correlation matrix for the correlation
    if resize_colorbar:
        ## take maximum over non-diagonal elements
        vmax = np.max(corr_matrix.flatten()[(np.eye(corr_matrix.shape[0]) == 0).flatten()])
    else:
        vmax = 1

    # Define a function to format the tick values
    def format_tick(value, tick_number):
        return f'{value:.2f}'

    # Create a FuncFormatter object based on the function
    formatter = FuncFormatter(format_tick)

    corr_fig = plt.figure(dpi=300)
    ax = corr_fig.add_subplot(111)
    sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', vmin=-vmax, vmax=vmax, square=square, cbar_kws={'ticks': [-vmax, vmax]})
    cax = ax.figure.axes[-1]
    cax.yaxis.set_major_formatter(formatter)  # Set the formatter for the y-axis
    cax.tick_params(labelsize=14*font_size_multiplier)

    if not axes_off:
        plt.xlabel(col1name, fontsize=16)
        plt.ylabel(col2name, fontsize=16)
        ax.set_xticklabels(effs_sorted, fontsize=16)
        for label in ax.xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax.set_yticklabels(effs_sorted, fontsize=16)
        for label in ax.yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
    else:
        plt.axis('off')

    # Plot the 2D correlation matrix for the p_values
    pvs_fig = plt.figure(dpi=300)
    ax = pvs_fig.add_subplot(111)
    sns.heatmap(pvs_matrix, annot=annot, cmap='coolwarm_r', vmin=0, vmax=1, square=square, cbar_kws={'ticks': [0, 1]})
    cax = ax.figure.axes[-1]
    cax.yaxis.set_major_formatter(formatter)  # Set the formatter for the y-axis
    cax.tick_params(labelsize=14*font_size_multiplier)
    if not axes_off:
        plt.xlabel(col1name, fontsize=16)
        plt.ylabel(col2name, fontsize=16)
        ax.set_xticklabels(effs_sorted, fontsize=16)
        for label in ax.xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax.set_yticklabels(effs_sorted, fontsize=16)
        for label in ax.yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
    else:
        plt.axis('off')

    return corr_fig, pvs_fig

# %% 

def compute_partial_2D_correlation_matrices(col1, col2, covar, effs_col1, effs_col2, effs_covar, semi = False):
    ''' Computes 2D Correlation Matrices and associated pv values 
    
    Arguments
    ---------
    col1 : pd.Series or np.ndarray
        The first column of data.
        Note the assumption that either col1, col2, effs_col1, effs_col2 are all pd.Series or all np.ndarray
    col2 : pd.Series or np.ndarray
        The second column of data.
    effs_col1 : pd.Series or np.ndarray
        The efficacies associated with the first column of data.
    effs_col2 : pd.Series or np.ndarray
        The efficacies associated with the second column of data.

    Returns
    -------
    corr_matrix : np.ndarray
        The correlation matrix.
    pvs_matrix : np.ndarray
        The p-values matrix.   
    
    '''

    if isinstance(col1, pd.DataFrame) or isinstance(col1, pd.Series):

        ### only keep those rows where both A and B are not np.nan
        mask = col1.apply(lambda x: type(x) == np.ndarray or type(x) == list) & col2.apply(lambda x: type(x) == np.ndarray or type(x) == list) & covar.apply(lambda x: type(x) == np.ndarray or type(x) == list)
        col1 = col1[mask]
        col2 = col2[mask]
        effs_col1 = effs_col1[mask]
        effs_col2 = effs_col2[mask]
        covar = covar[mask]
        effs_covar = effs_covar[mask]

        ### sort the col1 and col2 by the effs_col1 and effs_col2
        # col1 = col1.apply(lambda x, idx: sort_array_by_another(x, effs_col1.loc[idx]), args=(effs_col1.index,))
        # col2 = col2.apply(lambda x, idx: sort_array_by_another(x, effs_col2.loc[idx]), args=(effs_col2.index,))
        sorted_col1 = pd.Series(dtype=object)
        sorted_col2 = pd.Series(dtype=object)
        sorted_covar = pd.Series(dtype=object)

        for idx in col1.index:
            # sorted_col1.at[idx] = col1.loc[idx][np.argsort(effs_col1.loc[idx])]
            # sorted_col2.at[idx] = col2.loc[idx][np.argsort(effs_col2.loc[idx])]
            # sorted_covar.at[idx] = covar.loc[idx][np.argsort(effs_covar.loc[idx])]

            sorted_col1.at[idx] = [col1.loc[idx][i] for i in np.argsort(effs_col1.loc[idx])]
            sorted_col2.at[idx] = [col2.loc[idx][i] for i in np.argsort(effs_col2.loc[idx])]
            sorted_covar.at[idx] = [covar.loc[idx][i] for i in np.argsort(effs_covar.loc[idx])]
            
        col1 = sorted_col1
        col2 = sorted_col2
        covar = sorted_covar

        A = np.stack(col1.values).squeeze()
        B = np.stack(col2.values).squeeze()
        C = np.stack(covar.values).squeeze()

    else:
        ### sort the col1 and col2 by the effs_col1 and effs_col2
        ## we want to sort the columns by the efficacies on a row-by-row basis

        ## tile the efficacies to match the shape of the columns if necessary
        if len(effs_col1.shape) == 1:
            effs_col1 = np.tile(effs_col1, (col1.shape[0], 1))
            effs_col2 = np.tile(effs_col2, (col2.shape[0], 1))
            effs_covar = np.tile(effs_covar, (covar.shape[0], 1))

        sorted_col1 = []
        sorted_col2 = []
        sorted_covar = []

        print(effs_col1.shape)

        for i in range(len(col1)):
            sorted_col1.append(col1[i][np.argsort(effs_col1[i])])
            sorted_col2.append(col2[i][np.argsort(effs_col2[i])])
            sorted_covar.append(covar[i][np.argsort(effs_covar[i])])
            
        A = np.stack(sorted_col1)
        B = np.stack(sorted_col2)
        C = np.stack(sorted_covar)

    mask = ~np.isnan(A).any(axis=1) & ~np.isnan(B).any(axis=1) & ~np.isnan(C).any(axis=1)
    A = A[mask]
    B = B[mask]
    C = C[mask]
    
    n_efficacies = A.shape[1]
    
    corr_matrix = np.zeros((n_efficacies, n_efficacies))
    pvs_matrix = np.zeros((n_efficacies, n_efficacies))
    
    for i in range(n_efficacies):
        for j in range(n_efficacies):
            
            df_to_corr = pd.DataFrame({'A': A[:,i], 'B': B[:,j], 'C': C[:,i]})

            if not semi:
                stats = partial_corr(df_to_corr, x='A', y='B', covar='C', method='pearson')
            elif semi=='X':
                stats = partial_corr(df_to_corr, x='A', y='B', y_covar='C', method='pearson')
            elif semi=='Y':
                stats = partial_corr(df_to_corr, x='A', y='B', x_covar='C', method='pearson')

            print(stats)

            corr_matrix[i, j] = stats['r']
            pvs_matrix[i, j] = stats['p-val']

    return corr_matrix, pvs_matrix

#def compute_partial_2D_correlation(col1, col2, covar, effs_col1, effs_col2,effs_covar, col1name="Column 1", col2name = "Column 2",  effs_sorted = np.arange(0, 1.01, 0.125), semi=False, annot=True, groups= None, resize_colorbar=False, square=False):
def compute_partial_2D_correlation(col2, col1, covar, effs_col2, effs_col1,effs_covar, col2name="Column 2", col1name = "Column 1",  effs_sorted = np.arange(0, 1.01, 0.125), semi=False, annot=True, groups= None, resize_colorbar=False, square=False, axes_off=False, font_size_multiplier=1):


    if groups is None:
        corr_matrix, pvs_matrix = compute_partial_2D_correlation_matrices(col1, col2, covar, effs_col1, effs_col2, effs_covar, semi=semi)
    else:
        corr_matrix_g1, pvs_matrix_g1 = compute_partial_2D_correlation_matrices(col1[~groups], col2[~groups], covar[~groups], effs_col1[~groups], effs_col2[~groups], effs_covar[~groups], semi=semi)
        corr_matrix_g2, pvs_matrix_g2 = compute_partial_2D_correlation_matrices(col1[groups], col2[groups], covar[groups], effs_col1[groups], effs_col2[groups], effs_covar[groups], semi=semi)

        corr_matrix = sum(~groups) / len(groups) * corr_matrix_g1 + sum(groups) / len(groups) * corr_matrix_g2
        pvs_matrix = sum(~groups) / len(groups) * pvs_matrix_g1 + sum(groups) / len(groups) * pvs_matrix_g2


    # Plot the 2D correlation matrix for the correlation
    if resize_colorbar:
        ## take maximum over non-diagonal elements
        vmax = np.max(corr_matrix.flatten()[(np.eye(corr_matrix.shape[0]) == 0).flatten()])
    else:
        vmax = 1

    # Define a function to format the tick values
    def format_tick(value, tick_number):
        return f'{value:.2f}'

    # Create a FuncFormatter object based on the function
    formatter = FuncFormatter(format_tick)

    corr_fig = plt.figure(dpi=300)
    ax = corr_fig.add_subplot(111)
    sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', vmin=-vmax, vmax=vmax, square=square, cbar_kws={'ticks': [-vmax, vmax]})
    cax = ax.figure.axes[-1]
    cax.yaxis.set_major_formatter(formatter)  # Set the formatter for the y-axis
    cax.tick_params(labelsize=14 * font_size_multiplier)

    if not axes_off:
        plt.xlabel(col1name, fontsize=16)
        plt.ylabel(col2name, fontsize=16)
        ax.set_xticklabels(effs_sorted, fontsize=16)
        for label in ax.xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax.set_yticklabels(effs_sorted, fontsize=16)
        for label in ax.yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
    else:
        plt.axis('off')

    # Plot the 2D correlation matrix for the p_values
    pvs_fig = plt.figure(dpi=300)
    ax = pvs_fig.add_subplot(111)
    sns.heatmap(pvs_matrix, annot=annot, cmap='coolwarm_r', vmin=0, vmax=1, cbar_kws={'ticks': [0, 1]})
    cax = ax.figure.axes[-1]
    cax.yaxis.set_major_formatter(formatter)  # Set the formatter for the y-axis
    cax.tick_params(labelsize=14 * font_size_multiplier)
    if not axes_off:
        plt.xlabel(col1name, fontsize=16)
        plt.ylabel(col2name, fontsize=16)
        ax.set_xticklabels(effs_sorted, fontsize=16)
        for label in ax.xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax.set_yticklabels(effs_sorted, fontsize=16)
        for label in ax.yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
    else:
        plt.axis('off')

    return corr_fig, pvs_fig

# %% LOAD SIMULATED PARTICIPANTS

def load_simulated_participants(simulated_participants_folder, modelname, sim_timestamp, include_sleep=False):

    with open(os.path.join(simulated_participants_folder, modelname, '%s_rewss_taus.pkl' %sim_timestamp), 'rb') as file:
        sim_rews = np.array(pickle.load(file))

    with open(os.path.join(simulated_participants_folder, modelname, '%s_perturbed_counters_peeks_taus.pkl' %sim_timestamp), 'rb') as file:
        sim_obs = np.array(pickle.load(file))

    if include_sleep:
        with open(os.path.join(simulated_participants_folder, modelname, '%s_perturbed_sleep_errs_taus_ape.pkl' %sim_timestamp), 'rb') as file:
            sim_sleep = np.array(pickle.load(file))
    else:
        sim_sleep = None

    return sim_rews, sim_obs, sim_sleep

def load_simulated_participants_across_models(simulated_participants_folder, ape_models, sim_timestamp, include_sleep = False):

    sim_rewss, sim_obss, sim_sleepss = [], [], []

    for mname in ape_models:
        sim_rews, sim_obs, sim_sleep = load_simulated_participants(simulated_participants_folder, str(mname), sim_timestamp, include_sleep)
        sim_rewss.append(sim_rews)
        sim_obss.append(sim_obs)
        sim_sleepss.append(sim_sleep)

    sim_rewss = np.stack(sim_rewss)
    sim_obss = np.stack(sim_obss)
    sim_sleepss = np.stack(sim_sleepss)

    return sim_rewss, sim_obss, sim_sleepss

def load_simulated_estimates_across_models(simulated_participants_folder, ape_models, sim_timestamp, include_sleep = False):

    sim_estss = []

    for modelname in ape_models:
        with open(os.path.join(simulated_participants_folder, str(modelname), '%s_perturbed_control_errs_taus_ape.pkl' %sim_timestamp), 'rb') as file:
            sim_ests = np.array(pickle.load(file))
        sim_estss.append(sim_ests)

    sim_estss = np.stack(sim_estss)

    return sim_estss

# %% FACTOR ANALYSIS

def get_factor_analysis_details(data):
    fa = FactorAnalyzer(rotation="varimax")
    fa.fit(data)
    variance = np.sum(fa.get_factor_variance()[1])
    return fa.loadings_, len(fa.loadings_[0]), variance

def compute_similarity(loadings1, loadings2):
    similarities = []
    for i in range(loadings1.shape[1]):
        similarity = cosine_similarity(loadings1[:, i].reshape(1, -1), loadings2[:, i].reshape(1, -1))
        similarities.append(similarity[0][0])
    return np.mean(similarities)

def bootstrap_similarity(data1, data2, n_iterations=100):
    similarities = []
    for _ in range(n_iterations):
        sample1 = resample(data1)
        sample2 = resample(data2)
        loadings1, _, _ = get_factor_analysis_details(sample1)
        loadings2, _, _ = get_factor_analysis_details(sample2)
        sim = compute_similarity(loadings1, loadings2)
        similarities.append(sim)
    return np.percentile(similarities, [2.5, 97.5])

# %% COMPETITIVE REGRESSION

# Fisher z-transform function
def fisher_transform(correlation_matrix):
    return 0.5 * np.log((1 + correlation_matrix) / (1 - correlation_matrix + np.finfo(float).eps))

def competitive_corr_regression(corr_data, corr_hyps, do_fisher_transform = True):

    if do_fisher_transform:
        # Applying the transform to each matrix
        z_observed = fisher_transform(corr_data)
        z_candidates = [fisher_transform(corr_hyp) for corr_hyp in corr_hyps]
    else:
        z_observed = corr_data
        z_candidates = corr_hyps

    # Flattening the matrices to 1D arrays, removing the diagonal elements [assuming symmetric matrix and 1s on the diagonal]
    #z_observed_flat = z_observed[np.triu_indices(z_observed.shape[0], k = 1)]
    #z_candidate_1_flat = z_candidate_1[np.triu_indices(z_candidate_1.shape[0], k = 1)]
    #z_candidate_2_flat = z_candidate_2[np.triu_indices(z_candidate_2.shape[0], k = 1)]
    z_observed_flat = z_observed.flatten()
    z_candidates_flat = [z_candidate.flatten() for z_candidate in z_candidates]

    # Building the regression model
    #X = np.column_stack(( z_candidate_1_flat, z_candidate_2_flat))#np.random.random((36,))))#z_candidate_2_flat, z_candidate_1_flat,))
    X = np.column_stack(z_candidates_flat)#np.random.random((36,))))#z_candidate_2_flat, z_candidate_1_flat,))
    X = sm.add_constant(X)  # Adding a constant term to the predictors
    model = sm.OLS(z_observed_flat, X).fit()

    # Displaying the results
    print(model.summary())

    return model

def upper_tri_masking(A):
    """
    Returns the upper triangular part of a 2-D array A.

    Args:
        A: 2-D array

    Returns:
        Upper triangular part of A as a 1-D array.

    Examples:
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        upper_tri_masking(A)  # Output: array([2, 3, 6])
    """
    
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

# %%



def competitive_ridge_corr_regression(corr_data, corr_hyps, alpha=None, do_fisher_transform = True):

    if fisher_transform:
        # Applying the transform to each matrix
        z_observed = fisher_transform(corr_data)
        z_candidates = [fisher_transform(corr_hyp) for corr_hyp in corr_hyps]
    else:
        z_observed = corr_data
        z_candidates = corr_hyps

    # Flattening the matrices to 1D arrays, removing the diagonal elements [assuming symmetric matrix and 1s on the diagonal]
    #z_observed_flat = z_observed[np.triu_indices(z_observed.shape[0], k = 1)]
    #z_candidate_1_flat = z_candidate_1[np.triu_indices(z_candidate_1.shape[0], k = 1)]
    #z_candidate_2_flat = z_candidate_2[np.triu_indices(z_candidate_2.shape[0], k = 1)]
    z_observed_flat = z_observed.flatten()
    z_candidates_flat = [z_candidate.flatten() for z_candidate in z_candidates]

    # Building the regression model
    # #X = np.column_stack(( z_candidate_1_flat, z_candidate_2_flat))#np.random.random((36,))))#z_candidate_2_flat, z_candidate_1_flat,))
    X = np.column_stack(z_candidates_flat)#np.random.random((36,))))#z_candidate_2_flat, z_candidate_1_flat,))
    X = sm.add_constant(X)  # Adding a constant term to the predictors
    
    # # Applying ridge regression
    # if alpha is not None:
    #     ridge_model = Ridge(alpha=alpha)  # alpha is the regularization strength
    # else:
    #     ridge_model = Ridge()  # alpha is the regularization strength
    # ridge_model.fit(X[:, 1:], z_observed_flat)  # note: Ridge automatically adds a constant term

     ## WITH STATSMODELS APPROACH AND SUMMARY
    model = sm.OLS(z_observed_flat, X)
    #print(model.summary())
    if alpha is not None:
       results_fr = model.fit_regularized(L1_wt = 0, alpha=alpha)
    else:
        results_fr = model.fit_regularized(L1_wt = 0, alpha=1)

    pinv_wexog,_ = pinv_extended(model.wexog)
    normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))

    final = sm.regression.linear_model.OLSResults(model, 
                                              results_fr.params, 
                                              normalized_cov_params)

    print(final.summary())


    # Display results
    #print("Ridge coefficients: ", ridge_model.coef_)

    return model


def competitive_lasso_corr_regression(corr_data, corr_hyps, alpha=None, do_fisher_transform = True):

    if do_fisher_transform:
        # Applying the transform to each matrix
        z_observed = fisher_transform(corr_data)
        z_candidates = [fisher_transform(corr_hyp) for corr_hyp in corr_hyps]
    else:
        z_observed = corr_data
        z_candidates = corr_hyps

    # Flattening the matrices to 1D arrays, removing the diagonal elements [assuming symmetric matrix and 1s on the diagonal]
    #z_observed_flat = z_observed[np.triu_indices(z_observed.shape[0], k = 1)]
    #z_candidate_1_flat = z_candidate_1[np.triu_indices(z_candidate_1.shape[0], k = 1)]
    #z_candidate_2_flat = z_candidate_2[np.triu_indices(z_candidate_2.shape[0], k = 1)]
    z_observed_flat = z_observed.flatten()
    z_candidates_flat = [z_candidate.flatten() for z_candidate in z_candidates]

    # Building the regression model
    #X = np.column_stack(( z_candidate_1_flat, z_candidate_2_flat))#np.random.random((36,))))#z_candidate_2_flat, z_candidate_1_flat,))
    X = np.column_stack(z_candidates_flat)#np.random.random((36,))))#z_candidate_2_flat, z_candidate_1_flat,))
    X = sm.add_constant(X)  # Adding a constant term to the predictors
    
    # Applying ridge regression
    #model = Lasso(alpha=alpha)  # alpha is the regularization strength
    #model.fit(X[:, 1:], z_observed_flat)  # note: Ridge automatically adds a constant term

    ## WITH STATSMODELS APPROACH AND SUMMARY
    model = sm.OLS(z_observed_flat, X)
    #print(model.summary())
    if alpha is not None:
       results_fr = model.fit_regularized(L1_wt=alpha)
    else:
        results_fr = model.fit_regularized()

    pinv_wexog,_ = pinv_extended(model.wexog)
    normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))

    final = sm.regression.linear_model.OLSResults(model, 
                                              results_fr.params, 
                                              normalized_cov_params)

    print(final.summary())

    # Display results
    #print("Lasso coefficients: ", model.coef_)

    return model


# %%

def combine_train_test(train, test, effs_train, effs_test):
    
    combined = np.concatenate((train, test), axis=0)
    combined_effs = np.concatenate((effs_train, effs_test), axis=0)

    # sort by effs
    combined = [combined[i] for i in np.argsort(combined_effs)]

    return np.stack(combined)

# %% CORRELATION MATRIX SIMILARITY

def correlation_matrix_similarity(R1, R2):
    '''
    Calculate the Frobenius norms of R1 and R2
    implemented according to https://stats.stackexchange.com/questions/14673/measures-of-similarity-or-distance-between-two-covariance-matrices#comment26212_14676
    '''
    norm_R1 = np.linalg.norm(R1, 'fro')
    norm_R2 = np.linalg.norm(R2, 'fro')
    
    # Calculate the trace of the product of R1 and R2
    trace_product = np.trace(np.dot(R1, R2))
    
    # Calculate the similarity metric
    similarity = trace_product / (norm_R1 * norm_R2)
    
    return similarity

# %% CREATE ORDINAL MIXED EFFECTS GLM

def fit_mixed_effects_glm(df, target, categorical_efficacy = False):

    series = df[target]

    df_glm = pd.DataFrame(series.explode())

    df_glm['efficacy'] = df['effs'].explode().astype(float)

    ## merge with group
    df_glm['group'] = df['group']
    df_glm['pid'] = df_glm.index.astype(str)
    df_glm[target] = df_glm[target].astype(int)

    df_glm['efficacy_C'] = pd.Categorical(df_glm['efficacy'], ordered=True)
    df_glm['group_C'] = pd.Categorical(df_glm['group'], ordered=False)

    if categorical_efficacy:
        mixed_glm = smf.mixedlm(target + " ~ C(efficacy_C) + C(group_C)", df_glm, groups=df_glm["pid"]).fit()
    else:
        mixed_glm = smf.mixedlm(target + " ~ efficacy + C(group_C)", df_glm, groups=df_glm["pid"]).fit()

    return mixed_glm