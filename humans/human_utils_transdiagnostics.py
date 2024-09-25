# Kai Sandbrink
# 2023-08-14
# This script groups utility functions that are used for transdiagnostic analyses

# %% LIBRARY IMPORTS

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from human_utils_behavioral_analysis import get_mask_behav
from human_utils_project import get_clean_data

import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pingouin import partial_corr

from sklearn.decomposition import PCA

# %% 

def get_clean_transdiagnostics(exp_date = '24-01-22-29', file_base = '', facsimile = True):

    if exp_date == '24-01-22-29' and facsimile:

        files = [
            'transdiagnostics_22A.csv',
            'transdiagnostics_22B.csv',
            'transdiagnostics_29A.csv',
            'transdiagnostics_29B.csv',
        ]
    else:
        raise ValueError('exp_date not found')

    df = pd.DataFrame()

    for file in files:
        df = pd.concat([df, pd.read_csv(os.path.join(file_base,'data/%s' %file), index_col=0)])

    return df

def get_clean_combined_data(day = 1, exp_date='24-01-22-29', group='groupA', day1_test_mask_cutoff=None, file_base='', facsimile=True):

    df_behav, effs_train, effs_test, test_start = get_clean_data(day = day, group=group, exp_date=exp_date, day1_test_mask_cutoff=day1_test_mask_cutoff, file_base=file_base)
    df_trans = get_clean_transdiagnostics(exp_date = exp_date, file_base=file_base, facsimile=facsimile)

    padded_indices = []
    for pid in df_trans.index:
        idx = int(pid)
        id_str = ''
        if idx < 100:
            id_str += '0'
        if idx < 10:
            id_str += '0'
        id_str += str(idx)
        padded_indices.append(id_str)
    df_trans.index = padded_indices

    df = pd.merge(df_behav, df_trans, left_index=True, right_index=True)

    return df, effs_train, effs_test, test_start

# %% 


def compute_2D_correlation_matrices_transdiagnostics(col1, col2, effs_col1, partial=None):
    ''' Computes 2D Correlation Matrices and associated pv values 
    
    Arguments
    ---------
    col1 : pd.Series or np.ndarray - the n_efficacies variable
        The first column of data.
        Note the assumption that either col1, col2, effs_col1, effs_col2 are all pd.Series or all np.ndarray
    col2 : pd.Series or np.ndarray - the 1D variable
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
        if partial is not None:
            partial = partial[mask]

        ### sort the col1 and col2 by the effs_col1 and effs_col2
        # col1 = col1.apply(lambda x, idx: sort_array_by_another(x, effs_col1.loc[idx]), args=(effs_col1.index,))
        # col2 = col2.apply(lambda x, idx: sort_array_by_another(x, effs_col2.loc[idx]), args=(effs_col2.index,))
        sorted_col1 = pd.Series(dtype=object)

        for idx in col1.index:
            sorted_col1.at[idx] = col1.loc[idx][np.argsort(effs_col1.loc[idx])]

        col1 = sorted_col1

        A = np.stack(col1.values)
        B = np.stack(col2.values)
        partial = np.stack(partial.values)

    else:
        ### sort the col1 and col2 by the effs_col1 and effs_col2
        ## we want to sort the columns by the efficacies on a row-by-row basis

        ## tile the efficacies to match the shape of the columns if necessary
        if len(effs_col1.shape) == 1:
            effs_col1 = np.tile(effs_col1, (col1.shape[0], 1))

        sorted_col1 = []
        sorted_partial = []

        print(effs_col1.shape)

        for i in range(len(col1)):
            sorted_col1.append(col1[i][np.argsort(effs_col1[i])])
            if partial is not None:
                sorted_partial.append(partial[i][np.argsort(effs_col1[i])])

        A = np.stack(sorted_col1)
        B = col2.values.squeeze()
        if partial is not None:
            partial = np.stack(sorted_partial)

    mask = ~np.isnan(A).any(axis=1) & ~np.isnan(B).any(axis=1)
    A = A[mask]
    B = B[mask]
    if partial is not None:
        partial = partial[mask]

    n_efficacies = A.shape[1]
    n_tds = B.shape[1]
    
    corr_matrix = np.zeros((n_efficacies,n_tds))
    pvs_matrix = np.zeros((n_efficacies,n_tds))
    
    for i in range(n_efficacies):
        for j in range(n_tds):
            if partial is None:
                corr, pv = pearsonr(A[:,i], B[:,j])
                corr, pv = spearmanr(A[:,i], B[:,j])
                corr_matrix[i, j] = corr
                pvs_matrix[i, j] = pv
            else:
                df = pd.DataFrame({'A': A[:,i], 'B': B[:,j], 'C': partial[:,i]})
                result = partial_corr(data=df, x='A', y='B', covar='C')
                corr_matrix[i, j] = result['r'][0]
                pvs_matrix[i, j] = result['p-val'][0]

    if n_tds == 1:
        corr_matrix = np.expand_dims(corr_matrix, 0)
        pvs_matrix = np.expand_dims(pvs_matrix, 0)

    return corr_matrix.T, pvs_matrix.T

def compute_2D_correlation_transdiagnostics(col1, col2, effs_col1, col1name="Column 1", col2names = ["Column 2"],  effs_sorted = np.arange(0, 1.01, 0.125), groups = None, partial=None ):

    if groups is None:
        corr_matrix, pvs_matrix = compute_2D_correlation_matrices_transdiagnostics(col1, col2, np.stack(effs_col1.values),)
    else:
        corr_matrix_g1, pvs_matrix_g1 = compute_2D_correlation_matrices_transdiagnostics(col1[~groups], col2[~groups], np.stack(effs_col1[~groups].values), partial[~groups] if partial is not None else None)
        corr_matrix_g2, pvs_matrix_g2 = compute_2D_correlation_matrices_transdiagnostics(col1[groups], col2[groups], np.stack(effs_col1[groups].values), partial[groups] if partial is not None else None)

        corr_matrix = sum(~groups) / len(groups) * corr_matrix_g1 + sum(groups) / len(groups) * corr_matrix_g2
        pvs_matrix = sum(~groups) / len(groups) * pvs_matrix_g1 + sum(groups) / len(groups) * pvs_matrix_g2

    # Plot the 2D correlation matrix for the correlation
    corr_fig = plt.figure(figsize=(6, 2), dpi=300)
    ax = corr_fig.add_subplot(111)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Efficacy for ' + col1name)
    plt.ylabel('Transdiagnostics')
    ax.set_yticklabels(col2names)
    ax.set_xticklabels(effs_sorted)

    # Plot the 2D correlation matrix for the p_values
    pvs_fig = plt.figure(figsize=(6, 2), dpi=300)
    ax = pvs_fig.add_subplot(111)
    sns.heatmap(pvs_matrix, annot=True, cmap='coolwarm_r', vmin=0, vmax=1)
    plt.xlabel('Efficacy for ' + col1name)
    plt.ylabel('Transdiagnostics')
    ax.set_xticklabels(effs_sorted)
    ax.set_yticklabels(col2names)

    return corr_fig, pvs_fig

# %% QUANTILE ANALYSIS

def preprocess_for_quantile_analysis(col1, col2, effs_col1):
    """
    Preprocesses the input data by removing rows with missing values and sorting col1 based on effs_col1.

    Parameters
    ----------
    col1 : pd.Series or np.ndarray
        The first column of data, expected to be a series of lists or ndarrays.
    col2 : pd.Series or np.ndarray
        The second column of data.
    effs_col1 : pd.Series or np.ndarray
        The efficacies associated with the first column of data.

    Returns
    -------
    A, B : np.ndarray
        The preprocessed and sorted col1 and col2 ready for further analysis.
    """
    ### TODO: THIS IF IS QUARANTINED FOR DELETION AS IT DOES NOT APPEAR TO WORK PROPERLY (AND I AM USUALLY CONVERTING TO NP ARRAY ANYWAY)
    if isinstance(col1, pd.Series) or isinstance(col1, pd.DataFrame):
        # Filter out rows where col1 or col2 is not a list or ndarray
        mask = col1.apply(lambda x: isinstance(x, (np.ndarray, list))) & col2.apply(lambda x: isinstance(x, (np.ndarray, list)))
        col1_filtered = col1[mask]
        col2_filtered = col2[mask]
        effs_col1_filtered = effs_col1[mask]

        # Sort col1 by effs_col1
        sorted_col1 = pd.Series([col1_filtered.loc[idx][np.argsort(effs_col1_filtered.loc[idx])] for idx in col1_filtered.index], index=col1_filtered.index)

        # Convert to np.ndarray
        A = np.stack(sorted_col1.values)
        B = np.stack(col2_filtered.values)
        
    else:  # Assume np.ndarray for else case
        sorted_col1 = []
        for i in range(len(col1)):
            if len(effs_col1.shape) > 1:
                sorted_indices = np.argsort(effs_col1[i])
            else:
                sorted_indices = np.argsort(effs_col1)
            sorted_col1.append(col1[i][sorted_indices])

        A = np.stack(sorted_col1)
        B = np.array(col2).squeeze()  # Ensure col2 is correctly shaped
        
    return A, B

def compute_quantile_analysis(col1, col2, effs_col1, num_bins=10, take_quantile_on_col1 = True, flip_leastmost=False, bins=None):
    '''
    Performs quantile analysis on the given columns.

    Arguments
    ---------
    col1 : pd.Series or np.ndarray
        The first column of data, containing lists of different values in every row.
    col2 : pd.Series or np.ndarray
        The second column of data, to be averaged within quantile bins of col1.
    effs_col1 : pd.Series or np.ndarray
        The efficacies associated with the first column of data.
    num_bins : int, optional
        The number of quantile bins to use.

    Returns
    -------
    quantile_matrix : np.ndarray
        The matrix containing the average col2 value for each efficacy x quantile bin combination.
    '''

    if flip_leastmost:
        col2 = -col2

    # Initialize the result matrix
    if len(effs_col1.shape) > 1:
        n_efficacies = len(effs_col1[0]) if isinstance(effs_col1[0], (list, np.ndarray)) else effs_col1.shape[1]
    else:
        n_efficacies = len(effs_col1)
    quantile_matrix = np.zeros((num_bins, n_efficacies))

    col1, col2 = preprocess_for_quantile_analysis(col1, col2, effs_col1)

    binss = []

    for i in range(n_efficacies):
        # Extracting all i-th efficacy values across rows
        all_effs = np.array([row[i] for row in col1])

        # Calculating quantile bins for the i-th efficacy values
        if bins is None:
            if take_quantile_on_col1:
                bins = np.quantile(all_effs, q=np.linspace(0, 1, num_bins+1))
                binss.append(bins)
            else:
                bins = np.quantile(col2, q=np.linspace(0, 1, num_bins+1))
                binss = bins
        else:
            binss = bins
        
        # For each bin, calculate the average col2 score
        for bin_index in range(num_bins):
            if take_quantile_on_col1:
                in_bin = (all_effs >= bins[bin_index]) & (all_effs < bins[bin_index+1])
                if bin_index == num_bins - 1:  # Ensure the last bin includes the upper bound
                    in_bin |= (all_effs == bins[-1])
                
                # Averaging col2 values where col1 values fall within the current bin
                if np.any(in_bin):  # Avoid division by zero if bin is empty
                    avg_value = np.mean(col2[in_bin])
                else:
                    avg_value = np.nan  # Use NaN for bins without any data points
            else:
                in_bin = (col2 >= bins[bin_index]) & (col2 < bins[bin_index+1])
                if bin_index == num_bins - 1:
                    in_bin |= (col2 == bins[-1])

                ## average col1 values where col2 values fall within the current bin
                if np.any(in_bin):
                    avg_value = np.mean(all_effs[in_bin])
                else:
                    avg_value = np.nan
                    
            quantile_matrix[bin_index, i] = avg_value

    ## replace nans with first non-nan value in the same column
    for i in range(n_efficacies):
        nan_indices = np.where(np.isnan(quantile_matrix[:,i]))[0]
        nonnan_indices = np.where(~np.isnan(quantile_matrix[:,i]))[0]
        if len(nan_indices) > 0:
            for nan_index in nan_indices:
                ### find closest nonnan index
                assert len(nonnan_indices) > 0, "nonnan indices should be >0, instead we have bin indices %s and nan: %s and nonnnan: %s for input col1: %s and col2: %s" %(str(binss), str(nan_indices), str(nonnan_indices), str(col1), str(col2))
                quantile_matrix[nan_index, i] = quantile_matrix[nonnan_indices[np.argmin(np.abs(nonnan_indices - nan_index))], i]
                #quantile_matrix[nan_index, i] = quantile_matrix[nan_indices[-1]+1, i]

    return quantile_matrix, binss

def plot_quantile_analysis_results(col1, col2, effs_col1, col1name="Column 1", effs_sorted=np.arange(0, 1.01, 0.125), groups=None, num_bins = 10, square=False):
    """
    Computes quantile analysis for two groups (if provided) and plots the results.

    Arguments
    ---------
    col1 : pd.Series or np.ndarray
        The first column of data, containing lists of different values in every row.
    col2 : pd.Series or np.ndarray
        The second column of data, to be averaged within quantile bins of col1.
    effs_col1 : pd.Series or np.ndarray
        The efficacies associated with the first column of data.
    col1name : str, optional
        The name for the first column, used in plotting.
    effs_sorted : np.ndarray, optional
        The sorted efficacy levels, used for x-axis labels in plotting.
    groups : pd.Series or np.ndarray, optional
        A boolean array indicating group membership. If None, no group division is considered.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """

    # Perform quantile analysis
    if groups is None:
        quantile_matrix, bins = compute_quantile_analysis(col1, col2, effs_col1, num_bins=num_bins)
    else:
        quantile_matrix_g1, bins_g1 = compute_quantile_analysis(col1[~groups], col2[~groups], effs_col1[~groups], num_bins=num_bins)
        quantile_matrix_g2, bins_g2 = compute_quantile_analysis(col1[groups], col2[groups], effs_col1[groups], num_bins=num_bins)

        # For visualization, we'll keep them separate but you could combine them or compare them as needed

    # Plotting
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=300)
    
    sns.heatmap(quantile_matrix_g1, ax=axs[0], annot=True, cmap='viridis', cbar_kws={'label': 'Average Value'}, square=square)
    axs[0].set_title('Group 1')
    axs[0].set_xlabel('Efficacy for ' + col1name)
    axs[0].set_ylabel('Quantile Bins ' + col1name)
    axs[0].set_xticklabels(effs_sorted)
    #axs[0].set_yticklabels([f'{bin:.2f}' for bin in bins_g1[:-1]], rotation=0)

    assert not flip_leastmost, "Still need to implement quantile bin labels"

    sns.heatmap(quantile_matrix_g2, ax=axs[1], annot=True, cmap='viridis', cbar_kws={'label': 'Average Value'}, square=square)
    axs[1].set_title('Group 2')
    axs[1].set_xlabel('Efficacy for ' + col1name)
    #axs[1].set_ylabel('Quantile Bins for ' + col1name)
    axs[1].set_xticklabels(effs_sorted)
    #axs[1].set_yticklabels([f'{bin:.2f}' for bin in bins_g2[:-1]], rotation=0)

    plt.tight_layout()
    return fig

def plot_td_quantile_analysis_results(col1, col2, effs_col1, col1name="Column 1", col2name="Column 2", effs_sorted=np.arange(0, 1.01, 0.125), groups=None, num_bins = 10, combine_groups = False, annot=True, flip_leastmost = False, square=False, axes_off=False, font_size_multiplier=1.4, groupmins = None, groupmaxes = None):
    """
    Computes quantile analysis for two groups (if provided) and plots the results.

    Arguments
    ---------
    col1 : pd.Series or np.ndarray
        The first column of data, containing lists of different values in every row.
    col2 : pd.Series or np.ndarray
        The second column of data, to be averaged within quantile bins of col1.
    effs_col1 : pd.Series or np.ndarray
        The efficacies associated with the first column of data.
    col1name : str, optional
        The name for the first column, used in plotting.
    effs_sorted : np.ndarray, optional
        The sorted efficacy levels, used for x-axis labels in plotting.
    groups : pd.Series or np.ndarray, optional
        A boolean array indicating group membership. If None, no group division is considered.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """

    # Perform quantile analysis
    if groups is None:
        quantile_matrix, bins = compute_quantile_analysis(col1, col2, effs_col1, num_bins=num_bins, take_quantile_on_col1=False, flip_leastmost = flip_leastmost)
    else:
        quantile_matrix_g1, bins_g1 = compute_quantile_analysis(col1[~groups], col2[~groups], effs_col1[~groups], num_bins=num_bins, take_quantile_on_col1=False, flip_leastmost = flip_leastmost)
        quantile_matrix_g2, bins_g2 = compute_quantile_analysis(col1[groups], col2[groups], effs_col1[groups], num_bins=num_bins, take_quantile_on_col1=False, flip_leastmost= flip_leastmost)

        # For visualization, we'll keep them separate but you could combine them or compare them as needed

    # Plotting
    if not combine_groups and groups is not None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=300)
        
        sns.heatmap(quantile_matrix_g1, ax=axs[0], annot=annot, cmap='viridis', cbar_kws={'label': 'Average Value'}, square=square)
        axs[0].set_title('Group 1')
        axs[0].set_xlabel('Efficacy for ' + col1name)
        #axs[0].set_ylabel('Quantile Bins ' + col1name)
        axs[0].set_xticklabels(effs_sorted)
        axs[0].set_yticklabels([f'{bin:.2f}' for bin in bins_g1[:-1]], rotation=0)

        sns.heatmap(quantile_matrix_g2, ax=axs[1], annot=annot, cmap='viridis', cbar_kws={'label': 'Average Value'}, square=square)
        axs[1].set_title('Group 2')
        axs[1].set_xlabel('Efficacy for ' + col1name)
        #axs[1].set_ylabel('Quantile Bins for ' + col1name)
        axs[1].set_xticklabels(effs_sorted)
        axs[1].set_yticklabels([f'{bin:.2f}' for bin in bins_g2[:-1]], rotation=0)

        plt.tight_layout()

    else:
        ## normalize matrices between 0 and 1
        if groups is not None:
            value_range = [(np.nanmin(quantile_matrix_g1), np.nanmin(quantile_matrix_g2)), (np.nanmax(quantile_matrix_g1), np.nanmax(quantile_matrix_g2))]

            if groupmins is None:
                groupmins = [np.nanmin(quantile_matrix_g1), np.nanmin(quantile_matrix_g2)]
            if groupmaxes is None:
                groupmaxes = [np.nanmax(quantile_matrix_g1), np.nanmax(quantile_matrix_g2)] 
            quantile_matrix_g1 = (quantile_matrix_g1 - groupmins[0])/(groupmaxes[0] - groupmins[0])
            quantile_matrix_g2 = (quantile_matrix_g2 - groupmins[1])/(groupmaxes[1] - groupmins[1])

            quantile_matrix = (quantile_matrix_g1*sum(~groups) + quantile_matrix_g2*sum(groups))/len(groups)

            ticklabels = ['%.1f/\n%.1f' % (value_range[0][0], value_range[0][1]), '%.1f/\n%.1f' % (value_range[1][0], value_range[1][1])]
        else:
            value_range = [np.nanmin(quantile_matrix), np.nanmax(quantile_matrix)]
            ticklabels = ['%.1f' % value_range[0], '%.1f' % value_range[1]]

        print(quantile_matrix)

        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)

        #sns.heatmap(quantile_matrix, annot=annot, ax=ax, cmap='coolwarm', cbar_kws={'ticks': [0,1],}, square=square)
        sns.heatmap(quantile_matrix, annot=annot, ax=ax, cmap='coolwarm', cbar_kws={'ticks': [quantile_matrix.min(),quantile_matrix.max()],}, square=square)
        #sns.heatmap(quantile_matrix, annot=annot, ax=ax, cmap='coolwarm', square=square)
        
        # Set custom labels on the colorbar
        cbar = plt.gcf().axes[-1]
        #cbar.set_yticklabels(['Min', 'Max'])
        cbar.set_yticklabels(ticklabels)
        cbar.tick_params(labelsize=14*font_size_multiplier)  # Set the font size to 20

        if not axes_off:
            if not flip_leastmost:
                ylabel = 'Most ' + col2name + ' - Least ' + col2name + ' (Noniles)'
            else:
                ylabel = 'Least ' + col2name + ' - Most ' + col2name + ' (Noniles)'

            ax.set_xlabel('Efficacy for ' + col1name)
            ax.set_ylabel(ylabel)
            ax.set_xticklabels(effs_sorted)
            ax.set_yticklabels(np.arange(1,10))

        # Suppress the x and y axes
        else:
            plt.axis('off')
    
    return fig

def getPCs(td1, td2, enforceminusfirst=False, enforceminussecond=False):
    '''Computes and returns the principal components of two transdiagnostic scores

    Arguments
    ---------
    td1 : pd.Series or np.ndarray
        The first transdiagnostic score.
    td2 : pd.Series or np.ndarray
        The second transdiagnostic score.   

    Returns
    -------
    PCs : np.ndarray
        The principal components of the two transdiagnostic scores.
    '''

    if isinstance(td1, pd.Series) or isinstance(td1, pd.DataFrame):
        td1 = np.stack(td1.values)
        td2 = np.stack(td2.values)

    pca = PCA(n_components=2)
    PCs = pca.fit_transform(np.stack([td1, td2]).T)

    ### PRINT PC LOADINGS

    print('PC1 loadings: ', pca.components_[0])
    print('PC2 loadings: ', pca.components_[1])

    ### print variance explained
    print('Variance explained by PC1: ', pca.explained_variance_ratio_[0])
    print('Variance explained by PC2: ', pca.explained_variance_ratio_[1])

    for i, pccomps in enumerate(pca.components_):
        if any(pccomps < 0):
            if pccomps[0] > 0 and enforceminusfirst:
                PCs[:,i] = -PCs[:,i]
                print(f"Flipped PC{i+1} to enforce first component to be negative.")
            if pccomps[0] < 0 and enforceminussecond:
                PCs[:,i] = -PCs[:,i]
                print(f"Flipped PC{i+1} to enforce second component to be negative.")

    return PCs


    
# %%
