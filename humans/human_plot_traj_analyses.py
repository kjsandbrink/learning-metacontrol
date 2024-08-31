# Kai Sandbrink
# 2023-06-27
# This script contains plotting functions for trajectory analyses

# %% LIBRARY IMPORT

import numpy as np
import pandas as pd

from matplotlib.path import Path

import seaborn as sns
import matplotlib.pyplot as plt

from utils import format_axis
from matplotlib.ticker import FormatStrFormatter


# %% 

def plot_violin(test, effs_test, train=None, effs_train = None, ylabel='Mean Interval per Participant', xlabel="Efficacy", median_over_mean = False, yjitter=False, ylim=None):
    ''' Makes a violin plot
    
    Arguments
    ---------
    test : np.array of floats [n_participants, n_effs], metric of participant performance on test data
    effs_test : np.array of floats [n_effs], efficacy values for test data
    train : np.array of floats [n_participants, n_effs], metric of participant performance on train data
    effs_train : np.array of floats [n_effs], efficacy values for train data
    ylabel : str, label for y-axis
    xlabel : str, label for x-axis
    median_over_mean : bool, whether to plot median instead of mean
    yjitter : float, amount of jitter for y-axis

    Returns
    -------
    fig : matplotlib figure
    '''
        
    plot_df = pd.DataFrame({
        'x': np.tile(effs_test, test.shape[0]).flatten(),
        'y': test.flatten()
    })

    plot_df = plot_df.dropna()

    width = 0.1
    alpha_violin = 1
    yjitter_mag = 0.2

    color_test = '#006600'
    color_train = '#33cc33'

    if train is not None:
        plot_df_train = pd.DataFrame({
            'x': np.tile(effs_train, train.shape[0]).flatten(),
            'y': train.flatten()
        })

        plot_df_train = plot_df_train.dropna()

        width = 0.05

    ## PLOT SERIES OF VIOLIN PLOTS

    # Preparing data for violin plot
    data_list = [plot_df[plot_df['x'] == x]['y'].values for x in effs_test]

    # Create the violin plot
    fig, ax = plt.subplots(dpi=300)
    parts = ax.violinplot(data_list, positions=effs_test, showmeans=not median_over_mean, showmedians=median_over_mean, showextrema=False, widths=width)

    for pc in parts['bodies']:
        #pc.set_facecolor('C9')
        pc.set_facecolor(color_test)
        pc.set_edgecolor('black')
        pc.set_alpha(alpha_violin)

    # Set the color and linewidth of the mean lines
    if not median_over_mean:
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2) 
    else:
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)

    # Adding individual data points as a scatter plot
    jitter = 0.02 # adjust this value based on your actual data
    x_jitter = plot_df['x'] + np.random.uniform(-jitter, jitter, size=len(plot_df['x']))

    if not yjitter:
        ax.scatter(x_jitter, plot_df['y'], color='black', alpha=0.3, s=20)
    else:
        y_jitter = plot_df['y'] + np.random.uniform(-yjitter_mag, yjitter_mag, size=len(plot_df['y']))
        ax.scatter(x_jitter, y_jitter, color='black', alpha=0.3, s=20)

    # Overlaying with line plot
    if not median_over_mean:
        column_means = np.nanmean(test, axis=0)
    else:
        column_means = np.nanmedian(test, axis=0)
    #plt.plot(effs_test, column_means, color='C9', label="Test")
    plt.plot(effs_test, column_means, color=color_test, label="Without Cue")

    if train is not None:
        data_list = [plot_df_train[plot_df_train['x'] == x]['y'].values for x in effs_train]

        parts = ax.violinplot(data_list, positions=effs_train, showmeans=not median_over_mean, showmedians=median_over_mean, showextrema=False, widths=width)

        for pc in parts['bodies']:
            #pc.set_facecolor('C8')
            pc.set_facecolor(color_train)
            pc.set_edgecolor('black')
            pc.set_alpha(alpha_violin)

        # Set the color and linewidth of the mean lines
        if not median_over_mean:
            parts['cmeans'].set_color('black')
            parts['cmeans'].set_linewidth(2) 
        else:
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(2)

        # Adding individual data points as a scatter plot
        jitter = 0.02 # adjust this value based on your actual data
        x_jitter = plot_df_train['x'] + np.random.uniform(-jitter, jitter, size=len(plot_df_train['x']))

        if not yjitter:
            ax.scatter(x_jitter, plot_df_train['y'], color='black', alpha=0.3, s=20)
        else:
            y_jitter = plot_df_train['y'] + np.random.uniform(-yjitter_mag, yjitter_mag, size=len(plot_df_train['y']))
            ax.scatter(x_jitter, y_jitter, color='black', alpha=0.3, s=20)
        
        # Overlaying with line plot
        if not median_over_mean:
            column_means = np.nanmean(train, axis=0)
        else:
            column_means = np.nanmedian(train, axis=0)
        #plt.plot(effs_train, column_means, color='C8', label="Train")
        plt.plot(effs_train, column_means, color=color_train, label="With Cue")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()

    format_axis(ax)

    return fig

# %% 

def plot_violin_binarization(test, effs_test, train=None, effs_train = None, ylabel='Mean Interval per Participant', xlabel="Efficacy", median_over_mean = False, binarization=None, binarization_name = None, ylim=None):
    ''' Makes a violin plot
    
    Arguments
    ---------
    test : np.array of floats [n_participants, n_effs], metric of participant performance on test data
    effs_test : np.array of floats [n_effs], efficacy values for test data
    train : np.array of floats [n_participants, n_effs], metric of participant performance on train data
    effs_train : np.array of floats [n_effs], efficacy values for train data
    ylabel : str, label for y-axis
    xlabel : str, label for x-axis
    median_over_mean : bool, whether to plot median instead of mean
    binarization : list of bools [n_effs] or None, if None no binarization is performed, otherwise the list of bools is used to binarize the data
    binarization_name : str or None, if None no binarization is performed, otherwise the name is used to label the binarization

    Returns
    -------
    fig : matplotlib figure
    '''

    plot_df = pd.DataFrame({
        #'x': np.tile(effs_test, test.shape[0]).flatten(),
        'x': np.repeat(np.array(effs_test).reshape(1,-1), test.shape[0], axis=0).flatten(),
        'y': test.flatten(),
        'bin': np.repeat(binarization.values.reshape((-1,1)), test.shape[1], axis=1).flatten()
    })

    plot_df = plot_df.dropna()

    width = 0.1
    alpha_violin = {
        False: 1,
        True: 0.3
    }
    alpha_jitter = 0.1

    if train is not None:
        plot_df_train = pd.DataFrame({
            'x': np.tile(effs_train, train.shape[0]).flatten(),
            'y': train.flatten(),
            'bin': np.repeat(binarization.values.reshape((-1,1)), train.shape[1], axis=1).flatten()
        })

        plot_df_train = plot_df_train.dropna()

        width = 0.05

    ## PLOT SERIES OF VIOLIN PLOTS

    # Create the violin plot
    fig, ax = plt.subplots(dpi=300, figsize=(10, 5))

    for binary in [False, True]:

        # Preparing data for violin plot
        data_list = [plot_df[(plot_df['x'] == x) & (plot_df['bin'] == binary)]['y'].values for x in effs_test]

        parts = ax.violinplot(data_list, positions=effs_test, showextrema=False, widths=width)#, label='Top 50\% Most ' + binarization_name)

        for pc, eff in zip(parts['bodies'], effs_test):
            pc.set_facecolor('C9')
            pc.set_edgecolor('black')
            pc.set_alpha(alpha_violin[binary])

            if not binary:
                test_notbin_body = pc
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
                #pc.set_clip_path(Path([(eff, -np.inf), (eff, np.inf), (np.inf, np.inf), (np.inf, -np.inf)]), transform=ax.transData)
            else:
                test_bin_body = pc
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
                
        # Set the color and linewidth of the mean lines
        # if not median_over_mean:
        #     parts['cmeans'].set_color('black')
        #     parts['cmeans'].set_linewidth(2) 
        # else:
        #     parts['cmedians'].set_color('black')
        #     parts['cmedians'].set_linewidth(2)

        # Adding individual data points as a scatter plot
        jitter = 0.0075 # adjust this value based on your actual data
        x_jitter = plot_df['x'] - width/4 + 1/2*width*plot_df['bin'] + np.random.uniform(-jitter, jitter, size=len(plot_df['x']))

        ax.scatter(x_jitter, plot_df['y'], color='black', alpha=alpha_jitter, s=20)

        # Overlaying with line plot
        # if not median_over_mean:
        #     column_means = np.nanmean(test, axis=0)
        # else:
        #     column_means = np.nanmedian(test, axis=0)
        # plt.plot(effs_test, column_means, color='C9', label="Test")

        if train is not None:
            data_list = [plot_df_train[(plot_df_train['x'] == x) & (plot_df['bin'] == binary)]['y'].values for x in effs_train]

            train_parts = ax.violinplot(data_list, positions=effs_train, showextrema=False, widths=width)#, label='Bottom 50\% Least' + binarization_name)

            for pc, eff in zip(train_parts['bodies'], effs_train):
                pc.set_facecolor('C8')
                pc.set_edgecolor('black')
                pc.set_alpha(alpha_violin[binary])
            
                if not binary:
                    train_notbin_body = pc
                    m = np.mean(pc.get_paths()[0].vertices[:, 0])
                    pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
                    #pc.set_clip_path(Path([(eff, -np.inf), (eff, np.inf), (np.inf, np.inf), (np.inf, -np.inf)]), transform=ax.transData)
                else:
                    train_bin_body = pc
                    m = np.mean(pc.get_paths()[0].vertices[:, 0])
                    pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
                    
            # Set the color and linewidth of the mean lines
            # if not median_over_mean:
            #     train_parts['cmeans'].set_color('black')
            #     train_parts['cmeans'].set_linewidth(2) 
            # else:
            #     train_parts['cmedians'].set_color('black')
            #     train_parts['cmedians'].set_linewidth(2)

            # Adding individual data points as a scatter plot
            jitter = 0.0075 # adjust this value based on your actual data
            x_jitter = plot_df_train['x'] - width/4 + 1/2*width*plot_df_train['bin'] + np.random.uniform(-jitter, jitter, size=len(plot_df_train['x']))
            ax.scatter(x_jitter, plot_df_train['y'], color='black', alpha=alpha_jitter, s=20)
            
            # Overlaying with line plot
            # if not median_over_mean:
            #     column_means = np.nanmean(train, axis=0)
            # else:
            #     column_means = np.nanmedian(train, axis=0)
            # plt.plot(effs_train, column_means, color='C8', label="Train")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if train is not None:
        plt.legend([test_notbin_body, train_notbin_body, test_bin_body, train_bin_body], ['Test 50% Least ' + binarization_name, 'Train 50% Least ' + binarization_name, 'Test 50% Most ' + binarization_name, 'Train 50% Most ' + binarization_name])

    format_axis(ax)

    if ylim is not None:
        plt.ylim(ylim)

    return fig
        
# %%

def plot_line_scatter(test, effs_test, train=None, effs_train=None, ylabel='Mean Interval per Participant', xlabel="Efficacy", xjitter=0.1, yjitter=0.1, ylim=None, median_over_mean = False, true_values=None, effs_true=None, true_label='Mean of APE-trained', true_color='grey'):
    ''' Makes a line plot with overlaid lines connecting the mean with error bars given by the standard deviation of the mean
    
    Arguments
    ---------
    test : np.array of floats [n_participants, n_effs], metric of participant performance on test data
    effs_test : np.array of floats [n_effs], efficacy values for test data
    train : np.array of floats [n_participants, n_effs], metric of participant performance on train data
    effs_train : np.array of floats [n_effs], efficacy values for train data
    ylabel : str, label for y-axis
    xlabel : str, label for x-axis
    xjitter : float, amount of jitter for x-axis

    Returns
    -------
    fig : matplotlib figure
    '''

    fig, ax = plt.subplots(dpi=300)

    alpha_scatter = 0.10
    alpha_shade = 0.2      
    alpha_line = 0.9    

    color_test = 'C9'
    color_train = 'C8'

    # Calculate mean and SEM
    def get_summary_stats(data, use_median=False):
        if use_median:
            central_tendency = np.median(data, axis=0)
            variability = np.median(np.abs(data - np.median(data, axis=0)), axis=0) / np.sqrt(data.shape[0])
        else:
            central_tendency = np.mean(data, axis=0)
            variability = np.std(data, axis=0) / np.sqrt(data.shape[0])
        return central_tendency, variability
    
        # Apply jitter to data
    # def apply_jitter(data, effs):
    #     effs_jittered = effs + np.random.uniform(-xjitter, xjitter, size=data.shape)
    #     data_jittered = data + np.random.uniform(-yjitter, yjitter, size=data.shape)
    #     return effs_jittered, data_jittered

    def apply_jitter(data, effs):
        effs_jittered = effs + np.random.uniform(-xjitter, xjitter, size=data.shape)
        data_jittered = data + np.random.uniform(-yjitter, yjitter, size=data.shape)

        # Ensure that values do not go below 0
        effs_jittered = np.clip(effs_jittered, 0, None)
        data_jittered = np.clip(data_jittered, 0, None)

        return effs_jittered, data_jittered
    
    # If true_values and effs_true are provided, plot the true line
    if true_values is not None and effs_true is not None:
        ax.plot(effs_true, true_values, color=true_color, label=true_label, linestyle='--', alpha=alpha_line)
    
    
    # Plot for the test data
    test_central, test_variability = get_summary_stats(test, median_over_mean)
    ax.plot(effs_test, test_central, color=color_test, label='Test', alpha=alpha_line)
    ax.fill_between(effs_test, test_central - test_variability, test_central + test_variability, color=color_test, alpha=alpha_shade)
    effs_test_jittered, test_jittered = apply_jitter(test, effs_test)
    ax.scatter(effs_test_jittered, test_jittered.ravel(), color=color_test, alpha=alpha_scatter)
    
    # If train data is provided
    if train is not None and effs_train is not None:
        train_central, train_variability = get_summary_stats(train, median_over_mean)
        ax.plot(effs_train, train_central, color=color_train, label='Train', alpha=alpha_line)
        ax.fill_between(effs_train, train_central - train_variability, train_central + train_variability, color=color_train, alpha=alpha_shade)
        effs_train_jittered, train_jittered = apply_jitter(train, effs_train)
        ax.scatter(effs_train_jittered, train_jittered.ravel(), color=color_train, alpha=alpha_scatter)

    if ylim is not None:
        plt.ylim(ylim)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend()

    format_axis(ax)#, plot_line_multiplier=3)
    
    return fig
  
# %%

def plot_line_scatter_humans_ape_noape(test, effs_test, train=None, effs_train=None, ylabel='Mean Interval per Participant', xlabel="Efficacy", xjitter=0.1, yjitter=0.1, ylim=None, median_over_mean = False, true_values=None, true_stderr=None, effs_true=None, noape_values=None, noape_stderr= None, effs_noape=None, true_label='Mean of APE-trained', true_color='C1', noape_label='Mean of no-APE-trained', noape_color='C2'):
    ''' Makes a line plot with overlaid lines connecting the mean with error bars given by the standard deviation of the mean
    
    Arguments
    ---------
    test : np.array of floats [n_participants, n_effs], metric of participant performance on test data
    effs_test : np.array of floats [n_effs], efficacy values for test data
    train : np.array of floats [n_participants, n_effs], metric of participant performance on train data
    effs_train : np.array of floats [n_effs], efficacy values for train data
    ylabel : str, label for y-axis
    xlabel : str, label for x-axis
    xjitter : float, amount of jitter for x-axis

    Returns
    -------
    fig : matplotlib figure
    '''

    #fig, ax = plt.subplots(dpi=300)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    alpha_scatter = 0.10
    alpha_shade = 0.2      
    alpha_line = 0.9    

    color_test = '#006600'
    color_train = '#33cc33'

    # Calculate mean and SEM
    def get_summary_stats(data, use_median=False):
        if use_median:
            central_tendency = np.median(data, axis=0)
            variability = np.median(np.abs(data - np.median(data, axis=0)), axis=0) / np.sqrt(data.shape[0])
        else:
            central_tendency = np.mean(data, axis=0)
            variability = np.std(data, axis=0) / np.sqrt(data.shape[0])
        return central_tendency, variability
    
        # Apply jitter to data
    def apply_jitter(data, effs):
        effs_jittered = effs + np.random.uniform(-xjitter, xjitter, size=data.shape)
        data_jittered = data + np.random.uniform(-yjitter, yjitter, size=data.shape)

        # Ensure that values do not go below 0
        effs_jittered = np.clip(effs_jittered, 0, None)
        data_jittered = np.clip(data_jittered, 0, None)

        return effs_jittered, data_jittered
    
    # If true_values and effs_true are provided, plot the true line
    if true_values is not None and effs_true is not None:
        ax.plot(effs_true, true_values, color=true_color, label=true_label, alpha=alpha_line)
        #print(true_values)
        #print(true_stderr)
        ax.fill_between(effs_true, true_values - true_stderr, true_values + true_stderr, color=true_color, alpha=alpha_shade)

    if noape_values is not None and effs_noape is not None:
        ax.plot(effs_noape, noape_values, color=noape_color, label=noape_label, alpha=alpha_line)
        ax.fill_between(effs_noape, noape_values - noape_stderr, noape_values + noape_stderr, color=noape_color, alpha=alpha_shade)
    
    # Plot for the test data
    test_central, test_variability = get_summary_stats(test, median_over_mean)
    ax.scatter(effs_test, test_central, color=color_test, label='Without Cue', marker = "D", s=70)
    #ax.plot(effs_test, test_central, color=color_test, label='Test', alpha=alpha_line)
    #ax.fill_between(effs_test, test_central - test_variability, test_central + test_variability, color=color_test, alpha=alpha_shade)
    effs_test_jittered, test_jittered = apply_jitter(test, effs_test)
    ax.scatter(effs_test_jittered, test_jittered.ravel(), color=color_test, alpha=alpha_scatter)
    
    # If train data is provided
    if train is not None and effs_train is not None:
        train_central, train_variability = get_summary_stats(train, median_over_mean)
        ax.scatter(effs_train, train_central, color=color_train, label='With Cue', marker="D", s=70)
        # ax.plot(effs_train, train_central, color=color_train, label='Train', alpha=alpha_line)
        # ax.fill_between(effs_train, train_central - train_variability, train_central + train_variability, color=color_train, alpha=alpha_shade)
        effs_train_jittered, train_jittered = apply_jitter(train, effs_train)
        ax.scatter(effs_train_jittered, train_jittered.ravel(), color=color_train, alpha=alpha_scatter)

    if ylim is not None:
        plt.ylim(ylim)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend()

    format_axis(ax, plot_line_multiplier=2)
    
    return fig

# %%

def plot_line_scatter_humans_ape_noape_group(test, effs_test, train=None, effs_train=None, ylabel='Mean Interval per Participant', xlabel="Efficacy", xjitter=0.1, yjitter=0.1, ylim=None, median_over_mean = False, true_values=None, true_stderr=None, effs_true=None, noape_values=None, noape_stderr= None, effs_noape=None, true_label='Mean of APE-trained', true_color='C1', noape_label='Mean of no-APE-trained', noape_color='C2'):
    ''' Makes a line plot with overlaid lines connecting the mean with error bars given by the standard deviation of the mean
    
    Arguments
    ---------

    Returns
    -------
    fig : matplotlib figure
    '''

    #fig, ax = plt.subplots(dpi=300)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    alpha_scatter = 0.10
    alpha_shade = 0.2      
    alpha_line = 0.9    
    alpha_diamonds = 1

    color_test = '#006600'
    color_train = '#33cc33'

    # Calculate mean and SEM
    def get_summary_stats(data, use_median=False):
        if use_median:
            central_tendency = np.median(data, axis=0)
            variability = np.median(np.abs(data - np.median(data, axis=0)), axis=0) / np.sqrt(data.shape[0])
        else:
            central_tendency = np.mean(data, axis=0)
            variability = np.std(data, axis=0) / np.sqrt(data.shape[0])
        return central_tendency, variability
    
        # Apply jitter to data
    def apply_jitter(data, effs):
        effs_jittered = effs + np.random.uniform(-xjitter, xjitter, size=data.shape)
        data_jittered = data + np.random.uniform(-yjitter, yjitter, size=data.shape)

        # Ensure that values do not go below 0
        effs_jittered = np.clip(effs_jittered, 0, None)
        data_jittered = np.clip(data_jittered, 0, None)

        return effs_jittered, data_jittered
    
    if noape_values is not None and effs_noape is not None:
        ax.plot(effs_noape, noape_values, color=noape_color, label=noape_label, alpha=alpha_line*0.8)
        ax.fill_between(effs_noape, noape_values - noape_stderr, noape_values + noape_stderr, color=noape_color, alpha=alpha_shade*0.8)

    
    # If true_values and effs_true are provided, plot the true line
    if true_values is not None and effs_true is not None:
        ax.plot(effs_true, true_values, color=true_color, label=true_label, alpha=alpha_line)
        #print(true_values)
        #print(true_stderr)
        ax.fill_between(effs_true, true_values - true_stderr, true_values + true_stderr, color=true_color, alpha=alpha_shade)


    for current_group in range(len(test)):
        
        # If train data is provided
        if train is not None and effs_train is not None:
            current_train = train[current_group]
            current_effs_train = effs_train[current_group]

            if current_group == 0:
                label = 'With Cue'
            else:
                label = None

            train_central, train_variability = get_summary_stats(current_train, median_over_mean)
            ax.scatter(current_effs_train, train_central, color=color_train, label=label, marker="D", s=70, alpha=alpha_diamonds)
            # ax.plot(effs_train, train_central, color=color_train, label='Train', alpha=alpha_line)
            # ax.fill_between(effs_train, train_central - train_variability, train_central + train_variability, color=color_train, alpha=alpha_shade)
            effs_train_jittered, train_jittered = apply_jitter(current_train, current_effs_train)
            ax.scatter(effs_train_jittered, train_jittered.ravel(), color=color_train, alpha=alpha_scatter)


    for current_group in range(len(test)):

        current_test = test[current_group]
        current_effs_test = effs_test[current_group]

        if current_group == 0:
            label = 'Without Cue'
        else:
            label = None
        
        # Plot for the test data
        test_central, test_variability = get_summary_stats(current_test, median_over_mean)
        ax.scatter(current_effs_test, test_central, color=color_test, label=label, marker = "D", s=70, alpha=alpha_diamonds)
        #ax.plot(effs_test, test_central, color=color_test, label='Test', alpha=alpha_line)
        #ax.fill_between(effs_test, test_central - test_variability, test_central + test_variability, color=color_test, alpha=alpha_shade)
        effs_test_jittered, test_jittered = apply_jitter(current_test, current_effs_test)
        ax.scatter(effs_test_jittered, test_jittered.ravel(), color=color_test, alpha=alpha_scatter)
        
        if ylim is not None:
            plt.ylim(ylim)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend()

        format_axis(ax, plot_line_multiplier=2)
    
    return fig

# %% 

def plot_line_scatter_group(test, effs_test, train=None, effs_train=None, ylabel='Mean Interval per Participant', xlabel="Controllability", xjitter=0.1, yjitter=0.1, ylim=None, median_over_mean=False):
    ''' Makes a line plot with overlaid scatter plots for test and train data, connecting all test data points with a single line, including shaded error bars for these connections.
    '''

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    
    alpha_scatter = 0.10
    alpha_shade = 0.2      
    alpha_line = 1  
    alpha_diamonds = 1

    color_test = '#006600'
    color_train = '#33cc33'

    # Calculate mean and SEM
    def get_summary_stats(data, use_median=False):
        if use_median:
            central_tendency = np.median(data, axis=0)
            variability = np.median(np.abs(data - np.median(data, axis=0)), axis=0) / np.sqrt(data.shape[0])
        else:
            central_tendency = np.mean(data, axis=0)
            variability = np.std(data, axis=0) / np.sqrt(data.shape[0])
        return central_tendency, variability
    
        # Apply jitter to data
    def apply_jitter(data, effs):
        effs_jittered = effs + np.random.uniform(-xjitter, xjitter, size=data.shape)
        data_jittered = data + np.random.uniform(-yjitter, yjitter, size=data.shape)

        # Ensure that values do not go below 0
        effs_jittered = np.clip(effs_jittered, 0, None)
        data_jittered = np.clip(data_jittered, 0, None)

        return effs_jittered, data_jittered

    # Initialize lists to collect values
    all_test_central = []
    all_effs_test = []
    all_test_variability = []

    # Combine and plot data for train and test groups
    for current_group in range(len(test)):
        current_test = test[current_group]
        current_effs_test = effs_test[current_group]
        test_central, test_variability = get_summary_stats(current_test, median_over_mean)

        # Collect values
        all_test_central.extend(test_central)
        all_effs_test.extend(current_effs_test)
        all_test_variability.extend(test_variability)

        if train is not None and effs_train is not None:
            current_train = train[current_group]
            current_effs_train = effs_train[current_group]
            effs_train_jittered, train_jittered = apply_jitter(current_train, current_effs_train)
            ax.scatter(effs_train_jittered, train_jittered.ravel(), color=color_train, alpha=alpha_scatter)

            train_central, train_variability = get_summary_stats(current_train, median_over_mean)

        # Scatter plots for jittered data
        effs_test_jittered, test_jittered = apply_jitter(current_test, current_effs_test)
        ax.scatter(effs_test_jittered, test_jittered.ravel(), color=color_test, alpha=alpha_scatter)

        ### Plot means/medians
        if current_group == 0:
            label_test = "Without Cue"
            label_train = "With Cue"
        else:
            label_test = None
            label_train = None
        ax.scatter(current_effs_test, test_central, marker='D', s=70, color=color_test, alpha=alpha_diamonds, label=label_test)
        if train is not None and effs_train is not None:
            ax.scatter(current_effs_train, train_central, marker='D', s=70, color=color_train, alpha=alpha_diamonds, label=label_train)
            
    # Sort the collected test values by effs_test
    sorted_indices = np.argsort(all_effs_test)
    sorted_test_central = np.array(all_test_central)[sorted_indices]
    sorted_effs_test = np.array(all_effs_test)[sorted_indices]
    sorted_test_variability = np.array(all_test_variability)[sorted_indices]

    # Plot the sorted values
    ax.plot(sorted_effs_test, sorted_test_central, color=color_test, alpha=alpha_line)
    ax.fill_between(sorted_effs_test, sorted_test_central - sorted_test_variability, sorted_test_central + sorted_test_variability, color=color_test, alpha=alpha_shade)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    #ax.legend(['Test', 'Train'], loc='best')
    ax.legend()


    format_axis(ax)

    return fig

# %% 

# %%

def frac_takes_lineplot(taus, peek_probs,ylim=None, smoothing_window = 1):
    ''' makes a lineplot showing the probability of observing vs. betting / peeking v. betting for different tau values
    
    Arguments
    ---------
    taus : np.array of floats [n_taus] : tested tau values

    Returns
    -------
    fig : matplotlib.fig item

    '''

    mean_peek_probs = peek_probs.mean(axis=0)
    stderr_peek_probs = peek_probs.std(axis=0)/np.sqrt(peek_probs.shape[0])

    ### smoothing based on rolling average
    if smoothing_window > 1:
        mean_peek_probs = pd.DataFrame(mean_peek_probs).apply(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean(), axis=1).values
        stderr_peek_probs = pd.DataFrame(stderr_peek_probs).apply(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean(), axis=1).values

    ## create figure
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111)

    steps = np.arange(1, mean_peek_probs.shape[1]+1)

    for i in range(len(mean_peek_probs)):
        # ax.plot(steps, mean_peek_probs[i], label=1-taus[i], color='C%d' %i)
        # ax.fill_between(steps, mean_peek_probs[i] - stderr_peek_probs[i], mean_peek_probs[i] + stderr_peek_probs[i], color='C%d' %i, alpha=0.2)
        ax.plot(steps, mean_peek_probs[i], label=1-taus[i], color='C%d' %i)
        ax.fill_between(steps, mean_peek_probs[i] - stderr_peek_probs[i], mean_peek_probs[i] + stderr_peek_probs[i], color='C%d' %i, alpha=0.2)

    #ax.bar(range(len(policy)))

    ax.legend(title="Efficacy")

    #ax.set_xlabel("Steps")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Proportion of Observations")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_xlim(0, 50)

    format_axis(ax)

    #ax.set_title("Peek-Probability for Sample Model")
    #plt.tight_layout()

    return fig
# %%


def frac_correct_takes_lineplot(taus, trajss_rw, includes_sleep = False, smoothing_window = 1, ylim=None):
    ''' makes a lineplot showing the probability of observing vs. betting / peeking v. betting for different tau values
    
    Arguments
    ---------
    taus : np.array of floats [n_taus] : tested tau values
    logitss : np.array of floats [n_taus, n_episodes, n_steps, n_actions] , policy as log probabilities
    pss : np.array of floats [n_taus, n_episodes, n_steps, n_arms], integers representing actions

    Returns
    -------
    fig : matplotlib.fig item

    '''

    corr = trajss_rw == 1
    incorr = trajss_rw == 0

    total_prob_take = corr + incorr
    cond_prob = corr / total_prob_take

    ## compute fractions
    
    mean_probs = np.nanmean(cond_prob,axis=0)
    stderr_probs = np.nanstd(cond_prob, axis=0)/np.sqrt(cond_prob.shape[0])
    
    ### smoothing based on rolling average
    if smoothing_window > 1:
        mean_probs = pd.DataFrame(mean_probs).apply(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean(), axis=1).values
        stderr_probs = pd.DataFrame(stderr_probs).apply(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean(), axis=1).values


    ## compute mean and stderr

    ## create figure
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    steps = range(mean_probs.shape[1])

    for i in range(len(mean_probs)):
        # ax.plot(steps, mean_peek_probs[i], label=1-taus[i], color='C%d' %i)
        # ax.fill_between(steps, mean_peek_probs[i] - stderr_peek_probs[i], mean_peek_probs[i] + stderr_peek_probs[i], color='C%d' %i, alpha=0.2)
        ax.plot(steps, mean_probs[i], label=1-taus[i], color='C%d' %i)
        ax.fill_between(steps, mean_probs[i] - stderr_probs[i], mean_probs[i] + stderr_probs[i], color='C%d' %i, alpha=0.2)

    #ax.bar(range(len(policy)))

    ax.legend(title="Efficacy")

    ax.set_xlabel("Steps")
    ax.set_ylabel("Proportion Take Correct")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if ylim is not None:
        ax.set_ylim(ylim)

    format_axis(ax)

    #ax.set_title("Peek-Probability for Sample Model")

    plt.tight_layout()

    return fig
# %%
