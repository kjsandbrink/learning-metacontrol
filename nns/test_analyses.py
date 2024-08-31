# Kai Sandbrink
# 2022-10-30
# This script contains some analyses to run on traces of PeekTake Models

# %% LIBRARY IMPORTS

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

from utils import format_axis

# %% STACKPLOT FUNCTION

def policy_barplot(logitss, pss = None, episode = None):
    ''' makes a stackplot showing average policy for saved log probs
    
    Arguments
    ---------
    logitss : np.array of floats [n_episodes, n_steps, n_actions] or [n_steps, n_actions] , policy as log probabilities
    pss : np.array of floats [n_episodes, n_steps, n_arms], payout probabilities for both arms (used for coloring)
    episode : int or None, if None take average else select the given episode

    Returns
    -------
    fig : matplotlib.fig item

    '''

    ## Reduce dimension if necessary (i.e. received data for > 1 episodes)
    probs = np.exp(logitss)

    if len(probs.shape) > 2 and episode is None:
        policy = probs.mean(axis=0)
        assert pss is None, "averaging method not implemented when pss are included"
    elif episode is not None:
        policy = probs[episode]
        pss = pss[episode]
    else:
        policy = probs

    #action_probs = mean_policy.T.tolist()

    assert policy.shape[1] <= 4, "not implemented yet for PeekVTake Tasks!"
    
    ## switch sleep to final position (if the action is included)
    if policy.shape[1] == 4:
        temp = np.empty_like(policy)
        temp[:,0] = policy[:,0]
        temp[:,1:3] = policy[:,2:4]
        temp[:,3] = policy[:,1]
        policy = temp

    ## check to make sure correct arm is listed first
    '''
    greater_arm = np.argmax(np.sum(policy[:, 1:3], axis=0))
    if greater_arm == 1:
        temp = policy.copy()
        policy[:, 1] = temp[:,2]
        policy[:,2] = temp[:,1]
    '''

    assert pss.shape[1] == 2 or pss is None, "pss method not implemented for >2 arms"

    ## relabel policy such that correct arm is listed first
    if pss is not None:
        correct_arm = np.where(pss[:,0] > pss[:,1], 0, 1)
        #0 where arm 0 is greater arm, 1 where arm 1 is greater arm
        temp = policy.copy()
        policy[:,1] = np.where(~correct_arm.astype(bool), temp[:,1], temp[:,2])
        policy[:,2] = np.where(correct_arm.astype(bool), temp[:,1], temp[:,2])

        actions = ['Peek', 'Take (Correct)', 'Take (Incorrect)', 'Sleep'] #after keeping correction
        colors = ['blue', 'green', 'red', 'grey']

    else:
        actions = ['Peek', 'Take Arm 1', 'Take Arm 2', 'Sleep'] 
        colors = ['blue', 'yellow', 'grey', 'blue']

    ## create figure
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111)

    steps = range(len(policy))

    #ax.stackplot(range(len(policy)), policy.T, colors=['blue', 'green', 'red', 'grey'])

    for i in range(policy.shape[1]):
        ax.bar(steps, policy[:,i], bottom=policy[:,0:i].sum(axis=1), label=actions[i], color=colors[i])

    #ax.bar(range(len(policy)))

    ax.legend()

    ax.set_xlim([-0.5,len(policy)-0.5])
    #ax.set_ylim([0,1])

    ax.set_xlabel("Step")
    ax.set_ylabel("Policy")

    format_axis(ax)

    ax.set_title("Agent Policy in Episode of Observe-Bet-Efficacy Task")

    plt.tight_layout()

    return fig

# %% MULTIPLE EPISODES POLICY LINEPLOT

def frac_takes_lineplot(taus, logitss, ylim=None, ylabel="Probability of Observing", smoothing_window = 1, cmap = None):
    ''' makes a lineplot showing the probability of observing vs. betting / peeking v. betting for different tau values
    
    Arguments
    ---------
    taus : np.array of floats [n_taus] : tested tau values
    logitss : np.array of floats [n_taus, n_episodes, n_steps, n_actions] or [n_models, n_taus, n_episodes, n_steps, n_actions], policy as log probabilities
    ### exp : bool, is this a plot for the experimental or control group? (decides color scheme etc)

    Returns
    -------
    fig : matplotlib.fig item

    '''

    #note: peek action is action 0, action 1 is take L, action 2 is take R

    logitss = np.array(logitss)

    ## depending on shape, infer that we have multiple models
    if len(logitss.shape) > 4:
        peek_probs = np.exp(logitss[:,:,:,:,0])
        peek_probs = np.mean(peek_probs, axis=2)

        mean_peek_probs = peek_probs.mean(axis=0)
        stderr_peek_probs = peek_probs.std(axis=0)/np.sqrt(peek_probs.shape[0])        
    else:

        peek_probs = np.exp(logitss[:,:,:,0])

        mean_peek_probs = peek_probs.mean(axis=1)
        stderr_peek_probs = peek_probs.std(axis=1)/np.sqrt(peek_probs.shape[1])

    ### smoothing based on rolling average
    if smoothing_window > 1:
        mean_peek_probs = pd.DataFrame(mean_peek_probs).apply(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean(), axis=1).values
        stderr_peek_probs = pd.DataFrame(stderr_peek_probs).apply(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean(), axis=1).values

    ## create figure
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)

    steps = np.arange(1, mean_peek_probs.shape[1]+1)

    for i in range(len(mean_peek_probs)):
        # ax.plot(steps, mean_peek_probs[i], label=1-taus[i], color='C%d' %i)
        # ax.fill_between(steps, mean_peek_probs[i] - stderr_peek_probs[i], mean_peek_probs[i] + stderr_peek_probs[i], color='C%d' %i, alpha=0.2)
        if cmap is None:
            ax.plot(steps, mean_peek_probs[i], label=r"$\xi=$" + str(1-taus[i]), color='C%d' %i)
            ax.fill_between(steps, mean_peek_probs[i] - stderr_peek_probs[i], mean_peek_probs[i] + stderr_peek_probs[i], color='C%d' %i, alpha=0.2)
        else:
            ax.plot(steps, mean_peek_probs[i], label=r"$\xi=$" + str(1-taus[i]), color=cmap((len(mean_peek_probs) - i)/len(mean_peek_probs)))
            ax.fill_between(steps, mean_peek_probs[i] - stderr_peek_probs[i], mean_peek_probs[i] + stderr_peek_probs[i], color=cmap((len(mean_peek_probs) - i)/len(mean_peek_probs)), alpha=0.2)

    #ax.bar(range(len(policy)))

    #ax.legend(title="Efficacy")
    ax.legend()

    #ax.set_xlabel("Steps")
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlim(0, 50)

    format_axis(ax)

    #ax.set_title("Peek-Probability for Sample Model")
    #plt.tight_layout()

    return fig

# %% MULTIPLE EPISODES POLICY LINEPLOT

def frac_correct_takes_lineplot(taus, logitss, pss, includes_sleep = False, smoothing_window = 1, ylim=None, cmap=None, ylabel="Probability of Correct Bet"):
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

    #note: peek action is action 0, action 1 is take L, action 2 is take R

    probss = np.exp(np.array(logitss))
    pss = np.array(pss)

    ### FIGURE OUT CORRECT TAKES

    correct_arm = np.where(pss[...,0] > pss[...,1], 0, 1)

    if not includes_sleep:
        arm1 = 1
        arm2 = 2
    else:
        arm1 = 2
        arm2 = 3

    prob_correct_take = np.where(~correct_arm.astype(bool), probss[...,arm1], probss[...,arm2])
    prob_incorrect_take = np.where(correct_arm.astype(bool), probss[...,arm1], probss[...,arm2])
    total_prob_take = prob_correct_take + prob_incorrect_take
    cond_prob = prob_correct_take / total_prob_take

    ## compute fractions
    
    ## depending on shape, infer that we have multiple models
    if len(logitss.shape) <= 4:
        mean_probs = cond_prob.mean(axis=1)
        stderr_probs = cond_prob.std(axis=1)/np.sqrt(cond_prob.shape[1])    
    else:
        cond_prob = cond_prob.mean(axis=2)
    
        mean_probs = cond_prob.mean(axis=0)
        stderr_probs = cond_prob.std(axis=0)/np.sqrt(cond_prob.shape[1])

    ### smoothing based on rolling average
    if smoothing_window > 1:
        mean_probs = pd.DataFrame(mean_probs).apply(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean(), axis=1).values
        stderr_probs = pd.DataFrame(stderr_probs).apply(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean(), axis=1).values


    ## compute mean and stderr

    ## create figure
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)

    steps = np.arange(1, mean_probs.shape[1]+1)

    for i in range(len(mean_probs)):
        # ax.plot(steps, mean_peek_probs[i], label=1-taus[i], color='C%d' %i)
        # ax.fill_between(steps, mean_peek_probs[i] - stderr_peek_probs[i], mean_peek_probs[i] + stderr_peek_probs[i], color='C%d' %i, alpha=0.2)
        if cmap is None:
            #ax.plot(steps, mean_probs[i], label=1-taus[i], color='C%d' %i)
            ax.plot(steps, mean_probs[i], label=r"$\xi=$" + str(1-taus[i]), color='C%d' %i)
            ax.fill_between(steps, mean_probs[i] - stderr_probs[i], mean_probs[i] + stderr_probs[i], color='C%d' %i, alpha=0.2)
        else:
            ax.plot(steps, mean_probs[i], label=r"$\xi=$" + str(1-taus[i]), color=cmap((len(mean_probs) - i)/len(mean_probs)))
            ax.fill_between(steps, mean_probs[i] - stderr_probs[i], mean_probs[i] + stderr_probs[i], color=cmap((len(mean_probs) - i)/len(mean_probs)), alpha=0.2)

    #ax.bar(range(len(policy)))

    #ax.legend(title="Efficacy")
    ax.legend()

    #ax.set_xlabel("Steps")
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    format_axis(ax)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlim(0, 50)

    #ax.set_title("Peek-Probability for Sample Model")

    plt.tight_layout()

    return fig

# %% FRAC SLEEPS LINEPLOT

def frac_sleeps_lineplot(taus, logitss, ylim=None, smoothing_window = 1, cmap=None, ylabel="Probability of Sleeps"):
    ''' makes a lineplot showing the probability of sleeping for different tau values
    
    Arguments
    ---------
    taus : np.array of floats [n_taus] : tested tau values
    logitss : np.array of floats [n_taus, n_episodes, n_steps, n_actions] , policy as log probabilities
    ### exp : bool, is this a plot for the experimental or control group? (decides color scheme etc)

    Returns
    -------
    fig : matplotlib.fig item

    '''

    #note: peek action is action 0, action 1 is take L, action 2 is take R

    logitss = np.array(logitss)

    ## depending on shape, infer that we have multiple models
    if len(logitss.shape) > 4:
        peek_probs = np.exp(logitss[:,:,:,:,1])
        peek_probs = np.mean(peek_probs, axis=2)

        mean_probs = peek_probs.mean(axis=0)
        stderr_probs = peek_probs.std(axis=0)/np.sqrt(peek_probs.shape[0])        
    else:

        peek_probs = np.exp(logitss[:,:,:,1])

        mean_probs = peek_probs.mean(axis=1)
        stderr_probs = peek_probs.std(axis=1)/np.sqrt(peek_probs.shape[1])

    ### smoothing based on rolling average
    if smoothing_window > 1:
        mean_probs = pd.DataFrame(mean_probs).apply(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean(), axis=1).values
        stderr_probs = pd.DataFrame(stderr_probs).apply(lambda x: x.rolling(window=smoothing_window, min_periods=1).mean(), axis=1).values

    ## create figure
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)

    steps = np.arange(1, mean_probs.shape[1]+1)

    for i in range(len(mean_probs)):
        # ax.plot(steps, mean_peek_probs[i], label=1-taus[i], color='C%d' %i)
        # ax.fill_between(steps, mean_peek_probs[i] - stderr_peek_probs[i], mean_peek_probs[i] + stderr_peek_probs[i], color='C%d' %i, alpha=0.2)
        # ax.plot(steps, mean_peek_probs[i], label=1-taus[i], color='C%d' %i)
        # ax.fill_between(steps, mean_peek_probs[i] - stderr_peek_probs[i], mean_peek_probs[i] + stderr_peek_probs[i], color='C%d' %i, alpha=0.2)
        if cmap is None:
            #ax.plot(steps, mean_probs[i], label=1-taus[i], color='C%d' %i)
            ax.plot(steps, mean_probs[i], label=r"$\xi=$" + str(1-taus[i]), color='C%d' %i)
            ax.fill_between(steps, mean_probs[i] - stderr_probs[i], mean_probs[i] + stderr_probs[i], color='C%d' %i, alpha=0.2)
        else:
            ax.plot(steps, mean_probs[i], label=r"$\xi=$" + str(1-taus[i]), color=cmap((len(mean_probs) - i)/len(mean_probs)))
            ax.fill_between(steps, mean_probs[i] - stderr_probs[i], mean_probs[i] + stderr_probs[i], color=cmap((len(mean_probs) - i)/len(mean_probs)), alpha=0.2)


    #ax.bar(range(len(policy)))

    #ax.legend(title="Efficacy")
    ax.legend()

    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    format_axis(ax)

    ax.set_xlim(0, 50)

    #ax.set_title("Peek-Probability for Sample Model")

    #plt.tight_layout()

    return fig


# %% APE ESTIMATE ACCURACY LINEPLOT

def ape_accuracy_lineplot(taus, control_errss):
    ''' makes a lineplot showing the probability of observing vs. betting / peeking v. betting for different tau values
    
    Arguments
    ---------
    taus : np.array of floats [n_taus] : tested tau values
    control_errss : np.array of floats [n_taus, n_episodes, n_steps] , error in tau prediction

    Returns
    -------
    fig : matplotlib.fig item

    '''

    ## compute fractions
    control_errss = np.array(control_errss)
    mean_probs = control_errss.mean(axis=1)
    stderr_probs = control_errss.std(axis=1)/np.sqrt(control_errss.shape[1])

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
    ax.set_ylabel("Squared Error APE Prediction")

    format_axis(ax)

    #ax.set_title("Peek-Probability for Sample Model")

    plt.tight_layout()

    return fig

# %% LINE PLOT SHOWING CONTROL ESTIMATE

def within_episode_efficacy_lineplot(controlss, tau=None, episode = None):
    ''' makes a stackplot showing average policy for saved log probs
    
    Arguments
    ---------
    controlss : np.array of floats [n_episodes, n_steps] or [n_steps]
    episode : int or None, selected episode. if not None controlss assumed to be of shape [n_episodes, n_steps], else [n_steps]

    Returns
    -------
    fig : matplotlib.fig item

    '''

    if episode is not None:
        controlss = controlss[episode]

    ## create figure
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111)

    steps = range(len(controlss))

    ax.plot(steps, 1-controlss, label='Agent Estimation')
        ## to account for fact that efficacy signal helps code for % chance of failure

    ax.set_xlim((0,len(controlss)))

    if tau is not None:
        ax_x_lim = ax.get_xlim()

        ax.plot(ax_x_lim, [1-tau]*len(ax_x_lim), color='red', linewidth=2, label='True Efficacy')

    ax.set_ylim((0,1))

    ax.legend()

    ax.set_xlabel("Step")
    ax.set_ylabel("Estimated Efficacy")

    format_axis(ax)

    ax.set_title("Agent Estimation of Efficacy over Time, Efficacy=%d" %(100*(1-tau)))

    plt.tight_layout()

    return fig

# %% COMPARISON CURVES SEVERAL RUNS

def plot_comparison_curves_several_runs(x_exp = None, y_exp = None, x_control = None, y_control = None, x_baselines = None, y_baselines = None, x_theory = None, y_theory = None, title = '', axis_xlabel = '', axis_ylabel ='', label_exp = '', label_control = '', label_baselines = '', label_theory = '', x_units = None, ylim = None, marker='',  shaded_intervals=None, color_exp = 'C0', color_control ='C1', color_baselines = 'C2'):
    """ Plots comparison curves over means of metrics (perf, number actions taken per episode etc) for models

    Arguments
    ---------
    x_exp : iterable 1D
    y_exp : np.array or list of lists [nr_x_axis_points, nr_models]

    Returns
    -------
    fig : matplotlib.fig item
    """

    ### preprocessing
    if y_exp is not None:
        y_exp = np.array(y_exp)
        means_exp = y_exp.mean(axis=1)
        stderr_exp = y_exp.std(axis=1)/np.sqrt(len(y_exp))

    if y_control is not None:
        y_control = np.array(y_control)
        means_control = y_control.mean(axis=1)
        stderr_control = y_control.std(axis=1)/np.sqrt(len(y_control))

    if y_baselines is not None:
        y_baselines = np.array(y_baselines)
        if len(y_baselines.shape) == 2:
            means_baselines = y_baselines.mean(axis=1)
            stderr_baselines = y_baselines.std(axis=1)/np.sqrt(len(y_baselines))
        else:
            means_baselines = y_baselines
            stderr_baselines = np.zeros_like(means_baselines)

    if x_units == 'k':
        x_exp = np.array(x_exp)
        x_exp/=1000

        if x_control is not None:
            x_control = np.array(x_control)
            x_control/=1000

        if x_baselines is not None:
            x_baselines = np.array(x_baselines)
            x_baselines/=1000
        
    ### set up figure
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)

    ## shade regions
    if shaded_intervals is not None:
        for interval in shaded_intervals:
            ax.axvspan(interval[0], interval[1], facecolor='grey', alpha=0.2, edgecolor=None, closed=False)

    ### plot
    if y_exp is not None:
        ax.plot(x_exp, means_exp, color=color_exp, label=label_exp, marker=marker)
        ax.fill_between(x_exp, means_exp - stderr_exp, means_exp + stderr_exp, color=color_exp, alpha=0.2)

    if y_control is not None:
        ax.plot(x_control, means_control, color=color_control, label=label_control, marker=marker)
        ax.fill_between(x_control, means_control - stderr_control, means_control + stderr_control, color=color_control, alpha=0.2)

    if y_baselines is not None:
        ax.plot(x_baselines, means_baselines, color=color_baselines, label=label_baselines, marker=marker)
        ax.fill_between(x_baselines, means_baselines - stderr_baselines, means_baselines + stderr_baselines, color=color_baselines, alpha=0.2)

    if y_theory is not None:
        ax.scatter(x_theory, y_theory, color='red', label=label_theory, marker='+')

    ### format axis
    ax.set_xlabel(axis_xlabel)
    ax.set_ylabel(axis_ylabel)
    #ax.set_title(title)

    if x_units == 'k':
        ax.xaxis.set_major_formatter(FormatStrFormatter('%dk'))

    if label_exp != '' or label_control != '' or label_baselines != '':
        ax.legend()

    if ylim is not None:
        ax.set_ylim(ylim)

    format_axis(ax)

    return fig

# %% COMPARISON CURVES SEVERAL RUNS

def plot_behavior_mistrained(mistrained_xs, mistrained_model_ys, original_xs=None, original_model_ys=None, axis_xlabel=None, axis_ylabel= None, target_taus = None, cmap = None, figsize=None, perturbation_taus = None, font_size_multiplier = 1.5):

    ### set up figure
    if figsize is None:
        fig = plt.figure(dpi=400)
    else:
        fig = plt.figure(dpi=400, figsize=figsize)
    ax = fig.add_subplot(111)

    if target_taus is None and perturbation_taus is None:
        target_taus = mistrained_xs
    elif perturbation_taus is not None and target_taus is None:
        target_taus = perturbation_taus

    if original_xs is not None and original_model_ys is not None:   
        
        counter_peeks_mean = np.mean(original_model_ys, axis=0)
        counter_peeks_stderr = np.std(original_model_ys, axis=0)/np.sqrt(len(mistrained_model_ys))

        ax.plot(original_xs, counter_peeks_mean, color='black', label='Original', linewidth=2.5)
        ax.fill_between(original_xs, counter_peeks_mean - counter_peeks_stderr, counter_peeks_mean + counter_peeks_stderr, color='black', alpha=0.2)

    ## plot lines corresponding to models mistrained to particular values
    for i_tau, tau in enumerate(target_taus):

        counter_peeks_mean = np.mean(mistrained_model_ys[:,i_tau], axis=0)
        counter_peeks_stderr = np.std(mistrained_model_ys[:,i_tau], axis=0)/np.sqrt(len(mistrained_model_ys))

        if cmap is None:
            color= 'C%d' %i_tau
        else:
            if perturbation_taus is None or tau != 0:
                color = cmap((len(target_taus) - i_tau)/len(target_taus))
            else:
                color = 'black'

        if perturbation_taus is not None:
            label = - tau
            if tau != 0:
                linestyle='dashed'
                linewidth=None
                alpha=0.9
                alpha_shade=0.15
            else:
                label = 0
                color = 'black'
                linestyle='solid'
                linewidth = 2.5
                alpha=1
                alpha_shade=0.2
        else:
            label = 1 - tau
            linestyle='dashed'
            linewidth=None
            alpha=0.9
            alpha_shade=0.15

        # ax.plot(mistrained_xs, counter_peeks_mean, color=color, label=label, linestyle='dashed', alpha=0.9)
        # ax.fill_between(mistrained_xs, counter_peeks_mean - counter_peeks_stderr, counter_peeks_mean + counter_peeks_stderr, color=color, alpha=0.15)
        ax.plot(mistrained_xs, counter_peeks_mean, color=color, label=label, linestyle=linestyle, alpha=alpha, linewidth=linewidth)
        ax.fill_between(mistrained_xs, counter_peeks_mean - counter_peeks_stderr, counter_peeks_mean + counter_peeks_stderr, color=color, alpha=alpha_shade)

    ax.set_xlabel(axis_xlabel)
    ax.set_ylabel(axis_ylabel)

    ax.legend()
    format_axis(ax, font_size_multiplier = font_size_multiplier)
        
    return fig
    
# %% UNCERTAINTY CALCULATION

def uncertainty_calc(actionss, valuess, episode = None):

    if episode is not None:
        actions = actionss[episode]
        values = actionss[episode]

# %%

def plot_evidence_ratios(evidence_ratios, effs_sorted, cmap, ylabel, xlabel = 'Time since last observe', jitter = False, ylim=None, xlim=None):
    ''' Create the evidence ratio plot (evidence ratio is metric as a function of last observation) for the data from the sample network trajectories

    Arguments
    ---------
    evidence_ratios : np.array of floats [tau_levels, n_episodes, n_steps], given metric
    effs_sorted : np.array of floats [tau_levels]
    cmap : matplotlib colormap
    ylabel : str
    jitter : bool, whether to add jitter to the data points
    ylim : tuple of floats, limits of y axis

    Returns
    -------
    fig : matplotlib figure
    
    '''

    #evidence_ratios = np.exp(evidence_ratios)
    n_participants = len(evidence_ratios)

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    # for i in range(n_participants):
    #     for j, eff in enumerate(effs_sorted):
    #         ax.plot(range(evidence_ratios[i,j]), evidence_ratios[i,j], color=cmap(eff), alpha=0.5)

    #ax.legend()

    for j, eff in enumerate(effs_sorted):
            
        mean_sorted = np.nanmean(evidence_ratios[j],axis=0)
        stderr_sorted = np.nanstd(evidence_ratios[j],axis=0)/np.sqrt(n_participants)

        #print(mean_sorted, eff)

        ax.plot(range(len(mean_sorted)), mean_sorted, color=cmap((eff+0.2)/1.2), label='Eff ' + str(eff), linewidth=3.5)
        ax.fill_between(range(len(mean_sorted)), mean_sorted - stderr_sorted, mean_sorted + stderr_sorted, color=cmap((eff+0.2)/1.2), alpha=0.2)

    if jitter:
        print("Still needs to be implemented")
        # jitter_strength = 0.02
        # # Adding individual data points as a scatter plot
        # x_jitter = np.array(effs_sorted) + np.random.uniform(-jitter, jitter, size=evidence_ratios.shape)

        # ax.scatter(x_jitter, evidence_ratios, color='black', alpha=0.3, s=20)

    ### format axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.legend()
    format_axis(ax)
    
    return fig
# %%
