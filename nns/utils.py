# Kai Sandbrink
# 2022-03-09
# Various utility functions

# %% LIBRARY IMPORTS

import matplotlib.pyplot as plt
import numpy as np
import base64

import yaml, os, copy, math
from datetime import datetime

# %% GENERAL FUNCTIONS

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def combine_tuples(tuples):
    '''
    
    Arguments
    ---------
    tuples : list of tuples
    
    Returns
    -------
    combined : tuple
    '''
    
    combined = []
    [combined.extend(list(x)) for x in tuples]

    return tuple(combined)

def str2bool(v):
    ''' https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python '''
    return v.lower() in ("yes", "true", "t", "1")

def make_timed_results_folder(base_folder = None, addition=''):
    results_folder = '%s%s' %(datetime.now().strftime('%Y%m%d%H%M%S'), addition)
    if base_folder is not None:
        results_folder = os.path.join(base_folder, results_folder)
    
    os.makedirs(results_folder, exist_ok=True)
    return results_folder

def one_hot_encode_int(a, max_a, tensor_shape = False):
    one_hot_encoded = np.zeros((max_a,))
    one_hot_encoded[a] = 1
    if tensor_shape:
        one_hot_encoded = one_hot_encoded.reshape((1,-1))
    return one_hot_encoded

def flatten(l):
    ''' Reference: https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists '''
    return [item for sublist in l for item in sublist]

def get_timestamp():
    return datetime.now().strftime('%Y%m%d%H%M%S')

# %% PLOTTING FUNCTIONS

def format_axis(ax, line_width_multiplier=2, font_size_multiplier=1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=8*font_size_multiplier)
    ax.yaxis.set_tick_params(size=8*font_size_multiplier)

    ## SET AXIS WIDTHS
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2*line_width_multiplier)

    # increase tick width
    ax.tick_params(width=2*line_width_multiplier)

    ax.xaxis.label.set_fontsize(16*font_size_multiplier)
    ax.yaxis.label.set_fontsize(16*font_size_multiplier)

    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(14*font_size_multiplier)

    if ax.get_legend() is not None:
        for item in ax.get_legend().get_texts():
            item.set_fontsize(13*font_size_multiplier)
        
        for legobj in ax.get_legend().legendHandles:
            legobj.set_linewidth(2.0*line_width_multiplier)

        if ax.get_legend().get_title() is not None:
            ax.get_legend().get_title().set_fontsize(13*font_size_multiplier)

    for line in ax.lines:
        line.set_linewidth(2.5*font_size_multiplier)

    ax.title.set_fontsize(24*font_size_multiplier)

def plot_learning_curve(rews, smoothing_window=1, name='Q-learner', several_runs=False):
    """

    Parameters
    ----------
    rews
    smoothing_window
    name
    several_runs : bool, indicates whether rews is a nested list of rewards per run or not
    """

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    if several_runs:
        for rew_r in rews:
            ax.plot(range(len(rew_r)-smoothing_window+1), np.convolve(rew_r, np.ones(smoothing_window)/smoothing_window, mode='valid'), alpha=0.3, c='C0', label='individual')

        rews = np.array(rews).mean(axis=0)

    ax.plot(range(len(rews)-smoothing_window+1), np.convolve(rews, np.ones(smoothing_window)/smoothing_window, mode='valid'), c='C0', label='mean')

    ax.set_xlabel('episodes')
    ax.set_ylabel('total reward')
    ax.set_title('performance of %s, %d-smoothed' %(name, smoothing_window))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend()

    format_axis(ax)

    return fig

def plot_learning_curves_comparison(episodes, rewss_exp, rewss_control, smoothing_window = 1, name_exp='test', axis_ylim = None):
    """ plot learning curves for experimental and control groups side-by-side

    Arguments
    ---------
    episodes : episodes of training
    rewss_exp : np.array of floats or None, None indicates that this shouldn't be plotted
    rewss_control : np.array of floats or None, None indicates that this shouldn't be plotted

    """

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)

    ## do smoothing if necessary
    if smoothing_window > 1:
        rewss_exp = copy.deepcopy(rewss_exp) #to avoid changing original list
        rewss_control = copy.deepcopy(rewss_control)
        #ax.set_xlim((episodes[0], episodes[-1])) #before truncation


    if rewss_exp is not None:
        for i in range(len(rewss_exp)):
            rewss_exp[i] = np.convolve(rewss_exp[i], np.ones(smoothing_window)/smoothing_window, mode='valid')
    if rewss_control is not None:
        for i in range(len(rewss_control)):
            rewss_control[i] = np.convolve(rewss_control[i], np.ones(smoothing_window)/smoothing_window, mode='valid')

    if smoothing_window %2 == 0 and smoothing_window != 0:
        episodes = episodes[smoothing_window//2:-smoothing_window//2+1]

    if rewss_exp is not None:
        rewss_exp = np.array(rewss_exp)
        mean_rews_exp = np.array(rewss_exp).mean(axis=0)
        std_err_rews_exp = np.array(rewss_exp).std(axis=0)/np.sqrt(len(rewss_exp))

        ax.plot(episodes, mean_rews_exp, color='C0', label=name_exp)
        ax.fill_between(episodes, mean_rews_exp - std_err_rews_exp, mean_rews_exp + std_err_rews_exp, color='C0', alpha=0.2)

    if rewss_control is not None:
        rewss_control = np.array(rewss_control)
        mean_rews_control = np.array(rewss_control).mean(axis=0)
        std_err_rews_control = np.array(rewss_control).std(axis=0)/np.sqrt(len(rewss_control))
        ax.plot(episodes, mean_rews_control, color='C1', label='control')
        ax.fill_between(episodes, mean_rews_control - std_err_rews_control, mean_rews_control + std_err_rews_control, color='C1', alpha=0.2)

    ax.set_xlabel('k episodes')
    ax.set_ylabel('total reward')
    #ax.set_title('learning of %s model' %name_exp)

    ax.legend()

    if axis_ylim is not None:
        ax.set_ylim(axis_ylim)

    format_axis(ax)

    return fig

def plot_curves_several_runs(rews, title, title_y_axis, smoothing_window=1, several_runs=False, average_runs = True):
    """

    Parameters
    ----------
    rews
    smoothing_window
    name
    several_runs : bool, indicates whether rews is a nested list of rewards per run or not
    """

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    if several_runs:
        if average_runs:
            ind_alpha = 0.2
        else:
            ind_alpha = 0.8

        for i_r, rew_r in enumerate(rews):
            ax.plot(range(len(rew_r)-smoothing_window+1), np.convolve(rew_r, np.ones(smoothing_window)/smoothing_window, mode='valid'), alpha=ind_alpha, label='run %d' %i_r)

        if average_runs:
            rews = np.array(rews).mean(axis=0)

    if average_runs:
        ax.plot(range(len(rews)-smoothing_window+1), np.convolve(rews, np.ones(smoothing_window)/smoothing_window, mode='valid'), c='C0', label='mean')

    ax.set_xlabel('episodes')
    ax.set_ylabel(title_y_axis)
    ax.set_title(title)

    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles[-2:], labels[-2:])
    ax.legend()

    format_axis(ax)

    return fig

# %% CONFIG

class Config(dict):
    """ This Config can be used to access members with dot notation, e.g., config.attribute """

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def save_config_file(self, save_file):
        """ adapted from Pranav Mamidanna, DeepDraw code 
        """
        #mydict = {copy.copy(self.__dict__)}
        mydict = {}
        # Convert to python native types for better readability
        for (key, value) in self.__dict__.items():
            if isinstance(value, np.generic):
                mydict[key] = float(value)
            elif isinstance(value, list) or isinstance(value, np.ndarray):
                if isinstance(value[0], np.generic):
                    mydict[key] = [float(item) for item in value]
                        ## TODO: Make sure this works properly for dictionaries
            else:
                mydict[key] = value

        # Save yaml file in the model's path
        #path_to_yaml_file = os.path.join(model.model_path, 'config.yaml')
        with open(save_file, 'w') as myfile:
            yaml.dump(mydict, myfile, default_flow_style=False)

        return

    def load_config_file(self, load_file):
        
        try:
            with open(load_file, 'r') as f:
                mydict = yaml.load(f, Loader=yaml.FullLoader)
        except: ## TODO: GET RID OF THIS ONCE FIXED
            print("THERE IS AN ERROR IN LOADING THE TASK OPTIONS, REVERTING TO DEFAULT")
            with open('/home/kai/Documents/Projects/meta-peek-take/models/20240410150606/task_options.yaml', 'r') as f:
                mydict = yaml.load(f, Loader=yaml.FullLoader)

        self.update(mydict)

# %%
