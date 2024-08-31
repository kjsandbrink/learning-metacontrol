# Kai Sandbrink
# 2022-03-09
# Various utility functions

import matplotlib.pyplot as plt
import numpy as np

import yaml, os
from datetime import datetime

# %% GENERAL FUNCTIONS

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

# %% PLOTTING FUNCTIONS

def format_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=6)
    ax.yaxis.set_tick_params(size=6)

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


def get_timestamp():
    return datetime.now().strftime('%Y%m%d%H%M%S')


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
                mydict[key] = [int(item) for item in value]
            else:
                mydict[key] = value

        # Save yaml file in the model's path
        #path_to_yaml_file = os.path.join(model.model_path, 'config.yaml')
        with open(save_file, 'w') as myfile:
            yaml.dump(mydict, myfile, default_flow_style=False)

        return

    def load_config_file(self, load_file):
        with open(load_file, 'r') as f:
            mydict = yaml.load(f, Loader=yaml.FullLoader)
        
        self.__init__(mydict)