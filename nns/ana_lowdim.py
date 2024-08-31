# Kai Sandbrink
# 2023-04-27
# 

# %% LIBRARY IMPORT

import numpy as np
from utils import get_timestamp
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import sem
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon

from utils import format_axis, get_timestamp

# %% 2D

def create_pca_2D(hidden_states, n_comps=2):
    
    n_timepoints = hidden_states.shape[1]
    n_units = hidden_states.shape[2]
    hidden_states_2D = hidden_states.reshape((-1, n_units))
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(hidden_states_2D)

    pca = PCA(n_components=n_comps)
    pca.fit(scaled_data)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    hidden_states_pca = pca.transform(scaled_data).reshape((-1, n_timepoints, n_comps))

    return hidden_states_pca


def plot_hidden_states_pca(hidden_states_pca, n_samples_per_eff = 10):
    """
    Args:
    hidden_states_pca (ndarray): The hidden states for PCA. Shape: (num_samples, num_time_steps, num_features)
    n_samples_per_eff (int, optional): The number of samples per efficacy. Defaults to 10.

    Returns:
        matplotlib.figure.Figure: The plotted figure.
    """

    fig = plt.figure(dpi=360)
    ax1 = fig.add_subplot(111)

    for i, e in enumerate(np.arange(0, 1.01, 0.25)):
        for j in range(n_samples_per_eff):
            if j != 0:
                label_prefix = '_'
            else:
                label_prefix = ''
            sample_index = i*200 + j
            ax1.plot(hidden_states_pca[sample_index,:,0], hidden_states_pca[sample_index,:,1], color='C%d' %i, label=label_prefix+str(e))

    #ax1.set_title('PCA of Hidden States')
    ax1.set_xlabel('PCA 1')
    ax1.set_ylabel('PCA 2')
    #ax1.show()

    #ax2 = fig.add_subplot(122)

    plt.legend(title="Efficacy", title_fontsize= 13)

    format_axis(ax1)

    return fig


def plot_hidden_states_avg(hidden_states_pca, n_samples_per_eff=100, cmap='viridis'):
    fig = plt.figure(dpi=360)
    ax = fig.add_subplot(111)
    
    # Get the colormap
    colormap = plt.get_cmap(cmap)
    # Determine the number of unique efficacy indices (steps in the loop)
    num_eff_indices = len(np.arange(0, 1.01, 0.25))
    
    for i, e in enumerate(np.arange(0, 1.01, 0.25)[::-1]):
        samples_indices = np.arange(i*200, i*200+n_samples_per_eff)
        samples = hidden_states_pca[samples_indices, :, :]

        # Calculate mean and SEM for each point along the PCA dimensions
        mean_trajectory = np.mean(samples, axis=0)

        # Normalize current index and select a color from the colormap
        color = colormap((num_eff_indices - i) / (num_eff_indices))

        # Plot the mean trajectory with the selected color
        ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], color=color, label=r"$\xi=$" + str(e), marker='o')

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    #plt.legend(title="Efficacy", title_fontsize=13)
    plt.legend()

    # Assuming format_axis is a function you've defined elsewhere
    format_axis(ax)

    return fig


# %%

if __name__ == '__main__':
        
    # %% PARAMETERS

    ape_models =  [
        20230427201627,
        20230427201629,
        20230427201630,
        20230427201632,
        20230427201633,
        20230427201644,
        20230427201646,
        20230427201647,
        20230427201648,
        20230427201649
    ]

    control_models = [
        20230427201636,
        20230427201637,
        20230427201639,
        20230427201640,
        20230427201642,
        20230427201657,
        20230427201656,
        20230427201655,
        20230427201653,
        20230427201652
    ]

    n_comps = 3

    timestamp = get_timestamp()
    base_data_folder = 'data/reps'

    reps_folder_suffix = '20230524191845_100cases'

    analysis_folder = os.path.join('analysis', 'explore-exploit', 'lowdim')

    # %% LOAD DATA

    model = ape_models[0]

    model_data_folder = os.path.join(base_data_folder, str(model), reps_folder_suffix)
    checkpoint_str = ''

    cell_states = np.load(os.path.join(model_data_folder, 'cell_states%s.npy' %checkpoint_str))
    hidden_states = np.load(os.path.join(model_data_folder, 'hidden_states%s.npy' %checkpoint_str))

    combined_hidden_states = np.concatenate((cell_states, hidden_states), axis=2)

    efficacies = np.load(os.path.join(model_data_folder, 'efficacies%s.npy' %checkpoint_str)).flatten()


    # %%

    ape_pca = create_pca_2D(combined_hidden_states)
    fig = plot_hidden_states_pca(ape_pca)

    fig.savefig(os.path.join(analysis_folder, '%s_ape_pca.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_ape_pca.svg' %get_timestamp()))

    # %%

    model = control_models[0]

    model_data_folder = os.path.join(base_data_folder, str(model), reps_folder_suffix)
    checkpoint_str = ''

    cell_states = np.load(os.path.join(model_data_folder, 'cell_states%s.npy' %checkpoint_str))
    hidden_states = np.load(os.path.join(model_data_folder, 'hidden_states%s.npy' %checkpoint_str))

    combined_hidden_states = np.concatenate((cell_states, hidden_states), axis=2)

    na_pca = create_pca_2D(combined_hidden_states)
    fig = plot_hidden_states_pca(na_pca)


    fig.savefig(os.path.join(analysis_folder, '%s_no_pca.png' %get_timestamp()))
    fig.savefig(os.path.join(analysis_folder, '%s_no_pca.svg' %get_timestamp()))

    # %%
