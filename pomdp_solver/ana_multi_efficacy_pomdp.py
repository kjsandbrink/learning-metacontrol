# Kai Sandbrink
# 2022-09-25
# This script makes plots based on the saved alphavector policies outputted by QuickPOMDPS

# %% LIBRARY IMPORTS

import numpy as np
import matplotlib.pyplot as plt
import os
import sympy as sym
from sympy.abc import x
import matplotlib

from datetime import datetime
from utils import format_axis, get_timestamp

import xml.etree.ElementTree as ET

import pickle

# %% PARAMETERS

#results_datetime = '20230712153609'
results_datetime = '20230822002348'
analysis_datetime = datetime.now().strftime('%Y%m%d%H%M%S')

results_folder = os.path.join('results', 'pepe', results_datetime)
analysis_folder = os.path.join('analysis', 'pepe', results_datetime, analysis_datetime)

efficacies = np.arange(0, 1.125, 0.125)

# %% IMPORT ALPHAVECTORS

alphass = []
actionss= []

for eff in efficacies:

    eff_results_folder = os.path.join(results_folder, 'eff%d' %(eff*1000))

    # Load and parse the XML file into a tree.
    tree = ET.parse(os.path.join(eff_results_folder, 'policy.out'))

    # Get the root of the tree. This is the node of the tree where we start.
    root = tree.getroot()

    # Initialize empty lists for alpha vectors and actions
    alpha_vectors = []
    actions = []

    # Iterate over all 'Vector' elements in the XML
    for vector in root.iter('Vector'):
        # 'action' attribute in 'Vector' tag is the action, append it to actions list
        actions.append(float(vector.attrib['action']))
        
        # text in 'Vector' tag is the alpha vector. 
        # It's a space separated string, so we split by space to get a list.
        # Then we use list comprehension to convert each item in the list to float
        alpha_vector = [float(x) for x in vector.text.split()]
        
        # append alpha_vector to alpha_vectors list
        alpha_vectors.append(alpha_vector)

    alphass.append(alpha_vectors)
    actionss.append(actions)

# %% CALCULATE PEEK-TAKE BOUNDARY

os.makedirs(analysis_folder, exist_ok=True)

alphas_eff = np.array(alphass[0])
actions_eff = actionss[0]

n_steps = 50

n_beliefs = 1000

def calc_evidence_ratio_effs(alphas_eff, actions_eff):

    req_evidence_ratio_effs = []

    for t in range(n_steps):
    #t = 0

        for bL in np.arange(0.5, 1.001, 1/n_beliefs*2): ##solution assumes symmetry between beliefs on L and R

            b = np.array([bL, 1-bL])
            Vt = np.array(alphas_eff)[:, t*2:(t+1)*2]

            Vtb = np.dot(Vt, b)

            chosen_action = actions_eff[np.argmax(Vtb)]

            if chosen_action == 1:
                break

        req_evidence_ratio_effs.append(bL)

    return req_evidence_ratio_effs

req_evidence_ratio = []

for i, alphas_eff, actions_eff in zip(range(len(alphass)), alphass, actionss):
    
    print(i)
    req_evidence_ratio.append(calc_evidence_ratio_effs(alphas_eff, actions_eff))

# %% SAVE DATA

os.makedirs(analysis_folder, exist_ok=True)

with open(os.path.join(analysis_folder, '%s_req_evidence_ratio.pkl' %get_timestamp()), 'wb') as f:
    pickle.dump(req_evidence_ratio, f)

# %% LOAD DATA

datafile = 'analysis/pepe/20230822002348/20230822174359/20230822191659_req_evidence_ratio.pkl'

with open(datafile, 'rb') as f:
    req_evidence_ratio = pickle.load(f)

# %% MAKE PLOTS

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

matplotlib.rcParams.update({'font.size': 14})

n_eff_levels = len(efficacies)

ax.imshow(np.array(req_evidence_ratio), cmap='viridis', aspect='auto', extent=[0, n_steps, 0, 1], origin='lower')

#ax.plot(range(n_steps), belief_trajectory, marker='.', color='red', label='SARSOP')

#ax.set_title('Evidence Required to Take')
ax.set_xlabel('Step')
ax.set_ylabel('Efficacy')

cbar = fig.colorbar(ax.images[0], ax=ax)

cbar.set_label('Minimum Belief Required to Bet')

#format_axis(ax)

fig.savefig(os.path.join(analysis_folder, '%s_boundaries_peektake_%dbeliefs.png' %(get_timestamp(), n_beliefs)))
fig.savefig(os.path.join(analysis_folder, '%s_boundaries_peektake_%dbeliefs.svg' %(get_timestamp(), n_beliefs)))
# %%
