# learning-metacontrol

This is code for the publication, "Understanding human meta-control and its pathologies using deep neural networks" (Kai Sandbrink, Laurence Hunt, Christopher Summerfield). Preprint: https://osf.io/preprints/osf/5ezxs

### Instructions

The main and supplementary figures are generated from the "fig_\*.ipynb" and "supp_fig_\*.ipynb" notebooks in the "nns" folder. All main figure panels beginning with Figure 2 are reproducible using the included data (human behavioral data and recorded neural network trajectories). The notebooks contain samples of expected output.

To run these, the conda environment needs to be installed using the command

```
conda env create -f environment.yml
conda activate metacontrol
```

With the build environment active, the "humans" folder then needs to be installed as a Python package by calling

``` 
pip install -e . 
```

from the "humans" directory. Both steps together should take a few minutes on a normal computer.

To train the neural networks, run the "run_\*.py" files from the folder "nns". Then, run the needed "sample_\*.py" and "save_\*.py" files to generate the synthetic behavioral and neural data, updating the timestamps as needed.

To generate the POMPD solutions, run the "ovb_pepe_testcases_j_takes.jl" script followed by "ana_multi_efficacy_pomdp.py". 

A few individual supplementary panels are generated directly from the individual "ana_\*.py" scripts. The "stats_likelihood.ipynb" notebook analyzes the likelihood of human choices under neural network policies.

The human behavioral data is in the "data" folder. The scripts used for data collection are in the "website" folder. Neural network trajectories are in the "data" subfolder in the "nns" folder.

### Versions

The neural network and analysis code uses Python v3.10.4 and associated numerous public libraries, as specified in the environment.yml file. The POMDP solver uses Julia v1.8.1 and associated libraries, as specified in the Manifest.toml. Code was operated on Ubuntu 22.04.3 LTS.

The website uses JSPsych v7.1.2.

### Description of data

#### Human

The human data contains behavioral and transdiagnostic data. The behavioral data is composed of pickled Pandas dataframes that contain the following fields for each participant:

- **transitions_ep**, np.array [n_episodes, n_timepoints, 3] : for each episode and timepoint, this contains the arm that paid out (element 0), the action chosen by the participant (0=bet on arm 0, 0.5=observe, 1=bet on arm 1), and the action that was taken (sam encoding)

- **transitions_ep_rightwrong**, np.array [n_controllability_levels, n_timepoints] : for each episodeand timepoint, this indicates if the correct bet was executed (0=incorrect bet, 0.5=observe, 1=correct bet)

- **ps**, np.array [n_controllability_levels, n_timepoints, 2] : payout probabilities of both arms for each controllability level and timepoint

- **effs**, np.array [n_episodes] : controllability level at each episode

- **rewards_tallies**, np.array [n_episodes] : rewards earned by the participant for each episode

- **n_observes**, np.array [n_episodes] : number of observe actions chosen by the participant in each episode

- **intended_correct**, np.array [n_episodes] : fraction of bets the participant intended to place on the correct arm

- **efficacy_estimates**, np.array [n_episodes] : participants' estimates of the controllability level for each episode

- **group**, boolean : False if the participant was in group A, True if the participant was in group B

The transdiagnostic data is stored in CSV files containing for each participant the corresponding AD, Compul, and SW scores.

#### Neural Networks

The neural network data consists of pickled NumPy arrays. The neural network data consists of evaluation trajectories and summary statistics, perturbed trajectories and summary statistics, and simulated perturbed populations. The folders are named after the timestamp that designates the networks, for which the correspondence can be found in the file "settings_ana.py".

The evaluation and perturbation data contains the following pickled arrays for each network:

- **counters_peeks_taus**, np.array [n_episodes]: the average number of observes for each of the tested controllability levels [1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0]

- **rewss_taus**, np.array [n_episodes]: the average number of rewards for each of the tested controllability levels 

- **traj_actionss_taus**, list [n_episodes] of np.arrays [n_sampled_trajectories, n_timesteps] : actions chosen by the network (0=observe, 1=bet on first arm, 2=bet on second arm)

- **traj_logitss_taus**, list [n_episodes] of np.arrays [n_sampled_trajectories, n_timesteps, 3] : the logits corresponding to each of the possible actions

- **traj_controlss_taus**, list [n_episodes] of np.arrays [n_sampled_trajectories, n_timesteps] : the controllability readout value returned by the network

- **traj_controlss_errss_taus**, list [n_episodes] of np.arrays [n_sampled_trajectories, n_timesteps] : the error of the controllability readout value

- **traj_pss_taus**, list [n_episodes] of np.arrays [n_sampled_trajectories, n_timesteps, 2] : the payout probabilities of the two arms

For the second task, the data additionally contains the **sleep_errs_taus_ape** organized similarly, and adds a final action (3=sleep) where appropriate.

The simulated perturbed participants contian the following pickled numpy arrays:

- **perturbed_control_errs_taus_ape**, np.array [n_controllability_levels, n_simulated_participants]: the error of the controllability readout in each episode for each of the simulated partiicpants

- **perturbed_counters_peeks_taus**, np.array [n_controllability_levels, n_simulated_participants]: the number of observe actions in each episode for each of the simulated partiicpants

- **perturbed_test_taus**, np.array [n_controllaility_levels]: the tested controllability levels, in the format tau = 1 - controllability

For the second task, the folder contains the additional array **perturbed_sleep_errs_taus_ape**.