# Kai Sandbrink
# 2024-03-08
# This script aggregates settings for import by the various analysis scripts.

# %% TASK 1 MODELS

#### WITH BIAS = 0.4

## 04/08/24
pepe_nn_ape_models = [

    ## NEW - WITHOUT MANUAL SEED 
    20230922164924,
    20230922164923,
    20230922164921,
    20230922164920,
    20230922164918,

    ## ALREADY GENERATED REPS -- WITHOUT MANUAL SEED
    20240215102214,
    20240215102212,
    20240215102211,
    20240215102209,
    20240215102208,
]

## 04/08/24
pepe_nn_control_models = [
    ## FC LAYER
    20240404140419,
    20240404140417,
    20240404140416,
    20240404140414,
    20240404140413,
    20240404140411,
    20240404140410,
    20240404140408,
    20240404140407,
    20240404140405,
]

pepe_nn_baseline_models = {
    ## tau notation for keys
    0: [
        20240410150618,
        20240410150616,
        20240410150614,
        20240410150613,
        20240410150611,
        20240410150609,
        20240410150607,
        20240410150606,
        20240410150604,
        20240410150602,
    ],
    0.125: [
        20240410150634,
        20240410150632,
        20240410150631,
        20240410150629,
        20240410150627,
        20240410150625,
        20240410150624,
        20240410150622,
        20240410150621,
        20240410150619,
    ],
    0.25: [
        20240411113809,
        20240411113808,
        20240411113806,
        20240411113805,
        20240411113803,
        20240410150640,
        20240410150638,
        20240410150636,
        20240410150635,
        20240410150634,
    ],
    0.375: [
        20240411113825,
        20240411113823,
        20240411113822,
        20240411113820,
        20240411113819,
        20240411113817,
        20240411113815,
        20240411113814,
        20240411113812,
        20240411113811,
    ],
    0.5: [
        20240412072023,
        20240412072021,
        20240412072020,
        20240412072018,
        20240412072017,
        20240411113831,
        20240411113830,
        20240411113829,
        20240411113827,
        20240411113825,
    ],
    0.625: [
        20240412072038,
        20240412072037,
        20240412072035,
        20240412072034,
        20240412072032,
        20240412072031,
        20240412072029,
        20240412072027,
        20240412072026,
        20240412072024,
    ],
    0.75: [
        20240413025332,
        20240413025330,
        20240413025329,
        20240413025327,
        20240413025326,
        20240412072045,
        20240412072044,
        20240412072042,
        20240412072040,
        20240412072039,
    ],
    0.875: [
        20240413025347,
        20240413025346,
        20240413025345,
        20240413025343,
        20240413025341,
        20240413025340,
        20240413025338,
        20240413025337,
        20240413025335,
        20240413025334,
    ],
    1: [
        20240413223207,
        20240413223205,
        20240413223204,
        20240413223202,
        20240413223201,
        20240413025354,
        20240413025353,
        20240413025351,
        20240413025350,
        20240413025348,
    ]
}

## EFFICACY-AT-INPUT MODELS
## 04/08/24

pepe_nn_efficacy_at_input_models = [
    ## WTIH FC
    20240407180045,
    20240407180044,
    20240407180042,
    20240407180040,
    20240407180039,
    20240407180037,
    20240407180036,
    20240407180034,
    20240407180032,
    20240407180031,
]

## EXTRA-NODE MODELS

pepe_nn_extra_node_models = [
    20240316231814,
    20240316231812,
    20240316231810,
    20240316231809,
    20240316231807,
    20240316231806,
    20240316231804,
    20240316231803,
    20240316231801,
    20240316231759,
]

#### WITH BIAS 0.5, VOLATILITY 0.2, AND NO HELDOUT TEST REGION
#### 10/06/23

pepe_human_ape_models = [
	20230923060019,
	20230923060017,
	20230923060016,
	20230923060014,
	20230923060013,
	20230922111420,
	20230922111418,
	20230922111417,
	20230922111415,
	20230922111413,
]

pepe_human_control_models = [

    ## FC
    20240405185747,
    20240405185746,
    20240405185744,
    20240405185742,
    20240405185741,
    20240405185739,
    20240405185738,
    20240405185736,
    20240405185735,
    20240405185733,
]

### NO HOLDOUT BIAS 0.5, VOL 0.1, 250k ENTROPY ANNEALING
levc_human_ape_models = [
    20240305173412,
    20240305173411,
    20240305173409,
    20240305173407,
    20240305173406,
    20240305173403,
    20240305173402,
    20240305173400,
    20240305173359,
    20240305173357,
]
#levc_human_control_traj_timestamp = '20240311133201'

levc_human_control_models = [
    20240406130255,
    20240406130254,
    20240406130252,
    20240406130251,
    20240406130249,
    20240405190151,
    20240405190150,
    20240405190148,
    20240405190147,
    20240405190145,
]


# %% SETTINGS FOR INDIVIDUAL DIFFERENCES

trait_simulated_participants_folder_t1 = 'data/sim_perturbed_participants/pepe/sim/mag100'
#trait_sim_timestamp_t1 = '20240219163433'
trait_sim_timestamp_t1 = '20240220100914'

random_simulated_participants_folder_t1 = 'data/sim_perturbed_participants/pepe/random/mag100'
#random_sim_timestamp_t1 = '20231015161128'
random_sim_timestamp_t1 = '20240220100914'

zeros_simulated_participants_folder_t1 = 'data/sim_perturbed_participants/pepe/nostruc/mag100'
#random_sim_timestamp_t1 = '20231015161128'
zeros_sim_timestamp_t1 = '20240220100914'

trait_simulated_participants_folder_t2 = 'data/sim_perturbed_participants/levc/sim/mag50bias-80'

### 150 PARTICIPANTS, 100 REPETITIONS OF EACH CASE
trait_sim_timestamp_t2 = '20240502232009' ## MAG 50 BIAS -80 ### CURRENT BEST

trait_sim_mag_perturbation=0.5
trait_sim_bias_perturbation=-0.8
random_simulated_participants_folder_t2 = 'data/sim_perturbed_participants/levc/random/mag50bias-80'
random_sim_timestamp_t2 = '20240502232009'

zeros_simulated_participants_folder_t2 = '/home/kai/Documents/Projects/meta-peek-take/data/sim_perturbed_participants/levc/nostruc/mag50bias-80'
zeros_sim_timestamp_t2 = '20240502232009' ## MAG 50 BIAS 80 ### CURRENT BEST


# %% TIMESTAMPS


pepe_nn_ape_lc_timestamp = '20240408223405'
pepe_nn_ape_traj_timestamp = '20240408223350'

pepe_nn_control_lc_timestamp = '20240408223405'
pepe_nn_extra_node_lc_timestamp = '20240318134420'

pepe_nn_efficacy_at_input_traj_timestamp = '20240429123622'
pepe_nn_efficacy_at_input_lc_timestamp = '20240411100946'

pepe_nn_decoding_timestamp = '20240409140548'

pepe_nn_baseline_lc_timestamp = '20240428225443'

pepe_human_ape_traj_timestamp = '20231006143445' ## MATCHING 10/06/23


pepe_human_ape_traj_timestamp = '20231006143445' ## MATCHING 10/06/23

pepe_human_control_traj_timestamp = '20240408223350' 

levc_human_ape_traj_timestamp = '20240311133201'

levc_human_control_traj_timestamp = '20240409115346'

timestamps = {
    'pepe_nn_ape_lc' : pepe_nn_ape_lc_timestamp,
    'pepe_nn_ape_traj': pepe_nn_ape_traj_timestamp,
    'pepe_nn_control_lc': pepe_nn_control_lc_timestamp,
    'pepe_nn_extra_node_lc': pepe_nn_extra_node_lc_timestamp,
    'pepe_nn_efficacy_at_input_traj': pepe_nn_efficacy_at_input_traj_timestamp,
    'pepe_nn_decoding': pepe_nn_decoding_timestamp,
    'pepe_nn_baseline_lc': pepe_nn_baseline_lc_timestamp,
    'pepe_human_ape_traj': pepe_human_ape_traj_timestamp,
    'pepe_human_control_traj': pepe_human_control_traj_timestamp,
    'levc_human_ape_traj': levc_human_ape_traj_timestamp,
    'levc_human_control_traj': levc_human_control_traj_timestamp,
}


# %% NAMES

names = {
    'ape': 'APE-trained',
    'noape': 'Standard',
    'humans': 'Humans',
    'sarsop': "SARSOP",
    'efficacy_at_input': r"$\xi$-Input",
    "extra_node": "Standard+",
    'baselines': 'Single-Setting',
}

# %% COLORS

color_ape = 'C0'
color_noape = 'C1'
color_humans = 'C2'

color_sarsop = 'red'
color_efficacy_input = 'C4'
color_extra_node = 'C5'
color_baselines = 'C6'

color_trait = 'C8'
color_random = 'grey'

color_ad = 'C3'
color_compul = 'C9'

colors = {
    'ape': color_ape,
    'noape': color_noape,
    'humans': color_humans,
    'sarsop': color_sarsop,
    'efficacy_at_input': color_efficacy_input,
    'extra_node': color_extra_node,
    'baselines': color_baselines,
    'trait': color_trait,
    'random': color_random,
    'ad': color_ad,
    'compul': color_compul
}

# %% DIMENSIONS

# document dimensions
INCH = 1
PAGEWIDTH = 8.5*INCH
PAGEHEIGHT = 11*INCH
TEXTWIDTH = 5.5 * INCH