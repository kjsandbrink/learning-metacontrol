# Kai Sandbrink
# 2023-05-26
# This script extracts the survey responses for participants

# %% LIBRARY IMPORT

import numpy as np
import pandas as pd
import os, json
import glob

from utils import get_timestamp

# %% PARAMETERS

participant_ids = ['xyz']

data_folder = os.path.join('data', 'day3B', 'data')
save_file = os.path.join('results', 'surveys', get_timestamp() + '_survey_responses_diff_effs_combined.csv')

match_keys = {
    'zung': 'SDS',
    'anxiety': 'STAI',
    'ocir': 'OCI',
    'leb': 'LSAS',
    'bis': 'BIS',
    'eat': 'EAT',
    'apathy': 'AES',
}

# %% FUNCTIONS

# %% WITH PRINT
def update_main_df(main_df, participant_df, participant_id, match_keys):
    # check if the participant id is in the main dataframe
    if participant_id in main_df.index:
        for idx, row in participant_df.iterrows():
            if isinstance(row['response'], str):
                #print(row['response'])
                # get the response dictionary
                response_dict = json.loads(row['response'])
                #print(f"Response dictionary: {response_dict}")
                # iterate over each question in the response
                for question, response in response_dict.items():
                    # extract the question prefix and id
                    question_prefix, question_id = question.split('.')
                    # get the main df question key using the match_keys dict
                    main_df_question_key = match_keys.get(question_prefix, None)
                    if main_df_question_key is not None:
                        # construct the main df column name
                        main_df_column_name = main_df_question_key + '_' + question_id
                        if main_df_column_name in main_df.columns:
                            # update the main df with the participant's response
                            main_df.loc[participant_id, main_df_column_name] = response
                            #print(f"Updated {main_df_column_name} for participant {participant_id}")
    return main_df

# %% INITIALIZE DATAFRAME

columns = ['SDS_1', 'SDS_2', 'SDS_3', 'SDS_4', 'SDS_5', 'SDS_6', 'SDS_7', 'SDS_8', 'SDS_9', 'SDS_10', 'SDS_11', 'SDS_12', 'SDS_13', 'SDS_14', 'SDS_15', 'SDS_16', 'SDS_17', 'SDS_18', 'SDS_19', 'SDS_20', 'STAI_1', 'STAI_2', 'STAI_3', 'STAI_4', 'STAI_5', 'STAI_6', 'STAI_7', 'STAI_8', 'STAI_9', 'STAI_10', 'STAI_11', 'STAI_12', 'STAI_13', 'STAI_14', 'STAI_15', 'STAI_16', 'STAI_17', 'STAI_18', 'STAI_19', 'STAI_20', 'OCI_1', 'OCI_2', 'OCI_3', 'OCI_4', 'OCI_5', 'OCI_6', 'OCI_7', 'OCI_8', 'OCI_9', 'OCI_10', 'OCI_11', 'OCI_12', 'OCI_13', 'OCI_14', 'OCI_15', 'OCI_16', 'OCI_17', 'OCI_18', 'LSAS_1', 'LSAS_2', 'LSAS_3', 'LSAS_4', 'LSAS_5', 'LSAS_6', 'LSAS_7', 'LSAS_8', 'LSAS_9', 'LSAS_10', 'LSAS_11', 'LSAS_12', 'LSAS_13', 'LSAS_14', 'LSAS_15', 'LSAS_16', 'LSAS_17', 'LSAS_18', 'LSAS_19', 'LSAS_20', 'LSAS_21', 'LSAS_22', 'LSAS_23', 'LSAS_24', 'BIS_1', 'BIS_2', 'BIS_3', 'BIS_4', 'BIS_5', 'BIS_6', 'BIS_7', 'BIS_8', 'BIS_9', 'BIS_10', 'BIS_11', 'BIS_12', 'BIS_13', 'BIS_14', 'BIS_15', 'BIS_16', 'BIS_17', 'BIS_18', 'BIS_19', 'BIS_20', 'BIS_21', 'BIS_22', 'BIS_23', 'BIS_24', 'BIS_25', 'BIS_26', 'BIS_27', 'BIS_28', 'BIS_29', 'BIS_30', 'SCZ_1', 'SCZ_2', 'SCZ_3', 'SCZ_4', 'SCZ_5', 'SCZ_6', 'SCZ_7', 'SCZ_8', 'SCZ_9', 'SCZ_10', 'SCZ_11', 'SCZ_12', 'SCZ_13', 'SCZ_14', 'SCZ_15', 'SCZ_16', 'SCZ_17', 'SCZ_18', 'SCZ_19', 'SCZ_20', 'SCZ_21', 'SCZ_22', 'SCZ_23', 'SCZ_24', 'SCZ_25', 'SCZ_26', 'SCZ_27', 'SCZ_28', 'SCZ_29', 'SCZ_30', 'SCZ_31', 'SCZ_32', 'SCZ_33', 'SCZ_34', 'SCZ_35', 'SCZ_36', 'SCZ_37', 'SCZ_38', 'SCZ_39', 'SCZ_40', 'SCZ_41', 'SCZ_42', 'SCZ_43', 'AUDIT_1', 'AUDIT_2', 'AUDIT_3', 'AUDIT_4', 'AUDIT_5', 'AUDIT_6', 'AUDIT_7', 'AUDIT_8', 'AUDIT_9', 'AUDIT_10', 'EAT_1', 'EAT_2', 'EAT_3', 'EAT_4', 'EAT_5', 'EAT_6', 'EAT_7', 'EAT_8', 'EAT_9', 'EAT_10', 'EAT_11', 'EAT_12', 'EAT_13', 'EAT_14', 'EAT_15', 'EAT_16', 'EAT_17', 'EAT_18', 'EAT_19', 'EAT_20', 'EAT_21', 'EAT_22', 'EAT_23', 'EAT_24', 'EAT_25', 'EAT_26', 'AES_1', 'AES_2', 'AES_3', 'AES_4', 'AES_5', 'AES_6', 'AES_7', 'AES_8', 'AES_9', 'AES_10', 'AES_11', 'AES_12', 'AES_13', 'AES_14', 'AES_15', 'AES_16', 'AES_17', 'AES_18']

df = pd.DataFrame(columns=columns, index=participant_ids)

# %% READ IN DATA

for pid in participant_ids:
    p_survey_files = glob.glob(os.path.join(data_folder, '*' + pid + '*survey.txt'))
    if len(p_survey_files) > 0:
        p_survey_file = p_survey_files[0]
        p_survey = pd.read_csv(p_survey_file)

        df = update_main_df(df, p_survey, pid, match_keys)

df = df.fillna(0)

# %% SAVE

df.to_csv(save_file)

# %%
