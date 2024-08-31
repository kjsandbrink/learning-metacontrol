# Kai Sandbrink
# 2023-04-27
# Evaluate survey responses

# %% LIBRARY IMPORT

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from human_utils_project import sort_train_test

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

from sklearn.linear_model import Lasso, Ridge

from utils import format_axis, get_timestamp

# %% PARAMETERS4

survey_files = [
    'results/surveys/20240214133158_survey_responses_diff_effs_combined_22A.csv',
    'results/surveys/20240214133504_survey_responses_diff_effs_combined_22B.csv',
    'results/surveys/20240214133618_survey_responses_diff_effs_combined_29A.csv',
    'results/surveys/20240214133648_survey_responses_diff_effs_combined_29B.csv',
    ]

AD_questions = ['SDS_11', 'SDS_12', 'SDS_14', 'SDS_15', 'SDS_17', 'SDS_18', 'SDS_20', 
                'STAI_1', 'STAI_3', 'STAI_4', 'STAI_5', 'STAI_7', 'STAI_10', 'STAI_13', 'STAI_16', 'STAI_19',
                'BIS_9', 'BIS_13', 'BIS_20',
                'AES_1', 'AES_2', 'AES_7', 'AES_8', 'AES_16', 'AES_17', 'AES_18']

targets = ['n_observes_train', 'n_observes_test', 'total_step_max_ll_ape_train', 'total_step_max_ll_ape_test', 'total_mean_odl_ape_train', 'total_mean_odl_ape_test',] + \
            ['n_observes_train_loweff', 'n_observes_train_higheff', 'n_observes_test_loweff', 'n_observes_test_higheff', 'step_max_ll_ape_train_loweff', 'step_max_ll_ape_train_higheff', 'step_max_ll_ape_test_loweff', 'step_max_ll_ape_test_higheff', 'mean_odl_ape_train_loweff', 'mean_odl_ape_train_higheff', 'mean_odl_ape_test_loweff', 'mean_odl_ape_test_higheff'] + \
            ['total_mean_ll_ape_train', 'total_mean_ll_ape_test', 'total_mean_ll_control_train', 'total_mean_ll_control_test', ] + \
            ['mean_ll_ape_train_loweff', 'mean_ll_ape_train_higheff', 'mean_ll_ape_test_loweff', 'mean_ll_ape_test_higheff', 'mean_ll_control_train_loweff', 'mean_ll_control_train_higheff', 'mean_ll_control_test_loweff', 'mean_ll_control_test_higheff']


test_start = 5

# %% DATA READ IN

if survey_file2 is not None:
    survey_df1 = pd.read_csv(survey_file1, index_col=0)
    survey_df2 = pd.read_csv(survey_file2, index_col=0)

    survey_df = pd.concat([survey_df1, survey_df2])
else:
    survey_df = pd.read_csv(survey_file1, index_col=0)

if behavior_file2 is not None:
    behavior_df1 = pd.read_pickle(behavior_file1)
    behavior_df2 = pd.read_pickle(behavior_file2)

    behavior_df = pd.concat([behavior_df1, behavior_df2])
else:
    behavior_df = pd.read_pickle(behavior_file1)

df = behavior_df.join(survey_df, how='inner')

# %%

nobs_tr, nobs_te = sort_train_test(df['n_observes'].values, df['effs'].values, test_start)
df['n_observes_train'], df['n_observes_test'] = nobs_tr.sum(axis=1), nobs_te.sum(axis=1)

df['n_observes_train_loweff'] = nobs_tr[:,:2].sum(axis=1)
df['n_observes_train_higheff'] = nobs_tr[:,3:].sum(axis=1)
df['n_observes_test_loweff'] = nobs_te[:,:2].sum(axis=1)
df['n_observes_test_higheff'] = nobs_te[:,2:].sum(axis=1)

nst_tr, nst_te = sort_train_test(df['step_max_ll_ape'].values, df['effs'].values, test_start)
df['total_step_max_ll_ape_train'], df['total_step_max_ll_ape_test'] = nst_tr.sum(axis=1), nst_te.sum(axis=1)

df['step_max_ll_ape_train_loweff'] = nst_tr[:,:2].sum(axis=1)
df['step_max_ll_ape_train_higheff'] = nst_tr[:,3:].sum(axis=1)
df['step_max_ll_ape_test_loweff'] = nst_te[:,:2].sum(axis=1)
df['step_max_ll_ape_test_higheff'] = nst_te[:,2:].sum(axis=1)

df['total_mean_odl_ape_train'], df['total_mean_odl_ape_test'] = df['mean_odl_ape_train'].apply(sum), df['mean_odl_ape_test'].apply(sum)

tr = np.stack(df['mean_odl_ape_train'].values)
te = np.stack(df['mean_odl_ape_test'].values)

df['mean_odl_ape_train_loweff'] = tr[:,:2].sum(axis=1)
df['mean_odl_ape_train_higheff'] = tr[:,3:].sum(axis=1)
df['mean_odl_ape_test_loweff'] = te[:,:2].sum(axis=1)
df['mean_odl_ape_test_higheff'] = te[:,2:].sum(axis=1)


df['total_mean_ll_ape_train'], df['total_mean_ll_ape_test'] = df['mean_ll_ape_train'].apply(sum), df['mean_ll_ape_test'].apply(sum)

tr = np.stack(df['mean_ll_ape_train'].values)
te = np.stack(df['mean_ll_ape_test'].values)

df['mean_ll_ape_train_loweff'] = tr[:,:2].sum(axis=1)
df['mean_ll_ape_train_higheff'] = tr[:,3:].sum(axis=1)
df['mean_ll_ape_test_loweff'] = te[:,:2].sum(axis=1)
df['mean_ll_ape_test_higheff'] = te[:,2:].sum(axis=1)


df['total_mean_ll_control_train'], df['total_mean_ll_control_test'] = df['mean_ll_control_train'].apply(sum), df['mean_ll_control_test'].apply(sum)

tr = np.stack(df['mean_ll_control_train'].values)
te = np.stack(df['mean_ll_control_test'].values)

df['mean_ll_control_train_loweff'] = tr[:,:2].sum(axis=1)
df['mean_ll_control_train_higheff'] = tr[:,3:].sum(axis=1)
df['mean_ll_control_test_loweff'] = te[:,:2].sum(axis=1)
df['mean_ll_control_test_higheff'] = te[:,2:].sum(axis=1)

# %% CREATE TRAINING AND TEST SETS

X = df[AD_questions]  # replace with your actual column names

# Split the data into training and test datasets
test_size=0.2
random_state=42
X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

# %% RANDOM FOREST REGRESSION

def train_and_evaluate(target_column):
    # Get the target variable
    y_train = df.loc[X_train.index, target_column]
    y_test = df.loc[X_test.index, target_column]

    # Create and fit the model
    #rf = RandomForestRegressor()
    # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, max_depth=5)# Train the model on training data
    # rf.fit(X_train, y_train)

    #model = Lasso(alpha=1)
    model = Ridge(alpha=300)
    model.fit(X_train, y_train)

    # Make predictions on the test and training datasets
    # predictions_test = rf.predict(X_test)
    # predictions_train = rf.predict(X_train)
    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)

    # Calculate MSE and MAE for the test dataset
    mse_test = mean_squared_error(y_test, predictions_test)
    mae_test = mean_absolute_error(y_test, predictions_test)

    # Calculate MSE and MAE for the training dataset
    mse_train = mean_squared_error(y_train, predictions_train)
    mae_train = mean_absolute_error(y_train, predictions_train)

    # Calculate r2 score for the training and test datasets
    r2_test = r2_score(y_test, predictions_test)
    r2_train = r2_score(y_train, predictions_train)
    
    return model, (mse_test, mae_test, r2_test), (mse_train, mae_train, r2_train)

# Lists to store models and results
models = []
test_metrics = []
train_metrics = []

for col in targets:
    print(col)
    model, test_metric, train_metric = train_and_evaluate(col)
    models.append(model)
    test_metrics.append(test_metric)
    train_metrics.append(train_metric)

# %% PRINT RESULTS

# Create an empty DataFrame to store results
results = pd.DataFrame(columns=["Target", "MSE_Test", "MAE_Test", "r2_Test",
                                "MSE_Train", "MAE_Train", "r2_Train"])

# Populate the DataFrame with the model performance metrics
for col, model, test_metric, train_metric in zip(targets, models, test_metrics, train_metrics):
    results = results.append({
        "Target": col,
        "MSE_Test": test_metric[0],
        "MAE_Test": test_metric[1],
        "r2_Test": test_metric[2],
        "MSE_Train": train_metric[0],
        "MAE_Train": train_metric[1],
        "r2_Train": train_metric[2]
    }, ignore_index=True)

# Print the results
print(results)

# %%

def baseline_metrics(target_column):
    # Get the target variable
    y_train = df.loc[X_train.index, target_column]
    y_test = df.loc[X_test.index, target_column]

    # Predict the mean of y_train for all instances
    predictions_test = np.full(y_test.shape, y_train.mean())
    predictions_train = np.full(y_train.shape, y_train.mean())

    # Calculate MSE and MAE for the test dataset
    mse_test = mean_squared_error(y_test, predictions_test)
    mae_test = mean_absolute_error(y_test, predictions_test)

    # Calculate MSE and MAE for the training dataset
    mse_train = mean_squared_error(y_train, predictions_train)
    mae_train = mean_absolute_error(y_train, predictions_train)

    # Since all predictions are the same, t-test is not applicable for baseline model
    p_value_test = np.nan
    p_value_train = np.nan

    return mse_test, mae_test, p_value_test, mse_train, mae_train, p_value_train

# %% 

# Add columns to store the baseline model performance metrics
results["Baseline_MSE_Test"] = 0
results["Baseline_MAE_Test"] = 0
results["Baseline_MSE_Train"] = 0
results["Baseline_MAE_Train"] = 0

for i, col in enumerate(targets):
    mse_test, mae_test, _, mse_train, mae_train, _ = baseline_metrics(col)
    
    results.loc[i, "Baseline_MSE_Test"] = mse_test
    results.loc[i, "Baseline_MAE_Test"] = mae_test
    results.loc[i, "Baseline_MSE_Train"] = mse_train
    results.loc[i, "Baseline_MAE_Train"] = mae_train

# Print the results
print(results)

# %%

analysis_folder = os.path.join('analysis','AD', '525_528_619')
if not os.path.exists(analysis_folder):
    os.makedirs(analysis_folder)

results.to_csv(os.path.join(analysis_folder, 'Ridge_alpha100_results.csv'))
results.to_pickle(os.path.join(analysis_folder, 'Ridge_alpha100_results.pkl'))

# %% MAKE PLOT

metrics_to_plot = ['n_observes_train', 'n_observes_test', 
                   'total_mean_odl_ape_train', 'total_mean_odl_ape_test',
                     'n_observes_train_loweff', 'n_observes_train_higheff',
                       'n_observes_test_loweff', 'n_observes_test_higheff',
                       'mean_odl_ape_train_loweff', 'mean_odl_ape_train_higheff',
                       'mean_odl_ape_test_loweff', 'mean_odl_ape_test_higheff',]

df.reset_index(inplace=True)

df = results.loc[results['Target'].isin(metrics_to_plot)]

df.reset_index(inplace=True)

# Separate "Train" and "Test" metrics
df_train = df[df['Target'].str.contains('_train')]
df_test = df[df['Target'].str.contains('_test')]

# Create new column to denote whether the metric was from the training or test set
df_train['Set'] = 'Train'
df_test['Set'] = 'Test'

# Remove "_train" and "_test" from the Target column to allow grouping
df_train['Target'] = df_train['Target'].str.replace('_train', '')
df_test['Target'] = df_test['Target'].str.replace('_test', '')

# Concatenate the dataframes and filter for necessary metrics
df_concat = pd.concat([df_train, df_test])
df_concat = df_concat[df_concat['Target'].isin([metric.replace('_train', '').replace('_test', '') 
                                                for metric in metrics_to_plot])]
# Define bar width
bar_width = 0.35

# Create position of bars on X axis
r1 = np.arange(len(df_concat[df_concat['Set'] == 'Train']))
r2 = [x + bar_width for x in r1]

fig = plt.figure(figsize=(12, 6), dpi=300)
ax = fig.add_subplot(111)

# Make the plot
plt.bar(r1, df_concat[df_concat['Set'] == 'Train']['r2_Test'], color='b', width=bar_width, edgecolor='grey', label='Train')
plt.bar(r2, df_concat[df_concat['Set'] == 'Test']['r2_Test'], color='r', width=bar_width, edgecolor='grey', label='Test')

# Add xticks on the middle of the group bars
plt.xlabel('Metrics', fontweight='bold')
plt.ylabel('Predictive r2_Test Score')
plt.xticks([r + bar_width / 2 for r in range(len(df_concat[df_concat['Set'] == 'Train']))], 
           df_concat[df_concat['Set'] == 'Train']['Target'], rotation=45, ha='right')

plt.title('Predictive r2_Test Scores for Metrics')
plt.tight_layout()
plt.legend(['Train (Color Cue)', 'Test (No Cue)'])

format_axis(ax)

fig.savefig(os.path.join(analysis_folder, '%s_r2_Test_Scores_for_Metrics.png' %get_timestamp()), dpi=300)
fig.savefig(os.path.join(analysis_folder, '%s_r2_Test_Scores_for_Metrics.svg' %get_timestamp()), dpi=300)

# %%

