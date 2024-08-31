# Kai Sandbrink
# 2023-05-27
# This script compares behavior and transdiagnostic scores

# %% LIBRARY IMPORT

from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA

from utils import format_axis, get_timestamp
from human_utils_project import sort_overall, sort_train_test

from human_utils_project import plot_reward_tally, plot_n_observes, plot_prob_intended_correct_takes, plot_train_test_comp, plot_scatter_linefit

# %% PARAMETERS

behavior_file = 'results/behavior/20230623182952_behavior_diff_effs_6-19_days2and3.pkl'

td_file = 'results/transdiagnostic_scores/20230623183327_transdiagnostic_scores_diff_effs_combined_day3.csv'

analysis_folder = os.path.join('analysis', 'transdiagnostics')

effs_sorted_test = [0, 0.25,   0.75,  1.0]
effs_sorted_train = [0.125, 0.375, 0.5,0.785, 0.875]

test_start = len(effs_sorted_train)

# %% DATA READ IN

#behavior_df = pd.read_csv(behavior_file, index_col=0)
behavior_df = pd.read_pickle(behavior_file)
td_df = pd.read_csv(td_file, index_col=0)

df = pd.merge(behavior_df, td_df, left_index=True, right_index=True)
df = df.dropna()

# %% DATA PREPROCESSING OF BEHAVIOR DATA

# Convert the rewards_tallies from behavior_df to a 1D list, then calculate the sum
def parse_float_list(s):
    return [float(x) for x in s.strip('[]').split()]

def parse_float_nested_list(s):
    s = s.replace("[", "").replace("]", "")
    outer_list = s.split(', ')
    return [[float(x) for x in inner_list.split()] for inner_list in outer_list]

for day in ['day2', 'day3']:

    df['total_rewards_%s' %day] = df['rewards_tallies_%s' %day].apply(
        lambda x: np.nansum(parse_float_list(x)) if isinstance(x, str) else np.nansum(np.array(x))
    )

    # Using the new function to count observe actions
    df['observe_count_%s' %day] = df['transitions_ep_rightwrong_%s' %day].apply(
        lambda x: np.count_nonzero(np.array(parse_float_nested_list(x)) == 0.5) if isinstance(x, str) else np.count_nonzero(np.array(x) == 0.5)
    )

    nobs_tr, nobs_te = sort_train_test(df['n_observes_%s' %day].values, df['effs_%s' %day].values, test_start)
    df['n_observes_train_%s' %day], df['n_observes_test_%s' %day] = nobs_tr.sum(axis=1), nobs_te.sum(axis=1)

    nrews_tr, nrews_te = sort_train_test(df['rewards_tallies_%s' %day].values, df['effs_%s' %day].values, test_start)
    df['total_rewards_train_%s' %day], df['total_rewards_test_%s' %day] = nrews_tr.sum(axis=1), nrews_te.sum(axis=1)

# %% MLS ALL 3 REGRESSORS

import statsmodels.api as sm

dependent_vars = ['max_ll_ape_train_day2', 'max_ll_ape_train_day3', 'max_ll_ape_test_day2', 'max_ll_ape_test_day3',
                  'max_ll_control_train_day2', 'max_ll_control_train_day3', 'max_ll_control_test_day2', 'max_ll_control_test_day3']

# Create an empty DataFrame to store the summary statistics
summary_df = pd.DataFrame()

# For each dependent variable, we perform a regression analysis
for dv in dependent_vars:
    # Define the dependent variable
    Y = df[dv]
    
    # Define the independent variables
    X = df[['AD', 'Compul', 'SW']]
    
    # Add a constant to the independent variables matrix
    X = sm.add_constant(X)
    
    # Perform the multiple linear regression
    model = sm.OLS(Y, X)
    results = model.fit()

    # Store the p-values of the individual regressors and the overall p-value and R-squared
        # Store the p-values of the individual regressors and the overall p-value and R-squared
    summary_df = summary_df.append(pd.Series({
        'Dependent Variable': dv,
        'AD_p-value': results.pvalues['AD'],
        'Compul_p-value': results.pvalues['Compul'],
        'SW_p-value': results.pvalues['SW'],
        'Overall_p-value': results.f_pvalue,
        'R-squared': results.rsquared,
        'AD_coefficient': results.params['AD'],
        'Compul_coefficient': results.params['Compul'],
        'SW_coefficient': results.params['SW']
    }), ignore_index=True)

# Print the summary statistics
print(summary_df)

summary_df.to_csv(os.path.join(analysis_folder ,"%s_MLR_statistics.csv" %get_timestamp()))
# %% CORRELATION MATRIX

# Select columns
cols = ['AD', 'Compul', 'SW', 'max_ll_ape_train_day2', 'max_ll_ape_train_day3', 'max_ll_ape_test_day2', 'max_ll_ape_test_day3',
                  'max_ll_control_train_day2', 'max_ll_control_train_day3', 'max_ll_control_test_day2', 'max_ll_control_test_day3']

# Subset dataframe with selected columns
df_subset = df[cols]

# Calculate correlation matrix
corr = df_subset.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)

# Create a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

ax.set_title("Correlation Matrix")

#plt.show()

fig.savefig(os.path.join(analysis_folder, "%s_correlation_matrix.png" %get_timestamp()))
fig.savefig(os.path.join(analysis_folder, "%s_correlation_matrix.svg" %get_timestamp()))
