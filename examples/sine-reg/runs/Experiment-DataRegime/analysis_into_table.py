#%%
""" Analysis of the results of the sine regression experiment. """

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scienceplots

sns.set_theme(context='talk', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})

# plt.style.use(['science', 'no-latex'])

## Set the following matplotlib parameters 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

## Results in results.csv
results = pd.read_csv("results_all.csv", header=0)

print(results.columns)
print(results.head())

print("Total number of rows:", len(results))

## Columnms are: 
#       ['method', 'num_envs', 'taylor_order', 'context_size',
#       'gradient_updates', 'outer_steps', 'mse_ind', 'ci_ind', 'mse_ood',
#       'ci_ood']

## Open the zintgraff dataset
results_zg = pd.read_csv("results_zintgraff.csv", header=0)

## Add a column taylor order with 0
results_zg['taylor_order'] = 0

# print(results_zg.head())

#%%

## Get everywhere CAVIA, gradient_updates = 1, num_envs = 250

condition = {'method': 'CAVIA', 'gradient_updates': 1, 'num_envs': 250}
cavia_results = results[(results['method'] == condition['method']) & (results['gradient_updates'] == condition['gradient_updates']) & (results['num_envs'] == condition['num_envs'])]

## Average all the mse_ind and mse_ood and get the stds as well
mean = cavia_results[['mse_ind', 'mse_ood']].mean()
std = cavia_results[['mse_ind', 'mse_ood']].std()

print("mean mse_ind, mse_ood for CAVIA, gradient_updates = 1, num_envs = 250:", mean, std)


#%%

## We want to do all the above, but for all gradient updates and all num_envs. The results will be placed in a different dataframe

out_results = []

gradient_updates = [1, 5, 100]
num_envs = [250, 1000, 12500]

for gu in gradient_updates:
    for ne in num_envs:
        condition = {'method': 'CAVIA', 'gradient_updates': gu, 'num_envs': ne}
        cavia_results = results[(results['method'] == condition['method']) & (results['gradient_updates'] == condition['gradient_updates']) & (results['num_envs'] == condition['num_envs'])]
        mean = cavia_results[['mse_ind', 'mse_ood']].mean()
        std = cavia_results[['mse_ind', 'mse_ood']].std()
        method_name = f"FlashCAVIA-{gu}"
        out_results.append([method_name, ne, mean['mse_ind'], std['mse_ind'], mean['mse_ood'], std['mse_ood']])

        ## Let's also add the results for the zintgraff dataset
        condition = {'method': 'CAVIA', 'gradient_updates': gu, 'num_envs': ne}
        cavia_results = results_zg[(results_zg['method'] == condition['method']) & (results_zg['gradient_updates'] == condition['gradient_updates']) & (results_zg['num_envs'] == condition['num_envs'])]
        mean = cavia_results[['mse_ind', 'mse_ood']].mean()
        std = cavia_results[['mse_ind', 'mse_ood']].std()
        method_name = f"CAVIA-{gu}"
        out_results.append([method_name, ne, mean['mse_ind'], std['mse_ind'], mean['mse_ood'], std['mse_ood']])


        ## Lets do the same for MAML in the zintgraff dataset
        condition = {'method': 'MAML', 'gradient_updates': gu, 'num_envs': ne}
        maml_results = results_zg[(results_zg['method'] == condition['method']) & (results_zg['gradient_updates'] == condition['gradient_updates']) & (results_zg['num_envs'] == condition['num_envs'])]
        mean = maml_results[['mse_ind', 'mse_ood']].mean()
        std = maml_results[['mse_ind', 'mse_ood']].std()
        method_name = f"MAML-{gu}"
        out_results.append([method_name, ne, mean['mse_ind'], std['mse_ind'], mean['mse_ood'], std['mse_ood']])

        ## Let's do teh same for the NCF dataset
        condition = {'method': 'NCF', 'num_envs': ne}
        cavia_results = results[(results['method'] == condition['method']) & (results['num_envs'] == condition['num_envs'])]
        mean = cavia_results[['mse_ind', 'mse_ood']].mean()
        std = cavia_results[['mse_ind', 'mse_ood']].std()
        method_name = f"NCF"
        out_results.append([method_name, ne, mean['mse_ind'], std['mse_ind'], mean['mse_ood'], std['mse_ood']])

out_results = pd.DataFrame(out_results, columns=['method', 'num_envs', 'mse_ind_mean', 'mse_ind_std', 'mse_ood_mean', 'mse_ood_std'])

# print(out_results)


## Save the results to a csv file
out_results.to_csv("results_table.csv", index=False)


#%%

## Turn this into a hierchical table, with 10 rows for each method, and 3 columns for each num_envs. each column is subdivided into two further columns for mse_ind and mse_ood. (put the stds in parenthesis)

## We will use the multiindex feature of pandas

## First, we need to create a multiindex
methods = ['MAML-1', 'CAVIA-1', 'FlashCAVIA-1', 'MAML-5', 'CAVIA-5', 'FlashCAVIA-5', 'MAML-100', 'CAVIA-100', 'FlashCAVIA-100', 'NCF']
num_envs = [250, 1000, 12500]

index = pd.MultiIndex.from_product([methods, num_envs], names=['method', 'num_envs'])

## Now we create a new dataframe with this index
hierarchical_results = pd.DataFrame(index=index, columns=['mse_ind_mean', 'mse_ind_std', 'mse_ood_mean', 'mse_ood_std'])

## Now we fill in the values
for i, row in out_results.iterrows():
    method = row['method']
    num_envs = row['num_envs']
    mse_ind_mean = row['mse_ind_mean']
    mse_ind_std = row['mse_ind_std']
    mse_ood_mean = row['mse_ood_mean']
    mse_ood_std = row['mse_ood_std']

    hierarchical_results.loc[(method, num_envs), 'mse_ind_mean'] = mse_ind_mean
    hierarchical_results.loc[(method, num_envs), 'mse_ind_std'] = mse_ind_std
    hierarchical_results.loc[(method, num_envs), 'mse_ood_mean'] = mse_ood_mean
    hierarchical_results.loc[(method, num_envs), 'mse_ood_std'] = mse_ood_std

print(hierarchical_results)

## Save the results to a csv file
hierarchical_results.to_csv("hierarchical_results_table.csv")