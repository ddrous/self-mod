#%%
# Copyright (c) 2021 ddrous
# This script is to load and parse the output of the CelebA experiment

import re
import os
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# mpl.rcParams['lines.markersize'] = 1000

#%%

def parse_losses(path, verbose=True):
    ## Define the regex pattern to extract the relevant information
    """
        SOME FOR TRAIN SUPPORT and QUERY    
        Test loss value: 1.43e-02 ± 5.99e-03
        Train loss value for criterion 0: 1.43e-02
        Creating a new model with taylor order 0 ...
        SOME SIMILAR STUFF FOR VAL/ADAPT SUPPORT and QUERY
    """

    pattern = re.compile(r'Test loss value: (\d+\.\d+e-\d+)[±\de+\-\s.\n]+Train loss value for criterion 0: (\d+\.\d+e-\d+)')

    ## Create an empty list to add the regex matched in dictionary format
    data_list = []

    ## Iterate over the folders in the path
    for folder in os.listdir(path):
        if len(folder) != 13:
            continue

        ## Scan for a .py file in the folder
        for file in os.listdir(os.path.join(path, folder)):
            if file.endswith('.py'):
                ## Get the N and K values from the file name
                N, K = file.split('-')[1:]
                N = int(N[1:])
                K = int(K[1:-3])
                break

        if verbose:
            print(f"Folder name: {folder}")
            print(f'Currently parsing results for N={N}, K={K} ...')

        ## Get the path to the file containing the output of the experiment
        file_path = os.path.join(path, folder, 'nohup.log')
        ## Open the file and read the content
        with open(file_path, 'r') as f:
            content = f.read()
        ## Find all the matches of the pattern in the content
        matches = pattern.findall(content)

        if verbose:
            print("\tRegex matches found: ", matches)

        ## Extract the relevant information from the matches
        train_support = float(matches[0][1])
        train_query = float(matches[0][0])
        val_support = float(matches[1][1])
        val_query = float(matches[1][0])
        adapt_support = float(matches[2][1])
        adapt_query = float(matches[2][0])

        ## Append the information to the dictionary
        data_list.append({'N': N, 
                        'K': K, 
                        'train_support': train_support, 
                        'train_query': train_query, 
                        'val_support': val_support, 
                        'val_query': val_query, 
                        'adapt_support': adapt_support, 
                        'adapt_query': adapt_query})

    ## Actually make the dataframe
    losses = pd.DataFrame(data_list)

    return losses

#%%

ncf_losses = parse_losses(path='NCF/', verbose=False)
cavia_losses = parse_losses(path='CAVIA/', verbose=False)

display(ncf_losses.head(), cavia_losses.head())

#%%

# ## Fix K=10, the plot the losses for NCF and CAVIA with varying N


# fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# sns.lineplot(data=ncf_losses[ncf_losses['K'] == 10], x='N', y='train_support', label='NCF Train Support', ax=ax[0])
# sns.lineplot(data=cavia_losses[cavia_losses['K'] == 10], x='N', y='train_support', label='CAVIA Train Support', ax=ax[0])

# sns.lineplot(data=ncf_losses[ncf_losses['K'] == 10], x='N', y='train_query', label='NCF Train Query', ax=ax[1])
# sns.lineplot(data=cavia_losses[cavia_losses['K'] == 10], x='N', y='train_query', label='CAVIA Train Query', ax=ax[1])

# ax[0].set_title('Train Support Losses for K=10')
# ax[1].set_title('Train Query for K=10')

# plt.show()


#%%

## Fix K=10, the plot the losses for NCF and CAVIA with varying N
sns.set(context="poster", style="ticks")

def plot_losses(ax, metric='train_support', cmpas=["magma_r", None], legend=True):

    sns.lineplot(data=ncf_losses, 
                x='N', 
                y=metric, 
                hue='K', 
                style="K", 
                ax=ax, 
                dashes=False, 
                palette=cmpas[0] if cmpas[0] else None, 
                markers=["o"]*5
                )
    sns.lineplot(data=cavia_losses, 
                x='N', 
                y=metric, 
                hue='K', 
                style="K", 
                palette=cmpas[1] if cmpas[1] else None, 
                markers=["X"]*5, 
                dashes=True, 
                #  size=10,
                ax=ax, 
                legend=True,
                )

    # ax.set_yscale('log')
    ax.set_ylabel('MSE Loss')
    ax.set_title(metric.capitalize().replace('_', ' '))
    # ax.set_title('Train Query for K=10')
    ax.set_xscale('log', base=4)

    ## Set xticks and labels to [6, 12, 60, 252, 1020]
    ax.set_xticks([6, 12, 60, 252, 1020])
    ax.set_xticklabels([6, 12, 60, 252, 1020])

    ## Add a custom legend
    handles, labels = ax.get_legend_handles_labels()

    if legend == True:
        ## Draw the legend at the very top of the plot centered horizontally
        ax.legend(handles=handles[:], labels=labels[:], title=r'K — (NCF$\bullet$ CAVIA $\times$)', loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=10, title_fontsize='small')
    else:
        ax.legend().remove()





#%%
metrics = ['train_support', 'train_query', 'val_support', 'val_query', 'adapt_support', 'adapt_query']
fig, ax = plt.subplots(len(metrics), 1, figsize=(20, 10*len(metrics)), sharex=False)

for i, metric in enumerate(metrics):
    plot_losses(ax[i], metric=metric, cmpas=["magma_r", None], legend=True if i == 0 else False)
plt.show()

## Save the figure
fig.savefig('celeba_losses.png', bbox_inches='tight', dpi=300)