#%%
# Copyright (c) 2021 ddrous
# This script is to visualize the CelebA images for NCF and CAVIA size by size

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v3 as iio

#%%

def parse_paths(path, verbose=True):
    ## Define the empty dictionnary to store the paths
    paths = {}

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

        paths[(N, K)] = os.path.join(path, folder)

    return paths

cavia_paths = parse_paths('./CAVIA', verbose=False)
ncf_paths = parse_paths('./NCF', verbose=False)

print(f"CAVIA paths: {len(ncf_paths)}")

#%%

Ns = [6, 12, 60, 252, 1020]
Ks = [10, 50, 100, 500, 1000]

for N in Ns:
    for K in Ks:

        ## Get the paths for the current N and K (CAVIA couln't be run for N=1020, K=1000)
        try:
            cavia_path = cavia_paths[(N, K)]
        except:
            print(f"Error loading CAVIA data for N={N}, K={K}")
            cavia_path = None

        ncf_path = ncf_paths[(N, K)]

        ## Load the data
        if cavia_path is not None:
            cavia_data_train = iio.imread(os.path.join(cavia_path, 'few_shots_ind.png'))
            cavia_data_adapt = iio.imread(os.path.join(cavia_path, 'adapt/few_shots_ood.png'))
        else:
            cavia_data_train = np.zeros((32, 32, 3))
            cavia_data_adapt = np.zeros((32, 32, 3))

        ncf_data_train = iio.imread(os.path.join(ncf_path, 'few_shots_ind.png'))
        ncf_data_adapt = iio.imread(os.path.join(ncf_path, 'adapt/few_shots_ood.png'))

        ## Plot the images
        fig, axes = plt.subplots(2, 2, figsize=(25, 40))

        for i in range(4):
            if i == 0:
                axes[i//2, i%2].imshow(cavia_data_train)
                axes[i//2, i%2].set_title('CAVIA Meta-Train', fontsize=30)
            elif i == 2:
                axes[i//2, i%2].imshow(cavia_data_adapt)
                axes[i//2, i%2].set_title('CAVIA Meta-Test', fontsize=30)
            elif i == 1:
                axes[i//2, i%2].imshow(ncf_data_train)
                axes[i//2, i%2].set_title('NCF Meta-Train', fontsize=30)
            elif i == 3:
                axes[i//2, i%2].imshow(ncf_data_adapt)
                axes[i//2, i%2].set_title('NCF Meta-Test', fontsize=30)

            axes[i//2, i%2].axis('off')

        plt.tight_layout()
        plt.suptitle(f'N={N}, K={K}', fontsize=60, y=1.02)
        plt.show()

        ## Save the figure
        fig.savefig(f'Visualisations/N{N:04d}_K{K:04d}.png', bbox_inches='tight')

    #     break
    # break
