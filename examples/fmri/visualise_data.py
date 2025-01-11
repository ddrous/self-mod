#%%
# from selfmod import *
import scipy.io as sio

## Load the two matlab matrices in the data folder: 1. NBKO_PLRNN_dataset_RestReference.mat and 2. NBKO_PLRNN_dataset.mat

# Load the data using scipy
# data_ref = sio.loadmat('data/NBKO_PLRNN_dataset_RestReference.mat')
data = sio.loadmat('data/NBKO_PLRNN_dataset.mat')

# print(data.keys())
# print(data['Data'].shape)

X = data['Data'] ## shape (1,26)
print(type(X[0,0]))
print(X[0])



# ## Visualise the data
# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(10, 5))
# plt.plot(X[0, :])
