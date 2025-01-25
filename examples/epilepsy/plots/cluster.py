#%%[markdown]
# Hierarchical Shallow Piece-Wise Recurrent Neural Network on Epilepsy Data

#%%
# %load_ext autoreload
# %autoreload 2

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["JAX_PLATFORMS"] = 'cpu'

from selfmod import *
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)

from matplotlib import animation


#%%

## For reproducibility
seed = 2022
np.random.seed(seed)
torch.manual_seed(seed)

## Dataloader hps
nb_families = 2
nb_experts = nb_families

use_small_train_set = True
num_envs_train = 60 if use_small_train_set else 80
nb_envs_per_fam = (num_envs_train//nb_experts, 11420//nb_experts)   ## (Expected)

num_envs = (num_envs_train, 11420)
num_shots = (-1, -1)
num_workers = 24
shuffle = False
train_proportion = 1.0  ## Min proporrion of the trajectory for training
test_proportion = 1.0

## Learner/model hps
context_pool_size = 1
context_size = 10
taylor_orders = (0, 0)
skip_steps = 1
loss_contributors = num_envs[0]//nb_experts
max_ret_env_states = num_envs[0]
split_context = False
shift_context = True

meta_learner = "hier-shPLRNN"
data_size = 1
hidden_size = 16
latent_size = data_size
same_expert_init = True
tf_alpha_min = 0.5  ## Teacher forcing alpha (1. means no teacher forcing)

## Train and adapt hps
init_lrs = (1e-3, 1e-3)
sched_factor = 1.0
max_train_batches = 1
max_adapt_batches = 1
proximal_betas = (10., 10.)       ## For the model, context and the gate, in that order

nb_outer_steps = 1000
nb_inner_steps = (12, 12)
nb_adapt_epochs = 10000
validate_every = 10
print_error_every = (10, 10)

gate_update_strategy = "least_squares"      ## "least_squares" or "gradient_descent"
gate_update_every = 1                       ## Update the gate every x inner steps (useful in least_squares mode)
gating_hyperparams = {"max_kmeans":20, "convergence_tol":1e-3, "noise_level":1e-4}

context_regularization = False               ## Regularize the context with an L1 penalty
same_expert_optstate = False                  ## Use the same optstate for all experts
self_reweighting = False                     ## Reweight the outer loss by its own softmax

meta_train = True
meta_test = False

run_folder = None if meta_train else "./"
# run_folder = "./runs/250103-123848-Test/" if meta_train else "./"

data_folder = "./" if meta_train else "../../data/"



#%%[markdown]
# ## Meta-training



#%%

## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

train_file = "train_small.npz" if use_small_train_set else "train.npz"
train_dataloader = NumpyLoader(EpilepsyDataset(data_dir=data_folder+train_file, 
                                               skip_steps=skip_steps, 
                                               traj_prop_min=train_proportion), 
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

data = train_dataloader.dataset.dataset.squeeze()

data.shape
# %%

## Do PCA clustering on the data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)

kmeans = KMeans(n_clusters=nb_families, random_state=seed)
kmeans.fit(pca_data)
colors_map = {0:"r", 1:"b"}
## PLot the class 1 first
plt.scatter(pca_data[kmeans.labels_==1, 0], pca_data[kmeans.labels_==1, 1], label="0")
plt.scatter(pca_data[kmeans.labels_==0, 0], pca_data[kmeans.labels_==0, 1], label="1")
plt.legend()


## Get the true labels
true_labels = np.load("train.npz")['condition'][:60].astype(int)

plt.figure()
plt.scatter(pca_data[true_labels==1, 0], pca_data[true_labels==1, 1], label="0")
plt.scatter(pca_data[true_labels==0, 0], pca_data[true_labels==0, 1], label="1")

plt.legend()
