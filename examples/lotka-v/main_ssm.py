#%%[markdown]

# ## SSM idea
# Can we use ideas from SSM for meta-learning
"""
\begin{align}
h_{t+1} &= A_{\theta}(h_t, x_0) + B_{\theta}(u_t, x_0) \\
x_{t+1} &= C_{\theta}(h_t, x_0)
\end{align}
"""


#%%
import jax

print("Available devices:", jax.devices())

# from jax import config
# config.update("jax_debug_nans", True)

import jax.numpy as jnp

import numpy as np
from scipy.integrate import solve_ivp

import equinox as eqx
import diffrax

# import matplotlib.pyplot as plt
from selfmod import *

import optax
import time


#%%

SEED = 2025
main_key = jax.random.PRNGKey(SEED)

## Model hps
latent_size = 32
mlp_hidden_size = 64
mlp_depth = 3
data_size = 2

## Optimiser hps
init_lr = 1e-3

## Training hps
print_every = 1000
nb_epochs = 10000
batch_size = -1
variational = False
control_traj_id = 0

## Testing hps
test_env_vis = 0

train = True

## Data hps
data_folder = "./data/" if train else "../../data/"
run_folder = "./runs/241213-150028-Test/" if train else "./"
# run_folder = None if train else "./"


#%%
### Create and setup the run folder
if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)
_ = setup_run_folder(run_folder, os.path.basename(__file__))


#%%
train_data = np.load(data_folder+'train_data.npz')['X']
test_data = np.load(data_folder+'test_data.npz')['X']

print("train data shape:", train_data.shape)
print("test data shape:", test_data.shape)


# %%

class RNN(eqx.Module):
    data_size: int
    latent_size: int
    variational: bool

    A: eqx.Module
    B: eqx.Module
    C: eqx.Module

    def __init__(self, data_size, latent_size, mlp_hidden_size, mlp_depth, variational, key=None):
        self.data_size = data_size
        self.latent_size = latent_size
        self.variational = variational

        keys = jax.random.split(key, num=4)
        self.A = eqx.nn.MLP(data_size + latent_size, 
                            latent_size, 
                            mlp_hidden_size, 
                            mlp_depth, 
                            use_bias=True, 
                            activation=jax.nn.softplus,
                            key=keys[0])
        self.B = eqx.nn.MLP(data_size + data_size, 
                            latent_size,
                            mlp_hidden_size, 
                            mlp_depth, 
                            use_bias=True, 
                            activation=jax.nn.softplus, 
                            key=keys[1])
        self.C = eqx.nn.MLP(data_size + latent_size, 
                            data_size,
                            mlp_hidden_size, 
                            mlp_depth, 
                            use_bias=True, 
                            activation=jax.nn.softplus, 
                            key=keys[1])

    def __call__(self, x0s, us, key):
        """ Forward call: one control trajectory, many init conditions from the same environment """

        def rollout(x0):
            def f(carry, u):
                h = carry
                h_next = self.A(jnp.concatenate([h, x0])) + self.B(jnp.concatenate([u, x0]))
                x_next = self.C(jnp.concatenate([h, x0]))
                return h_next, (h_next, x_next)

            h0 = jnp.zeros(self.latent_size)
            _, (hs, xs) = jax.lax.scan(f, h0, us)

            return xs, hs

        return eqx.filter_vmap(rollout)(x0s)


# %%

model_keys = jax.random.split(main_key, num=2)

model = RNN(data_size=data_size, 
            latent_size=latent_size, 
            variational=variational,
            mlp_hidden_size=mlp_hidden_size, 
            mlp_depth=mlp_depth, 
            key=model_keys[0])

## Print the total number of learnable paramters in the model components
print(f"Number of learnable parameters in the model: {count_params(model)/1000:3.1f} k")

# %%

def loss_fn(model, batch, key):
    X = batch    ## X: (nb_envs, nb_trajs, traj_len, data_size), U: (batch_size, seq_len, data_size)

    X0s = X[:, :, 0, :]
    Us = X[:, control_traj_id, :, :]

    Xs, _ = eqx.filter_vmap(model, in_axes=(0, 0, None))(X0s, Us, key)

    l2 = jnp.mean((Xs - X)**2)
    linf = jnp.max(jnp.abs(Xs - X))

    ## TODO: implement a loss to stop the B model from being forgotten


    return l2, (linf, )


@eqx.filter_jit
def train_step(model, batch, opt_state, key):
    print('\nCompiling function "train_step" ...')

    (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, key)

    updates, opt_state = opt_node.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, aux_data


# def sample_batch_portion(outputs, t_evals, traj_prop=traj_prop_train):
#     """Returns a trajectory of len 20% of the original trajectory """

#     if traj_prop == 1.0:
#         return outputs, t_evals

#     min_len = int(t_evals.shape[0] * traj_prop)
#     start_idx = np.random.randint(0, t_evals.shape[0] - min_len)
#     end_idx = start_idx + min_len

#     ts = t_evals[start_idx:end_idx]
#     trajs = outputs[:, start_idx:end_idx, :]

#     return trajs, ts




#%%

if train:
    sched_node = optax.exponential_decay(init_value=init_lr, transition_steps=10, decay_rate=0.99)
    opt_node = optax.adam(sched_node)
    opt_state_node = opt_node.init(eqx.filter(model, eqx.is_array))

    train_key, _ = jax.random.split(main_key)

    nb_data_points = train_data.shape[1]
    losses = []

    print(f"\n\n=== Beginning Training ... ===")
    start_time = time.time()

    batch = train_data  ## (nb_envs, nb_trajs, traj_len, data_size) - TODO No batching is done

    for epoch in range(nb_epochs):

        # nb_batches = 0
        # loss_sum_node = 0.

        # for i in range(0, nb_data_points, batch_size):

        #     train_key, _ = jax.random.split(train_key)
        #     model, opt_state_node, loss, (rec_loss, kl_loss) = train_step(model, batch, opt_state_node, train_key)

        #     loss_sum_node += loss

        #     nb_batches += 1

        # loss_epoch_node = loss_sum_node/nb_batches
        # losses.append(loss_epoch_node)


        model, opt_state_node, loss, (linf_loss,) = train_step(model, batch, opt_state_node, train_key)
        losses.append(loss)

        if epoch%print_every==0 or epoch<=3 or epoch==nb_epochs-1:
            print(f"    Epoch: {epoch:-5d}      L2_loss: {loss:.12f}      Linf_loss: {linf_loss:.12f}", flush=True)
            eqx.tree_serialise_leaves(f"{run_folder}/checkpoints/model_{epoch:05d}.eqx", model)

    wall_time = time.time() - start_time
    time_in_hmsecs = seconds_to_hours(wall_time)
    print("\nTotal GD training time: %d hours %d mins %d secs" %time_in_hmsecs)

    print(f"Training Complete, saving model to folder: {run_folder}")
    eqx.tree_serialise_leaves(run_folder+"model.eqx", model)
    np.save(run_folder+"losses.npy", np.array(losses))

else:
    model = eqx.tree_deserialise_leaves(run_folder+"model_lfads.eqx", model)
    try:
        losses = np.load(run_folder+"losses_lfads.npy")
    except:
        losses = []
    print("Model loaded from folder")


# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax = sbplot(np.array(losses), x_label='Epoch', y_label='L2', y_scale="log", label='Train Losses', ax=ax);
plt.legend()
plt.draw()
plt.savefig(run_folder+"loss.png", dpi=100, bbox_inches='tight')



# %%

## Test the model
def test_model(model, batch, env_id=test_env_vis):
    X = batch
    x0 = X[env_id, :, 0, :]
    us = X[env_id, control_traj_id, :, :]
    xs = X[env_id, :, :, :]

    x_hat, x_lats = model(x0, us, main_key)
    return x_hat, xs, x_lats

X_hat, X, X_lat = test_model(model, test_data)

print(f"Test MSE: {jnp.mean((X - X_hat)**2):.8f}")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'yellow', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
colors = colors*10

## Visualize channels trajectories
traj_id = np.random.randint(0, X_hat.shape[0])
ch_vis_start = 0
t = np.arange(X_hat.shape[1])
for i in range(ch_vis_start, ch_vis_start+2):
    if i==0:
        sbplot(t, X_hat[traj_id, :,i], "+-", x_label='Time Step', y_label='x', label=f'Pred', title=f'Trajectories', ax=ax, alpha=0.5, color=colors[i])
        sbplot(t, X[traj_id, :,i], "-", lw=1, label=f'True', ax=ax, color=colors[i])
    else:
        sbplot(t, X_hat[traj_id, :,i], "x-", x_label='Time Step', y_label='x', ax=ax, alpha=0.5, color=colors[i])
        sbplot(t, X[traj_id, :,i], "-", lw=1, ax=ax, color=colors[i])

## Limit ax x and y axis to (-5,5)
plt.draw();
# plt.ylim(-30/10000, 30/10000)

## Save the plot
plt.savefig(run_folder+"results_lfads.png", dpi=100, bbox_inches='tight')

# ## Save the results to a npz file
np.savez(run_folder+"test_data.npz", latents=X_lat, gt_trajs=X, recons=X_hat)


# %%[markdown]
## Setup UMAP and plot the latents


# %%
## Visualize the latent space after a UMAP projection of X_lats
import umap

## Get all the latents
X_lats = []
envs = []
for e in range(9):
    _, _, x_lats = test_model(model, test_data, i)
    X_lats.append(x_lats)
    envs.append(np.ones(x_lats.shape[0])*e)
X_lats = np.concatenate(X_lats, axis=0)
envs = np.concatenate(envs, axis=0)

# %matplotlib inline
reducer = umap.UMAP(n_components=2, min_dist=0., n_neighbors=55, metric='euclidean')
embeddings = reducer.fit_transform(X_lats[:, -1, :]) # TEST

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for e in range(9):
    plt.scatter(embeddings[envs==e, 0], embeddings[envs==e, 1], alpha=0.5, label=f'Env {e}')
# plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5)

plt.legend()
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')


# %%

plt.draw()
plt.savefig(run_folder+"umap_latents.png", dpi=100, bbox_inches='tight');
