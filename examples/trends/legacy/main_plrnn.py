#%%[markdown]
## Teacher-Forcing Neural ODEs for Epilepsy Dataset

#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["JAX_PLATFORMS"] = 'cpu'

from selfmod import *
# jax.config.update('jax_platform_name', 'cpu')

from matplotlib import animation
# ## Import jax and debug NaNs
# import jax
# jax.config.update("jax_debug_nans", True)


#%%

## For reproducibility
seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)

## Dataloader hps
ode_count = 3          ## Total number of ODEs in the dataset
nb_experts = ode_count
nb_envs_per_fam = (600//nb_experts, 600//nb_experts)
top_k = 1

num_envs = (nb_envs_per_fam[0]*ode_count, 600)
num_shots = (-1, -1)
num_workers = 24
shuffle = False
train_proportion = 1.0  ## Min proporrion of the trajectory for training
test_proportion = 1.0

## Learner/model hps
context_pool_size = 20
context_size = 10
taylor_orders = (1, 0)
skip_steps = 1
loss_contributors = 600
max_ret_env_states = num_envs[0]
split_contexts = False

data_size = 1
hidden_size = 16*2

## Train and adapt hps
init_lrs = (1e-3, 1e-3)
sched_factor = 0.4
# transition_steps = 150
max_train_batches = 1
max_adapt_batches = 1
proximal_betas = (10., 10., 0.)       ## For the model, context and the gate, in that order

nb_outer_steps = 500
nb_inner_steps = (12, 12, 1)
nb_adapt_epochs = 100
validate_every = 10*1

print_error_every = (10, 10)

meta_train = True
save_trainer = True
meta_test = True

run_folder = None if meta_train else "./"
# run_folder = "./runs/250110-115030-Test/" if meta_train else "./"

data_folder = "./data/" if meta_train else "../../data/"


#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

adapt_folder = setup_run_folder(run_folder, os.path.basename(__file__), os.path.dirname(__file__), copy_ode_gen=False)







#%%
# ## Open the data file as a space saparated file
# import pandas as pd
# data = pd.read_csv(data_folder+"synthetic_control.data", sep=" ", header=None)
# print(data)

## Read the file line by line
time_series = []
with open(data_folder+"synthetic_control.data", 'r') as f:
    for line in f:
        time_series.append(list(map(float, line.split())))

print("Number of time series:", len(time_series))
print("Time series 0", time_series[0])
time_series = np.array(time_series)

## Normalise the time series
time_series = (time_series - np.mean(time_series, axis=0)) / np.std(time_series, axis=0)
# time_series = (time_series - np.mean(time_series, axis=0))
## scale it between -1 and 1
# time_series = (time_series - np.min(time_series, axis=0)) / (np.max(time_series, axis=0) - np.min(time_series, axis=0))

## Plot 6 randomly chosen time series
fig, ax = plt.subplots(2, 3, figsize=(6*3, 6))
ax = ax.flatten()

## Set the samme y limits for all plots
ylim = np.min(time_series), np.max(time_series)

np.random.seed(0)
for i in range(6):
    ts_id = np.random.randint(0, len(time_series))
    ax[i].plot(time_series[ts_id])
    ax[i].set_title(f"Time Series {ts_id}")
    ax[i].set_ylim(ylim)

plt.tight_layout()
plt.savefig(run_folder+"train_trajectories.png")

print("Time series shape:", type(time_series[0]), time_series.dtype, time_series.shape)

#%%

## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

train_dataloader = NumpyLoader(TrendsDataset(data_dir=data_folder, 
                                               skip_steps=skip_steps, 
                                               traj_prop_min=train_proportion,
                                               use_full_traj=True), 
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

val_dataloader = NumpyLoader(TrendsDataset(data_dir=data_folder, 
                                             skip_steps=skip_steps,
                                             traj_prop_min=test_proportion,
                                             use_full_traj=True),
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)



#%%

def xavier_uniform(key, shape):
    lim = 1 / np.sqrt(shape[0])
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)

class RootRNN(eqx.Module):
    root_utils: any
    network_size: int


    def __init__(self, data_size, latent_size, hidden_size, key=None):
        """ Shallow piece-wise linear RNN from Mannuel Brenner et al. 2024. Encoders and Decoders are identiy functions """
        super().__init__()
        D, L, M = data_size, hidden_size, latent_size

        keys = jax.random.split(key, 7)
        A = xavier_uniform(keys[0], (M, M))
        W1 = xavier_uniform(keys[1], (M, L))
        W2 = xavier_uniform(keys[2], (L, M))
        h2 = xavier_uniform(keys[3], (L,))
        h1 = xavier_uniform(keys[4], (M,))
        alpha = jnp.array([0.1])

        props = (data_size, latent_size, hidden_size, None, None)
        params = (A, W1, W2, h1, h2, alpha)

        _, shapes, treedef = flatten_pytree(params)
        self.root_utils = (shapes, treedef, props)
        self.network_size = sum(x.size for x in jax.tree_util.tree_leaves(params) if x is not None)

    def __call__(self, xs_gt, params):
        """ Predict based on the observation 
        x_gt: (T, D) 
        params: (A, W1, W2, h1, h2, alpha)
        """
        A, W1, W2, h1, h2, alpha = params
        z0 = jnp.zeros(xs_gt.shape[1])

        def f(z, x_gt):
            z_curr = alpha*z + (1-alpha)*x_gt     ## Teacher-Forcing
            z_next = A@z_curr + W1@jax.nn.relu(W2@z_curr + h2) + h1
            return z_next, z_next

        _, zs = jax.lax.scan(f, z0, xs_gt)

        return zs


# ## Define model and loss function for the learner
class Expert(eqx.Module):
    root_network: eqx.Module
    hyperlayer: list

    data_size: int
    latent_size: int
    # ctx_shift: jnp.ndarray

    def __init__(self, data_size, latent_size, hidden_size, context_size, ctx_shift=None, ctx_utils=None, key=None):
        self.data_size = data_size
        self.latent_size = latent_size

        self.root_network = RootRNN(data_size, data_size, hidden_size, key=key)

        in_hyper, out_hyper = context_size, self.root_network.network_size
        self.hyperlayer = eqx.nn.Linear(in_hyper, out_hyper, key=key, use_bias=True)

        # self.ctx_shift = jnp.array([ctx_shift], dtype=jnp.float32)     ## Shift the context by this much

    def __call__(self, xs, ctx):
        ## xts is in batch form

        # ctx = ctx + self.ctx_shift

        ## If there's Taylor Expansion to be done, has to be here !!!

        subject_weights = self.hyperlayer(ctx)
        # final_arr = self.root_weights + delta_arr

        shapes, treedef, _ = self.root_network.root_utils
        subject_params = unflatten_pytree(subject_weights, shapes, treedef)

        # xs = xts[0]
        # ts = jnp.broadcast_to(xts[1][None,:], (xts[0].shape[0], xts[1].shape[0]))     ## Broadcast along trajectories in the same environment

        # def predict(xs_):
        #     return self.root_network(xs_, subject_params)
        # return eqx.filter_vmap(predict)(xs_batch)

        return self.root_network(xs, subject_params)


# ## Define model and loss function for the learner
class MixER(eqx.Module):
    experts: list
    n_experts: int
    gate:dict
    is_moe: bool
    split_contexts: bool
    ctx_utils: any

    def __init__(self, data_size, latent_size, hidden_size, context_size, nb_experts, top_k, ctx_utils, key=None):
        keys = jax.random.split(key, nb_experts+2)
        self.split_contexts = False
        self.ctx_utils = ctx_utils

        ## Whether the context is split into tiny chunks for each expert
        if self.split_contexts:
            eff_context_size = context_size//nb_experts
        else:
            eff_context_size = context_size
        self.experts = [Expert(data_size, data_size, hidden_size, eff_context_size, ctx_utils=ctx_utils, key=keys[0]) for i in range(nb_experts)]

        lim = 1 / np.sqrt(context_size)
        gate_weight = jax.random.uniform(keys[-1], (context_size, nb_experts), minval=-lim, maxval=lim)

        def gating_function(gate, ctx):
            H = jax.lax.stop_gradient(gate["weight"].T) @ ctx
            G = jax.nn.softmax(H)       ## This works, but above doesn't
            return G

        # self.gate = {"weight":gate_weight, "temperature":gate_temp, "top_k":top_k, "function":gating_function}
        self.gate = {"weight":gate_weight, "temperature":[0.001], "top_k":top_k, "function":gating_function, "lsqr_factor":jnp.array([1e-3])}

        self.n_experts = nb_experts
        self.is_moe = True     ## Fix this !

    def __call__(self, xts, ctx):
        G = self.gate["function"](self.gate, ctx)
        if self.split_contexts:
            ctx_pieces = jnp.split(ctx, self.n_experts, axis=0)

        xs = xts[0]
        max_G = jnp.max(G)
        dy = jnp.zeros(xs.shape)
        for i in range(self.n_experts):
            if self.split_contexts:
                ctx_i = ctx_pieces[i]
            else:
                ctx_i = ctx

            contribution = jax.lax.cond(G[i]>max_G-1e-6, 
                                        lambda in_dat: self.experts[i](*in_dat), 
                                        lambda in_dat: jnp.zeros(in_dat[0].shape), 
                                        (xts, ctx_i))
            dy += contribution

        return dy


def env_loss_fn(model, ctx, y_hat, y):
    """
    Loss function for one environment. Leading dimension of y_hat corresponds to the pool size !
    """

    # term1 = jnp.mean((y_hat[...,-1]-y[...,-1])**2)
    term1 = jnp.mean((y_hat-y)**2)
    # term2 = jnp.mean(jnp.abs(ctx))
    # term3 = params_norm_squared(model)

    # term2 = jnp.abs(model.vectorfield.neuralnet.gate(ctx).squeeze())

    # loss_val = term1 + 1e-3*term2 + 1e-3*term3
    # loss_val = term1 + 1e-3*term2
    loss_val = term1

    return loss_val, (term1, 0., 0.)
    # return loss_val, (term1, term2, 0.)

## Example context to use
contexts = ArrayContextParams(nb_envs=num_envs[0], context_size=context_size, key=None)

gen_key, enc_key, dec_key = jax.random.split(model_key, num=3)
model = CatchAllModel(MixER(data_size=data_size,
                            latent_size=data_size,
                            hidden_size=hidden_size, 
                            context_size=contexts.eff_context_size, 
                            nb_experts=nb_experts, 
                            top_k=top_k, 
                            ctx_utils=None,
                            key=gen_key),
                    taylor_order=taylor_orders[0])

learner = Learner(model=model,
                context_size=contexts.eff_context_size, 
                context_pool_size=context_pool_size,
                env_loss_fn=env_loss_fn, 
                contexts=contexts,
                reuse_contexts=True,
                loss_contributors=loss_contributors,
                pool_filling="NF",      ## TODO. Put back NF as soon as mem permits
                loss_filling="NF",      ## The environment with the biggest loss is picked up
                key=model_key)

model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)) if x is not None)
print("\n\nTotal number of parameters in the model:", model_params)
print("Total number of parameters in one context:", contexts.eff_context_size)


#%%

## Define optimiser and train the model
init_lr_model, init_lr_ctx = init_lrs

total_steps = nb_outer_steps*nb_inner_steps[0]
bd_scales = {total_steps//3:sched_factor, 2*total_steps//3:sched_factor}
sched_model = optax.piecewise_constant_schedule(init_value=init_lr_model, boundaries_and_scales=bd_scales)
sched_ctx = optax.piecewise_constant_schedule(init_value=init_lr_ctx, boundaries_and_scales=bd_scales)
opt_model = optax.adabelief(sched_model)
opt_ctx = optax.adabelief(init_lr_ctx)

trainer = NCFTrainer(learner, (opt_model, opt_ctx), key=trainer_key)

#%%

## Use this loss criterion instead ...
# loss_criterion = lambda y, y_hat: jnp.quantile((y - y_hat)**2, q=q, axis=(-1, -2, -3))

## Meta-training
if meta_train == True:
    trainer_save_path = run_folder if save_trainer == True else False
    trainer.meta_train_gated(dataloader=train_dataloader, 
                        nb_epochs=1, 
                        nb_outer_steps=nb_outer_steps, 
                        nb_inner_steps=nb_inner_steps, 
                        inner_tols=(1e-16, 1e-16, 1e-16), 
                        proximal_betas=proximal_betas, 
                        max_train_batches=max_train_batches, 
                        print_error_every=print_error_every, 
                        save_checkpoints=True, 
                        validate_every=validate_every, 
                        save_path=trainer_save_path, 
                        val_dataloader=val_dataloader, 
                        val_nb_steps=nb_adapt_epochs,
                        val_criterion_id=0, 
                        max_val_batches=max_train_batches,
                        key=trainer_key)
else:
    print("Skipping meta-training ...")
    restore_folder = run_folder
    trainer.restore_trainer(path=run_folder)

#%%
## Test and visualise the results on a test dataloader
visualtester = DynamicsVisualTester(trainer, key=test_key)

ind_crit, all_ind_crit = visualtester.evaluate(train_dataloader, 
                                    taylor_order=taylor_orders[1], 
                                    nb_steps=nb_adapt_epochs,
                                    print_error_every=print_error_every, 
                                    criterion_id=0,
                                    verbose=True,
                                    val_dataloader=val_dataloader,
                                    max_ret_env_states=max_ret_env_states,
                                    max_adapt_batches=max_adapt_batches,
                                    stochastic=False)

visualtester.visualize_artefacts(save_path=run_folder+"artefacts.png", ylim=None)
print("Loss per InD environment:", all_ind_crit[0].tolist())


#%%
visualtester.visualize_dynamics(save_path=run_folder+"dynamics.png",
                                data_loader=val_dataloader,
                                # envs=[142, 143, 192, 193, 199, 200, 202, 203, 215, 232, 240, 242],
                                envs=jnp.arange(0, nb_envs_per_fam[0]*ode_count, 100).tolist(),
                                dims=(0,0),
                                traj=0,
                                share_axes=False,
                                key=test_key)


#%%
## Inspect the context, and evalualte the gate layer
contexts = learner.contexts
network = trainer.learner.model.vectorfield.neuralnet

# print("These the gate weights:", network.gate.weight.squeeze())

@eqx.filter_vmap
def gate_fn(ctx):
    return network.gate["function"](network.gate, ctx)

gate_vals = gate_fn(contexts.params)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7*2, 6))
## sort and plot histogram of gate values
# gate_vals = jnp.sort(gate_vals.flatten())
ax.hist(gate_vals.flatten(), bins=50);

ax.set_title(f"Gate Histogram with Top-K = {top_k}")
# print(gate_vals)

## inshow on ax2
img = ax2.imshow(gate_vals, aspect='auto', cmap='turbo', interpolation=None)
plt.colorbar(img)
ax2.set_xlabel("Experts")
ax2.set_ylabel("Environments")

## Set yticks in steps of 16
y_labels = np.arange(0, nb_envs_per_fam[0]*ode_count, nb_envs_per_fam[0])
ax2.set_yticks(y_labels)
ax2.set_yticklabels(y_labels)

x_labels = np.arange(0, nb_experts, 1)
ax2.set_xticks(x_labels)
ax2.set_xticklabels(x_labels)

ax2.set_title("Gate Values")

plt.draw()
plt.savefig(run_folder+"gate_histogram_big.png")




#%%

@eqx.filter_vmap(in_axes=(None, 0))
def gate_anim_fn(network, ctx):
    return network.gate["function"](network.gate, ctx)

## We want to do an animation of how the gate values change over time
all_gate_vals = []
for outer_step in list(range(0, nb_outer_steps, print_error_every[0]))+[nb_outer_steps-1]:
    contexts_ = eqx.tree_deserialise_leaves(run_folder+f"checkpoints/contexts_outstep_{outer_step:06d}.eqx", learner.contexts)
    network_ = eqx.tree_deserialise_leaves(run_folder+f"checkpoints/model_outstep_{outer_step:06d}.eqx", learner.model).vectorfield.neuralnet

    all_gate_vals.append(gate_anim_fn(network_, contexts_.params))

all_gate_vals = jnp.stack(all_gate_vals, axis=0)

#%%
## Plot the gate values as an animation
fig, ax = plt.subplots(1, 1, figsize=(6, 7))
img = ax.imshow(all_gate_vals[0], aspect='auto', cmap='turbo', interpolation="nearest")
plt.colorbar(img)
ax.set_xlabel("Experts")
ax.set_ylabel("Environments")

ax.set_title(f"Outer Step {0}")

ax.set_yticks(y_labels)
ax.set_yticklabels(y_labels)

ax.set_xticks(x_labels)

def animate(i):
    img.set_data(all_gate_vals[i])
    ax.set_title(f"Outer Step {i*print_error_every[0]}")
    return img,

ani = animation.FuncAnimation(fig, animate, frames=len(all_gate_vals), interval=100, blit=True)
ani.save(run_folder+"gate_vals_animation.gif", writer='pillow', fps=20)



#%%

perp = ode_count if ode_count > 1 else 4
visualtester.visualize_context_clusters(perplexities=(perp, perp),
                                        key=test_key,
                                        # key=jax.random.PRNGKey(time.time_ns()),
                                        save_path=run_folder+"context_clusters.png")

#%%
X = learner.contexts.params
# 1-100   Normal
# 101-200 Cyclic
# 201-300 Increasing trend
# 301-400 Decreasing trend
# 401-500 Upward shift
# 501-600 Downward shift

## We have 600 samples and 6 classes as above. Create the labels
labels = np.zeros((600,), dtype=int)
labels[100:200] = 1 
labels[200:300] = 2
labels[300:400] = 3
labels[400:500] = 4
labels[500:600] = 5

color_table = {0:"royalblue", 1:"crimson", 2:"forestgreen", 3:"darkorange", 4:"purple", 5:"black"}
colors = [color_table[l] for l in labels]

conditions = {0:"Normal", 1:"Cyclic", 2:"Increasing trend", 3:"Decreasing trend", 4:"Upward shift", 5:"Downward shift"}

## Use PCA instead
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
# X_reduced = X

# Plotting
plt.figure(figsize=(10, 7))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=50, c=colors)

markers = {0:'o', 1:'x', 2:'^', 3:'s', 4:'D', 5:'P'}
for class_label in range(6):
    marker = markers[class_label]
    plt.scatter(X_reduced[labels==class_label, 0], X_reduced[labels==class_label, 1], s=50, c=color_table[class_label], label=conditions[class_label], marker=marker)

plt.legend()

plt.title("Contexts Clustering", fontsize=24)
plt.xlabel("PC 1")
plt.ylabel("PC 2")

plt.draw()
plt.savefig(run_folder+"pc_contexts.png", bbox_inches='tight');




#%%

## Use Umap instead
import umap
reducer = umap.UMAP(n_components=2)
X_reduced = reducer.fit_transform(X)

# Plotting
plt.figure(figsize=(10, 7))

markers = {0:'o', 1:'x', 2:'^', 3:'s', 4:'D', 5:'P'}
for class_label in range(6):
    marker = markers[class_label]
    plt.scatter(X_reduced[labels==class_label, 0], X_reduced[labels==class_label, 1], s=50, c=color_table[class_label], label=conditions[class_label], marker=marker)

plt.legend()

plt.title("Contexts Clustering", fontsize=24)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

plt.draw()
plt.savefig(run_folder+"umap_contexts.png", bbox_inches='tight');







#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")

#%%
