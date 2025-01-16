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
context_pool_size = 2
context_size = 256
taylor_orders = (2, 0)
ivp_args = {"return_traj":True, "max_steps":256*16, "integrator":diffrax.Dopri5(), "rtol": 1e-3, "atol":1e-6, "clip_sol":None, "adjoint": diffrax.RecursiveCheckpointAdjoint()}
skip_steps = 3
loss_contributors = 60
max_ret_env_states = num_envs[0]
split_contexts = False

data_size = 1
hidden_size = 16*1
depth = 1

## Train and adapt hps
init_lrs = (1e-3, 1e-3)
sched_factor = 0.4
# transition_steps = 150
max_train_batches = 1
max_adapt_batches = 1
proximal_betas = (10., 10., 0.)       ## For the model, context and the gate, in that order

nb_outer_steps = 2000
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
                                               use_full_traj=False), 
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

val_dataloader = NumpyLoader(TrendsDataset(data_dir=data_folder, 
                                             skip_steps=skip_steps,
                                             traj_prop_min=test_proportion,
                                             use_full_traj=False),
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

#%%

# ## Define model and loss function for the learner
class Expert(eqx.Module):
    layers_data: list
    activations_data: list
    layers_main: list
    # layers_ctx: list
    activations_main: list
    # activations_ctx: list
    data_size:int

    ctx_utils:any
    depth_data:int
    depth_main:int

    # ctx_shift: jnp.ndarray

    def __init__(self, data_size, hidden_size, depth_data, depth_main, context_size, ctx_utils=None, key=None):
        self.ctx_utils = ctx_utils
        self.depth_data = depth_data
        self.depth_main = depth_main
        depth_ctx = depth_data
        self.data_size = data_size

        intermediate_size = hidden_size//2

        # keys_ctx = jax.random.split(key, num=depth_ctx+1)
        # hid_ctx_size = (context_size + intermediate_size) // 2
        # self.activations_ctx = [Swish(key=k) for k in keys_ctx[:depth_ctx]]
        # self.layers_ctx = [eqx.nn.Linear(context_size, hid_ctx_size, key=keys_ctx[0])]
        # self.layers_ctx += [eqx.nn.Linear(hid_ctx_size, hid_ctx_size, key=keys_ctx[i]) for i in range(1, depth_ctx)]
        # self.layers_ctx += [eqx.nn.Linear(hid_ctx_size, intermediate_size, key=keys_ctx[depth_ctx])]

        keys = jax.random.split(key, num=depth_data+depth_main+2)
        hid_ctx_size = (data_size + intermediate_size) // 2
        self.activations_data = [Swish(key=k) for k in keys[:depth_data]]
        self.layers_data = [eqx.nn.Linear(data_size, hid_ctx_size, key=keys[0])]
        self.layers_data += [eqx.nn.Linear(hid_ctx_size, hid_ctx_size, key=keys[i]) for i in range(1, depth_data)]
        self.layers_data += [eqx.nn.Linear(hid_ctx_size, intermediate_size, key=keys[depth_data])]

        self.activations_main = [Swish(key=k) for k in keys[depth_data+2:]]
        self.layers_main = [eqx.nn.Linear(2*intermediate_size, hidden_size, key=keys[depth_data+1])]
        self.layers_main += [eqx.nn.Linear(hidden_size, hidden_size, key=keys[depth_data+i+1]) for i in range(1, depth_main)]
        self.layers_main += [eqx.nn.Linear(hidden_size, data_size, key=keys[depth_data+depth_main+1])]

        assert len(self.layers_data) == len(self.activations_data)+1, f"Total number of layers {len(self.layers_data)} and activations {len(self.activations_data)} mismatch in the data network"
        assert len(self.layers_main) == len(self.activations_main)+1, f"Total number of layers {len(self.layers_main)} and activations {len(self.activations_main)} mismatch in the main network"

        # self.ctx_shift = jnp.array([ctx_shift])


    def __call__(self, t, y, ctx_flat):
        # for layer, activation in zip(self.layers_ctx[:-1], self.activations_ctx):
        #     ctx = activation(layer(ctx))
        # ctx = self.layers_ctx[-1](ctx)

        ex_shapes, ex_treedef, ex_static, _ = self.ctx_utils
        params = unflatten_pytree(ctx_flat, ex_shapes, ex_treedef)
        ctx_fun = eqx.combine(params, ex_static)
        ctx = ctx_fun(jnp.array([t]))

        # y = jnp.concatenate([t_arr, y], axis=0)
        for layer, activation in zip(self.layers_data[:-1], self.activations_data):
            y = activation(layer(y))
        y = self.layers_data[-1](y)

        ## Apply the context at each layer (except the very last)
        y = jnp.concatenate([y, ctx], axis=0)
        for layer, activation in zip(self.layers_main[:-1], self.activations_main):
            y = activation(layer(y))
        y = self.layers_main[-1](y)

        # return jax.nn.tanh(y).reshape((self.latent_size, -1))
        return y


# ## Define model and loss function for the learner
class NeuralNet(eqx.Module):
    experts: list
    n_experts: int
    gate:dict
    is_moe: bool
    split_contexts: bool
    ctx_utils: any

    def __init__(self, data_size, hidden_size, depth, context_size, nb_experts, top_k, ctx_utils, key=None):
        keys = jax.random.split(key, nb_experts+2)
        self.split_contexts = False
        self.ctx_utils = ctx_utils

        ## Whether the context is split into tiny chunks for each expert
        if self.split_contexts:
            eff_context_size = context_size//nb_experts
        else:
            eff_context_size = context_size
        self.experts = [Expert(data_size, hidden_size, 2, depth, eff_context_size, ctx_utils=ctx_utils, key=keys[0]) for i in range(nb_experts)]

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

    def __call__(self, t, y, ctx):
        G = self.gate["function"](self.gate, ctx)
        # G = jax.lax.stop_gradient(self.gate["function"](self.gate, ctx))

        if self.split_contexts:
            ctx_pieces = jnp.split(ctx, self.n_experts, axis=0)

        latent_size = y.shape[0]
        data_size = self.experts[0].data_size

        max_G = jnp.max(G)
        dy = jnp.zeros(data_size)
        for i in range(self.n_experts):
            if self.split_contexts:
                ctx_i = ctx_pieces[i]
            else:
                ctx_i = ctx

            contribution = jax.lax.cond(G[i]>max_G-1e-6, 
                                        lambda in_dat: self.experts[i](*in_dat), 
                                        lambda in_dat: jnp.zeros(data_size), 
                                        (t, y, ctx_i))
            dy += contribution

        return dy

        # return self.experts[0](t, y, ctx) 




def env_loss_fn(model, ctx, y_hat, y):
    """
    Loss function for one environment. Leading dimension of y_hat corresponds to the pool size !
    """

    term1 = jnp.mean((y_hat-y)**2)
    # term2 = jnp.mean(jnp.abs(ctx))
    # term3 = params_norm_squared(model)

    # term2 = jnp.abs(model.vectorfield.neuralnet.gate(ctx).squeeze())

    # loss_val = term1 + 1e-3*term2 + 1e-3*term3
    # loss_val = term1 + 1e-3*term2
    loss_val = term1

    # return loss_val, (term1, term2, 0.)
    return loss_val, (term1, 0., 0.)

## Example context to use
# contexts = ArrayContextParams(nb_envs=num_envs[0], context_size=context_size, key=None)
contexts = InfDimContextParams(nb_envs=num_envs[0], input_dim=1, output_dim=hidden_size//2, hidden_size=32*1, depth=3, key=None)

gen_key, enc_key, dec_key = jax.random.split(model_key, num=3)
neuralnet = NeuralNet(data_size=data_size,
                    hidden_size=hidden_size, 
                    depth=depth,
                    context_size=contexts.eff_context_size, 
                    nb_experts=nb_experts, 
                    top_k=top_k, 
                    ctx_utils=contexts.ctx_utils,
                    key=gen_key) 

model = NeuralODE(neuralnet=neuralnet,
                taylor_order=taylor_orders[0],
                ivp_args=ivp_args,
                t_eval=None,    ## t_eval is provided with each model call
                taylor_ad_mode="forward")

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
# opt_model = optax.chain(optax.clip(1.), optax.adam(sched_model))
opt_ctx = optax.adabelief(init_lr_ctx)
# opt_ctx = optax.chain(optax.clip(1.), optax.adam(init_lr_ctx))

# sched_model = optax.exponential_decay(init_value=init_lr_model, transition_steps=transition_steps, decay_rate=0.99)
# opt_model = optax.adam(sched_model)
# sched_ctx = optax.exponential_decay(init_value=init_lr_ctx, transition_steps=transition_steps, decay_rate=0.99)
# opt_ctx = optax.adam(sched_ctx)

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
