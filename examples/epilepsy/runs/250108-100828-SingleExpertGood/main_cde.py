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
seed = 2028
np.random.seed(seed)
torch.manual_seed(seed)

## Dataloader hps
ode_count = 2          ## Total number of ODEs in the dataset
nb_experts = ode_count
nb_envs_per_fam = (80//nb_experts, 80//nb_experts)
top_k = 1

num_envs = (nb_envs_per_fam[0]*ode_count, nb_envs_per_fam[1]*ode_count)
num_shots = (-1, -1)
num_workers = 24
shuffle = False
train_proportion = 1.0  ## Min proporrion of the trajectory for training
test_proportion = 1.0

## Learner/model hps
context_pool_size = 3
context_size = 10
taylor_orders = (2, 0)
# ivp_args = {"return_traj":True, "max_steps":256*16, "integrator":diffrax.Tsit5(), "rtol": 1e-3, "atol":1e-6, "clip_sol":None, "adjoint": diffrax.BacksolveAdjoint()}
ivp_args = {"return_traj":True, "max_steps":256*16, "integrator":diffrax.Dopri5(), "rtol": 1e-3, "atol":1e-6, "clip_sol":None, "adjoint": diffrax.RecursiveCheckpointAdjoint()}
skip_steps = 1
loss_contributors = nb_envs_per_fam[0]*1
max_ret_env_states = num_envs[0]
split_contexts = False

data_size = 1
latent_size = 16
hidden_size = 32*2
depth = 3

## Train and adapt hps
init_lrs = (1e-3, 1e-3)
sched_factor = 0.4
# transition_steps = 150
max_train_batches = 1
max_adapt_batches = 1
proximal_betas = (10., 10., 0.)       ## For the model, context and the gate, in that order

nb_outer_steps = 40
nb_inner_steps = (10, 10, 1)
nb_adapt_epochs = 10
validate_every = 10*1

print_error_every = (10*1, 10*1)

meta_train = False
save_trainer = True
meta_test = True

run_folder = None if meta_train else "./"
# run_folder = "./runs/250103-123848-Test/" if meta_train else "./"

data_folder = "./data/" if meta_train else "../../data/"


#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

adapt_folder = setup_run_folder(run_folder, os.path.basename(__file__), os.path.dirname(__file__), copy_ode_gen=False)

#%%

## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

train_dataloader = NumpyLoader(EpilepsyDataset(data_dir=data_folder+"train.npz", 
                                               skip_steps=skip_steps, 
                                               traj_prop_min=train_proportion), 
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

val_dataloader = NumpyLoader(EpilepsyDataset(data_dir=data_folder+"train.npz", 
                                             skip_steps=skip_steps,
                                             traj_prop_min=test_proportion),
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

#%%

# ## Plot the trajectories in the a few environments

## Alternative way to gather the data
(outs, ts), _ = next(iter(train_dataloader))

print("Shapes of data and t_eval:", outs.shape, ts.shape)

E_plot = 5

# fig, ax = plt.subplots(E_plot, 1, figsize=(6, E_plot*3))
fig, ax = plt.subplots(1, E_plot, figsize=(6*E_plot, 3))
ax = ax.flatten()
if E_plot==1:
    ax = [ax]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'r', 'g', 'b', 'c', 'm', 'y']
ylim = outs.min(), outs.max()
xlim = 0, 1
for e, e_ in enumerate(np.random.choice(outs.shape[0], E_plot)):
    ax[e].plot(ts[e_], outs[e_].squeeze(), '-', color=colors[e])
    ax[e].set_title(f"Env {e}")
    ax[e].set_xlabel("Normalised Time")
    ax[e].set_ylabel(f"$EEG$")
    ax[e].set_ylim(ylim)
    ax[e].set_xlim(xlim)

plt.tight_layout()
plt.draw()
plt.savefig(run_folder+"train_trajectories.png")




#%%

class RootNetwork(eqx.Module):
    network: list
    root_utils: any
    network_size: int     ## The effective/actual size of a root network (flattened neural network)

    def __init__(self, input_dim, output_dim, hidden_size, depth, activation=jax.nn.softplus, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        self.network = MLP(input_dim, output_dim, hidden_size, depth, activation, key=key)
        
        props = (input_dim, output_dim, hidden_size, depth, activation)
        params, static = eqx.partition(self.network, eqx.is_array)
        _, shapes, treedef = flatten_pytree(params)
        self.root_utils = (shapes, treedef, static, props)

        self.network_size = sum(x.size for x in jax.tree_util.tree_leaves(params) if x is not None)

    def __call__(self, x):
        return self.network(x)


# ## Define model and loss function for the learner
class Expert(eqx.Module):
    root_weights: jnp.ndarray
    hyperlayer: list
    root_utils: list

    data_size: int
    latent_size: int

    def __init__(self, data_size, latent_size, hidden_size, depth, context_size, key=None):
        self.data_size = data_size
        self.latent_size = latent_size

        root = RootNetwork(latent_size, (data_size+1)*latent_size, hidden_size, depth, Swish(key=key), key=key)
        self.root_utils = root.root_utils
        root_params, static = eqx.partition(root.network, eqx.is_array)
        self.root_weights = flatten_pytree(root_params)[0]

        in_hyper, out_hyper = context_size, root.network_size
        self.hyperlayer = eqx.nn.Linear(in_hyper, out_hyper, key=key, use_bias=False)

    def __call__(self, t, y, ctx):
        delta_arr = self.hyperlayer(ctx)
        final_arr = self.root_weights + delta_arr

        shapes, treedef, static, _ = self.root_utils
        params = unflatten_pytree(final_arr, shapes, treedef)
        root_fun = eqx.combine(params, static)

        return root_fun(y)


# ## Define model and loss function for the learner
class Generator(eqx.Module):
    experts: list
    n_experts: int
    gate:dict
    is_moe: bool
    split_contexts: bool

    def __init__(self, data_size, latent_size, hidden_size, depth, context_size, nb_experts, top_k, key=None):
        keys = jax.random.split(key, nb_experts+2)
        self.split_contexts = False

        ## Whether the context is split into tiny chunks for each expert
        if self.split_contexts:
            eff_context_size = context_size//nb_experts
        else:
            eff_context_size = context_size
        self.experts = [Expert(data_size, latent_size, hidden_size, depth, eff_context_size, key=keys[0]) for i in range(nb_experts)]

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
        ctx_pieces = jnp.split(ctx, self.n_experts, axis=0)

        latent_size = y.shape[0]
        data_size = self.experts[0].data_size

        max_G = jnp.max(G)
        dy = jnp.zeros(latent_size*(data_size+1), )
        for i in range(self.n_experts):
            if self.split_contexts:
                ctx_i = ctx_pieces[i]
            else:
                ctx_i = ctx

            contribution = jax.lax.cond(G[i]>max_G-1e-6, 
                                        lambda in_dat: self.experts[i](*in_dat), 
                                        # lambda in_dat: jnp.zeros((latent_size, data_size+1)), 
                                        lambda in_dat: jnp.zeros(latent_size*(data_size+1), ), 
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
contexts = ArrayContextParams(nb_envs=num_envs[0], context_size=context_size, key=None)

gen_key, enc_key, dec_key = jax.random.split(model_key, num=3)
neuralnet = Generator(data_size=data_size,
                        latent_size=latent_size,
                        hidden_size=hidden_size, 
                        depth=depth,
                        context_size=context_size, 
                        nb_experts=nb_experts, 
                        top_k=top_k, 
                        key=gen_key) 
encoder = eqx.nn.MLP(in_size=data_size,     ## For the initial conditions only !
                    out_size=latent_size, 
                    width_size=hidden_size, 
                    depth=depth, 
                    use_bias=True, 
                    activation=jax.nn.softplus,
                    key=enc_key)
decoder = eqx.nn.Linear(in_features=latent_size,           ## For all time steps !
                        out_features=data_size, 
                        use_bias=True, 
                        key=dec_key)

model = NeuralCDE(neuralnet=neuralnet,
                taylor_order=taylor_orders[0],
                ivp_args=ivp_args,
                encoder=encoder,
                decoder=decoder,
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
opt_model = optax.adam(sched_model)
# opt_model = optax.chain(optax.clip(1.), optax.adam(sched_model))
opt_ctx = optax.adam(init_lr_ctx)
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
                                envs=jnp.arange(0, nb_envs_per_fam[0]*ode_count, 10).tolist(),
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
labels = np.load(data_folder+"train.npz")["condition"].astype(int)
print("Labels:", labels)
color_table = {0:"royalblue", 1:"crimson"}
colors = [color_table[l] for l in labels]

conditions = {0:"Healthy", 1:"Epileptic"}

# import umap
# umap_reducer = umap.UMAP(n_components=2, random_state=time.time_ns()%(2**32), min_dist=0.0, spread=1.0, metric="euclidean")
# Fit and transform the data
# X_reduced = umap_reducer.fit_transform(X)

## Use PCA instead
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plotting
plt.figure(figsize=(10, 7))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=50, c=colors)

for class_label in [0,1]:
    marker = "o" if class_label==0 else "x"
    plt.scatter(X_reduced[labels==class_label, 0], X_reduced[labels==class_label, 1], s=50, c=color_table[class_label], label=conditions[class_label], marker=marker)

plt.legend()

plt.title("Training Context Dimensionality Reduction", fontsize=24)
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
plt.xlabel("PC 1")
plt.ylabel("PC 2")

# # Adding annotations for each point
# for i in range(0, X_reduced.shape[0], nb_envs_per_fam[0]):
#     label = labels[i]
#     # label = i
#     plt.text(X_reduced[i, 0], X_reduced[i, 1]+5e-1, str(label), fontsize=16, ha='left', va='bottom', color='black', weight='bold')

plt.draw()
plt.savefig(run_folder+"pc_contexts.png", bbox_inches='tight');

print("X0", X[0])
print("X1", X[1])


#%%

X = learner.contexts.params
y = np.load(data_folder+"train.npz")["condition"].astype(int)

# ## Let's use Gaussian Mixture Models to cluster the contexts
# from sklearn.mixture import GaussianMixture
# # gmm = GaussianMixture(n_components=2, random_state=seed)
# gmm = GaussianMixture(n_components=2)
# ## Initialise the mean in a supervised way
# gmm.means_ = np.array([X[y==i].mean(axis=0) for i in range(2)])
# gmm.fit(X)

# ## Predict the clusters
# y_pred = gmm.predict(X)


## Let's use Random Forest to classify the contexts
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=seed)
clf.fit(X, y)
y_pred = clf.predict(X)

## Calculate accuracy with sklearn metrics
from sklearn.metrics import accuracy_score
acc = accuracy_score(y, y_pred)
print("Accuracy:", acc)


print("Y = ", y)
print("Y_pred = ", y_pred)

















#%%
## Adapt the model to the new dataset
if meta_test:
    adapt_id = nb_envs_per_fam[1]*1+1     ## The single environment to adapt to (the difficult rectangular one)

    adapt_dataset = EpilepsyDataset(data_dir=data_folder+"adapt.npz", 
                                             skip_steps=skip_steps,
                                             traj_prop_min=test_proportion,
                                             adaptation=True)
    adapt_dataset.total_envs = 1
    adapt_dataset.dataset = adapt_dataset.dataset[adapt_id:, :, :, :]
    adapt_dataset.t_eval = adapt_dataset.t_eval[adapt_id:, :]

    adapt_dataloader = NumpyLoader(dataset=adapt_dataset,
                                # batch_size=num_envs[1], 
                                batch_size=1, 
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    adapt_dataset_test = EpilepsyDataset(data_dir=data_folder+"adapt.npz", 
                                             skip_steps=skip_steps,
                                             traj_prop_min=test_proportion,
                                             adaptation=True)
    adapt_dataset_test.total_envs = 1
    adapt_dataset_test.dataset = adapt_dataset_test.dataset[adapt_id:, :, :, :]
    adapt_dataset_test.t_eval = adapt_dataset_test.t_eval[adapt_id:, :]

    adapt_dataloader_test = NumpyLoader(dataset=adapt_dataset_test,
                                batch_size=1,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    ood_crit, all_ood_crit = visualtester.evaluate(adapt_dataloader, 
                                        taylor_order=taylor_orders[1], 
                                        nb_steps=nb_adapt_epochs,
                                        print_error_every=print_error_every, 
                                        criterion_id=0,
                                        verbose=True,
                                        val_dataloader=adapt_dataloader_test,
                                        max_ret_env_states=1,
                                        max_adapt_batches=max_adapt_batches,
                                        stochastic=False)
    print("Loss per OoD environment:", all_ood_crit[0].tolist())

#%%
visualtester.visualize_artefacts(save_path=adapt_folder+"artefacts_adapt.png", adaptation=True)

visualtester.visualize_dynamics(save_path=adapt_folder+"dynamics_adapt.png",
                                data_loader=adapt_dataloader_test,
                                nb_envs=1,
                                traj=0,
                                dims=(0,0),     ## The Data is 1-dimensional
                                share_axes=False,
                                key=test_key)

#%%

perp = ode_count if ode_count > 1 else 4
visualtester.visualize_context_clusters(perplexities=(perp, perp),
                                        # key=test_key,
                                        key=jax.random.PRNGKey(time.time_ns()),
                                        save_path=adapt_folder+"context_clusters.png")

#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")

#%%
