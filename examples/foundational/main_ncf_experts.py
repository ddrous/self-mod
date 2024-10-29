#%%
# %load_ext autoreload
# %autoreload 2

import os
from selfmod import *

#%%

## For reproducibility
seed = 2029

## Dataloader hps
# num_envs = (9*28, 4*28)
num_envs = (16*10, 16*10)
num_shots = (-1, -1)
num_workers = 0
shuffle = False
train_proportion = 0.2  ## Min proporrion of the trajectory for training
test_proportion = 1.0

## Learner/model hps
context_pool_size = 2
context_size = 64*1*2
taylor_orders = (2, 0)
# ivp_args = {"return_traj":True, "max_steps":256*2, "dt_min":1e-4, "integrator":diffrax.Tsit5()}
ivp_args = {"return_traj":True, "max_steps":256*16, "dt_init":1e-2, "integrator":diffrax.Tsit5(), "rtol": 1e-2, "atol":1e-4, "clip_sol":None, "adjoint": diffrax.RecursiveCheckpointAdjoint()}
skip_steps = 2
# loss_contributors = 16*5//2
loss_contributors = 16*1
max_ret_env_states = num_envs[0]

## Train and adapt hps
init_lrs = (5e-4, 5e-4)
sched_factor = 1.0
max_train_batches = 1
max_adapt_batches = 1

proximal_betas = (0., 0.)

nb_outer_steps = 10000
nb_inner_steps = (1, 1)
nb_adapt_epochs = 5000
validate_every = 50*1

print_error_every = (100*1, 100*1)

meta_train = True
save_trainer = True
meta_test = True

# run_folder = "./241025-162623/"
run_folder = None
data_folder = "./data_2D/"


#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

adapt_folder = setup_run_folder(run_folder, os.path.basename(__file__))

#%%

## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

train_dataloader = NumpyLoader(ODEBenchDataset(data_dir=data_folder+"train.npz", 
                                               norm_consts=data_folder+"train_bounds.npy",   ## since more data in test/val sets
                                               num_shots=num_shots[0], 
                                               skip_steps=skip_steps, 
                                               traj_prop_min=train_proportion), 
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

val_dataloader = NumpyLoader(ODEBenchDataset(data_dir=data_folder+"test.npz", 
                                             norm_consts=data_folder+"train_bounds.npy",
                                             num_shots=num_shots[1], 
                                             skip_steps=skip_steps,
                                             traj_prop_min=test_proportion),
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

# ins, outs = next(iter(train_dataloader))
# ins.shape, outs.shape
# val_dataloader.num_batches

#%%

# # ## Plot all trajectories in the first 9 environments
# # plt_data = train_dataloader.dataset.dataset
# # plt_t = train_dataloader.dataset.t_eval

# ## Alternative way to gather the data
# (ins, ts), outs = next(iter(train_dataloader))
# plt_data = outs
# plt_t = ts

# print("Shapes of data and t_eval:", plt_data.shape, plt_t.shape)

# E_plot = 2
# E_ = 16

# fig, ax = plt.subplots(E_plot, 1, figsize=(6, E_plot*3))
# if E_plot==1:
#     ax = [ax]
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple']
# for e in range(E_plot):
#     e_plot_data = plt_data[e*E_:(e+1)*E_, :, :, 0]
#     e_t_eval = plt_t[e*E_:(e+1)*E_]
#     for e_ in range(E_):
#         # ax[e].plot(e_plot_data[e_].T, '-', color=colors[e_])
#         ax[e].plot(e_t_eval[e_], e_plot_data[e_].T, '-', color=colors[e_])
#         # if e==1:
#         #     print("t_eval is:", e_t_eval[e_])
#     ax[e].set_title(f"Environment {23+e}")
#     ax[e].set_xlabel("time")
#     ax[e].set_ylabel("y_0")

# plt.tight_layout()
# plt.show()



#%%


# ## Define model and loss function for the learner
class Expert(eqx.Module):
    layers_data: list
    activations_data: list
    layers_main: list
    layers_ctx: list
    activations_main: list
    activations_ctx: list

    ctx_utils:any
    depth_data:int
    depth_main:int

    def __init__(self, data_size, hidden_size, depth_data, depth_main, context_size, ctx_utils=None, key=None):
        self.ctx_utils = ctx_utils
        self.depth_data = depth_data
        self.depth_main = depth_main
        depth_ctx = depth_data

        layer_ctx_size = context_size
        # layer_ctx_size = context_size//depth_main  ## Size of the context to modulate each shared/main layer
        # assert context_size%depth_main==0, "Context size must be divisible by the depth of the main network"
        total_ctx_size = context_size * depth_main


        keys_ctx = jax.random.split(key, num=depth_ctx+1)
        hid_ctx_size = (context_size + total_ctx_size) // 2
        self.activations_ctx = [Swish(key=k) for k in keys_ctx[:depth_ctx]]
        self.layers_ctx = [eqx.nn.Linear(context_size, hid_ctx_size, key=keys_ctx[0])]
        self.layers_ctx += [eqx.nn.Linear(hid_ctx_size, hid_ctx_size, key=keys_ctx[i]) for i in range(1, depth_ctx)]
        self.layers_ctx += [eqx.nn.Linear(hid_ctx_size, total_ctx_size, key=keys_ctx[depth_ctx])]

        keys = jax.random.split(key, num=depth_data+depth_main+2)
        self.activations_data = [Swish(key=k) for k in keys[:depth_data]]
        self.layers_data = [eqx.nn.Linear(0+data_size, hidden_size, key=keys[0])]
        self.layers_data += [eqx.nn.Linear(hidden_size, hidden_size, key=keys[i]) for i in range(1, depth_data)]
        self.layers_data += [eqx.nn.Linear(hidden_size, layer_ctx_size, key=keys[depth_data])]

        self.activations_main = [Swish(key=k) for k in keys[depth_data+2:]]
        self.layers_main = [eqx.nn.Linear(2*layer_ctx_size, hidden_size, key=keys[depth_data+1])]
        self.layers_main += [eqx.nn.Linear(hidden_size+layer_ctx_size, hidden_size, key=keys[depth_data+i+1]) for i in range(1, depth_main)]
        self.layers_main += [eqx.nn.Linear(hidden_size, data_size, key=keys[depth_data+depth_main+1])]

        assert len(self.layers_data) == len(self.activations_data)+1, f"Total number of layers {len(self.layers_data)} and activations {len(self.activations_data)} mismatch in the data network"
        assert len(self.layers_main) == len(self.activations_main)+1, f"Total number of layers {len(self.layers_main)} and activations {len(self.activations_main)} mismatch in the main network"


    def __call__(self, in_dat):
        t, y, ctx = in_dat

        for layer, activation in zip(self.layers_ctx[:-1], self.activations_ctx):
            ctx = activation(layer(ctx))
        ctx = self.layers_ctx[-1](ctx)

        ## Split the context into parts for each layer
        ctx_parts = jnp.split(ctx[:], self.depth_main, axis=0)

        # y = jnp.concatenate([t_arr, y], axis=0)
        for layer, activation in zip(self.layers_data[:-1], self.activations_data):
            y = activation(layer(y))
        y = self.layers_data[-1](y)

        ## Apply the context at each layer (except the very last)
        for layer, activation, ctx_part in zip(self.layers_main[:-1], self.activations_main, ctx_parts):
            y = jnp.concatenate([y, ctx_part], axis=0)
            y = activation(layer(y))
        y = self.layers_main[-1](y)

        return y


# ## Define model and loss function for the learner
class Model(eqx.Module):
    experts: list
    n_experts: int
    top_k: int
    gate_weight: eqx.Module
    is_moe: bool

    def __init__(self, data_size, hidden_size, depth, context_size, nb_experts=10, top_k=2, key=None):
        keys = jax.random.split(key, nb_experts+2)

        self.experts = [Expert(data_size, hidden_size, 2, depth, context_size, key=keys[i]) for i in range(nb_experts)]

        self.gate_weight = jnp.zeros((nb_experts, context_size))

        self.n_experts = nb_experts
        self.top_k = top_k
        self.is_moe = True


    def gating_function(self, ctx):
        H = self.gate_weight@ctx

        topk_vals, topk_idx = jax.lax.top_k(H, self.top_k)
        infs = jnp.full_like(H, -jnp.inf)
        infs = infs.at[topk_idx].set(topk_vals / 0.1)
        G = jax.nn.softmax(infs)

        return G

    def __call__(self, t, y, ctx):
        G = self.gating_function(ctx)

        dy = jnp.zeros_like(y)
        for i in range(self.n_experts):
            dy += G[i]*self.experts[i]((t, y, ctx))

        return dy




def env_loss_fn(model, ctx, y_hat, y):
    """
    Loss function for one environment. Leading dimension of y_hat corresponds to the pool size !
    """

    term1 = jnp.mean((y_hat-y)**2)
    # term2 = jnp.mean(jnp.abs(ctx))
    # term3 = params_norm_squared(model)

    # term2 = jnp.abs(model.vectorfield.neuralnet.gate(ctx).squeeze())
    
    # loss_val = term1 + 1e-3*term2 + 1e-3*term3
    loss_val = term1

    return loss_val, (term1, 0., 0.)

## Example context to use
contexts = ArrayContextParams(nb_envs=num_envs[0], context_size=context_size, key=None)

neuralnet = Model(data_size=2,
                hidden_size=64*2, 
                depth=3,
                context_size=context_size,
                key=model_key) 

model = NeuralODE(neuralnet=neuralnet,
                    taylor_order=taylor_orders[0],
                    ivp_args=ivp_args,
                    t_eval=None,    ## t_eval is provided with each model call
                    taylor_ad_mode="forward")

# print("Model is ...", model)

learner = Learner(model=model,
                context_size=contexts.eff_context_size, 
                context_pool_size=context_pool_size,
                env_loss_fn=env_loss_fn, 
                contexts=contexts,
                reuse_contexts=True,
                loss_contributors=loss_contributors,
                pool_filling="NF",     ## TODO. Put back NF as soon as mem permits
                loss_filling="NF",   ## First only, we only need the first loss contributor
                key=model_key)


model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)) if x is not None)
print("\n\nTotal number of parameters in the model:", model_params)
print("Total number of parameters in one context:", contexts.eff_context_size)



#%%

## Define optimiser and train the model
init_lr_model, init_lr_ctx = init_lrs
total_steps = nb_outer_steps*nb_inner_steps[0]
# total_steps = nb_outer_steps
bd_scales = {total_steps//3:sched_factor, 2*total_steps//3:sched_factor}
sched_model = optax.piecewise_constant_schedule(init_value=init_lr_model, boundaries_and_scales=bd_scales)
# opt_model = optax.adam(sched_model)
opt_model = optax.chain(optax.clip(10.), optax.adam(sched_model))

opt_ctx = optax.adam(init_lr_ctx)

trainer = NCFTrainer(learner, (opt_model, opt_ctx), key=trainer_key)

#%%

## Use this loss criterion instead ...
# loss_criterion = lambda y, y_hat: jnp.quantile((y - y_hat)**2, q=q, axis=(-1, -2, -3))

## Meta-training
if meta_train == True:
    trainer_save_path = run_folder if save_trainer == True else False
    trainer.meta_train(dataloader=train_dataloader, 
                        nb_epochs=1, 
                        nb_outer_steps=nb_outer_steps, 
                        nb_inner_steps=nb_inner_steps, 
                        inner_tols=(1e-16, 1e-16), 
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
    # trainer.meta_train_noalm(dataloader=train_dataloader, 
    #                     nb_epochs=1, 
    #                     nb_outer_steps=nb_outer_steps, 
    #                     max_train_batches=max_train_batches, 
    #                     print_error_every=print_error_every, 
    #                     save_checkpoints=True,
    #                     validate_every=validate_every, 
    #                     save_path=trainer_save_path, 
    #                     val_dataloader=val_dataloader, 
    #                     val_nb_steps=nb_adapt_epochs,
    #                     val_criterion_id=0, 
    #                     max_val_batches=max_train_batches,
    #                     key=trainer_key)
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



# new_ctx = np.load(run_folder+"contexts_params.npy")
# learner.contexts = eqx.tree_at(lambda c:c.params, learner.contexts, new_ctx)
# eqx.tree_serialise_leaves(run_folder+"contexts.eqx", learner.contexts)


#%%
visualtester.visualize_dynamics(save_path=run_folder+"dynamics.png",
                                data_loader=val_dataloader,
                                nb_envs=16*5,
                                # envs=[142, 143, 192, 193, 199, 200, 202, 203, 215, 232, 240, 242],
                                # envs=jnp.arange(0, 16*4).tolist(),
                                traj=1,
                                share_axes=False,
                                key=test_key)

# exit()

#%%
## Inspect the context, and evalualte the gate layer
contexts = learner.contexts
network = trainer.learner.model.vectorfield.neuralnet

@eqx.filter_vmap
def gate_fn(ctx):
    ctx_fam, ctx_env = jnp.split(ctx, 2, axis=0)

    in_dat = jnp.concatenate((jnp.array([0.]), jnp.array([0.,0.]), ctx_fam), axis=0)
    H = network.gate_weight@ctx

    topk_vals, topk_idx = jax.lax.top_k(H, 2)
    infs = jnp.full_like(H, -jnp.inf)
    infs = infs.at[topk_idx].set(topk_vals)
    G = jax.nn.softmax(infs)

    return G

gate_vals = gate_fn(contexts.params)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
## sort and plot histogram of gate values
# gate_vals = jnp.sort(gate_vals.flatten())
ax.hist(gate_vals.flatten(), bins=50);

print(gate_vals)
# print(gate_vals.sum(axis=0))
# print(network.gate_weight)

plt.draw()
plt.savefig(run_folder+"gate_histogram.png")



#%%




























## Adapt the model to the new dataset
if meta_test:
    adapt_dataloader = NumpyLoader(ODEBenchDataset(data_dir=data_folder+"adapt_train.npz", 
                                                   adaptation=True,
                                                   norm_consts=data_folder+"adapt_train_bounds.npy",
                                                   num_shots=num_shots[0], 
                                                   skip_steps=skip_steps,
                                                   traj_prop_min=train_proportion),
                                batch_size=num_envs[1], 
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    adapt_dataloader_test = NumpyLoader(ODEBenchDataset(data_dir=data_folder+"adapt_test.npz", 
                                                        adaptation=True,
                                                        norm_consts=data_folder+"adapt_train_bounds.npy",
                                                        num_shots=num_shots[0], 
                                                        skip_steps=skip_steps,
                                                        traj_prop_min=test_proportion),
                                batch_size=num_envs[1],
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
                                        max_ret_env_states=max_ret_env_states,
                                        max_adapt_batches=max_adapt_batches,
                                        stochastic=False)
    print("Loss per OoD environment:", all_ood_crit[0].tolist())

    visualtester.visualize_artefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)

    visualtester.visualize_dynamics(save_path=adapt_folder+"dynamics_ood.png",
                                    data_loader=adapt_dataloader_test,
                                    nb_envs=4,
                                    traj=0,
                                    share_axes=False,
                                    key=test_key)

#%%

visualtester.visualize_context_clusters(perplexities=(3, 3),
                                        # key=test_key,
                                        key=jax.random.PRNGKey(time.time_ns()),
                                        save_path=run_folder+"context_clusters.png")

#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")

#%%
# adapt_dataloader_test.dataset.dataset.max()
# train_dataloader.dataset.dataset.max()
# val_dataloader.dataset.dataset.max()