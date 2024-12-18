#%%
# %load_ext autoreload
# %autoreload 2

import os
from selfmod import *

#%%

## For reproducibility
seed = 2026

## Dataloader hps
num_envs = (9*28, 4*28)
num_shots = (-1, -1)
num_workers = 0
shuffle = False
train_proportion = 0.1
test_proportion = 1.0

## Learner/model hps
context_pool_size = 4
context_size = 256
taylor_orders = (2, 0)
ivp_args = {"y0_pad_size":0, "return_traj":True, "max_steps":4096*1, "dt_init":1e-2}
skip_steps = 5
loss_contributors = 8
max_ret_env_states = 8

## Train and adapt hps
init_lrs = (5e-4, 1e-2)
sched_factor = 1.
max_train_batches = 1
max_adapt_batches = 1

nb_outer_steps = 6000
nb_inner_steps = (10, 10)
nb_adapt_epochs = 500
validate_every = 500

print_error_every = (100, 100)

meta_train = False
save_trainer = True
meta_test = True

run_folder = "./runs/241011-145434/"
# run_folder = None
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

train_dataloader = NumpyLoader(ODEBenchDataset(data_dir=data_folder+"train_data.npz", 
                                               num_shots=num_shots[0], 
                                               skip_steps=skip_steps,
                                               traj_prop=train_proportion), 
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

val_dataloader = NumpyLoader(ODEBenchDataset(data_dir=data_folder+"test_data.npz", 
                                             num_shots=num_shots[1], 
                                             skip_steps=skip_steps,
                                             traj_prop=test_proportion),
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

# ins, outs = next(iter(train_dataloader))
# ins.shape, outs.shape

#%%

# # ## Plot all trajectories in the first 9 environments
# # plt_data = train_dataloader.dataset.dataset
# # plt_t = train_dataloader.dataset.t_eval

# ## Alternative way to gather the data
# (ins, ts), outs = next(iter(train_dataloader))
# plt_data = outs
# plt_t = ts

# print("Shapes of data and t_eval:", plt_data.shape, plt_t.shape)

# E_plot = 1
# E_ = 9

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
#     ax[e].set_title(f"Environment {e}")
#     ax[e].set_xlabel("time")
#     ax[e].set_ylabel("y_0")

# plt.tight_layout()
# plt.show()



#%%












# ## Define model and loss function for the learner
class MultiMLP(eqx.Module):
    layers_data: list
    layers_shared: list
    activations: list
    ctx_utils:any

    ## A compression layer to bypass the model
    # layer_context: eqx.nn.Linear

    def __init__(self, data_size, hidden_size, int_size, context_size, ctx_utils, key=None):
        self.ctx_utils = ctx_utils

        keys = jax.random.split(key, num=12)
        self.activations = [Swish(key=key_i) for key_i in keys[:]]

        self.layers_data = [eqx.nn.Linear(1+data_size, hidden_size, key=keys[3]), self.activations[2], 
                            eqx.nn.Linear(hidden_size, hidden_size, key=keys[4]), self.activations[3], 
                            eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]), self.activations[9], 
                            eqx.nn.Linear(hidden_size, int_size, key=keys[5])]

        self.layers_shared = [eqx.nn.Linear(int_size+context_size, hidden_size, key=keys[6]), self.activations[4], 
                              eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]), self.activations[3], 
                              eqx.nn.Linear(hidden_size, hidden_size, key=keys[7]), self.activations[5], 
                              eqx.nn.Linear(hidden_size, hidden_size, key=keys[2]), self.activations[4], 
                              eqx.nn.Linear(hidden_size, hidden_size, key=keys[8]), self.activations[6], 
                              eqx.nn.Linear(hidden_size, data_size, key=keys[9])]

        # self.layer_context = eqx.nn.Linear(context_size, data_size, key=keys[0])

    def __call__(self, t, y, ctx_arr):

        ctx_shapes, ctx_treedef, ctx_static, _ = self.ctx_utils
        ctx_params = unflatten_pytree(ctx_arr, ctx_shapes, ctx_treedef)
        ctx_fun = eqx.combine(ctx_params, ctx_static)

        t_arr = jnp.array([t])
        ctx = ctx_fun(t_arr)

        y = jnp.concatenate([t_arr, y], axis=0)
        for layer in self.layers_data:
            y = layer(y)

        y = jnp.concatenate([y, ctx], axis=0)
        for layer in self.layers_shared:
            y = layer(y)

        # y = self.layer_context(ctx)

        return y




def env_loss_fn(model, ctx, y_hat, y):
    """
    Loss function for one environment. Leading dimension of y_hat corresponds to the pool size !
    """

    term1 = jnp.mean((y_hat-y)**2)
    # term2 = jnp.mean(jnp.abs(ctx))
    # term3 = params_norm_squared(model)

    # loss_val = term1 + 1e-3*term2 + 1e-3*term3
    loss_val = term1

    return loss_val, (term1, 0., 0.)

## Example context to use
contexts = InfDimContextParams(nb_envs=num_envs[0], 
                                input_dim=1,
                                output_dim=context_size,
                            hidden_size=64, 
                            depth=3, 
                            activation=Swish(key=model_key),
                            key=None)
# contexts = ArrayContextParams(nb_envs=num_envs[0], context_size=context_size)

neuralnet = MultiMLP(data_size=2,
                     int_size=context_size,
                     hidden_size=128, 
                     context_size=context_size,
                     ctx_utils=contexts.ctx_utils,
                     key=model_key) 

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
                reuse_contexts=False,
                loss_contributors=loss_contributors,
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
opt_model = optax.adam(sched_model)

opt_ctx = optax.adam(init_lr_ctx)

trainer = NCFTrainer(learner, (opt_model, opt_ctx), key=trainer_key)

#%%

## Meta-training
if meta_train == True:
    trainer_save_path = run_folder if save_trainer == True else False
    trainer.meta_train(dataloader=train_dataloader, 
                        nb_epochs=1, 
                        nb_outer_steps=nb_outer_steps, 
                        nb_inner_steps=nb_inner_steps, 
                        inner_tols=(1e-16, 1e-16), 
                        proximal_betas=(10., 10.), 
                        max_train_batches=max_train_batches, 
                        print_error_every=print_error_every, 
                        validate_every=validate_every, 
                        save_path=trainer_save_path, 
                        val_dataloader=val_dataloader, 
                        val_nb_steps=nb_adapt_epochs,
                        val_criterion_id=0, 
                        max_val_batches=max_train_batches,
                        key=trainer_key)
else:
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
                                    max_adapt_batches=max_adapt_batches)

visualtester.visualize_artefacts(save_path=run_folder+"artefacts.png")
print("Loss per InD environment:", all_ind_crit[0].tolist())
#%%
visualtester.visualize_dynamics(save_path=run_folder+"dynamics.png",
                                data_loader=val_dataloader,
                                nb_envs=15,
                                traj=0,
                                share_axes=False,
                                key=test_key)


#%%










## Adapt the model to the new dataset
if meta_test:
    adapt_dataloader = NumpyLoader(ODEBenchDataset(data_dir=data_folder+"adapt_train.npz", 
                                                   adaptation=True,
                                                   num_shots=num_shots[0], 
                                                   skip_steps=skip_steps,
                                                   traj_prop=train_proportion),
                                batch_size=num_envs[1], 
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    adapt_dataloader_test = NumpyLoader(ODEBenchDataset(data_dir=data_folder+"adapt_test.npz", 
                                                        adaptation=True,
                                                        num_shots=num_shots[0], 
                                                        skip_steps=skip_steps,
                                                        traj_prop=test_proportion),
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
                                        max_adapt_batches=max_adapt_batches)

    visualtester.visualize_artefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)
    print("Loss per OoD environment:", all_ood_crit[0].tolist())

visualtester.visualize_dynamics(save_path=adapt_folder+"dynamics_ood.png",
                                data_loader=adapt_dataloader_test,
                                nb_envs=4,
                                traj=0,
                                share_axes=False,
                                key=test_key)







#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")
