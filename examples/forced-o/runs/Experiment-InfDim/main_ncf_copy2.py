#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
# os.environ["EQX_ON_ERROR"] = 'breakpoint'

from selfmod import *
# jax.config.update("jax_debug_nans", True)



#%%

## For reproducibility
seed = 2024

## Dataloader hps
num_envs = (8, 6)
num_shots = (-1, -1)
num_workers = 0
shuffle = False

## Learner/model hps
context_pool_size = 2
context_size = 256
taylor_orders = (2, 0)
# taylor_weight_init = 10.        ## Pos for all Taylor, neg for no-Taylor, 0 for equal chances at the start
# ivp_args = {"T":1.0, "y0_pad_size":0, "return_traj":True, "adjoint":diffrax.DirectAdjoint()} 
ivp_args = {"T":1.0, "y0_pad_size":0, "return_traj":True, "max_steps":4096*1, "dt_init":1e-2}
skip_steps = 1
loss_contributors = 6
max_ret_env_states = 8

## Train and adapt hps
init_lrs = (5e-4, 1e-2)
sched_factor = 1.
max_train_batches = -1
max_adapt_batches = -1

nb_train_epochs = 1
nb_outer_steps = 2000
nb_inner_steps = (10, 10)

print_error_every = (10, 10)
validate_every = nb_outer_steps//20

# nb_adapt_epochs = 7500
nb_adapt_epochs = 1500       ## To use during evaluation and visulisation

meta_train = True
# run_folder = "./runs/240915-003956/"
run_folder = None
save_trainer = True

meta_test = True


#%%
mother_key = jax.random.PRNGKey(seed)

#%%

if meta_train == True:
    if not os.path.exists('./runs'):
        os.mkdir('./runs')

    # Run folder to store the result of this run
    if run_folder == None:
        run_folder = './runs/'+time.strftime("%y%m%d-%H%M%S")+'/'
    else:
        print("Using user-defined run folder:", run_folder)
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        print("Created a new run folder at:", run_folder)

    # Save the run scripts in that folder
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {run_folder}")

    # Save the selfmod module files as well
    os.system(f"cp -r ../../selfmod {run_folder}")
    print("Completed copied scripts ")
else:
    print("No training. Loading model and results from:", run_folder)

## Create a folder for the adaptation results
if meta_test:
    adapt_folder = run_folder+"adapt/"
    if not os.path.exists(adapt_folder):
        os.mkdir(adapt_folder)







#%%

## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

train_dataloader = NumpyLoader(DynamicsDataset(data_dir="./data/train_data.npz", 
                                               num_shots=num_shots[0], 
                                               skip_steps=skip_steps), 
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

val_dataloader = NumpyLoader(DynamicsDataset(data_dir="./data/test_data.npz", 
                                             num_shots=num_shots[1], 
                                             skip_steps=skip_steps),
                              batch_size=num_envs[0],
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)


# ins, outs = next(iter(train_dataloader))
# ins.shape, outs.shape



#%%












# ## Define model and loss function for the learner
class MultiMLP(eqx.Module):
    layers_data: list
    layers_shared: list
    activations: list
    ctx_utils:any

    def __init__(self, data_size, hidden_size, int_size, context_size, ctx_utils, key=None):
        self.ctx_utils = ctx_utils

        keys = jax.random.split(key, num=12)
        self.activations = [Swish(key=key_i) for key_i in keys[:7]]

        self.layers_data = [eqx.nn.Linear(1+data_size, hidden_size, key=keys[3]), self.activations[2], 
                            eqx.nn.Linear(hidden_size, hidden_size, key=keys[4]), self.activations[3], 
                            eqx.nn.Linear(hidden_size, int_size, key=keys[5])]

        self.layers_shared = [eqx.nn.Linear(int_size+context_size, hidden_size, key=keys[6]), self.activations[4], 
        # self.layers_shared = [eqx.nn.Linear(1+data_size+context_size, hidden_size, key=keys[6]), self.activations[4], 
                              eqx.nn.Linear(hidden_size, hidden_size, key=keys[7]), self.activations[5], 
                              eqx.nn.Linear(hidden_size, hidden_size, key=keys[8]), self.activations[6], 
                              eqx.nn.Linear(hidden_size, data_size, key=keys[9])]

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
        # y = jnp.concatenate([t_arr, y, ctx], axis=0)
        for layer in self.layers_shared:
            y = layer(y)

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

## Just so the model knows the kind of context to use
contexts = InfDimContextParams(nb_envs=num_envs[0], 
                                input_dim=1,
                                output_dim=context_size,
                            hidden_size=32, 
                            depth=2, 
                            activation=Swish(key=model_key),
                            key=None)
neuralnet = MultiMLP(data_size=2,
                     int_size=context_size,
                     hidden_size=128, 
                     context_size=context_size,
                     ctx_utils=contexts.ctx_utils,
                     key=model_key) 

model = NeuralODE(neuralnet=neuralnet,
                    taylor_order=taylor_orders[0],
                    ivp_args=ivp_args,
                    t_eval=train_dataloader.dataset.t_eval.tolist(),
                    taylor_ad_mode="forward")

learner = Learner(model=model,
                context_size=contexts.eff_context_size, 
                context_pool_size=context_pool_size,
                env_loss_fn=env_loss_fn, 
                contexts=contexts,     ## Optional, but good for saving !
                reuse_contexts=False,
                loss_contributors=loss_contributors,
                key=model_key)




model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)) if x is not None)
print("\n\nTotal number of parameters in the model:", model_params)
print("Total number of parameters in one context:", contexts.eff_context_size)



#%%

## Define optimiser and train the model
init_lr_model, init_lr_ctx = init_lrs

bd_scales = {nb_train_epochs//3:sched_factor, 2*nb_train_epochs//3:sched_factor}
sched_model = optax.piecewise_constant_schedule(init_value=init_lr_model, boundaries_and_scales=bd_scales)
opt_model = optax.adam(sched_model)

opt_ctx = optax.adam(init_lr_ctx)

# trainer = CAVIATrainer(learner, (opt_model, opt_ctx), key=trainer_key)
trainer = NCFTrainer(learner, (opt_model, opt_ctx), key=trainer_key)

#%%

# with jax.profiler.trace("data/jax-trace", create_perfetto_link=True, create_perfetto_trace=True):

## Meta-training
if meta_train == True:
    trainer_save_path = run_folder if save_trainer == True else False
    # trainer.meta_train(dataloader=train_dataloader,
    #                     nb_epochs=nb_train_epochs,
    #                     nb_inner_steps=nb_inner_steps, 
    #                     max_train_batches=max_train_batches,
    #                     print_error_every=print_error_every, 
    #                     save_path=trainer_save_path, 
    #                     val_dataloader=val_dataloader, 
    #                     val_criterion_id=0,
    #                     validate_every=nb_train_epochs//10,
    #                     backup_contexts=True,
    #                     key=trainer_key)
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
    print("\nNo training, loaded model and results from "+ run_folder +" folder ...\n")














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













## Adapt the model to the new dataset
if meta_test:
    adapt_dataloader = NumpyLoader(DynamicsDataset(data_dir="./data/adapt_train.npz", 
                                                   num_shots=num_shots[0], 
                                                   skip_steps=skip_steps),
                                batch_size=num_envs[1], 
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    all_shots_loader = NumpyLoader(DynamicsDataset(data_dir="./data/adapt_test.npz", 
                                                   num_shots=num_shots[0], 
                                                   skip_steps=skip_steps),
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
                                        val_dataloader=all_shots_loader,
                                        max_ret_env_states=max_ret_env_states,
                                        max_adapt_batches=max_adapt_batches)

    visualtester.visualize_artefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)

    print("Loss per OoD environment:", all_ood_crit[0].tolist())








#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")
