#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

from selfmod import *
# jax.config.update("jax_debug_nans", True)



#%%

## For the experiment
seed = 2026
# num_envs = (12500, 1000)
# taylor_orders = (0, 0)
# context_size = 2    ## from 2 to 50
csv_export_path = "results.csv"

## Dataloader hps
# num_envs = (12500, 1000)  ## (meta-train, meta-test) vary for low-high data regime
num_shots = (10, 100)
num_workers = 0
# shuffle = False

## Learner/model hps
context_pool_size = 1 if taylor_orders[0] == 0 else 2
# context_size = 2    ## from 2 to 50
# taylor_orders = (2, 0)
loss_contributors = 250
envs_batch_size = num_envs[0]

## Train and adapt hps
init_lrs = (1e-3, 1e-3)
sched_factor = 1.
max_train_batches = -1
max_adapt_batches = -1

nb_outer_steps = 10000
nb_inner_steps = (20, 20)

print_error_every = (100, 100)   ## every 1 epoch, every 1000 batches

nb_adapt_epochs = 5000

meta_train = True
# run_folder = "./runs/240715-025946-Test/"
run_folder = None
save_trainer = True

meta_test = True

max_ret_env_states = 250


#%%
mother_key = jax.random.PRNGKey(seed)

#%%

if meta_train == True:
    if not os.path.exists('./NCF'):
        os.mkdir('./NCF')

    # Run folder to store the result of this run
    if run_folder == None:
        run_folder = './NCF/'+time.strftime("%y%m%d-%H%M%S")+'/'
    else:
        print("Using user-defined run folder:", run_folder)
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        print("Created a new run folder at:", run_folder)

    # Save the run scripts in that folder
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {run_folder}")

    # Save the selfmod module files as well
    os.system(f"cp -r ../../../../selfmod {run_folder}")
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

train_dataloader = NumpyLoader(SinusoidDataset(num_envs=num_envs[0],
                                            num_shots=num_shots[0]), 
                              batch_size=envs_batch_size, 
                              shuffle=False,
                              num_workers=num_workers,
                              drop_last=False)

val_dataloader = NumpyLoader(SinusoidDataset(num_envs=num_envs[0],
                                            num_shots=num_shots[1]), ## TODO make sure the val set has the same environment size as the train set
                              batch_size=envs_batch_size, 
                              shuffle=False,
                              num_workers=num_workers,
                              drop_last=False)


# ins, outs = next(iter(train_dataloader))
# ins.shape, outs.shape




#%%












## Define model and loss function for the learner
class MultiMLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, context_size, hidden_size, key=None):
        keys = jax.random.split(key, num=4)

        self.layers = [eqx.nn.Linear(in_size+context_size, hidden_size, key=keys[0]),
                        jax.nn.softplus,
                        eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]),
                        jax.nn.softplus,
                        eqx.nn.Linear(hidden_size, out_size, key=keys[2]),
                        ]

    def __call__(self, x, ctx):

        y = jnp.concatenate((x, ctx))
        for layer in self.layers:
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

contexts = ArrayContextParams(nb_envs=envs_batch_size,
                            context_size=context_size)

neuralnet = MultiMLP(in_size=1,
                     out_size=1,
                     context_size=context_size,
                     hidden_size=40, 
                     key=model_key)

model = NeuralContextFlow(neuralnet=neuralnet, 
                            taylor_order=taylor_orders[0],
                            )

learner = Learner(model=model,
                context_size=context_size, 
                context_pool_size=context_pool_size,
                env_loss_fn=env_loss_fn, 
                reuse_contexts=False,
                loss_contributors=loss_contributors,
                key=model_key)




model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)) if x is not None)
print("\n\nTotal number of parameters in the model:", model_params)
print("Total number of parameters in one context:", contexts.eff_context_size)




#%%

## Define optimiser and train the model
init_lr_model, init_lr_ctx = init_lrs
nb_train_steps = nb_outer_steps*nb_inner_steps[0]*len(train_dataloader)
bd_scales = {nb_train_steps//3:sched_factor, 2*nb_train_steps//3:sched_factor}
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
    #                     nb_outer_steps=nb_train_epochs,
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
                        inner_tols=(1e-12, 1e-12), 
                        proximal_betas=(10., 10.), 
                        max_train_batches=max_train_batches,
                        print_error_every=print_error_every, 
                        save_path=trainer_save_path, 
                        val_dataloader=val_dataloader, 
                        max_val_batches=max_train_batches,
                        validate_every=nb_outer_steps//10,
                        val_criterion_id=0,
                        val_nb_steps=nb_adapt_epochs,
                        key=trainer_key)
else:
    restore_folder = run_folder
    trainer.restore_trainer(path=run_folder)
    print("\nNo training, loaded model and results from "+ run_folder +" folder ...\n")














#%%
## Test and visualise the results on a test dataloader
visualtester = SineVisualTester(trainer, key=test_key)

ind_crit, all_ind_crit = visualtester.evaluate(val_dataloader, 
                                    taylor_order=taylor_orders[1], 
                                    nb_steps=nb_adapt_epochs,
                                    print_error_every=print_error_every, 
                                    criterion_id=0,
                                    verbose=True,
                                    val_dataloader=val_dataloader,
                                    max_ret_env_states=max_ret_env_states,
                                    max_adapt_batches=max_adapt_batches)

visualtester.visualize_artefacts(save_path=run_folder+"artefacts.png")


#%%













## Adapt the model to the new dataset
if meta_test:
    adapt_dataloader = NumpyLoader(SinusoidDataset(num_envs=num_envs[1],
                                                num_shots=num_shots[0],
                                                adaptation=True), 
                                batch_size=num_envs[1], 
                                shuffle=False,
                                num_workers=num_workers,
                                drop_last=False)
    all_shots_loader = NumpyLoader(SinusoidDataset(num_envs=num_envs[1],
                                                num_shots=num_shots[1],
                                                adaptation=True), 
                                batch_size=num_envs[1], 
                                shuffle=False,
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


#%%

## Visualise the adaptation results

if meta_test:

    visualtester.visualize_artefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)








#%%

import scipy.stats as st
all_ind_crit_ = np.asarray(all_ind_crit[0])
all_losses_conf_ind = st.t.interval(0.95, len(all_ind_crit_)-1, loc=ind_crit, scale=st.sem(all_ind_crit_))
losses_conf_ind = np.mean(np.abs(np.array(all_losses_conf_ind) - ind_crit))
print(f"Losses with 95% confidence interval InD: {ind_crit} ± {losses_conf_ind}")

all_ood_crit_ = np.asarray(all_ood_crit[0])
all_losses_conf_ood = st.t.interval(0.95, len(all_ood_crit_)-1, loc=ood_crit, scale=st.sem(all_ood_crit_))
losses_conf_ood = np.mean(np.abs(np.array(all_losses_conf_ood) - ood_crit))
print(f"Losses with 95% confidence interval OoD: {ood_crit} ± {losses_conf_ood}")

if csv_export_path is not None:
    ## Export all hyperparamters and results to a csv: method,num_envs,taylor_order,context_size,gradient_updates,mse_ind,ci_ind,mse_ood,ci_ood
    with open(csv_export_path, 'a') as f:
        f.write(f"NCF,{num_envs[0]},{taylor_orders[0]},{context_size},{nb_inner_steps[0]},{nb_outer_steps},{ind_crit},{losses_conf_ind},{ood_crit},{losses_conf_ood}\n")



#%%
## After training, copy nohup.log to the runfolder
notebook_mode = True
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    if os.path.exists("nohup.log"):
        os.system(f"cp nohup.log {run_folder}")

