#%%

notebook_mode = True
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    notebook_mode = False

if notebook_mode:
    %load_ext autoreload
    %autoreload 2

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

from selfmod import *
# jax.config.update("jax_debug_nans", True)



#%%

## For reproducibility
seed = 2026

## Dataloader hps
num_envs = (240, 240)
num_shots = (10, 1)
num_workers = 0

## Learner/model hps
context_pool_size = 1
context_size = 4
taylor_orders = (0, 0)

## Train and adapt hps
init_lrs = (1e-3, 1e-1)
sched_factor = 1.
max_train_batches = -1
max_eval_batches = -1

nb_train_epochs = 100
nb_inner_steps = 5

print_error_every = 10

nb_adapt_epochs = 1
nb_inner_steps_eval = 5       ## To use during evaluation and visulisation

meta_train = True
run_folder = "./runs/240715-025946-Test/"
# run_folder = None
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

train_dataloader = NumpyLoader(SinusoidDataset(num_envs=num_envs[0],
                                            num_shots=num_shots[0]), 
                              batch_size=num_envs[0], 
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=False)


val_dataloader = NumpyLoader(SinusoidDataset(num_envs=num_envs[0],
                                            num_shots=num_shots[1]), 
                              batch_size=num_envs[0], 
                              shuffle=True,
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


neuralnet = MultiMLP(in_size=1,
                     out_size=1,
                     context_size=context_size,
                     hidden_size=4, 
                     key=model_key)

model = NeuralContextFlow(neuralnet=neuralnet, 
                            taylor_order=taylor_orders[0],
                            taylor_scale=100,
                            taylor_weight_init=-10.)  ## equal chances for taylor or not

learner = Learner(model=model,
                context_size=context_size, 
                context_pool_size=context_pool_size,
                env_loss_fn=env_loss_fn, 
                key=model_key)




model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)) if x is not None)
print("\n\nTotal number of parameters in the model:", model_params)





#%%

## Define optimiser and train the model
init_lr_model, init_lr_ctx = init_lrs

bd_scales = {nb_train_epochs//3:sched_factor, 2*nb_train_epochs//3:sched_factor}
sched_model = optax.piecewise_constant_schedule(init_value=init_lr_model, boundaries_and_scales=bd_scales)
opt_model = optax.adabelief(sched_model)

opt_ctx = optax.sgd(init_lr_ctx)

trainer = Trainer(learner, (opt_model, opt_ctx), key=trainer_key)

#%%

# with jax.profiler.trace("data/jax-trace", create_perfetto_link=True, create_perfetto_trace=True):

## Meta-training
if meta_train == True:
    trainer_save_path = run_folder if save_trainer == True else False
    trainer.meta_train_cavia(dataloader=train_dataloader,
                            nb_epochs=nb_train_epochs,
                            nb_inner_steps=nb_inner_steps, 
                            max_train_batches=max_train_batches,
                            print_error_every=print_error_every, 
                            save_path=trainer_save_path, 
                            val_dataloader=val_dataloader, 
                            val_criterion_id=0,
                            validate_every=nb_train_epochs//10,
                            backup_contexts=True,
                            key=trainer_key)
    # trainer.meta_train_proximal(dataloader=train_dataloader,
    #                             nb_epochs=nb_train_epochs,
    #                             nb_outer_steps=1,
    #                             nb_inner_steps=(1,5), 
    #                             inner_tols=(1e-12, 1e-12), 
    #                             proximal_betas=(10., 10.), 
    #                             max_train_batches=max_train_batches,
    #                             print_error_every=print_error_every, 
    #                             save_path=trainer_save_path, 
    #                             val_dataloader=val_dataloader, 
    #                             val_criterion_id=0,
    #                             key=trainer_key)
else:
    restore_folder = run_folder
    trainer.restore_trainer(path=run_folder)
    print("\nNo training, loaded model and results from "+ run_folder +" folder ...\n")














#%%
## Test and visualise the results on a test dataloader
visualtester = CelebAVisualTester(trainer, key=test_key)

ind_crit, _ = visualtester.evaluate(val_dataloader, 
                                    nb_inner_steps=nb_inner_steps,
                                    max_eval_batches=max_eval_batches)

# all_shots_dataloader = NumpyLoader(CelebADataset(data_folder, 
#                                             data_split="train",
#                                             num_shots=np.prod(resolution), 
#                                             order_pixels=False, 
#                                             seed=seed), 
#                               batch_size=envs_batch_size, 
#                               shuffle=True,
#                               num_workers=24,
#                               drop_last=False)


visualtester.visualizeArtefacts(save_path=run_folder+"artefacts.png")

# visualtester.visualizeFewShotsMulti(few_shots_loader=train_dataloader,
#                                     all_shots_loader=all_shots_dataloader,
#                                     nb_inner_steps=nb_inner_steps_eval,
#                                     num_envs=6,
#                                     save_path=run_folder+"few_shots_multi_ind.png",
#                                     key=jax.random.PRNGKey(time.time_ns())
#                              );

#%%













## Adapt the model to the new dataset
if meta_test:
    adapt_dataloader = NumpyLoader(SinusoidDataset(num_envs=num_envs[1],
                                                num_shots=num_shots[0]), 
                                batch_size=num_envs[1], 
                                shuffle=True,
                                num_workers=num_workers,
                                drop_last=False)

    opt_adapt = optax.sgd(init_lr_ctx)

    _, contexts, aux_data = trainer.meta_test(adapt_dataloader,
                                            nb_inner_steps=nb_inner_steps,
                                            taylor_order=taylor_orders[1],
                                            optimizer=opt_adapt,
                                            max_adapt_batches=max_train_batches,     ## JUST to set up adaptation for future tasks
                                            print_error_every=print_error_every, 
                                            save_path=adapt_folder)

    ood_crit, _ = visualtester.evaluate(adapt_dataloader, 
                                        taylor_order=taylor_orders[1], 
                                        nb_inner_steps=nb_inner_steps_eval,
                                        max_eval_batches=max_eval_batches)


#%%

## Visualise the adaptation results

if meta_test:
    all_shots_loader = NumpyLoader(SinusoidDataset(num_envs=num_envs[1],
                                                num_shots=num_shots[1]), 
                                batch_size=num_envs[1], 
                                shuffle=True,
                                num_workers=num_workers,
                                drop_last=False)

    visualtester.visualizeArtefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)

    # visualtester.visualizeFewShotsMulti(few_shots_loader=adapt_dataloader,
    #                             all_shots_loader=all_shots_loader,
    #                             nb_inner_steps=nb_inner_steps_eval,
    #                             num_envs=6,
    #                             save_path=adapt_folder+"few_shots_multi_ood.png",
    #                             key=jax.random.PRNGKey(time.time_ns())
    #                             );











#%%
## After training, copy nohup.log to the runfolder
if not notebook_mode and os.path.exists("nohup.log"):
    os.system(f"cp nohup.log {run_folder}")

