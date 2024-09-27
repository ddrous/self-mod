k_shots=100
taylor_orders=(2,0)
#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

from selfmod import *
# jax.config.update("jax_debug_nans", True)


#%%

## For reproducibility
seed = 2024

## Dataloader hps
# k_shots = 100
resolution = (32, 32)
data_folder="../../data/"
shuffle = False
num_workers = 24
input_dim = 2
output_dim = 3

## Train and adapt hps
context_size = 128
loss_contributors = 32
# taylor_orders = (0, 0)
context_pool_size = 1 if taylor_orders[0] == 0 else 3
init_lrs = (1e-3, 1e-3)
sched_factor = 1.0
envs_batch_size = 162770
envs_batch_size_val = 16*16*2
max_train_batches = 1
max_val_batches = 1

pool_filling_strategy = "NF"
loss_filling_strategy = "NF"

nb_outer_steps = int(4500 * 100 / k_shots)
# nb_outer_steps = 4
nb_inner_steps = (10, 10)
uq_train_contexts = 466

print_error_every = 160
validate_every = 30*5*1000000

nb_adapt_steps = 500

meta_train = True
# run_folder = "./runs/240831-091752-NEWTEST/"
run_folder = None

save_prefix = ""
meta_test = True


#%%
mother_key = jax.random.PRNGKey(seed)




#%%

if meta_train == True:
    # check that 'tmp' folder exists. If not, create it
    if not os.path.exists('./runs'):
        os.mkdir('./runs')

    # Run folder to store the result of this run
    if run_folder == None:
        run_folder = './runs/'+time.strftime("%y%m%d-%H%M%S")+'/'
        os.mkdir(run_folder)
        print("New run folder created successfuly:", run_folder)
    else:
        print("Using pre-existing run folder:", run_folder)

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

##### Numpy Loader
train_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="train",
                                            num_shots=k_shots, 
                                            order_pixels=False, 
                                            resolution=resolution,
                                            # seed=seed
                                            ), 
                              batch_size=envs_batch_size, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)
all_shots_train_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="train",
                                            num_shots=np.prod(resolution), 
                                            order_pixels=False, 
                                            resolution=resolution,
                                            # seed=seed,
                                            ), 
                              batch_size=envs_batch_size_val, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)


#%%


class MultiMLP(eqx.Module):
    layers_shared: list
    activations: list

    def __init__(self, in_size, out_size, hidden_size, context_size, key=None):
        keys = jax.random.split(key, 10)
        self.activations = [jax.nn.relu for key_i in keys[:5]]

        self.layers_shared = [eqx.nn.Linear(in_size+context_size, hidden_size, key=keys[5]), self.activations[0], 
                              eqx.nn.Linear(hidden_size, hidden_size, key=keys[6]), self.activations[1], 
                              eqx.nn.Linear(hidden_size, hidden_size, key=keys[7]), self.activations[2], 
                              eqx.nn.Linear(hidden_size, hidden_size, key=keys[8]), self.activations[3], 
                              eqx.nn.Linear(hidden_size, out_size, key=keys[9])]

    def __call__(self, x, ctx):
        y = jnp.concatenate([x, ctx], axis=0)
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



contexts = ArrayContextParams(nb_envs=envs_batch_size,
                            context_size=context_size)
neuralnet = MultiMLP(in_size=input_dim,
                     out_size=output_dim, 
                     hidden_size=128,
                     context_size=context_size,
                     key=model_key)

model = NeuralContextFlow(neuralnet=neuralnet, 
                          taylor_order=taylor_orders[0])

learner = Learner(model=model, 
                context_size=context_size, 
                context_pool_size=context_pool_size,
                pool_filling=pool_filling_strategy,
                contexts=contexts,
                reuse_contexts=False,
                env_loss_fn=env_loss_fn, 
                loss_contributors=loss_contributors,
                loss_filling=loss_filling_strategy,
                key=model_key)



model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)) if x is not None)
print("\n\nTotal number of parameters in the model:", model_params)
print("Total number of parameters in one context:", contexts.eff_context_size)




#%%

## Define optimiser and train the model
init_lr_model, init_lr_ctx = init_lrs

total_strain_steps = nb_outer_steps*nb_inner_steps[0]
bd_scales = {total_strain_steps//3:sched_factor, 2*total_strain_steps//3:sched_factor}        ## TODO fix this ! times nb_batches ?
sched_model = optax.piecewise_constant_schedule(init_value=init_lr_model, boundaries_and_scales=bd_scales)
opt_model = optax.adam(sched_model)

opt_ctx = optax.adam(init_lr_ctx)

trainer = NCFTrainer(learner, (opt_model, opt_ctx), key=trainer_key)

#%%

if meta_train == True:
    trainer.meta_train(dataloader=train_dataloader,
                        nb_epochs=1,
                        nb_outer_steps=nb_outer_steps,
                        nb_inner_steps=nb_inner_steps, 
                        inner_tols=(1e-16, 1e-16), 
                        max_train_batches=max_train_batches,
                        print_error_every=(1, print_error_every), 
                        save_path=run_folder, 
                        # val_dataloader=all_shots_train_dataloader, 
                        max_val_batches=max_val_batches,
                        val_criterion_id=0,
                        validate_every=validate_every,
                        val_nb_steps=nb_adapt_steps,
                        key=trainer_key)

    # trainer.meta_train_noalm(dataloader=train_dataloader,
    #                     nb_epochs=1,
    #                     nb_outer_steps=nb_outer_steps,
    #                     max_train_batches=max_train_batches,
    #                     print_error_every=(1, print_error_every), 
    #                     save_path=trainer_save_path, 
    #                     # val_dataloader=all_shots_train_dataloader, 
    #                     max_val_batches=max_val_batches,
    #                     val_criterion_id=0,
    #                     validate_every=validate_every,
    #                     val_nb_steps=nb_adapt_steps,
    #                     key=trainer_key)
else:
    restore_folder = run_folder
    trainer.restore_trainer(path=run_folder)
    print("\nNo training, loaded model and results from "+ run_folder +" folder ...\n")






#%%
## Test and visualise the results on a test dataloader
visualtester = CelebAVisualTester(trainer, key=test_key)

## Evaluation on the full train set with all shots 
if train_dataloader.batch_size == all_shots_train_dataloader.batch_size:
    ind_crit, _ = visualtester.evaluate(train_dataloader, 
                                        nb_steps=nb_adapt_steps,
                                        taylor_order=taylor_orders[1],
                                        print_error_every=print_error_every,
                                        max_adapt_batches=max_val_batches,
                                        val_dataloader=all_shots_train_dataloader,
                                        verbose=True)
else:
    print("Train dataloaders have different batch sizes. Skipping evaluation ...")

visualtester.visualize_artefacts(save_path=run_folder+"artefacts.png")

#%%

# visualtester.visualize_few_shots_multi(few_shots_loader=train_dataloader,
#                                 all_shots_loader=all_shots_train_dataloader,
#                                 nb_steps=nb_adapt_steps,
#                                 save_path=run_folder+"few_shots_ind.png",
#                                 key=jax.random.PRNGKey(time.time_ns())
#                              );
visualtester.visualize_few_shots_multi_uq(few_shots_loader=train_dataloader,
                                all_shots_loader=all_shots_train_dataloader,
                                nb_steps=nb_adapt_steps,
                                save_path=run_folder+save_prefix+"few_shots_ind_uq.svg",
                                taylor_order=taylor_orders[0],
                                num_envs=16,
                                uq_train_contexts=uq_train_contexts,
                                interp_method="cubic", ##  {'linear', 'nearest', 'cubic'}
                                key=test_key
                             );


#%%

## Evaluation on the actual validation set from CelebA with just a few shots 
val_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="val",
                                            num_shots=k_shots, 
                                            order_pixels=False, 
                                            resolution=resolution,
                                            # seed=seed
                                            ), 
                              batch_size=envs_batch_size_val, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)
all_shots_val_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="val",
                                            num_shots=np.prod(resolution), 
                                            order_pixels=False, 
                                            resolution=resolution,
                                            # seed=seed
                                            ), 
                              batch_size=envs_batch_size_val, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

if val_dataloader.batch_size == all_shots_val_dataloader.batch_size:
    ind_crit, _ = visualtester.evaluate(val_dataloader, 
                                        nb_steps=nb_adapt_steps,
                                        taylor_order=taylor_orders[1],
                                        print_error_every=print_error_every,
                                        max_adapt_batches=max_val_batches,
                                        val_dataloader=all_shots_val_dataloader,
                                        verbose=True)
else:
    print("Validation dataloaders have different batch sizes. Skipping evaluation ...")


#%%


## Adapt the model to the new dataset
if meta_test:
    adapt_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                                data_split="test",
                                                num_shots=k_shots, 
                                                order_pixels=False, 
                                                resolution=resolution,
                                                # seed=seed
                                                ), 
                                batch_size=envs_batch_size_val, 
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)
    all_shots_dataloader_test = NumpyLoader(CelebADataset(data_folder, 
                                                data_split="test",
                                                num_shots=np.prod(resolution),
                                                order_pixels=False, 
                                                resolution=resolution,
                                                # seed=seed
                                                ), 
                                batch_size=envs_batch_size_val, 
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    if adapt_dataloader.batch_size == all_shots_dataloader_test.batch_size:
        ood_crit, _ = visualtester.evaluate(adapt_dataloader, 
                                            nb_steps=nb_adapt_steps,
                                            taylor_order=taylor_orders[1],
                                            max_adapt_batches=max_train_batches,
                                            val_dataloader=all_shots_dataloader_test,
                                            print_error_every=print_error_every,
                                            verbose=True)
    else:
        print("Adaptation dataloaders have different batch sizes. Skipping evaluation ...")

#%%

## Visualise the adaptation results

if meta_test:
    visualtester.visualize_artefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)

    # visualtester.visualize_few_shots_multi(few_shots_loader=adapt_dataloader,
    #                                 all_shots_loader=all_shots_dataloader_test,
    #                                 nb_steps=nb_adapt_steps,
    #                                 save_path=adapt_folder+"few_shots_ood.png",
    #                                 key=jax.random.PRNGKey(time.time_ns())
    #                             );
    visualtester.visualize_few_shots_multi_uq(few_shots_loader=adapt_dataloader,
                                    all_shots_loader=all_shots_dataloader_test,
                                    nb_steps=nb_adapt_steps,
                                    save_path=adapt_folder+save_prefix+"few_shots_ood_uq.svg",
                                    taylor_order=taylor_orders[0],
                                    num_envs=7,
                                    uq_train_contexts=uq_train_contexts,
                                    interp_method="cubic",
                                    key=test_key
                                );

#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    if os.path.exists("nohup.log"):
        os.system(f"cp nohup.log {run_folder}")


#%%

