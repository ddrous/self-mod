#%%
%load_ext autoreload
%autoreload 2

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

from selfmod import *

# jax.config.update("jax_debug_nans", True)


#%%

seed = 2024

## Train and adapt hps
context_pool_size = 2
context_size = 8
init_lr = 5e-4
sched_factor = 0.5
envs_batch_size = 250

nb_outer_steps = 5
nb_inner_steps_max = 5
proximal_beta = 1e1
inner_tol_node = 2e-11
inner_tol_ctx = 1e-10

print_error_every = 100
nb_epochs_adapt = 1000

meta_train = True
run_folder = "./runs/240609-215946/"
# run_folder = None
save_trainer = True

meta_test = True
restore_adaptation = False and meta_test

## Dataset hps
data_folder="./data/pix0100_res32_ord0/"








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

    # Save the run and dataset scripts in that folder
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {run_folder}")
    os.system(f"cp dataset.py {run_folder}")

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

## Define dataloaders for training and validation
train_dataloader = RegMetaDataLoader(data_folder+"train_data.npz", envs_batch_size=envs_batch_size, points_batch_size=100, envs_shuffle=True, points_shuffle=True, key=data_key)

# val_dataloader = RegMetaDataLoader(data_folder+"test_data.npz", envs_batch_size=envs_batch_size, points_batch_size=32*32)
val_dataloader = RegDataLoader(data_folder+"test_data.npz", batch_size=32*32)

nb_envs = train_dataloader.nb_envs
nb_points_per_env = train_dataloader.nb_points_per_env
input_dim = train_dataloader.input_dim
output_dim = train_dataloader.output_dim

print("Training dataloader properties: \n", 
        "Number of environments:", nb_envs, "\n",
        "Number of points per environment:", nb_points_per_env, "\n",
        "Batch size (envs):", train_dataloader.envs_batch_size, "\n",
        "Batch size (datapoints):", train_dataloader.points_batch_size, "\n",
        "Input dimension:", input_dim, "\n",
        "Output dimension:", output_dim, "\n")










#%%

## Define model and loss function for the learner
class Swish(eqx.Module):
    beta: jnp.ndarray
    def __init__(self, key=None):
        self.beta = jax.random.uniform(key, shape=(1,), minval=0.01, maxval=1.0)
    def __call__(self, x):
        return x * jax.nn.sigmoid(self.beta * x)

class MultiMLP(eqx.Module):
    layers_data: list
    layers_context: list
    layers_shared: list
    activations: list

    def __init__(self, in_size, out_size, hidden_size, context_size, key=None):
        keys = jax.random.split(key, 10)
        self.activations = [Swish(key=key_i) for key_i in keys[:7]]

        self.layers_context = [eqx.nn.Linear(context_size, hidden_size, key=keys[0]), self.activations[0],
                               eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]), self.activations[1], 
                               eqx.nn.Linear(hidden_size, hidden_size, key=keys[2])]

        self.layers_data = [eqx.nn.Linear(in_size, hidden_size, key=keys[3]), self.activations[2], 
                            eqx.nn.Linear(hidden_size, hidden_size, key=keys[4]), self.activations[3], 
                            eqx.nn.Linear(hidden_size, hidden_size, key=keys[5])]

        self.layers_shared = [eqx.nn.Linear(2*hidden_size, hidden_size, key=keys[6]), self.activations[4], 
                              eqx.nn.Linear(hidden_size, hidden_size, key=keys[7]), self.activations[5], 
                              eqx.nn.Linear(hidden_size, hidden_size, key=keys[8]), self.activations[6], 
                              eqx.nn.Linear(hidden_size, out_size, key=keys[9])]

    def __call__(self, x, ctx):
        ctx = ctx
        for layer in self.layers_context:
            ctx = layer(ctx)

        y = x
        for layer in self.layers_data:
            y = layer(y)

        y = jnp.concatenate([y, ctx], axis=0)
        for layer in self.layers_shared:
            y = layer(y)

        return y


def loss_fn_ctx(model, batch, ctx, ctxs, key):
    X, Y = batch

    ind = jax.random.permutation(key, ctxs.shape[0])[:context_pool_size]
    ctx_pool = ctxs[ind, :]

    Y_hat, _ = jax.vmap(model, in_axes=(None, None, 0))(X, ctx, ctx_pool)
    Y_new = jnp.broadcast_to(Y, Y_hat.shape)

    term1 = jnp.mean((Y_hat-Y_new)**2)
    term2 = jnp.mean(jnp.abs(ctx))
    term3 = params_norm_squared(model)

    loss_val = term1 + 1e-3*term2 + 1e-3*term3

    return loss_val, (term3, term1, term2)


neuralnet = MultiMLP(in_size=input_dim, out_size=output_dim, hidden_size=64, context_size=context_size, key=model_key)

model = NeuralContextFlow(neuralnet=neuralnet, taylor_order=2)
contexts = ArrayContextParams(nb_envs=nb_envs, context_size=context_size)

learner = RegLearner(model=model, contexts=contexts, loss_fn_ctx=loss_fn_ctx, key=model_key)








model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)) if x is not None)
print("\n\nTotal number of parameters in the model:", model_params)
print("Total number of parameters in the contexts:", contexts.params.shape[0]*contexts.params.shape[1], "\n")








#%%

## Define optimiser and train the model
nb_total_epochs = nb_outer_steps * 1
bd_scales = {nb_total_epochs//3:sched_factor, 2*nb_total_epochs//3:sched_factor}

sched_model = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=bd_scales)
sched_ctx = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=bd_scales)

opt_model = optax.adam(sched_model)
opt_ctx = optax.adam(sched_ctx)

trainer = RegTrainer(learner, (opt_model, opt_ctx), key=trainer_key)

#%%
## Meta-training
if meta_train == True:
    trainer_save_path = run_folder if save_trainer == True else False
    trainer.train_proximal(super_dataloader=train_dataloader,
                           nb_outer_steps_max=nb_outer_steps, 
                           nb_inner_steps_max=nb_inner_steps_max, 
                           proximal_reg=proximal_beta, 
                           inner_tol_model=inner_tol_node, 
                           inner_tol_ctx=inner_tol_ctx,
                           print_error_every=print_error_every, 
                           save_path=trainer_save_path, 
                        #    val_dataloader=val_dataloader, 
                           key=trainer_key)
else:
    restore_folder = run_folder
    trainer.restore_trainer(path=run_folder)
    print("\nNo training, loaded model and results from "+ run_folder +" folder ...\n")




#%%
len(trainer.losses_model)
sbplot(np.vstack(trainer.losses_model), label="Model loss")











#%%
## Test and visualise the results on a test dataloader
visualtester = RegVisualTester(trainer, key=test_key)

ind_crit, _ = visualtester.test(val_dataloader)        ## TODO: Use val_dataloader
# print("In domain test error:", ind_crit)

visualtester.visualizeCelebA(val_dataloader,
                             few_shot_loader=train_dataloader,
                             resolution=32,
                             save_path=run_folder+"results_in_domain.png",
                            #  environment=0,
                             key=jax.random.PRNGKey(time.time_ns())
                             );

#%%










## Adapt the model to the new dataset
if meta_test:
    adapt_dataloader = RegMetaDataLoader(data_folder+"adapt_train.npz", envs_batch_size=envs_batch_size, point_batch_size=100, adaptation=True, key=data_key)

    sched_ctx_new = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=bd_scales)
    opt_adapt = optax.adabelief(sched_ctx_new)

    if restore_adaptation == False:
        trainer.adapt_bulk(adapt_dataloader,
                            nb_epochs=nb_epochs_adapt, 
                            taylor_order=0,
                            optimizer=opt_adapt, 
                            print_error_every=print_error_every, 
                            save_path=adapt_folder)
    else:
        trainer.restore_adapted_trainer(path=adapt_folder, data_loader=adapt_dataloader)
        print("Restored trained and adapated model from", adapt_folder)


#%%
if meta_test:
    # adapt_dataloader_test = RegMetaDataLoader(data_folder+"adapt_test.npz", envs_batch_size=envs_batch_size, point_batch_size=32*32, adaptation=True, key=data_key)
    adapt_dataloader_test = RegDataLoader(data_folder+"adapt_test.npz", batch_size=32*32, adaptation=True, key=data_key)

    ood_crit, _ = visualtester.test(adapt_dataloader_test)
    # print("Out of domain test error:", ood_crit)

    visualtester.visualizeCelebA(adapt_dataloader_test,
                                few_shot_loader=adapt_dataloader,
                                resolution=32, 
                                save_path=adapt_folder+"results_ood.png",
                                # environment=0,
                                key=jax.random.PRNGKey(time.time_ns())
                                );















#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    if os.path.exists("nohup.log"):
        os.system(f"cp nohup.log {run_folder}")


#%%

