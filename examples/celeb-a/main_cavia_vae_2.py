#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

from selfmod import *
# from selfmod.dataloader import CelebADataLoader

# jax.config.update("jax_debug_nans", True)

# ## To avoid perfetto profiler bug
# import socket
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


#%%

## For reproducibility
seed = 2024

## Dataloader hps
k_shots = 100
resolution = (32, 32)
data_folder="./data/" 

## Train and adapt hps
context_pool_size = 8
context_size = 128
taylor_orders = (2, 0)      ## Expansion orders for meta-training and meta-testing. TODO The same vector field cannot readily be used if increased !
init_lrs = (1e-4, 1e-1)
sched_factor = 1.
envs_batch_size = 64*1
max_train_batches = -1      ## TODO: should be -1

nb_train_epochs = 20
nb_inner_steps = 5

print_error_every = 1000

nb_adapt_epochs = 1
nb_inner_steps_eval = 5       ## To use during evaluation and visulisation

meta_train = True
# run_folder = "./runs/240609-215946-VAE-Test/"
run_folder = None
save_trainer = True

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
# train_dataloader = CelebADataLoader(data_folder, 
#                                     envs_batch_size=envs_batch_size, 
#                                     shots_batch_size=k_shots, 
#                                     data_split="train",
#                                     envs_shuffle=True, 
#                                     shots_shuffle=True, 
#                                     order_pixels=False, 
#                                     key=data_key)
# val_dataloader = CelebADataLoader(data_folder, 
#                                   envs_batch_size=envs_batch_size, 
#                                   shots_batch_size=k_shots, 
#                                   data_split="val",
#                                   envs_shuffle=True, 
#                                   shots_shuffle=True, 
#                                   order_pixels=False, 
#                                   key=data_key)



# ##### Pytorch dataloading #####
train_dataset = CelebADataset(data_folder, 
                            data_split="train",
                            num_shots=k_shots, 
                            order_pixels=False, 
                            seed=seed)
# train_dataloader = DataLoader(train_dataset, 
#                               batch_size=envs_batch_size, 
#                               shuffle=True,
#                             #   backend="jax",
#                               collate_fn=collate_to_jax,
#                             #   num_workers=24,
#                               drop_last=False)

##### Numpy Loader
train_dataloader = NumpyLoader(train_dataset, 
                              batch_size=envs_batch_size, 
                              shuffle=True,
                              num_workers=24,
                              drop_last=False)


val_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="val",
                                            num_shots=k_shots, 
                                            order_pixels=False, 
                                            seed=seed), 
                              batch_size=envs_batch_size, 
                              shuffle=True,
                              num_workers=24,
                              drop_last=False)


## Print all attributes of the dataloader
# print(train_dataloader.__dict__)

## Check data properties
# print(next(train_dataloader))

# for x, y in train_dataloader:
#     print(x.shape, y.shape)
#     break










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
    # layers_context: list
    layers_shared: list
    activations: list

    decoder: eqx.Module         ## The Decoder is finetuned as we GO !

    def __init__(self, in_size, out_size, hidden_size, context_size, key=None):
        keys = jax.random.split(key, 10)
        self.activations = [Swish(key=key_i) for key_i in keys[:7]]

        # self.layers_context = [eqx.nn.Linear(context_size, hidden_size, key=keys[0]), self.activations[0],
        #                        eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]), self.activations[1], 
        #                        eqx.nn.Linear(hidden_size, hidden_size, key=keys[2])]

        self.layers_data = [eqx.nn.Linear(in_size, hidden_size, key=keys[3]), self.activations[2], 
                            eqx.nn.Linear(hidden_size, hidden_size, key=keys[4]), self.activations[3], 
                            eqx.nn.Linear(hidden_size, hidden_size, key=keys[3]), self.activations[0], 
                            eqx.nn.Linear(hidden_size, out_size, key=keys[5])]

        # self.layers_shared = [eqx.nn.Linear(2*hidden_size, hidden_size, key=keys[6]), self.activations[4], 
        self.layers_shared = [eqx.nn.Linear(out_size+out_size, hidden_size, key=keys[6]), self.activations[4], 
                            #   eqx.nn.Linear(hidden_size, hidden_size, key=keys[7]), self.activations[5], 
                            #   eqx.nn.Linear(hidden_size, hidden_size, key=keys[8]), self.activations[6], 
                              eqx.nn.Linear(hidden_size, out_size, key=keys[9])]


        decoder = Decoder(img_size=[32, 32, 3], kernel_size=[3, 3], latent_dim=context_size, key=keys[7])
        # decoder = eqx.tree_deserialise_leaves("runs/240101-193230-VAE/decoder.eqx", decoder)      ## Pretrained VAE decoder
        self.decoder = decoder

    def __call__(self, x, noise):
        # ctx = ctx
        # for layer in self.layers_context:
        #     ctx = layer(ctx)

        # noise = jax.random.normal(model_key, shape=(ctx_fun.latent_dim,))       ## Here !
        img = self.decoder(noise)

        i = (x[0]*resolution[0]).astype(int)
        j = (x[1]*resolution[1]).astype(int)
        ctx = img[:, i, j]

        y = x
        for layer in self.layers_data:
            y = layer(y)

        y = jnp.concatenate([y, ctx], axis=0)       ## TODO a linear combination instead ?
        # y = jnp.concatenate([x, ctx], axis=0)
        for layer in self.layers_shared:
            y = layer(y)

        return y





def env_loss_fn(model, ctx, y_hat, y):
    """
    Loss function for one environment. Leading dimension of y_hat corresponds to the pool size !
    """

    term1 = jnp.mean((y_hat-y)**2)
    term2 = jnp.mean(jnp.abs(ctx))                  ## TODO make sure the ctx is from the normal distribution
    term3 = params_norm_squared(model)

    # loss_val = term1 + 1e-3*term2 + 1e-3*term3
    loss_val = term1

    return loss_val, (term1, term2, term3)


input_dim = 2
output_dim = 3

# ex_context = FuncContextParams(nb_envs=1, key=model_key)


neuralnet = MultiMLP(in_size=input_dim, 
                     out_size=output_dim, 
                     hidden_size=128, 
                     context_size=context_size, 
                     key=model_key)

model = NeuralContextFlow(neuralnet=neuralnet, 
                          taylor_order=taylor_orders[0])      ## TODO : taylor order=2

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
                            key=trainer_key)

else:
    restore_folder = run_folder
    trainer.restore_trainer(path=run_folder)
    print("\nNo training, loaded model and results from "+ run_folder +" folder ...\n")














#%%
## Test and visualise the results on a test dataloader
visualtester = CelebAVisualTester(trainer, key=test_key)

ind_crit, _ = visualtester.evaluate(val_dataloader, 
                                    nb_inner_steps=nb_inner_steps,
                                    max_eval_batches=10)
# print("In domain test error:", ind_crit)

# all_shots_dataloader = CelebADataLoader(data_folder, 
#                                         envs_batch_size=envs_batch_size, 
#                                         shots_batch_size=np.prod(resolution), 
#                                         data_split="train",
#                                         envs_shuffle=True, 
#                                         shots_shuffle=True, 
#                                         order_pixels=False, 
#                                         key=data_key)
all_shots_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="train",
                                            num_shots=np.prod(resolution), 
                                            order_pixels=False, 
                                            seed=seed), 
                              batch_size=envs_batch_size, 
                              shuffle=True,
                              num_workers=24,
                              drop_last=False)


visualtester.visualizeArtefacts(save_path=run_folder+"artefacts.png")

# visualtester.visualizeFewShots(few_shots_loader=train_dataloader,
#                                 all_shots_loader=all_shots_dataloader,
#                                 nb_inner_steps=nb_inner_steps_eval,
#                                 save_path=run_folder+"few_shots_ind.png",
#                                 key=jax.random.PRNGKey(time.time_ns())
#                              );

visualtester.visualizeFewShotsMulti(few_shots_loader=train_dataloader,
                                    all_shots_loader=all_shots_dataloader,
                                    nb_inner_steps=nb_inner_steps_eval,
                                    num_envs=6,
                                    save_path=run_folder+"few_shots_multi_ind.png",
                                    key=jax.random.PRNGKey(time.time_ns())
                             );

#%%













## Adapt the model to the new dataset
if meta_test:
    # adapt_dataloader = CelebADataLoader(data_folder, 
    #                                     envs_batch_size=envs_batch_size, 
    #                                     shots_batch_size=k_shots, 
    #                                     data_split="test",
    #                                     key=data_key)
    adapt_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                                data_split="test",
                                                num_shots=k_shots, 
                                                order_pixels=False, 
                                                seed=seed), 
                                batch_size=envs_batch_size, 
                                shuffle=True,
                                num_workers=24,
                                drop_last=False)



    # visualtester.visualizeArtefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)



    opt_adapt = optax.sgd(init_lr_ctx)

    _, contexts, aux_data = trainer.meta_test(adapt_dataloader,
                                            nb_inner_steps=nb_inner_steps,
                                            taylor_order=taylor_orders[1],
                                            optimizer=opt_adapt,
                                            max_adapt_batches=max_train_batches,     ## JUST to set up adaptation for future tasks
                                            print_error_every=print_error_every, 
                                            save_path=adapt_folder)

    # visualtester.visualizeArtefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)

    ood_crit, _ = visualtester.evaluate(adapt_dataloader, 
                                        taylor_order=taylor_orders[1], 
                                        nb_inner_steps=nb_inner_steps_eval)


#%%

## Visualise the adaptation results

if meta_test:
    # all_shots_loader = CelebADataLoader(data_folder, 
    #                                     envs_batch_size=envs_batch_size, 
    #                                     shots_batch_size=np.prod(resolution), 
    #                                     data_split="test",
    #                                     key=data_key)
    all_shots_loader = NumpyLoader(CelebADataset(data_folder, 
                                                data_split="test",
                                                num_shots=np.prod(resolution), 
                                                order_pixels=False, 
                                                seed=seed), 
                                batch_size=envs_batch_size, 
                                shuffle=True,
                                num_workers=24,
                                drop_last=False)

    visualtester.visualizeArtefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)

    # visualtester.visualizeFewShots(few_shots_loader=adapt_dataloader,
    #                             all_shots_loader=all_shots_loader,
    #                             nb_inner_steps=nb_inner_steps_eval,
    #                             save_path=adapt_folder+"few_shots_ood.png",
    #                             key=jax.random.PRNGKey(time.time_ns())
    #                             );

    visualtester.visualizeFewShotsMulti(few_shots_loader=adapt_dataloader,
                                all_shots_loader=all_shots_loader,
                                nb_inner_steps=nb_inner_steps_eval,
                                num_envs=6,
                                save_path=adapt_folder+"few_shots_multi_ood.png",
                                key=jax.random.PRNGKey(time.time_ns())
                                );


#%%
# learner.contexts_adapt

#%%

# ## Let's investigate the model

# model = trainer.learner.model

# print(model)

# losses, contexts, aux_data

# X, Y, Y_hat = aux_data

# print(Y_hat[5])




# fig, ax = plt.subplot_mosaic('A', figsize=(4*1, 3.7*1))
# img_size = (32, 32, 3)

# def make_image(xy_coords, rgb_pixels):
#     img = np.zeros(img_size)
#     x_coords = (xy_coords[:, 0] * img_size[0]).astype(int)
#     y_coords = (xy_coords[:, 1] * img_size[1]).astype(int)
#     img[x_coords, y_coords, :] = np.clip(rgb_pixels, 0., 1.)
#     return img

# X_plot, Y_plot, Y_hat_plot = X[0], Y[0], Y_hat[0]
# true_img = make_image(X_plot, Y_hat_plot)
# ax['A'].imshow(true_img)
# ax['A'].set_title('Test', fontsize=14)








#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    if os.path.exists("nohup.log"):
        os.system(f"cp nohup.log {run_folder}")


#%%

