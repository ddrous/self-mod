#%%
# %load_ext autoreload
# %autoreload 2

## Do not preallocate GPU memory
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = '\"platform\"'

import jax.flatten_util
from selfmod import *
# jax.config.update("jax_debug_nans", True)

## Execute jax on CPU
# jax.config.update("jax_platform_name", "cpu")





#%%

seed = 2026
# seed = int(np.random.randint(0, 10000))

context_pool_size = 4               ## Number of neighboring contexts j to use for a flow in env e
context_size = 64
# nb_epochs = 2400
nb_epochs_adapt = 5000
init_lr = 5e-4
sched_factor = 0.5            ## Multiply the lr by this factor at each third of the training

nb_outer_steps = 5000
nb_inner_steps_max = 10
proximal_beta = 1e1
inner_tol_node = 2e-11
inner_tol_ctx = 1e-10

print_error_every = 100


train = True
# run_folder = "./runs/240531-195125/"
run_folder = None
generate_data = False if run_folder else True

save_trainer = True

finetune = False

adapt_test = True
adapt_restore = False

integrator = diffrax.Dopri5
# integrator = RK4
ivp_args = {"dt_init":1e-4, "rtol":1e-3, "atol":1e-6, "max_steps":40000, "subdivisions":2}

#%%


if train == True and generate_data == True:

    # check that 'tmp' folder exists. If not, create it
    if not os.path.exists('./runs'):
        os.mkdir('./runs')

    # Make a new folder inside 'tmp' whose name is the current time
    run_folder = './runs/'+time.strftime("%y%m%d-%H%M%S")+'/'
    # run_folder = "./runs/23012024-163033/"
    os.mkdir(run_folder)
    print("Run folder created successfuly:", run_folder)

    # Save the run and dataset scripts in that folder
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {run_folder}")
    os.system(f"cp dataset.py {run_folder}")

    # Save the nodax module files as well
    os.system(f"cp -r ../../idncflow {run_folder}")
    print("Completed copied scripts ")


else:
    # run_folder = "./runs/22022024-112457/"  ## Needed for loading the model and finetuning TODO: opti
    print("No training. Loading data and results from:", run_folder)

## Create a folder for the adaptation results
adapt_folder = run_folder+"adapt/"
if not os.path.exists(adapt_folder):
    os.mkdir(adapt_folder)

# %%

if train == True and generate_data == True:
    # Run the dataset script to generate the data
    os.system(f'python dataset.py --split=train --savepath="{run_folder}" --seed="{seed}"')
    os.system(f'python dataset.py --split=test --savepath="{run_folder}" --seed="{seed*2}"')



#%%

## Define dataloader for training and validation
train_dataloader = DataLoader(run_folder+"train_data.npz", batch_size=-1, shuffle=True, key=seed)

nb_envs = train_dataloader.nb_envs
nb_trajs_per_env = train_dataloader.nb_trajs_per_env
nb_steps_per_traj = train_dataloader.nb_steps_per_traj
data_size = train_dataloader.data_size

val_dataloader = DataLoader(run_folder+"test_data.npz", shuffle=False)

#%%

## Define model and loss function for the learner

class Swish(eqx.Module):
    beta: jnp.ndarray
    def __init__(self, key=None):
        self.beta = jax.random.uniform(key, shape=(1,), minval=0.01, maxval=1.0)
    def __call__(self, x):
        return x * jax.nn.sigmoid(self.beta * x)

class Augmentation(eqx.Module):
    layers_data: list
    # layers_context: list
    layers_shared: list
    activations: list
    ctx_utils:any

    def __init__(self, data_size, int_size, context_size, ctx_utils, key=None):
        self.ctx_utils = ctx_utils

        keys = generate_new_keys(key, num=12)
        self.activations = [Swish(key=key_i) for key_i in keys[:7]]

        # self.layers_context = [eqx.nn.Linear(context_size, context_size//4, key=keys[0]), self.activations[0],
        #                        eqx.nn.Linear(context_size//4, int_size, key=keys[1]), self.activations[1], eqx.nn.Linear(int_size, int_size, key=keys[2])]

        self.layers_data = [eqx.nn.Linear(1+data_size, int_size, key=keys[3]), self.activations[2], 
                            eqx.nn.Linear(int_size, int_size, key=keys[4]), self.activations[3], 
                            eqx.nn.Linear(int_size, int_size, key=keys[5])]

        self.layers_shared = [eqx.nn.Linear(2*int_size, int_size, key=keys[6]), self.activations[4], 
                              eqx.nn.Linear(int_size, int_size, key=keys[7]), self.activations[5], 
                              eqx.nn.Linear(int_size, int_size, key=keys[8]), self.activations[6], 
                              eqx.nn.Linear(int_size, data_size, key=keys[9])]

    def __call__(self, t, y, ctx_arr):

        ctx_shapes, ctx_treedef, ctx_static = self.ctx_utils
        ctx_params = unflatten_pytree(ctx_arr, ctx_shapes, ctx_treedef)
        ctx_fun = eqx.combine(ctx_params, ctx_static)

        t_arr = jnp.array([t])

        ctx = ctx_fun(t_arr)
        # for layer in self.layers_context:
        #     ctx = layer(ctx)

        y = jnp.concatenate([t_arr, y], axis=0)
        for layer in self.layers_data:
            y = layer(y)

        y = jnp.concatenate([y, ctx], axis=0)
        for layer in self.layers_shared:
            y = layer(y)

        return y



# class ContextFlowVectorField(eqx.Module):
#     physics: eqx.Module
#     augmentation: eqx.Module

#     def __init__(self, augmentation, physics=None):
#         self.augmentation = augmentation
#         self.physics = physics

#     def __call__(self, t, x, ctxs):
#         if self.physics is None:
#             vf = lambda xi_: self.augmentation(t, x, xi_)
#         else:
#             vf = lambda xi_: self.physics(t, x, xi_) + self.augmentation(t, x, xi_)

#         gradvf = lambda xi_, xi: eqx.filter_jvp(vf, (xi_,), (xi-xi_,))[1]

#         ctx, ctx_ = ctxs
#         return vf(ctx_) + gradvf(ctx_, ctx)



class ContextFlowVectorField(eqx.Module):
    physics: eqx.Module
    augmentation: eqx.Module

    def __init__(self, augmentation, physics=None):
        self.augmentation = augmentation
        self.physics = physics

    def __call__(self, t, x, ctxs):
        ctx, ctx_ = ctxs

        if self.physics is None:
            vf = lambda xi: self.augmentation(t, x, xi)
        else:
            vf = lambda xi: self.physics(t, x, xi) + self.augmentation(t, x, xi)

        gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
        scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]

        return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term



# ## Define a custom loss function here
# def loss_fn_ctx(model, trajs, t_eval, ctx, all_ctx_s, key):

#     # ind = jax.random.randint(key, shape=(context_pool_size,), minval=0, maxval=all_ctx_s.shape[0])
#     ind = jax.random.permutation(key, all_ctx_s.shape[0])[:context_pool_size]
#     ctx_s = all_ctx_s[ind, :]

#     trajs_hat, nb_steps = jax.vmap(model, in_axes=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_s)
#     new_trajs = jnp.broadcast_to(trajs, trajs_hat.shape)

#     term1 = jnp.mean((new_trajs-trajs_hat)**2)  ## reconstruction
#     term2 = jnp.mean(ctx**2)             ## TODO regularisation... Make this L1
#     # term2 = jnp.mean(jnp.abs(ctx))

#     loss_val = term1 + 1e-3*term2
#     # loss_val = jnp.nan_to_num(term1, nan=0.0, posinf=0.0, neginf=0.0)
#     # loss_val = term1

#     return loss_val, (jnp.sum(nb_steps)/ctx_s.shape[0], term1, term2)


def loss_fn_ctx(model, trajs, t_eval, ctx, all_ctx_s, key):

    # ind = jax.random.randint(key, shape=(context_pool_size,), minval=0, maxval=all_ctx_s.shape[0])
    ind = jax.random.permutation(key, all_ctx_s.shape[0])[:context_pool_size]
    ctx_s = all_ctx_s[ind, :]

    # jax.debug.print("indices chosen for this loss {}", ind)

    trajs_hat, nb_steps = jax.vmap(model, in_axes=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_s)
    new_trajs = jnp.broadcast_to(trajs, trajs_hat.shape)

    term1 = jnp.mean((new_trajs-trajs_hat)**2)  ## reconstruction
    # term2 = jnp.mean(ctx**2)             ## regularisation
    term2 = jnp.mean(jnp.abs(ctx))             ## regularisation
    # term2 = params_norm_squared(ctx)
    term3 = params_norm_squared(model)

    loss_val = term1 + 1e-3*term2 + 1e-3*term3
    # loss_val = jnp.nan_to_num(term1, nan=0.0, posinf=0.0, neginf=0.0)
    # loss_val = term1

    return loss_val, (jnp.sum(nb_steps)/ctx_s.shape[0], term1, term2)









contexts = IDContextParams(nb_envs=nb_envs, context_size=context_size, hidden_size=32, depth=3, key=seed)
augmentation = Augmentation(data_size=2, int_size=context_size, context_size=context_size, ctx_utils=contexts.ctx_utils, key=seed)
vectorfield = ContextFlowVectorField(augmentation, physics=None)

learner = Learner(vectorfield, contexts, loss_fn_ctx, integrator, ivp_args, key=seed)

print("\n\nTotal number of parameters in the vector field:", sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(vectorfield,eqx.is_array)) if x is not None))
print("\n\nTotal number of parameters in the contexts:", contexts.params.shape[0]*contexts.params.shape[1], "\n")


#%%

## Define optimiser and traine the model

nb_total_epochs = nb_outer_steps * 1
sched_node = optax.piecewise_constant_schedule(init_value=init_lr,
                        boundaries_and_scales={nb_total_epochs//3:sched_factor, 2*nb_total_epochs//3:sched_factor})

sched_ctx = optax.piecewise_constant_schedule(init_value=init_lr,
                        boundaries_and_scales={nb_total_epochs//3:sched_factor, 2*nb_total_epochs//3:sched_factor})

opt_node = optax.adam(sched_node)
opt_ctx = optax.adam(sched_ctx)

trainer = Trainer(train_dataloader, learner, (opt_node, opt_ctx), key=seed)

#%%

trainer_save_path = run_folder if save_trainer == True else False
if train == True:
    # for i, prop in enumerate(np.linspace(0.25, 1.0, 3)):
    for i, prop in enumerate(np.linspace(1.0, 1.0, 1)):
        # trainer.dataloader.int_cutoff = int(prop*nb_steps_per_traj)
        # trainer.train(nb_epochs=nb_epochs*(2**0), print_error_every=print_error_every*(2**0), update_context_every=1, save_path=trainer_save_path, key=seed, val_dataloader=val_dataloader, int_prop=prop)
        trainer.train_proximal(nb_outer_steps_max=nb_outer_steps, nb_inner_steps_max=nb_inner_steps_max, proximal_reg=proximal_beta, inner_tol_node=inner_tol_node, inner_tol_ctx=inner_tol_ctx,
                               print_error_every=print_error_every*(2**0), save_path=trainer_save_path, key=seed, val_dataloader=val_dataloader, int_prop=prop)

else:
    # print("\nNo training, attempting to load model and results from "+ run_folder +" folder ...\n")

    restore_folder = run_folder
    # restore_folder = "./runs/27012024-155719/finetune_193625/"
    trainer.restore_trainer(path=restore_folder)



















#%%

## Test and visualise the results on a test dataloader

test_dataloader = DataLoader(run_folder+"test_data.npz", shuffle=False)
visualtester = VisualTester(trainer)
# ans = visualtester.trainer.nb_steps_node
# print(ans.shape)

ind_crit = visualtester.test(test_dataloader, int_cutoff=1.0)

savefigdir = run_folder+"results_in_domain.png"
visualtester.visualize(test_dataloader, int_cutoff=1.0, save_path=savefigdir);



#%%

























## Give the dataloader an id to help with restoration later on

if adapt_test and not adapt_restore:
    os.system(f'python dataset.py --split=adapt --savepath="{adapt_folder}" --seed="{seed*3}"');

if adapt_test:
    adapt_dataloader = DataLoader(adapt_folder+"adapt_train.npz", adaptation=True, data_id="170846", key=seed)

    # sched_ctx_new = optax.piecewise_constant_schedule(init_value=1e-5,
    #                         boundaries_and_scales={int(nb_epochs_adapt*0.25):1.,
    #                                                 int(nb_epochs_adapt*0.5):0.1,
    #                                                 int(nb_epochs_adapt*0.75):1.})
    sched_ctx_new = optax.piecewise_constant_schedule(init_value=init_lr,
                            boundaries_and_scales={nb_total_epochs//3:sched_factor, 2*nb_total_epochs//3:sched_factor})
    # sched_ctx_new = 1e-5
    opt_adapt = optax.adabelief(sched_ctx_new)

    if adapt_restore == False:
        trainer.adapt(adapt_dataloader, nb_epochs=nb_epochs_adapt, optimizer=opt_adapt, print_error_every=print_error_every, save_path=adapt_folder)
    else:
        print("Save_id for restoring trained adapation model:", adapt_dataloader.data_id)
        trainer.restore_adapted_trainer(path=adapt_folder, data_loader=adapt_dataloader)


#%%
if adapt_test:
    ood_crit = visualtester.test(adapt_dataloader, int_cutoff=1.0)      ## It's the same visualtester as before during training. It knows trainer

    visualtester.visualize(adapt_dataloader, int_cutoff=1.0, save_path=adapt_folder+"results_ood.png");




#%%
## If the nohup.log file exists, copy it to the run folder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    if os.path.exists("nohup.log"):
        os.system(f"cp nohup.log {run_folder}")


#%%

