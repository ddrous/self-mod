## This script is a walkthrough for the Self-Mod library on the CelebA dataset.

import os

from selfmod import *
import time
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

## For reproducibility
seed = 2024

## Dataloader hyperparameters
k_shots = 100                # Number of sample points per environment
resolution = (32, 32)        # Image resolution
data_folder = "./data/"      # Path to the dataset
shuffle = False              # Whether to shuffle the data
num_workers = 24             # Number of worker processes for data loading
input_dim = 2                # Input dimension (x,y coordinates)
output_dim = 3               # Output dimension (RGB values)

## Training and adaptation hyperparameters
context_pool_size = 2        # Number of contexts to maintain in the pool
context_size = 128           # Dimension of context vectors
loss_contributors = 32       # Number of environments contributing to the loss
taylor_orders = (2, 0)       # Taylor expansion orders for training and testing
init_lrs = (1e-3, 1e-3)      # Initial learning rates (model, context)
sched_factor = 1.0           # Learning rate scheduler factor
envs_batch_size = 162770     # Number of environments per batch for training
envs_batch_size_val = 100    # Number of environments per batch for validation
max_train_batches = 1        # Maximum number of training batches per epoch
max_val_batches = 1          # Maximum number of validation batches

# Strategies for filling pools and selecting contexts for loss computation
pool_filling_strategy = "NF"  # Nearest-First strategy for pool filling
loss_filling_strategy = "NF"  # Nearest-First strategy for loss contributors selection (StochasticNCF)

nb_outer_steps = 162000      # Number of outer loop steps
nb_inner_steps = (10, 10)    # Number of inner loop steps (adaptation)

print_error_every = 1620     # Print error every N steps
validate_every = 30*5*1000000  # Validate every N steps

nb_adapt_steps = 5000        # Number of adaptation steps for meta-testing

# Control flow flags
meta_train = True            # Whether to perform meta-training
meta_test = True             # Whether to perform meta-testing
save_prefix = ""             # Prefix for saved files
run_folder = None            # Folder to store results (created if None)

# Initialize random keys
mother_key = jax.random.PRNGKey(seed)

# Create run folder for storing results
if meta_train:
    # Create runs directory if it doesn't exist
    if not os.path.exists('./runs'):
        os.mkdir('./runs')

    # Create run folder with timestamp or use existing one
    if run_folder is None:
        run_folder = './runs/'+time.strftime("%y%m%d-%H%M%S")+'/'
        os.mkdir(run_folder)
        print("New run folder created successfully:", run_folder)
    else:
        print("Using pre-existing run folder:", run_folder)

    # Save the current script and the selfmod module
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {run_folder}")
    os.system(f"cp -r ../../selfmod {run_folder}")
    print("Completed copied scripts")
else:
    print("No training. Loading model and results from:", run_folder)

# Create folder for adaptation results
if meta_test:
    adapt_folder = run_folder+"adapt/"
    if not os.path.exists(adapt_folder):
        os.mkdir(adapt_folder)

# Split random keys for reproducibility
mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

# Create data loaders for training
# CelebADataset loads images from the CelebA dataset
# NumpyLoader wraps the dataset for efficient loading
train_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="train",
                                            num_shots=k_shots, 
                                            order_pixels=False, 
                                            resolution=resolution), 
                              batch_size=envs_batch_size, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

# Dataloader with all pixels for full image reconstruction
all_shots_train_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="train",
                                            num_shots=np.prod(resolution), 
                                            order_pixels=False, 
                                            resolution=resolution), 
                              batch_size=envs_batch_size_val, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)


# Define the model architecture
class MultiMLP(eqx.Module):
    """
    Multi-layer perceptron with shared layers.
    Takes input coordinates and context vector to produce RGB values.
    """
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
        # Concatenate input and context
        y = jnp.concatenate([x, ctx], axis=0)
        # Pass through all layers
        for layer in self.layers_shared:
            y = layer(y)
        return y


def env_loss_fn(model, ctx, y_hat, y):
    """
    Loss function for one environment.
    Leading dimension of y_hat corresponds to the pool size.
    
    Args:
        model: The neural network model
        ctx: Context vector
        y_hat: Predicted output
        y: Target output
        
    Returns:
        Tuple of (loss_value, (term1, term2, term3))
    """
    # Mean squared error loss
    term1 = jnp.mean((y_hat-y)**2)
    loss_val = term1
    return loss_val, (term1, 0., 0.)


# Initialize context parameters
contexts = ArrayContextParams(nb_envs=envs_batch_size,
                            context_size=context_size)

# Create neural network
neuralnet = MultiMLP(in_size=input_dim,
                     out_size=output_dim, 
                     hidden_size=128,
                     context_size=context_size,
                     key=model_key)

# Wrap network in NeuralContextFlow (NCF) model
# NCF models use Taylor expansion for information flow across environemnts
model = NeuralContextFlow(neuralnet=neuralnet, 
                          taylor_order=taylor_orders[0])

# Create learner to manage model training
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

# Print model information
model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)) if x is not None)
print("\n\nTotal number of parameters in the model:", model_params)
print("Total number of parameters in one context:", contexts.eff_context_size)

# Define optimizers for model and contexts
init_lr_model, init_lr_ctx = init_lrs

# Learning rate schedule with piecewise constant decay
total_strain_steps = nb_outer_steps*nb_inner_steps[0]
bd_scales = {total_strain_steps//3:sched_factor, 2*total_strain_steps//3:sched_factor}
sched_model = optax.piecewise_constant_schedule(init_value=init_lr_model, boundaries_and_scales=bd_scales)
opt_model = optax.adam(sched_model)

opt_ctx = optax.adam(init_lr_ctx)

# Create trainer to manage the training process
trainer = NCFTrainer(learner, (opt_model, opt_ctx), key=trainer_key)

# Meta-training phase
if meta_train:
    # Train the model without using alternating minimisation
    trainer.meta_train_noalm(dataloader=train_dataloader,
                        nb_epochs=1,
                        nb_outer_steps=nb_outer_steps,
                        max_train_batches=max_train_batches,
                        print_error_every=(1, print_error_every), 
                        save_path=run_folder, 
                        max_val_batches=max_val_batches,
                        val_criterion_id=0,
                        validate_every=validate_every,
                        val_nb_steps=nb_adapt_steps,
                        key=trainer_key)
else:
    # Load pre-trained model
    restore_folder = run_folder
    trainer.restore_trainer(path=run_folder)
    print("\nNo training, loaded model and results from "+ run_folder +" folder ...\n")

# Create visual tester for evaluation and visualization
visualtester = CelebAVisualTester(trainer, key=test_key)

# Evaluate on training set with all shots
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

# Visualize reconstruction artifacts
visualtester.visualize_artefacts(save_path=run_folder+"artefacts.png")

# Visualize few-shot learning with uncertainty quantification
visualtester.visualize_few_shots_multi_uq(few_shots_loader=train_dataloader,
                                all_shots_loader=all_shots_train_dataloader,
                                nb_steps=nb_adapt_steps,
                                save_path=run_folder+save_prefix+"few_shots_ind_uq.png",
                                taylor_order=taylor_orders[0],
                                num_envs=16,
                                uq_train_contexts=1000,
                                interp_method="cubic",  # Interpolation method: 'linear', 'nearest', 'cubic'
                                key=test_key
                             )

# Create validation dataloaders
val_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="val",
                                            num_shots=k_shots, 
                                            order_pixels=False, 
                                            resolution=resolution), 
                              batch_size=envs_batch_size, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

all_shots_val_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="val",
                                            num_shots=np.prod(resolution), 
                                            order_pixels=False, 
                                            resolution=resolution), 
                              batch_size=envs_batch_size_val, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

# Evaluate on validation set
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

# Meta-testing phase (adaptation to new data)
if meta_test:
    # Create test dataloaders
    adapt_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                                data_split="test",
                                                num_shots=k_shots, 
                                                order_pixels=False, 
                                                resolution=resolution), 
                                batch_size=envs_batch_size, 
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)
    
    all_shots_dataloader_test = NumpyLoader(CelebADataset(data_folder, 
                                                data_split="test",
                                                num_shots=np.prod(resolution),
                                                order_pixels=False, 
                                                resolution=resolution), 
                                batch_size=envs_batch_size_val, 
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    # Evaluate on test set
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

    # Visualize adaptation results
    visualtester.visualize_artefacts(save_path=adapt_folder+"artefacts.png", adaptation=True)

    # Visualize few-shot learning on test set with uncertainty quantification
    visualtester.visualize_few_shots_multi_uq(few_shots_loader=adapt_dataloader,
                                    all_shots_loader=all_shots_dataloader_test,
                                    nb_steps=nb_adapt_steps,
                                    save_path=adapt_folder+save_prefix+"few_shots_ood_uq.png",
                                    taylor_order=taylor_orders[0],
                                    num_envs=7,
                                    uq_train_contexts=1000,
                                    interp_method="cubic",
                                    key=test_key
                                )
