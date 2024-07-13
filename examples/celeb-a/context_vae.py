#%%
# """## Train a VAE and generate diverse CelebA samples """

# %load_ext autoreload
# %autoreload 2

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
import jax
import jax.numpy as jnp
import numpy as np
import optax

import time
import os

import equinox as eqx
from selfmod import sbplot, plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

#%%
MOTHER_KEY = jax.random.PRNGKey(2026)

# RUN_FOLDDER = './runs/240713-143455/'
RUN_FOLDDER = './runs/240713-134917-GoldenT0/'
print("Using run folder:", RUN_FOLDDER)

DATA_FOLDER=RUN_FOLDDER+"contexts/"
BATCH_SIZE = 64*4
EPOCHS = 200
LR = 1e-5

NB_TEST_SAMPLES = 128*16
CONTEXT_DIM = 256
NOISE_DIM = 64







#%%
## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
data_key, model_key, trainer_key, test_key = jax.random.split(MOTHER_KEY, num=4)


## Pytorch dataset for CelebA
class ContextDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, data_dir):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        context_files = os.listdir(data_dir)
        context_files = [data_dir+file for file in context_files]

        contexts = []
        for file in context_files:
            context = np.load(file)
            contexts.append(context)

        self.contexts = np.concatenate(contexts, axis=0)
        print("Total number of contexts loaded, and dimension:", self.contexts.shape)

    def __getitem__(self, index):
        return torch.from_numpy(self.contexts[index]).to(self.device)

    def __len__(self):
        return len(self.contexts)


train_dataset = ContextDataset(data_dir=DATA_FOLDER)

train_dataloader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=24)





#%%


class Encoder(eqx.Module):
    """ Encoder with convolutions and dense layers"""
    layers: list

    def __init__(self, key):
        layer_keys = jax.random.split(key, 4)

        self.layers = [
            eqx.nn.Linear(CONTEXT_DIM, 4*NOISE_DIM, key=layer_keys[0]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(4*NOISE_DIM, 2*NOISE_DIM, key=layer_keys[1]),
        ]

    def __call__(self, x):
        z = x
        for layer in self.layers:
            z = layer(z)
        return z


class Decoder(eqx.Module):
    """ Decoder with dense layers and deconvolutions"""
    layers: list

    def __init__(self, key):
        layer_keys = jax.random.split(key, 4)

        self.layers = [
            eqx.nn.Linear(NOISE_DIM, 4*NOISE_DIM, key=layer_keys[0]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(4*NOISE_DIM, CONTEXT_DIM, key=layer_keys[1]),
        ]

    def __call__(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        return x



class VAE(eqx.Module):
    """ Variational Autoencoder with convolutions and deconvolutions"""

    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, key):
        enc_key, dec_key = jax.random.split(key, 2)

        self.encoder = Encoder(key=enc_key)
        self.decoder = Decoder(key=dec_key)

    def __call__(self, x, key):
        mu, logvar = jnp.split(self.encoder(x), 2, axis=-1)
        eps = jax.random.normal(key=key, shape=mu.shape)
        z = mu + eps*jnp.exp(0.5*logvar)
        return self.decoder(z), mu, logvar

model = VAE(key=model_key)

## Count the number of parameters
count_enc = np.sum([p.size for p in jax.tree.leaves(model.encoder) if isinstance(p, jnp.ndarray)])
print(f"Number of parameters in the encoder: {count_enc}")

count_dec = np.sum([p.size for p in jax.tree.leaves(model.decoder) if isinstance(p, jnp.ndarray)])
print(f"Number of parameters in the decoder: {count_dec}")
print()

## Initialise the optimizer
nb_train_steps = len(train_dataloader)*EPOCHS
sched = optax.piecewise_constant_schedule(LR, {nb_train_steps//3: 1., 2*nb_train_steps//3: 0.1})
optimizer = optax.adam(sched)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

#%%

## Loss function
def loss_fn(model, xs, keys):
    recon_xs, mus, logvars = jax.vmap(model)(xs, keys)
    # BCE = jnp.sum(-xs*jnp.log(recon_xs) - (1-xs)*jnp.log(1-recon_xs), axis=(1,2,3))
    BCE = jnp.mean((xs - recon_xs)**2, axis=1)
    KLD = -0.5 * jnp.sum(1 + logvars - mus**2 - jnp.exp(logvars), axis=1)
    return jnp.mean(BCE + KLD)

@eqx.filter_jit
def train_step(model, xs, opt_state, key):
    keys = jax.random.split(key, xs.shape[0])
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, xs, keys)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


#%%
## Training loop


losses = []

start = time.time()

for epoch in range(EPOCHS):

    loss_epoch = 0.
    nb_batches = 0

    for batch_id, batch in enumerate(train_dataloader):

        ctxs = jnp.asarray(batch)
        trainer_key, _ = jax.random.split(trainer_key)

        model, opt_state, loss = train_step(model, ctxs, opt_state, trainer_key)

        loss_epoch += loss
        nb_batches += 1

        if batch_id % 10 == 0:
            print(f"Epoch {epoch:-5d}   Batch {batch_id:-5d}    Loss: {loss}", end="\r")

        # if batch_id>10:     ## Just for now !
        #     break

            losses.append(loss)
    # print()

end = time.time()
print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(end-start)))

#%%

sbplot(losses, title="ELBO Loss", x_label="Iterations", y_scale="log");
plt.savefig(DATA_FOLDER+"elbo_loss.png")






#%%

## Generate test data
zs = jax.random.normal(test_key, (NB_TEST_SAMPLES, NOISE_DIM))
samples = jax.vmap(model.decoder)(zs)



## Use PCA to transform the samples before plotting
# visualiser = PCA(n_components=2)
# reducer = KernelPCA(n_components=2)
# train_data_plot = reducer.fit_transform(train_dataset.contexts)
# sample_plot = reducer.transform(samples)

# ## Use TSNE to transform the samples before plotting
reducer = TSNE(n_components=2, random_state=2026)
all_data = jnp.concat([train_dataset.contexts, samples])
all_data_plot = reducer.fit_transform(all_data)

train_data_plot = all_data_plot[:len(train_dataset.contexts)]
sample_plot = all_data_plot[len(train_dataset.contexts):]


plt.figure(figsize=(6, 6))
plt.scatter(train_data_plot[:, 0], train_data_plot[:, 1], s=1, c='b', label="Train samples")
plt.scatter(sample_plot[:, 0], sample_plot[:, 1], s=1, c='orange', label="Generated samples")

plt.legend()
plt.draw()

## Save the figure in the run folder
plt.savefig(DATA_FOLDER+"t-SNE_visualisation.png");


## Save the samples into a file
np.save(DATA_FOLDER+"generated_samples.npy", samples)


#%%
# ## Save the model in the run folder
# eqx.tree_serialise_leaves(RUN_FOLDDER+"model.eqx", model)
# eqx.tree_serialise_leaves(RUN_FOLDDER+"decoder.eqx", model.decoder)

# ## In case we run with nohup
# if os.path.exists("nohup.log"):
#     os.system(f"cp nohup.log {RUN_FOLDDER}")