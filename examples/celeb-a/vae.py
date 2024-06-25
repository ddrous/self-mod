"""## Train a VAE and generate diverse CelebA samples """

%load_ext autoreload
%autoreload 2

import jax
import jax.numpy as jnp
import optax

import time
import os
import matplotlib.pyplot as plt

import equinox as eqx
from selfmod import CelebADataLoader


#%%
MOTHER_KEY = jax.random.PRNGKey(2024)

# RUN_FOLDDER = './runs/'+time.strftime("%y%m%d-%H%M%S")+'/'
RUN_FOLDDER = './runs/250609-112233-VAE/'
if not os.path.exists(RUN_FOLDDER):
    os.mkdir(RUN_FOLDDER)
print("New run folder created successfuly:", RUN_FOLDDER)

DATA_FOLDER="./data/"

EPOCHS = 1


#%%
## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
data_key, model_key, trainer_key, test_key = jax.random.split(MOTHER_KEY, num=4)
## Define dataloaders for training and validation
train_dataloader = CelebADataLoader(DATA_FOLDER, 
                                    envs_batch_size=32,
                                    shots_batch_size=32*32, 
                                    data_split="train",
                                    envs_shuffle=False, 
                                    shots_shuffle=False, 
                                    order_pixels=True, 
                                    key=data_key)

class VAE(eqx.Module):
    """ Variational Autoencoder with convolutions and deconvolutions"""

    img_size: list
    kernel_size: list
    latent_dim: int
    encoder: list
    decoder: list

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        layer_keys = jax.random.split(key, 8)

        self.encoder = [
            lambda x: x.reshape(self.img_size),
            eqx.nn.Conv2d(self.img_size[-1], 32, kernel_size, padding="SAME", key=layer_keys[0]),
            jax.nn.relu,
            eqx.nn.Conv2d(32, 64, kernel_size, padding="SAME", key=layer_keys[1]),
            jax.nn.relu,
            lambda x: x.flatten(),
            eqx.nn.Linear(self.img_size[0]*self.img_size[1]*64, 256, key=layer_keys[2]),
            jax.nn.relu,
            eqx.nn.Linear(256, 2*latent_dim, key=layer_keys[3])
        ]

        self.decoder = [
            eqx.nn.Linear(latent_dim, 256, key=layer_keys[4]),
            jax.nn.relu,
            eqx.nn.Linear(256, self.img_size[0]*self.img_size[1]*64, key=layer_keys[5]),
            jax.nn.relu,
            lambda x: x.reshape((self.img_size[0], self.img_size[1], 64)),
            eqx.nn.ConvTranspose2d(64, 32, kernel_size, padding="SAME", key=layer_keys[6]),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(32, self.img_size[-1], kernel_size, padding="SAME", key=layer_keys[7]),
            jax.nn.sigmoid
        ]

    def encode(self, x):
        z = x
        for layer in self.encoder:
            z = layer(z)
        return z
    
    def decode(self, z):
        x = z
        for layer in self.decoder:
            x = layer(x)
        return x

    def __call__(self, x, key):
        mu, logvar = jnp.split(self.encode(x), 2, axis=-1)
        eps = eqx.random.normal(mu.shape, key=key)
        z = mu + eps*jnp.exp(0.5*logvar)
        return self.decode(z), mu, logvar

model = VAE(img_size=[32, 32, 3], kernel_size=[3, 3], latent_dim=128, key=model_key)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

#%%

## Loss function
def loss_fn(model, xs, keys):
    recon_xs, mus, logvars = jax.vmap(model)(xs, keys)
    BCE = jnp.sum(-xs*jnp.log(recon_xs) - (1-xs)*jnp.log(1-recon_xs), axis=1)
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

for epoch in range(EPOCHS):
    loss_epoch = 0.

    for batch_id, batch in enumerate(train_dataloader):

        _, Ys = batch
        trainer_key, _ = jax.random.split(trainer_key)

        model, opt_state, loss = train_step(model, Ys, opt_state, trainer_key)
        loss_epoch += loss

        # if batch_id < 10:
        print(f"Epoch {epoch}   Batch {batch_id}    Loss: {loss_epoch}")

        if batch_id>10:     ## Just for now !
            break


#%%

## Test the model

zs = jax.random.normal(test_key, (64, 128))
samples = jax.vmap(model.decode)(zs).reshape((64, 1, 32, 32, 3))

# Plot the generated samples
plt.figure(figsize=(8, 8))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(samples[i].squeeze())
    plt.axis('off')
plt.show()
