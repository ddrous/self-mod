"""## Train a VAE and generate diverse CelebA samples """

# %load_ext autoreload
# %autoreload 2

import jax
import jax.numpy as jnp
import numpy as np
import optax

import time
import os

import equinox as eqx
import diffrax
from selfmod import CelebADataLoader, sbplot, plt, VNet

import torch
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# ## JAX debug the NaNs
# from jax import config
# config.update("jax_debug_nans", True)


#%%
MOTHER_KEY = jax.random.PRNGKey(2026)

RUN_FOLDDER = '../runs/'+time.strftime("%y%m%d-%H%M%S")+'/'
# RUN_FOLDDER = './runs/250624-193230-VAE/'
if not os.path.exists(RUN_FOLDDER):
    os.mkdir(RUN_FOLDDER)
print("Using run folder:", RUN_FOLDDER)
os.system(f"cp {__file__} {RUN_FOLDDER}")

DATA_FOLDER="../data/"
BATCH_SIZE = 256
EPOCHS = 10
LR=1e-3
IMG_SIZE = [64, 64, 3]
LATENT_DIM = np.prod(IMG_SIZE)
LEVELS = 4
DEPTH = 16


#%%
## Define 4 keys for dataloader(s), learner(s), trainer(s) and visualtester(s)
data_key, model_key, trainer_key, test_key = jax.random.split(MOTHER_KEY, num=4)

## Custom dataloader for training and validation
# train_dataloader = CelebADataLoader(DATA_FOLDER, 
#                                     envs_batch_size=BATCH_SIZE,
#                                     shots_batch_size=np.prod(IMG_SIZE[:2]), 
#                                     resolution=IMG_SIZE[:2],
#                                     data_split="train",
#                                     envs_shuffle=True, 
#                                     shots_shuffle=False, 
#                                     order_pixels=True, 
#                                     key=data_key)

## Pytorch dataloader for CelebA
class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, img_dir):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imgs_root = img_dir+"img_align_celeba/"

        partitions = pd.read_csv(img_dir+'list_eval_partition.txt', 
                                 header=None, 
                                 sep=r'\s+', 
                                 names=['filename', 'partition'])
        self.files = partitions[partitions['partition'] == 0]['filename'].values

        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                        transforms.Resize((IMG_SIZE[0], IMG_SIZE[1]), Image.LANCZOS),
                                        transforms.ToTensor(),
                                        ])

    def get_image(self, filename):
        img_path = os.path.join(self.imgs_root, filename)
        img = self.transform(img_path).float().to(self.device)
        # img = img * 2 - 1
        img = img.permute(1, 2, 0)
        return img
        # return jnp.asarray(img.numpy())

    def __getitem__(self, index):
        return self.get_image(self.files[index])

    def __len__(self):
        return len(self.files)


train_dataset = CelebaDataset(img_dir=DATA_FOLDER)

train_dataloader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=24) 
# print(len(train_dataset))






#%%

class Encoder(eqx.Module):
    """ Encoder with convolutions and ODE solver"""
    img_size: list
    kernel_size: list
    latent_dim: int

    layers: list

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        layer_keys = jax.random.split(key, 4)
        H, W, C = self.img_size

        self.layers = [
            lambda x: x.reshape((C*2, H, W)),

            VNet(input_shape=[C*2, H, W],
                 output_shape=[C*2, H, W], 
                 levels=LEVELS, 
                 depth=DEPTH, 
                 kernel_size=kernel_size, 
                 activation=jax.nn.relu, 
                 final_activation=lambda x: x, 
                 batch_norm=False, 
                 dropout_rate=0.,
                key=layer_keys[0]),

            lambda x: x.flatten(),
        ]

    def __call__(self, x):

        def vectorfield(t, y, args):
            # print("Inputput shapes:", y.shape)
            dy = y
            for layer in self.layers:
                dy = layer(dy)
            # print("Output shapes:", dy.shape)
            return dy

        ## Solve a differential equation from t=0 to t=1, usign diffrax
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vectorfield),
            diffrax.Dopri5(),
            t0=0,
            t1=1,
            dt0=0.1,
            y0=jnp.concatenate([x.flatten(), jnp.zeros_like(x).flatten()]),
            args=None,
            max_steps=512,
        )
        z = sol.ys[-1]
        return z


class Decoder(eqx.Module):
    """ Decoder with dense layers and deconvolutions"""
    img_size: list
    kernel_size: list
    latent_dim: int

    layers: list

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        layer_keys = jax.random.split(key, 4)
        H, W, C = self.img_size

        self.layers = [
            lambda z: z.reshape((C, H, W)),
            VNet(input_shape=[C, H, W],
                 output_shape=[C, H, W], 
                 levels=LEVELS, 
                 depth=DEPTH, 
                 kernel_size=kernel_size, 
                 activation=jax.nn.relu, 
                 final_activation=lambda x: x, 
                batch_norm=False,
                dropout_rate=0.,
                key=layer_keys[0]),
            lambda y: y.flatten(),
        ]

    def __call__(self, z):
        
        def vectorfield(t, y, args):
            dy = y
            for layer in self.layers:
                dy = layer(dy)
            return dy

        ## Solve a differential equation from t=0 to t=1, usign diffrax
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vectorfield),
            diffrax.Dopri5(),
            t0=0,
            t1=1,
            dt0=0.1,
            y0=z,
            max_steps=512,
        )
        x_recon = sol.ys.reshape(self.img_size[2], self.img_size[0], self.img_size[1])

        return jax.nn.sigmoid(x_recon)


class VAE(eqx.Module):
    """ Variational Autoencoder with convolutions and deconvolutions"""

    img_size: list
    kernel_size: list
    latent_dim: int
    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        enc_key, dec_key = jax.random.split(key, 2)

        self.encoder = Encoder(img_size, kernel_size, latent_dim, key=enc_key)
        self.decoder = Decoder(img_size, kernel_size, latent_dim, key=dec_key)

    def __call__(self, x, key):
        mu, logvar = jnp.split(self.encoder(x), 2, axis=-1)
        eps = jax.random.normal(key=key, shape=mu.shape)
        z = mu + eps*jnp.exp(0.5*logvar)
        return self.decoder(z), mu, logvar

model = VAE(img_size=IMG_SIZE, kernel_size=[3, 3], latent_dim=LATENT_DIM, key=model_key)

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
    recon_xs, mus, logvars = eqx.filter_vmap(model)(xs, keys)
    # print("ALl shapes:", xs.shape, recon_xs.shape, mus.shape, logvars.shape)
    BCE = jnp.mean(-xs*jnp.log(recon_xs) - (1-xs)*jnp.log(1-recon_xs), axis=(1,2,3))
    KLD = -0.5 * jnp.mean(1 + logvars - mus**2 - jnp.exp(logvars), axis=1)
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

@jax.jit
@jax.vmap
def transform_batch(x):
    return x.reshape((IMG_SIZE[-1], IMG_SIZE[0], IMG_SIZE[1]))

losses = []

start = time.time()

for epoch in range(EPOCHS):

    loss_epoch = 0.
    nb_batches = 0

    for batch_id, batch in enumerate(train_dataloader):

        # _, Ys = batch
        Ys = jnp.asarray(batch)

        images = transform_batch(Ys)
        trainer_key, _ = jax.random.split(trainer_key)

        model, opt_state, loss = train_step(model, images, opt_state, trainer_key)

        loss_epoch += loss
        nb_batches += 1

        # if batch_id < 10:
        if batch_id % 10 == 0:
            print(f"Epoch {epoch:-5d}   Batch {batch_id:-5d}    Loss: {loss}", end="\r")

        # if batch_id>10:     ## Just for now !
        #     break

        losses.append(loss)
    print()

end = time.time()
print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(end-start)))

#%%

sbplot(losses, title="ELBO Loss", x_label="Iterations");
plt.savefig(RUN_FOLDDER+"elbo_loss.png")






#%%

## Test and plot the decoder
zs = jax.random.normal(test_key, (64, LATENT_DIM))
samples = jax.vmap(model.decoder)(zs).reshape((64, 1, IMG_SIZE[0], IMG_SIZE[1], 3))

plt.figure(figsize=(8, 8))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(samples[i].squeeze())
    plt.axis('off')
plt.draw()

## Save the figure in the run folder
plt.savefig(RUN_FOLDDER+"samples.png");

#%%
## Save the model in the run folder
eqx.tree_serialise_leaves(RUN_FOLDDER+"model.eqx", model)
eqx.tree_serialise_leaves(RUN_FOLDDER+"decoder.eqx", model.decoder)

## In case we run with nohup
if os.path.exists("nohup.log"):
    os.system(f"cp nohup.log {RUN_FOLDDER}")