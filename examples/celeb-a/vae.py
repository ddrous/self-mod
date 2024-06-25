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
from selfmod import CelebADataLoader, sbplot, plt

import torch
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#%%
MOTHER_KEY = jax.random.PRNGKey(2026)

# RUN_FOLDDER = './runs/'+time.strftime("%y%m%d-%H%M%S")+'/'
RUN_FOLDDER = './runs/250624-193230-VAE/'
if not os.path.exists(RUN_FOLDDER):
    os.mkdir(RUN_FOLDDER)
print("Using run folder:", RUN_FOLDDER)

DATA_FOLDER="./data/"
BATCH_SIZE = 1024*4
EPOCHS = 25*10*5
LR=1e-2
IMG_SIZE = [32, 32, 3]


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

class Downsample2D(eqx.Module):
    """ Downsample 2D image by a factor: https://docs.kidger.site/equinox/examples/unet/ """
    factor: int
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, y):
        C, H, W = y.shape
        y = jnp.reshape(y, [C, H // self.factor, self.factor, W // self.factor, self.factor])
        return jnp.max(y, axis=[2, 4])


class Upsample2D(eqx.Module):
    """ Upsample 2D image by a factor: https://docs.kidger.site/equinox/examples/unet/ """
    factor: int
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, y):
        C, H, W = y.shape
        y = jnp.reshape(y, [C, H, 1, W, 1])
        y = jnp.tile(y, [1, 1, self.factor, 1, self.factor])
        return jnp.reshape(y, [C, H * self.factor, W * self.factor])


class Encoder(eqx.Module):
    """ Encoder with convolutions and dense layers"""
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
            eqx.nn.Conv2d(C, 4, kernel_size, padding="SAME", key=layer_keys[0]),
            Downsample2D(factor=2),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Conv2d(4, 8, kernel_size, padding="SAME", key=layer_keys[1]),
            Downsample2D(factor=2),
            eqx.nn.PReLU(init_alpha=0.),
            lambda x: x.flatten(),
            eqx.nn.Linear(8*H*W//(4*4), 32, key=layer_keys[2]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(32, 2*latent_dim, key=layer_keys[3])
        ]

    def __call__(self, x):
        z = x
        for layer in self.layers:
            z = layer(z)
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
            eqx.nn.Linear(latent_dim, 32, key=layer_keys[0]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(32, 8*H*W//(4*4), key=layer_keys[1]),
            eqx.nn.PReLU(init_alpha=0.),
            lambda x: x.reshape((8, H//4, W//4)),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(8, 4, kernel_size, padding="SAME", key=layer_keys[2]),
            eqx.nn.PReLU(init_alpha=0.),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(4, C, kernel_size, padding="SAME", key=layer_keys[3]),
            jax.nn.sigmoid
        ]

    def __call__(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        return x



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

model = VAE(img_size=IMG_SIZE, kernel_size=[3, 3], latent_dim=12, key=model_key)

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
    BCE = jnp.sum(-xs*jnp.log(recon_xs) - (1-xs)*jnp.log(1-recon_xs), axis=(1,2,3))
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
        # if batch_id % 100 == 0:
        print(f"Epoch {epoch}   Batch {batch_id}    Loss: {loss}", end="\r")

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
zs = jax.random.normal(test_key, (64, 12))
samples = jax.vmap(model.decoder)(zs).reshape((64, 1, 32, 32, 3))

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