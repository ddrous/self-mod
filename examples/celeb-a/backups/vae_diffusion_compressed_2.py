#%%
"""## Train a Shared-ODE VAE with Cycle Consistency """

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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
import torch
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Prevent JAX from preallocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

#%%
MOTHER_KEY = jax.random.PRNGKey(2026)
RUN_FOLDER = './runs_shared_ode/'
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
print("Using run folder:", RUN_FOLDER)

DATA_FOLDER = "../data/"  # Update this to your path
BATCH_SIZE = 128*16
EPOCHS = 5
LR = 1e-3
IMG_SIZE = [64, 64, 3]
LATENT_DIM = 256
CONV_BASE_DEPTH = 8

#%%
## Define keys
data_key, model_key, trainer_key, test_key = jax.random.split(MOTHER_KEY, num=4)

#%%
# ==========================================
# 1. Data Loading (Restored)
# ==========================================

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
        # img = img.permute(1, 2, 0)
        # img = img.permute(2, 0, 1)  # Change to (C, H, W)
        return img
        # return jnp.asarray(img.numpy())

    def __getitem__(self, index):
        return self.get_image(self.files[index])

    def __len__(self):
        return len(self.files)


train_dataset = CelebaDataset(img_dir=DATA_FOLDER)

# to_jax = lambda x: jnp.asarray(x.numpy())

train_dataloader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=24)

#%%
# ==========================================
# 2. Helper Layers (Restored)
# ==========================================

class Downsample2D(eqx.Module):
    factor: int
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, y):
        C, H, W = y.shape
        # Average pooling
        y = jnp.reshape(y, [C, H // self.factor, self.factor, W // self.factor, self.factor])
        return jnp.mean(y, axis=[2, 4])

class Upsample2D(eqx.Module):
    factor: int
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, y):
        C, H, W = y.shape
        y = jnp.reshape(y, [C, H, 1, W, 1])
        y = jnp.tile(y, [1, 1, self.factor, 1, self.factor])
        return jnp.reshape(y, [C, H * self.factor, W * self.factor])

# ==========================================
# 3. Architecture (Shared ODE)
# ==========================================

class Encoder(eqx.Module):
    compress_layers: list
    img_size: list
    latent_dim: int

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.latent_dim = latent_dim
        layer_keys = jax.random.split(key, 5)
        H, W, C = img_size

        self.compress_layers = [
            eqx.nn.Conv2d(C, CONV_BASE_DEPTH, kernel_size, padding="SAME", key=layer_keys[0]),
            Downsample2D(factor=2),
            eqx.nn.PReLU(init_alpha=0.01),
            eqx.nn.Conv2d(CONV_BASE_DEPTH, CONV_BASE_DEPTH*2, kernel_size, padding="SAME", key=layer_keys[1]),
            Downsample2D(factor=2),
            eqx.nn.PReLU(init_alpha=0.01),
            eqx.nn.Conv2d(CONV_BASE_DEPTH*2, CONV_BASE_DEPTH*4, kernel_size, padding="SAME", key=layer_keys[2]),
            Downsample2D(factor=2), # 8x8x128
            eqx.nn.PReLU(init_alpha=0.01),
            lambda x: x.flatten(),
            eqx.nn.Linear(CONV_BASE_DEPTH*4*8*8, latent_dim, key=layer_keys[3]),
            # Note: No activation here, linear mapping to "Noise Initial Condition"
        ]

    def __call__(self, x, vf_func):
        # 1. Spatial Compression
        # print("Encoder input shape:", x.shape, flush=True)
        z = x
        for layer in self.compress_layers:
            z = layer(z)
        
        # Save this state! This is "Initial Condition of Nosing Process"
        z_initial = z 

        # 2. Forward ODE (Nosing): t=0 -> t=1
        term = diffrax.ODETerm(vf_func)
        solver = diffrax.Dopri5()
        sol = diffrax.diffeqsolve(
            term, solver, t0=0, t1=1, dt0=0.1, y0=z, max_steps=128
        )
        z_latent = sol.ys[-1]
        
        return z_latent, z_initial

class Decoder(eqx.Module):
    decompress_layers: list
    img_size: list
    latent_dim: int

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.latent_dim = latent_dim
        layer_keys = jax.random.split(key, 5)
        H, W, C = img_size

        self.decompress_layers = [
            eqx.nn.Linear(latent_dim, CONV_BASE_DEPTH*4*8*8, key=layer_keys[0]),
            lambda x: x.reshape((CONV_BASE_DEPTH*4, 8, 8)),
            eqx.nn.PReLU(init_alpha=0.01),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(CONV_BASE_DEPTH*4, CONV_BASE_DEPTH*2, kernel_size, padding="SAME", key=layer_keys[1]),
            eqx.nn.PReLU(init_alpha=0.01),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(CONV_BASE_DEPTH*2, CONV_BASE_DEPTH, kernel_size, padding="SAME", key=layer_keys[2]),
            eqx.nn.PReLU(init_alpha=0.01),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(CONV_BASE_DEPTH, C, kernel_size, padding="SAME", key=layer_keys[3]),
            jax.nn.sigmoid
        ]

    def __call__(self, z, vf_func):
        # 1. Backward ODE (De-Nosing): t=1 -> t=0
        term = diffrax.ODETerm(vf_func)
        solver = diffrax.Dopri5()
        sol = diffrax.diffeqsolve(
            term, solver, t0=1, t1=0, dt0=-0.1, y0=z, max_steps=128
        )
        z_final = sol.ys[-1] # This is "Output of Denoising Process"

        # 2. Spatial Decompression
        x_recon = z_final
        for layer in self.decompress_layers:
            x_recon = layer(x_recon)
            
        return x_recon, z_final

class VAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    vf_layers: eqx.Module

    def __init__(self, img_size, kernel_size, latent_dim, key):
        enc_key, dec_key, vf_key = jax.random.split(key, 3)
        
        self.encoder = Encoder(img_size, kernel_size, latent_dim, key=enc_key)
        self.decoder = Decoder(img_size, kernel_size, latent_dim, key=dec_key)
        
        # Shared Vector Field MLP
        # Input: Latent + 1 (time)
        self.vf_layers = eqx.nn.MLP(
            in_size=latent_dim + 1,
            out_size=latent_dim,
            width_size=256,
            depth=3,
            activation=jax.nn.silu, # SiLU is common in diffusion/ODE
            key=vf_key
        )

    def __call__(self, x):
        # Define the vector field function closing over self.vf_layers
        def vf_func(t, y, args):
            t_vec = jnp.broadcast_to(t, (1,))
            inp = jnp.concatenate([t_vec, y], axis=-1)
            return self.vf_layers(inp)
            
        # Forward Pass
        z_latent, z_initial_enc = self.encoder(x, vf_func)
        
        # Backward Pass
        x_recon, z_final_dec = self.decoder(z_latent, vf_func)
        
        return x_recon, z_latent, z_initial_enc, z_final_dec

model = VAE(img_size=IMG_SIZE, kernel_size=[3, 3], latent_dim=LATENT_DIM, key=model_key)

# Print param counts
count = np.sum([p.size for p in jax.tree.leaves(model) if isinstance(p, jnp.ndarray)])
print(f"Total Parameters: {count}")

#%%
# ==========================================
# 4. Loss Function (Updated)
# ==========================================

def loss_fn(model, xs, keys):
    print("(Re)compiling loss function with shapes:", xs.shape, flush=True)

    # Run model on batch
    # x_recon: (B, H, W, C)
    # z_latent: (B, Latent) - The noise
    # z_initial_enc: (B, Latent) - The compressed features before ODE
    # z_final_dec: (B, Latent) - The features after reverse ODE
    x_recon, z_latent, z_initial_enc, z_final_dec = jax.vmap(model)(xs)
    
    # 1. Reconstruction Loss (Pixel MSE)
    recon_loss = jnp.mean((xs - x_recon) ** 2)
    
    # 2. Batch-based KL Divergence
    # We want the 'z_latent' cloud to be N(0, I)
    mu_batch = jnp.mean(z_latent, axis=0)
    var_batch = jnp.var(z_latent, axis=0)
    logvar_batch = jnp.log(var_batch + 1e-6)
    kl_loss = -0.5 * jnp.sum(1 + logvar_batch - mu_batch**2 - var_batch)
    
    # 3. Consistency Loss (Feature MSE)
    # Discrepancy between Encoder Start and Decoder End
    consistency_loss = jnp.mean((z_initial_enc - z_final_dec) ** 2)
    
    # Weights
    w_recon = 1.0       # Pixel reconstruction, TODO: remove this, as in JEPA
    w_kl = 1.0       # Weight for batch statistics
    w_consistency = 1.0 # Enforce strict invertibility of the ODE
    
    total_loss = (w_recon * recon_loss) + (w_kl * kl_loss) + (w_consistency * consistency_loss)
    
    return total_loss, (recon_loss, kl_loss, consistency_loss)

@eqx.filter_jit
def train_step(model, xs, opt_state, key):
    keys = jax.random.split(key, xs.shape[0])
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, xs, keys)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux

# Optimizer
optimizer = optax.adam(LR)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

#%%
# ==========================================
# 5. Training Loop
# ==========================================

losses = []
r_losses = []
kl_losses = []
cons_losses = []

print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_loss = 0.
    epoch_recon = 0.
    epoch_kl = 0.
    epoch_cons = 0.
    
    steps = 0
    
    # Convert PyTorch loader to iterator to handle potential issues gracefully
    if isinstance(train_dataloader, list):
         print("No data loader found. Skipping training loop.")
         break

    for i, batch in enumerate(train_dataloader):
        # Pytorch (N, C, H, W) -> JAX (N, H, W, C)
        # images = jnp.asarray(batch.numpy()).transpose(0, 2, 3, 1)

        images = jnp.asarray(batch.numpy())
        
        trainer_key, _ = jax.random.split(trainer_key)
        
        model, opt_state, loss, aux = train_step(model, images, opt_state, trainer_key)
        
        epoch_loss += loss
        epoch_recon += aux[0]
        epoch_kl += aux[1]
        epoch_cons += aux[2]
        steps += 1

        losses.append(loss)
        r_losses.append(aux[0])
        kl_losses.append(aux[1])
        cons_losses.append(aux[2])
        
        if i % 10 == 0:
            print(f"Ep {epoch} | Step {i} | L: {loss:.4f} | Rec: {aux[0]:.4f} | KL: {aux[1]:.4f} | Cons: {aux[2]:.4f}", end="\r")

    if steps > 0:
        avg_loss = epoch_loss / steps
        avg_recon = epoch_recon / steps
        avg_kl = epoch_kl / steps
        avg_cons = epoch_cons / steps
        print(f"\nEpoch {epoch} complete. Avg Loss: {avg_loss:.4f} | Rec: {avg_recon:.4f} | KL: {avg_kl:.4f} | Cons: {avg_cons:.4f}")

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f}s")

# Save model
eqx.tree_serialise_leaves(RUN_FOLDER + "model.eqx", model)

#%%
# print(losses, r_losses, kl_losses, cons_losses)
plt.figure(figsize=(8, 4))
plt.plot(losses, label="Total Loss")
plt.plot(r_losses, label="Reconstruction Loss")
plt.plot(kl_losses, label="KL Loss")
plt.plot(cons_losses, label="Consistency Loss")
plt.xlabel("Train Steps")
plt.yscale("log")
plt.ylabel("Loss")
plt.legend()
plt.draw()
plt.savefig(RUN_FOLDER + "loss_curve.png")



#%%
# ==========================================
# 6. Visualization (Corrected)
# ==========================================

# 1. Prepare Data
# Assumes DataLoader returns (N, C, H, W)
images_tensor = next(iter(train_dataloader))
images_nchw = jnp.asarray(images_tensor.numpy())
images_nhwc = images_nchw.transpose(0, 2, 3, 1)  # For matplotlib (N, H, W, C)

subset_nhwc = images_nhwc[:4]
subset_nchw = images_nchw[:4]

# 2. Inference
# Returns: Recon, Latent(Noise), Enc_Features(Start of ODE), Dec_Features(End of ODE)
recon_x, z_latent, z_initial_enc, z_final_dec = jax.vmap(model)(subset_nchw)
recon_nhwc = recon_x.transpose(0, 2, 3, 1)

# 3. Random Generation Check
def vf_func_inference(t, y, args):
    t_vec = jnp.broadcast_to(t, (1,))
    inp = jnp.concatenate([t_vec, y], axis=-1)
    return model.vf_layers(inp)

key_gen = jax.random.PRNGKey(int(time.time()))
rand_z = jax.random.normal(key_gen, z_latent.shape)
rand_recon, _ = jax.vmap(lambda z: model.decoder(z, vf_func_inference))(rand_z)
rand_recon_nhwc = rand_recon.transpose(0, 2, 3, 1)

# 4. Plot 1: Generative Quality & Latent Stat
fig1, axs = plt.subplots(4, 4, figsize=(12, 12))
cols = ["Original", "Reconstruction", "Latent Z (Noise)", "Random Gen"]

for ax, col in zip(axs[0], cols):
    ax.set_title(col, fontsize=12, fontweight='bold')

for i in range(4):
    axs[i, 0].imshow(np.clip(subset_nhwc[i], 0, 1))
    axs[i, 0].axis('off')
    
    axs[i, 1].imshow(np.clip(recon_nhwc[i], 0, 1))
    axs[i, 1].axis('off')
    
    axs[i, 2].bar(range(LATENT_DIM), z_latent[i], color='gray', alpha=0.8, width=1.0)
    axs[i, 2].set_ylim([-4, 4])
    axs[i, 2].axhline(0, color='k', linewidth=0.5)
    axs[i, 2].axis('off')
    
    axs[i, 3].imshow(np.clip(rand_recon_nhwc[i], 0, 1))
    axs[i, 3].axis('off')

plt.tight_layout()
plt.savefig(RUN_FOLDER + "results_generative.png")
plt.show()

# 5. Plot 2: ODE Consistency (Manifold vs Recovered)
fig2, axs = plt.subplots(4, 2, figsize=(10, 10))
fig2.suptitle("ODE Boundary Check: Structure vs Consistency", fontsize=14)
axs[0, 0].set_title("Start of ODE (Encoder Output)", fontsize=11, color='darkblue')
axs[0, 1].set_title("End of ODE (Decoder Output)", fontsize=11, color='darkgreen')

for i in range(4):
    # Initial Encoding (Should be structured/spiky, not pure noise)
    axs[i, 0].bar(range(LATENT_DIM), z_initial_enc[i], color='royalblue', alpha=0.9, width=1.0)
    axs[i, 0].set_ylim([-4, 4])
    axs[i, 0].axhline(0, color='k', linewidth=0.5)
    axs[i, 0].set_ylabel(f"Sample {i}")
    
    # Final Decoding (Should match the Left plot if cycle consistent)
    axs[i, 1].bar(range(LATENT_DIM), z_final_dec[i], color='forestgreen', alpha=0.9, width=1.0)
    axs[i, 1].set_ylim([-4, 4])
    axs[i, 1].axhline(0, color='k', linewidth=0.5)
    axs[i, 1].set_yticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(RUN_FOLDER + "results_ode_consistency.png")
plt.show()

# ---------------------------------------------------------
# 5. Interpretation
# ---------------------------------------------------------
print("\n[Visual Interpretation Guide]")
print("1. Generative Plot (Gray Bars): These should look like random noise (centered at 0, unit variance).")
print("2. Consistency Plot (Blue/Green Bars):")
print("   - If Blue looks like Gray (Noise): Your CNN encoder is doing all the work; the ODE is Identity.")
print("   - If Blue looks Structured (Spikes/Patterns): Your ODE is successfully transforming manifold -> noise.")
print("   - If Blue != Green: The ODE integration is not invertible (Consistency Loss is too high or steps too low).")



#%%[markdown]
# ---------------------------------------------------
# 6. Next Big Idea
# ---------------------------------------------------

# - We have two independent vf_layers MLPs in Encoder and Decoder. But the decoder's is an EMA of the encoder's? (like JEPA). We enforce this via stop gradient as well. 

# - This should avoid needing a reconstruciton loss in pixel space, like Yann LeCun keeps suggesting.


#%%
## Small experiment to check tha the vector field is different from its initialisation
orig_model = VAE(img_size=IMG_SIZE, kernel_size=[3, 3], latent_dim=LATENT_DIM, key=model_key)
trained_model = model

print(trained_model.vf_layers.layers[0].weight.shape)

vf_diff = jnp.mean(jnp.abs(orig_model.vf_layers.layers[0].weight - trained_model.vf_layers.layers[0].weight))

print(f"Mean absolute difference in first VF layer weights: {vf_diff:.6f}")



# %%
#%%

