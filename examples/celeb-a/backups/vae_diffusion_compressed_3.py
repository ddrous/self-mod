"""## Train a Shared-ODE VAE with Cycle Consistency and Early Stopping """

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
BATCH_SIZE = 128*8
EPOCHS = 3
PATIENCE = 200  # Stop if no improvement for this many steps
LR = 1e-4
IMG_SIZE = [64, 64, 3]
LATENT_DIM = 128
CONV_BASE_DEPTH = 32 

#%%
## Define keys
data_key, model_key, trainer_key, test_key = jax.random.split(MOTHER_KEY, num=4)

#%%
# ==========================================
# 1. Data Loading
# ==========================================

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, img_dir):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imgs_root = os.path.join(img_dir, "img_align_celeba/")

        try:
            partitions = pd.read_csv(os.path.join(img_dir, 'list_eval_partition.txt'), 
                                     header=None, 
                                     sep=r'\s+', 
                                     names=['filename', 'partition'])
            self.files = partitions[partitions['partition'] == 0]['filename'].values
        except:
            print("Warning: Partition file not found, using empty list")
            self.files = []

        self.transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((IMG_SIZE[0], IMG_SIZE[1]), Image.LANCZOS),
            transforms.ToTensor(), # Returns (C, H, W)
        ])

    def get_image(self, filename):
        img_path = os.path.join(self.imgs_root, filename)
        # Load directly to CPU tensor, conversion happens in DataLoader/JAX
        img = self.transform(img_path) 
        return img

    def __getitem__(self, index):
        return self.get_image(self.files[index])

    def __len__(self):
        return len(self.files)

# Initialize DataLoader
try:
    train_dataset = CelebaDataset(img_dir=DATA_FOLDER)
    train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True)
    print(f"Data Loaded: {len(train_dataset)} images")
except Exception as e:
    print(f"Data Loader failed: {e}")
    train_dataloader = []

#%%
# ==========================================
# 2. Helper Layers
# ==========================================

class Downsample2D(eqx.Module):
    factor: int
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, y):
        C, H, W = y.shape
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
# 3. Architecture: Split Compressor / Decompressor / VAE
# ==========================================

class Compressor(eqx.Module):
    """ Purely Spatial Compression (No ODE) """
    layers: list
    img_size: list
    latent_dim: int

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.latent_dim = latent_dim
        layer_keys = jax.random.split(key, 5)
        H, W, C = img_size

        self.layers = [
            eqx.nn.Conv2d(C, CONV_BASE_DEPTH, kernel_size, padding="SAME", key=layer_keys[0]),
            Downsample2D(factor=2),
            eqx.nn.PReLU(init_alpha=0.01),
            
            eqx.nn.Conv2d(CONV_BASE_DEPTH, CONV_BASE_DEPTH*2, kernel_size, padding="SAME", key=layer_keys[1]),
            Downsample2D(factor=2),
            eqx.nn.PReLU(init_alpha=0.01),
            
            eqx.nn.Conv2d(CONV_BASE_DEPTH*2, CONV_BASE_DEPTH*4, kernel_size, padding="SAME", key=layer_keys[2]),
            Downsample2D(factor=2), # 8x8 -> 64 * 4 channels
            eqx.nn.PReLU(init_alpha=0.01),
            
            lambda x: x.flatten(),
            eqx.nn.Linear(CONV_BASE_DEPTH*4*8*8, latent_dim, key=layer_keys[3]),
            # No final activation: these are the "structured features"
        ]

    def __call__(self, x):
        z = x
        for layer in self.layers:
            z = layer(z)
        return z

class Decompressor(eqx.Module):
    """ Purely Spatial Decompression (No ODE) """
    layers: list
    img_size: list
    latent_dim: int

    def __init__(self, img_size, kernel_size, latent_dim, key):
        self.img_size = img_size
        self.latent_dim = latent_dim
        layer_keys = jax.random.split(key, 5)
        H, W, C = img_size

        self.layers = [
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

    def __call__(self, z):
        x_recon = z
        for layer in self.layers:
            x_recon = layer(x_recon)
        return x_recon

class VAE(eqx.Module):
    compressor: Compressor
    decompressor: Decompressor
    vf_layers: eqx.Module

    def __init__(self, img_size, kernel_size, latent_dim, key):
        enc_key, dec_key, vf_key = jax.random.split(key, 3)
        
        self.compressor = Compressor(img_size, kernel_size, latent_dim, key=enc_key)
        self.decompressor = Decompressor(img_size, kernel_size, latent_dim, key=dec_key)
        
        # Shared Vector Field MLP
        # Input: Latent + 1 (time)
        self.vf_layers = eqx.nn.MLP(
            in_size=latent_dim + 1,
            out_size=latent_dim,
            width_size=256,
            depth=3,
            activation=jax.nn.silu, 
            key=vf_key
        )

    def encode(self, x):
        """ 
        1. Compress Image -> Features (z_enc)
        2. Solve ODE (Fwd) -> Noise (z_lat)
        """
        # 1. Compress
        z_enc = self.compressor(x)
        
        # 2. ODE Forward (Manifold -> Noise)
        def vf_func(t, y, args):
            t_vec = jnp.broadcast_to(t, (1,))
            inp = jnp.concatenate([t_vec, y], axis=-1)
            return self.vf_layers(inp)
        term = diffrax.ODETerm(vf_func)
        solver = diffrax.Dopri5()
        sol = diffrax.diffeqsolve(
            term, solver, t0=0, t1=1, dt0=0.1, y0=z_enc, max_steps=128
        )
        z_lat = sol.ys[-1]
        
        return z_enc, z_lat

    def decode_training(self, z_enc):
        """
        During Training: Decompress directly from Feature space.
        Does NOT use the ODE.
        """
        return self.decompressor(z_enc)

    def generate(self, z_noise):
        """
        During Inference: 
        1. Solve ODE (Reverse): Noise -> Features
        2. Decompress: Features -> Image
        """
        # 1. ODE Backward (Noise -> Manifold)
        def vf_func(t, y, args):
            t_vec = jnp.broadcast_to(t, (1,))
            inp = jnp.concatenate([t_vec, y], axis=-1)
            return self.vf_layers(inp)
        term = diffrax.ODETerm(vf_func)
        solver = diffrax.Dopri5()
        # Note: t=1 (Noise) -> t=0 (Manifold)
        sol = diffrax.diffeqsolve(
            term, solver, t0=1, t1=0, dt0=-0.1, y0=z_noise, max_steps=128
        )
        z_features = sol.ys[-1]
        
        # 2. Decompress
        img = self.decompressor(z_features)
        return img

    def __call__(self, x):
        """ Forward pass for training """
        # 1. Encode
        z_enc, z_lat = self.encode(x)
        
        # 2. Decode (Training path - bypasses ODE)
        x_rec = self.decode_training(z_enc)
        # x_rec = self.generate(z_lat)

        return x_rec, z_enc, z_lat

model = VAE(img_size=IMG_SIZE, kernel_size=[3, 3], latent_dim=LATENT_DIM, key=model_key)

# Print param counts
count = np.sum([p.size for p in jax.tree.leaves(model) if isinstance(p, jnp.ndarray)])
print(f"Total Parameters: {count}")

#%%
# ==========================================
# 4. Loss Function (Updated)
# ==========================================

def calculate_gaussian_kl(z):
    """ Calculates KL( N(mu, var) || N(0, 1) ) """
    mu = jnp.mean(z, axis=0)
    var = jnp.var(z, axis=0) + 1e-6
    logvar = jnp.log(var)
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_per_dim = -0.5 * (1 + logvar - mu**2 - var)
    return jnp.mean(kl_per_dim)

def mmd_loss_gaussian(z_batch, key, sigma=1.0):
    """ Maximum Mean Discrepancy between z_batch and N(0, 1) using IMQ kernel. """
    batch_size, n_features = z_batch.shape
    z_true = jax.random.normal(key, shape=z_batch.shape)
    
    def compute_kernel(x, y):
        C = 2.0 * n_features * sigma
        x_exp = jnp.expand_dims(x, 1)
        y_exp = jnp.expand_dims(y, 0)
        dists_sq = jnp.sum((x_exp - y_exp)**2, axis=-1)
        return C / (C + dists_sq)

    k_zz = compute_kernel(z_batch, z_batch)
    k_tt = compute_kernel(z_true, z_true)
    k_zt = compute_kernel(z_batch, z_true)
    
    return jnp.mean(k_zz) + jnp.mean(k_tt) - 2 * jnp.mean(k_zt)


def swd_loss_gaussian(z_batch, key, num_projections=50):
    """ Sliced Wasserstein Distance between z_batch and N(0, 1). """
    batch_size, n_features = z_batch.shape
    key_z, key_theta = jax.random.split(key)
    
    z_true = jax.random.normal(key_z, shape=z_batch.shape)
    theta = jax.random.normal(key_theta, shape=(n_features, num_projections))
    theta = theta / jnp.sqrt(jnp.sum(theta**2, axis=0, keepdims=True))
    
    proj_z = jnp.dot(z_batch, theta)       # (B, P)
    proj_true = jnp.dot(z_true, theta)     # (B, P)
    
    proj_z_sorted = jnp.sort(proj_z, axis=0)
    proj_true_sorted = jnp.sort(proj_true, axis=0)
    
    return jnp.mean((proj_z_sorted - proj_true_sorted)**2)
    

def loss_fn(model, xs, key):
    # keys = jax.random.split(key, xs.shape[0])

    # x_rec: Reconstructed image
    # z_enc: Compressed features (Input to ODE)
    # z_lat: Latent noise (Output of ODE)
    x_rec, z_enc, z_lat = jax.vmap(model)(xs)
    
    # 1. Reconstruction Loss (MSE)
    #    Strictly connects Compressor <-> Decompressor
    recon_loss = jnp.mean((xs - x_rec) ** 2)
    # recon_loss = -jnp.mean(xs * jnp.log(x_rec + 1e-6) + (1 - xs) * jnp.log(1 - x_rec + 1e-6))
    
    # 2. Latent KL (Minimize)
    #    Forces the END of the ODE (z_lat) to be Gaussian
    # kl_latent = calculate_gaussian_kl(z_lat)
    # kl_latent = swd_loss_gaussian(z_lat, key, num_projections=100)
    kl_latent = mmd_loss_gaussian(z_lat, key, sigma=1.0)
    
    # 3. Feature Negative KL (Minimize negative => Maximize Divergence)
    #    Forces the START of the ODE (z_enc) to be NON-Gaussian
    # kl_feature = calculate_gaussian_kl(z_enc)
    kl_feature = swd_loss_gaussian(z_enc, key, num_projections=100)
    consistency_loss = jax.nn.relu(1.0 - kl_feature) + 1e-4 ## offset is good for plotting 

    # consistency_loss = 1e-1

    # Weights
    w_recon = 1.0
    w_kl = 0.1          # Standard VAE weight
    w_consistency = 1.0 # Don't push too hard or it might explode
    
    total_loss = (w_recon * recon_loss) + (w_kl * kl_latent) + (w_consistency * consistency_loss)
    
    return total_loss, (recon_loss, kl_latent, consistency_loss)

@eqx.filter_jit
def train_step(model, xs, opt_state, key):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, xs, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux

optimizer = optax.adam(LR)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

#%%
# ==========================================
# 5. Training Loop with Early Stopping
# ==========================================

losses = []
aux_history = {"recon":[], "kl_lat":[], "neg_kl_feat":[], "total":[]}

# --- EARLY STOPPING STATE ---
best_train_loss = np.inf
steps_without_improvement = 0
stop_training = False
# -----------------------------

print(f"Starting training with Patience: {PATIENCE} steps...")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_loss = 0.
    epoch_aux = np.zeros(3)
    steps = 0
    
    if isinstance(train_dataloader, list):
         print("No data loader found.")
         break

    for i, batch in enumerate(train_dataloader):
        # Assumes Loader returns (N, C, H, W)
        images = jnp.asarray(batch.numpy())
        
        trainer_key, _ = jax.random.split(trainer_key)
        model, opt_state, loss, aux = train_step(model, images, opt_state, trainer_key)
        
        current_loss_val = loss.item()
        
        # --- EARLY STOPPING CHECK (Step Level) ---
        if current_loss_val < best_train_loss:
            best_train_loss = current_loss_val
            steps_without_improvement = 0
            # Save the best model immediately
            eqx.tree_serialise_leaves(RUN_FOLDER + "best_model.eqx", model)
        else:
            steps_without_improvement += 1
            
        if steps_without_improvement >= PATIENCE:
            print(f"\n\n[Early Stopping] No improvement for {PATIENCE} steps. Best Loss: {best_train_loss:.6f}")
            stop_training = True
            break
        # ------------------------------------------

        epoch_loss += loss
        epoch_aux += np.array(aux)
        steps += 1
        aux_history["recon"].append(aux[0])
        aux_history["kl_lat"].append(aux[1])
        aux_history["neg_kl_feat"].append(aux[2])
        aux_history["total"].append(loss)
        
        if i % 10 == 0:
            # print(f"Ep {epoch} | St {i} | L: {loss:.4f} | Best: {best_train_loss:.4f} | Pat: {steps_without_improvement}/{PATIENCE}", end="\r")
            print(f"Ep {epoch} | Step {i} | Loss(Total): {loss:.4f} | Best(Total): {best_train_loss:.4f} | Recons: {aux[0]:.4f} | KL(Lat): {aux[1]:.4f} | -KL(Feat): {aux[2]:.4f} | Pat: {steps_without_improvement}/{PATIENCE}", end="\r")

    if stop_training:
        break

    if steps > 0:
        avg_loss = epoch_loss / steps
        avg_aux = epoch_aux / steps
        losses.append(avg_loss)
        print(f"\nEpoch {epoch} Done. Loss: {avg_loss:.4f} | Rec: {avg_aux[0]:.4f} | KL(N): {avg_aux[1]:.4f} | -KL(F): {avg_aux[2]:.4f}")

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f}s")

# IMPORTANT: Load the best model back for visualization
print("Loading best model for visualization...")
model = eqx.tree_deserialise_leaves(RUN_FOLDER + "best_model.eqx", model)


#%%
# Plot Loss 
plt.figure(figsize=(10, 5))
# plt.plot(losses, label="Total", linewidth=2)
plt.plot(aux_history["total"], label="Total", linestyle="-", linewidth=2)
plt.plot(aux_history["recon"], label="Recon (MSE)", linestyle="--")
plt.plot(aux_history["kl_lat"], label="KL Latent (Min)", linestyle="--")
plt.plot(aux_history["neg_kl_feat"], label="Neg KL Feature (Min)", linestyle="--")
plt.yscale('symlog', linthresh=1e-3)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Dynamics")
plt.savefig(RUN_FOLDER + "loss_curve.png")
plt.show()

#%%
# ==========================================
# 6. Visualization
# ==========================================

# 1. Prepare Batch
try:
    images_raw = next(iter(train_dataloader))
    images_nchw = jnp.asarray(images_raw.numpy())
    images_nhwc = images_nchw.transpose(0, 2, 3, 1) # For Plotting
    
    subset_nchw = images_nchw[:4]
    subset_nhwc = images_nhwc[:4]
except:
    print("Viz data failed")
    subset_nchw = jnp.zeros((4, 3, 64, 64))

# 2. Inference (Training Mode)
# Gets reconstruction via "shortcut" (z_enc -> Decoder)
x_rec, z_enc, z_lat = jax.vmap(model)(subset_nchw)
x_rec_nhwc = x_rec.transpose(0, 2, 3, 1)

# 3. Generation (Inference Mode)
# Gets generation via ODE (Noise -> ODE -> Decoder)
key_gen = jax.random.PRNGKey(int(time.time()))
rand_noise = jax.random.normal(key_gen, z_enc.shape) # Match batch/dim
x_gen = jax.vmap(model.generate)(rand_noise)
x_gen_nhwc = x_gen.transpose(0, 2, 3, 1)

# 4. Plot Generative Results
fig1, axs = plt.subplots(4, 4, figsize=(12, 12))
cols = ["Original", "Recon (Shortcut)", "Latent (Final)", "Gen (From Rd Noise)"]

for ax, col in zip(axs[0], cols):
    ax.set_title(col, fontsize=12, fontweight='bold')

for i in range(4):
    # Original
    axs[i, 0].imshow(np.clip(subset_nhwc[i], 0, 1))
    axs[i, 0].axis('off')
    
    # Recon (Shortcut)
    axs[i, 1].imshow(np.clip(x_rec_nhwc[i], 0, 1))
    axs[i, 1].axis('off')
    
    # Latent (Should be Gaussian)
    axs[i, 2].bar(range(LATENT_DIM), z_lat[i], color='gray', alpha=0.8, width=1.0)
    axs[i, 2].set_ylim([-4, 4])
    axs[i, 2].axhline(0, color='k', linewidth=0.5)
    axs[i, 2].axis('off')
    
    # Generated (ODE)
    axs[i, 3].imshow(np.clip(x_gen_nhwc[i], 0, 1))
    axs[i, 3].axis('off')

plt.tight_layout()
plt.savefig(RUN_FOLDER + "results_generative.png")
plt.show()

# 5. Plot ODE Work (Feature vs Noise)
fig2, axs = plt.subplots(4, 2, figsize=(10, 10))
fig2.suptitle("ODE Boundary Check: Features vs Noise", fontsize=14)
axs[0, 0].set_title("Features (Compressed Input)\nGoal: Non-Gaussian", fontsize=11, color='darkblue')
axs[0, 1].set_title("Latent (ODE Output)\nGoal: Gaussian", fontsize=11, color='gray')

for i in range(4):
    # Features (z_enc)
    axs[i, 0].bar(range(LATENT_DIM), z_enc[i], color='royalblue', alpha=0.9, width=1.0)
    mu_e, std_e = jnp.mean(z_enc[i]), jnp.std(z_enc[i])
    axs[i, 0].text(0, -5, f"$\mu$={mu_e:.2f}, $\sigma$={std_e:.2f}", fontsize=9)
    axs[i, 0].set_ylim([-6, 6]) # Wider range likely
    axs[i, 0].axhline(0, color='k', linewidth=0.5)
    
    # Latent (z_lat)
    axs[i, 1].bar(range(LATENT_DIM), z_lat[i], color='gray', alpha=0.9, width=1.0)
    mu_l, std_l = jnp.mean(z_lat[i]), jnp.std(z_lat[i])
    axs[i, 1].text(0, -3.5, f"$\mu$={mu_l:.2f}, $\sigma$={std_l:.2f}", fontsize=9)
    axs[i, 1].set_ylim([-4, 4])
    axs[i, 1].axhline(0, color='k', linewidth=0.5)
    axs[i, 1].set_yticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(RUN_FOLDER + "results_ode_work.png")
plt.show()
