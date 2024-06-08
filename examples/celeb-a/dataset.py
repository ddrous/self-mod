#%%
import torch
from torchvision.transforms import transforms
import jax.numpy as jnp
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

## Parse the three arguments from the command line: "train", the foldername, and the seed

import argparse


if _in_ipython_session:
	args = argparse.Namespace(split='adapt', savepath="./data/", resolution=32, seed=2026, train_pixels=100, order_pixels=0, verbose=1)
else:
	parser = argparse.ArgumentParser(description='CelebA preprocessing script.')
	parser.add_argument('--split', type=str, help='Generate "train", "test", "adapt", "adapt_test", or "adapt_huge" data', default='train', required=False)
	parser.add_argument('--savepath', type=str, help='Description of optional argument', default='data/', required=False)
	parser.add_argument('--seed',type=int, help='Seed to gnerate the data', default=2026, required=False)
	parser.add_argument('--resolution', type=int, help='Transform the images to shape (res, res, 3)', default=32, required=False)
	parser.add_argument('--train_pixels',type=int, help='Number of training pixels per image (if training)', default=10, required=False)
	parser.add_argument('--order_pixels', type=int, help='Whether to order the training pixels', default=0, required=False)
	parser.add_argument('--verbose',type=int, help='Whether to print details or not ?', default=1, required=False)

	args = parser.parse_args()


split = args.split
assert split in ["train", "test", "adapt", "adapt_test"], "Invalid split. Choose from 'train', 'test', 'adapt', 'adapt_test'"

savepath = args.savepath
seed = args.seed
resolution = args.resolution
img_size = (resolution, resolution, 3)
train_pixels = args.train_pixels
order_pixels = args.order_pixels
verbose = args.verbose

if args.verbose != 0:
  print("Running this script in ipython (Jupyter) session ?", _in_ipython_session)
  print('=== Parsed arguments to generate data ===')
  print(' Split:', split)
  print(' Savepath:', savepath)
  print(' Seed:', seed)
  print(' Image size:', img_size)
  print(' Train pixels:', train_pixels)
  print(' Order pixels:', order_pixels)
  print(' Verbose:', verbose)
  print('=========================================')


## Set numpy seed for reproducibility
np.random.seed(seed)


#%%


## Read list_eval_partiions.txt to get the train(0), val(1), test(2) splits
partitions = pd.read_csv(savepath+'/list_eval_partition.txt', header=None, sep=r'\s+', names=['filename', 'partition'])

# if split == "train":    ## Train set
#   files = partitions[partitions['partition'] == 0]['filename'].values
# elif split == "test":   ## Test set
#   files = partitions[partitions['partition'] == 2]['filename'].values
# else:                   ## Validation set (Never used at this point)
#   files = partitions[partitions['partition'] == 1]['filename'].values
# assert len(files) > 0, "No files found for the split"


if split in ["train", "test"]:              ## meta-training set (masked and full pixels)
  files = partitions[partitions['partition'] == 0]['filename'].values
elif split in ["adapt", "adapt_test"]:      ## meta-testing set (masked and full pixels)
  files = partitions[partitions['partition'] == 2]['filename'].values
# else:                                       ## validation set as defined by celebA (Never used at this point)
#   files = partitions[partitions['partition'] == 1]['filename'].values
    ## Error
assert len(files) > 0, "No files found for the split"


## Transformations of the images

imgs_root = './data/img_align_celeba'
transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                      transforms.Resize((img_size[0], img_size[1]), Image.LANCZOS),
                                      transforms.ToTensor(),
                                      ])
def get_image(filename):
    img_path = os.path.join(imgs_root, filename)
    img = transform(img_path).float()
    # img = img * 2 - 1
    img = img.permute(1, 2, 0)
    return img

def sample_pixels(img, nb_train_pixels, order_pixels):
    if order_pixels:
        flattened_indices = list(range(img_size[0] * img_size[1]))[:nb_train_pixels]
    else:
        flattened_indices = np.random.choice(list(range(img_size[0] * img_size[1])), nb_train_pixels, replace=False)

    x, y = np.unravel_index(flattened_indices, (img_size[0], img_size[1]))
    coordinates = np.vstack((x, y)).T
    coords = torch.from_numpy(coordinates).float()

    pixel_values = img[coords[:, 0].long(), coords[:, 1].long(), :]

    # normalise coordinates
    # coords[:, 0] /= img_size[0]
    # coords[:, 1] /= img_size[1]

    return coords, pixel_values



## Create and fill the dataset tensor
nb_envs = len(files)

if split == "train" or split == "adapt":
  nb_points_per_env = train_pixels
elif split == "test" or split == "adapt_test":
  nb_points_per_env = img_size[0] * img_size[1]
inputs_dim = 2
outputs_dim = 3
X = np.zeros((nb_envs, nb_points_per_env, inputs_dim))
Y = np.zeros((nb_envs, nb_points_per_env, outputs_dim))


if verbose:
  print(f"Creating dataset with {nb_envs} images and {nb_points_per_env} pixels per image ...")

for env, imgname in enumerate(files):
    img = get_image(imgname)
    coords, pixel_values = sample_pixels(img, nb_points_per_env, order_pixels)
    X[env, :, 0] = coords[:, 0] / img_size[0]
    X[env, :, 1] = coords[:, 1] / img_size[1]
    Y[env, :, :] = pixel_values

    if verbose and (env%100==0 or env==nb_envs-1) == 0:
      print(f"Processed {env+1}/{nb_envs} images", end='\r')





# Save t_eval and the solution to a npz file
if split == "train":
  filename = savepath+'train_data.npz'
elif split == "test":
  filename = savepath+'test_data.npz'
elif split == "adapt":
  filename = savepath+'adapt_train.npz'
elif split == "adapt_test":
  filename = savepath+'adapt_test.npz'

## Check if nan or inf in data
if np.isnan(X).any() or np.isinf(X).any() or np.isnan(Y).any() or np.isinf(Y).any():
  print("NaN or Inf in data. Exiting without saving...")
else:
  np.savez(filename, X=X, Y=Y)
  print(f"Saved data to {filename}")




#%%


## Set seaborn style
import seaborn as sns
sns.set_theme(style='ticks', palette='muted', font_scale=1.5, context='paper')


## Visualize one image at random
if _in_ipython_session and verbose:

    idx = np.random.randint(0, nb_envs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ## Display the few shot pixels in a blank image
    blank_img = np.zeros(img_size)
    x_coords = (X[idx, :, 0] * img_size[0]).astype(int)
    y_coords = (X[idx, :, 1] * img_size[1]).astype(int)
    blank_img[x_coords, y_coords, :] = Y[idx, :, :]
    ax1.imshow(blank_img, extent=[0, img_size[1], img_size[0], 0])
    ax1.set_title('Few-shot pixels', fontsize=20)

    ## Display the full image
    full_img = get_image(files[idx]).numpy()
    ax2.imshow(full_img, extent=[0, img_size[1], img_size[0]+1, 0])
    ax2.set_title('All pixels', fontsize=20)

    plt.tight_layout()
    plt.show()

# %%
