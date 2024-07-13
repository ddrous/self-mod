#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

from selfmod import *


#%%

## For reproducibility
seed = 2028

## Dataloader hps
resolution = (32, 32)
img_size = (3, resolution[0], resolution[1])

## Learner/model hps
context_size = 256
nb_images = 6*6

# run_folder = "./runs/240713-130822/"
# run_folder = "./runs/240713-134917-GoldenT0/"
run_folder = "./runs/240713-143455/"


#%%


## Define model and loss function for the learner
class MultiCNN(eqx.Module):
    layers_context: list
    vnet: eqx.Module

    def __init__(self, kernel_size, hidden_chans, vnet_base_chans, context_size, key=None):
        keys = jax.random.split(key, num=4)

        self.layers_context = [eqx.nn.Linear(context_size, context_size*2, key=keys[0]),
                                eqx.nn.PReLU(init_alpha=0.),
                                eqx.nn.Linear(context_size*2, np.prod(resolution), key=keys[0]),
                                eqx.nn.PReLU(init_alpha=0.),
                                lambda x: x.reshape((1, resolution[0], resolution[1])),
                                ]

        ## The VNet to process the context
        self.vnet = VNet(input_shape=(1, *resolution),
                        output_shape=(3, *resolution),
                        levels=4,
                        depth=vnet_base_chans,
                        kernel_size=3,
                        activation=eqx.nn.PReLU(init_alpha=0.),
                        final_activation=jax.nn.sigmoid,
                    #   final_activation=lambda x:x,
                        batch_norm=False,
                        dropout_rate=0.,
                        key=keys[3]
                    )

    def __call__(self, x, ctx):

        ctx = ctx
        for layer in self.layers_context:
            ctx = layer(ctx)

        y = ctx
        y = self.vnet(y)

        x_coords = (x[:,0]*resolution[0]).astype(int)
        y_coords = (x[:,1]*resolution[1]).astype(int)
        rgbs = jnp.transpose(y, axes=(1,2,0))[x_coords, y_coords, :]

        return rgbs


mother_key = jax.random.PRNGKey(seed)
neuralnet = MultiCNN(kernel_size=(3,3),
                     hidden_chans=6,
                     vnet_base_chans=16, 
                     context_size=context_size, 
                     key=mother_key)

model = NonBatchedNeuralContextFlow(neuralnet=neuralnet, taylor_order=2)

model_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)) if x is not None)
print("\n\nTotal number of parameters in the model:", model_params)



## Load trained from model.eqx
model = eqx.tree_deserialise_leaves(run_folder+"model.eqx", model)


#%%



## Define model and loss function for the learner
class Generator(eqx.Module):
    contextnet: eqx.Module
    vnet: eqx.Module
    alpha:float

    def __init__(self, model):
        self.contextnet = model.neuralnet.layers_context
        self.vnet = model.neuralnet.vnet

        self.alpha = jax.nn.sigmoid(model.taylor_scale*model.taylor_weight[0])

    def __call__(self, noise):
        ctx = noise
        for layer in self.contextnet:
            ctx = layer(ctx)

        return (1-self.alpha)*self.vnet(ctx)

generator = Generator(model)


#%%

## Generate random context of size context_size
mother_key, _ = jax.random.split(mother_key)
# contexts = jax.random.normal(mother_key, (nb_images, context_size))


## Load contexts from run_folder/contexts/file.npy
context_folder = run_folder+"contexts/"
context_files = os.listdir(context_folder)
context_files = [context_folder+file for file in context_files]

# context = np.load(context_files[0])
# context

contexts = []
for file in context_files:
    context = np.load(file)
    contexts.append(context)

# print("Number of batched contexts loaded:", len(contexts))

contexts = np.concatenate(contexts, axis=0)
print("Total number of contexts loaded, and dimension:", contexts.shape)

##### Randomly pick nb_images contexts  #####
# contexts = contexts[:nb_images]
# contexts = jax.random.permutation(mother_key, contexts)[:nb_images]

##### Perform a linear combination of all existing contexts #####
# weights = jax.random.normal(mother_key, (nb_images, contexts.shape[0]))
# contexts = weights @ contexts

##### Transform the contexts in a 2D plot using t-SNE   #####
# tsne = TSNE(n_components=3, random_state=seed)
# contexts_plot = tsne.fit_transform(contexts)
# title_plot = "t-SNE plot of the contexts"

##### Transform the contexts in a 2D plot using PCA   #####
pca = PCA(n_components=3)
contexts_plot = pca.fit_transform(contexts)
title_plot = "PCA plot of the contexts"

#%%

## 2D plot of the contexts
# fig, ax = plt.subplots(1,1, figsize=(8,8))
# ax.scatter(contexts_plot[:,0], contexts_plot[:,1], s=1)
# ax.set_title(title_plot)


## 3D plot of the contexts
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(contexts_plot[:,0], contexts_plot[:,1], contexts_plot[:,2], s=1)
# ax.set_title(title_plot)

## 3D interactive plot of the contexts
fig = px.scatter_3d(x=contexts_plot[:,0], y=contexts_plot[:,1], z=contexts_plot[:,2], title=title_plot)
fig.update_layout(width=400, height=400)
fig.update_traces(marker=dict(size=1))

fig.show()


#%%

## Normalise the contexts into a standard normal distribution
scaler = StandardScaler()
contexts = scaler.fit_transform(contexts)

print("Mean of the contexts:", contexts.mean())
print("Standard deviation of the contexts:", contexts.std())


## Generate new gaussian samples and rescale them
# mother_key, _ = jax.random.split(mother_key)
contexts = jax.random.normal(mother_key, (nb_images, context_size))
contexts = scaler.inverse_transform(contexts)


#%%

# contexts = contexts.at[0, :200].set(0.)
# print(contexts[1])

images = eqx.filter_vmap(generator)(contexts)
images = jnp.transpose(images, axes=(0,2,3,1))

# print(images[0].max())


## Remove the blue channel
# images = images.at[:,:,:,2].set(0.)
# images = images.at[:,:,:,1].set(0.)


## Visualise the generated images
sq_nb_images = int(np.sqrt(nb_images))
# plt.style.use("ggplot")

fig, ax = plt.subplots(sq_nb_images, sq_nb_images, figsize=(2*sq_nb_images, 2*sq_nb_images))

# images = jnp.ones_like(images)
for i in range(sq_nb_images):
    for j in range(sq_nb_images):
        ax[i,j].imshow(images[i*sq_nb_images+j])
        ax[i,j].axis("off")

        ## Put boundary between axis
        ax[i,j].spines['left'].set_color('white')
plt.tight_layout()


## Save the figure
# save_path = run_folder+"generated_faces.png"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

plt.savefig(run_folder+"generated_faces.png", bbox_inches='tight')


#%%

# ## Load the plot the file vnet.png
# from PIL import Image
# img = Image.open("../../vnet.png")
# ## downscale to 32x64x4
# img = img.resize((64, 32))


# img = np.array(img) / 255.
# img = img[:32, :32, :3]
# print(img)

# plt.imshow(img)
# plt.axis("off")















