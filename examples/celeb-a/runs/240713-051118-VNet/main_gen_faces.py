#%%
# %load_ext autoreload
# %autoreload 2

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

from selfmod import *


#%%

## For reproducibility
seed = 2028

## Dataloader hps
resolution = (32, 32)
img_size = (3, resolution[0], resolution[1])

## Learner/model hps
context_size = 256
nb_images = 8*8

run_folder = "./analysis/"



#%%


## Define model and loss function for the learner
class MultiCNN(eqx.Module):
    layers_context: list
    vnet: eqx.Module
    taylor_weight: jnp.ndarray

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
                        levels=3,
                        depth=vnet_base_chans,
                        kernel_size=3,
                        activation=eqx.nn.PReLU(init_alpha=0.),
                        final_activation=jax.nn.sigmoid,
                    #   final_activation=lambda x:x,
                        batch_norm=False,
                        dropout_rate=0.,
                        key=keys[3]
                    )

        self.taylor_weight = jnp.array([0.])        ## We start with full power to the Taylor expansion !

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
model = eqx.tree_deserialise_leaves("model.eqx", model)


#%%



## Define model and loss function for the learner
class Generator(eqx.Module):
    contextnet: eqx.Module
    vnet: eqx.Module

    def __init__(self, model):
        self.contextnet = model.neuralnet.layers_context
        self.vnet = model.neuralnet.vnet

    def __call__(self, noise):
        ctx = noise
        for layer in self.contextnet:
            ctx = layer(ctx)

        return self.vnet(ctx)

generator = Generator(model)


#%%

## Generate random context of size context_size
mother_key, _ = jax.random.split(mother_key)
context = jax.random.normal(mother_key, (nb_images, context_size))

images = eqx.filter_vmap(generator)(context)
images = jnp.transpose(images, axes=(0,2,3,1))

## Remove the blue channel
images = images.at[:,:,:,2].set(0.)
images = images.at[:,:,:,1].set(0.)


## Visualise the generated images
sq_nb_images = int(np.sqrt(nb_images))
plt.style.use("ggplot")

fig, ax = plt.subplots(sq_nb_images, sq_nb_images, figsize=(3*sq_nb_images, 3*sq_nb_images))

# images = jnp.ones_like(images)
for i in range(sq_nb_images):
    for j in range(sq_nb_images):
        ax[i,j].imshow(images[i*sq_nb_images+j])
        ax[i,j].axis("off")

        ## Put boundary between axis
        ax[i,j].spines['left'].set_color('white')

## Save the figure
plt.savefig(run_folder+"generated_faces.png", bbox_inches='tight')


#%%

## Load the plot the file vnet.png
from PIL import Image
img = Image.open("../../vnet.png")
## downscale to 32x64x4
img = img.resize((64, 32))


img = np.array(img) / 255.
img = img[:32, :32, :3]
print(img)

plt.imshow(img)
plt.axis("off")















