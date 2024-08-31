
#%%
%load_ext autoreload
%autoreload 2
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
from selfmod import *
jax.config.update("jax_debug_nans", True)

## Make sure jax prints all elements of a large array
# jax.config.update("jax_array_printer_threshold", 1000)
np.set_printoptions(threshold=10000)

key = jax.random.PRNGKey(time.time_ns())
res = (32, 32)

scene = init_gaussians(key, res, 1000)
scene = scene + 0.001
image = render_image(scene, res)

print("has Nan", jnp.any(jnp.isnan(image)))
print("All image", image)

fig, (ax) = plt.subplots(1, 1, figsize=(6, 6))
sbimshow(image, title="Random init", ax=ax)
# sbimshow(ref_image, title="Reference", ax=ax[1])




#%%
## Check if pos sem def
# mat = jnp.array([[ 0.00104693, -0.00402184], [-0.00402184,  0.02340409]])
# print("mat\n", mat)
# print("eig", jnp.linalg.eigh(mat))
# print("Deternminant", jnp.linalg.det(mat))


10. * max(2, 2) * jnp.finfo(jnp.float32).eps
# jnp.finfo(jnp.float64).eps
# jnp.finfo?