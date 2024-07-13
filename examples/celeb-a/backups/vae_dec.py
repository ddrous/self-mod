import jax
import jax.numpy as jnp
import equinox as eqx


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
            eqx.nn.Linear(latent_dim, 48, key=layer_keys[0]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(48, 12*H*W//(4*4), key=layer_keys[1]),
            eqx.nn.PReLU(init_alpha=0.),
            lambda x: x.reshape((12, H//4, W//4)),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(12, 8, kernel_size, padding="SAME", key=layer_keys[2]),
            eqx.nn.PReLU(init_alpha=0.),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(8, C, kernel_size, padding="SAME", key=layer_keys[3]),
            jax.nn.sigmoid
        ]

    def __call__(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        return x