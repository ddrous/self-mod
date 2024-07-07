from ._utils import *



class Learner:
    def __init__(self, model, env_loss_fn, context_size, context_pool_size, key=None):
        if key is None:
            raise ValueError("You must provide a key for the learner.")
        self.key = key

        self.model = model
        self.context_size = context_size
        self.context_pool_size = context_pool_size

        def env_loss_fn_(model, batch, ctx, ctxs, key):
            """ Wrapping the loss function before vectorizing it below """
            X, Y = batch

            ind = jax.random.permutation(key, ctxs.shape[0])[:self.context_pool_size]
            ctx_pool = ctxs[ind, :]

            Y_hat = jax.vmap(model, in_axes=(None, None, 0))(X, ctx, ctx_pool)
            Y_new = jnp.broadcast_to(Y, Y_hat.shape)

            return env_loss_fn(model, ctx, Y_new, Y_hat)

        def loss_fn(model, contexts, batch, weightings, key):
            keys = jax.random.split(key, num=contexts.params.shape[0])

            losses, (term1, terms2, terms3) = jax.vmap(env_loss_fn_, in_axes=(None, 0, 0, None, 0))(model, batch, contexts.params, contexts.params, keys)

            return jnp.sum(losses*weightings), (term1, terms2, terms3)

        self.loss_fn = loss_fn              ## Meta loss function
        self.env_loss_fn = env_loss_fn_      ## Base loss function

    def save_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalid path provided. The path must end with /"
        eqx.tree_serialise_leaves(path+"model.eqx", self.model)
        if hasattr(self, "contexts"):
            eqx.tree_serialise_leaves(path+"contexts.eqx", self.contexts)

    def load_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"
        self.model = eqx.tree_deserialise_leaves(path+"model.eqx", self.model)
        if hasattr(self, "contexts"):
            self.contexts = eqx.tree_deserialise_leaves(path+"contexts.eqx", self.contexts)







class MLP(eqx.Module):
    """ An MLP """
    layers: jnp.ndarray

    def __init__(self, in_size, out_size, hidden_size, depth, activation, key=None):
        keys = jax.random.split(key, num=depth+1)

        self.layers = []

        for i in range(depth):
            if i==0:
                layer = eqx.nn.Linear(in_size, hidden_size, use_bias=True, key=keys[i])
            elif i==depth-1:
                layer = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=keys[i])
            else:
                layer = eqx.nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[i])

            self.layers.append(layer)

            if i != depth-1:
                self.layers.append(activation)

    def __call__(self, x):
        """ Returns y such that y = MLP(x) """
        y = x
        for layer in self.layers:
            y = layer(y)
        return y





class ArrayContextParams(eqx.Module):
    params: jnp.ndarray
    def __init__(self, nb_envs, context_size):
        self.params = jnp.zeros((nb_envs, context_size))
    def __call__(self):
        return self.params






class NeuralContextFlow(eqx.Module):
    neuralnet: eqx.Module
    taylor_order: int

    def __init__(self, neuralnet, taylor_order):
        self.neuralnet = neuralnet
        self.taylor_order = taylor_order

    def __call__(self, xs, ctx, ctx_):

        def point_predict(x):

            vf = lambda xi: self.neuralnet(x, xi)

            if self.taylor_order==0:
                return vf(ctx)

            elif self.taylor_order==1:
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                return vf(ctx_) + 1.0*gradvf(ctx_)

            elif self.taylor_order==2:
                gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
                scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
                return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

            else:
                raise NotImplementedError("Higher order terms are not implemented yet.")

        ys = eqx.filter_vmap(point_predict)(xs)

        return ys





class SelfModVectorField(eqx.Module):
    neuralnet: eqx.Module
    taylor_order: int

    def __init__(self, neuralnet, taylor_order):
        self.neuralnet = neuralnet
        self.taylor_order = taylor_order

    def __call__(self, t, x, args):
        ctx, ctx_ = args

        vf = lambda xi: self.neuralnet(x, xi)

        if self.taylor_order==0:
            return vf(ctx)

        elif self.taylor_order==1:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            return vf(ctx_) + 1.0*gradvf(ctx_)

        elif self.taylor_order==2:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
            return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

        else:
            raise NotImplementedError("Higher order terms are not implemented yet.")



class NeuralODE(eqx.Module):
    vectorfield: eqx.Module
    ivp_args: dict
    taylor_order: int

    def __init__(self, neuralnet, taylor_order, ivp_args=None):
        self.ivp_args = ivp_args if ivp_args is not None else {}
        self.vectorfield = SelfModVectorField(neuralnet, taylor_order=taylor_order)
        self.taylor_order = taylor_order

    def __call__(self, xs, ctx, ctx_):

        def integrate(y0):
            sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(self.vectorfield),
                    self.ivp_args.get("integrator", diffrax.Dopri5()),
                    args=(ctx, ctx_.squeeze()),
                    t0=0.,
                    t1=self.ivp_args.get("T", 1.),
                    dt0=self.ivp_args.get("dt_init", 1e-2),
                    y0=jnp.concat([y0, jnp.zeros((self.ivp_args.get("y0_pad_size", 1),))], axis=0),
                    stepsize_controller=diffrax.PIDController(rtol=self.ivp_args.get("rtol", 1e-3), 
                                                                atol=self.ivp_args.get("atol", 1e-6)),
                    # saveat=diffrax.SaveAt(ts=t_eval),
                    adjoint=self.ivp_args.get("adjoint", diffrax.RecursiveCheckpointAdjoint()),
                    max_steps=self.ivp_args.get("max_steps", 4096*1)
                )
            return sol.ys[-1]

        return jax.vmap(integrate)(xs)






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


class VAEDecoder(eqx.Module):
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
            eqx.nn.Linear(latent_dim, 1024, key=layer_keys[0]),
            eqx.nn.PReLU(init_alpha=0.),
            eqx.nn.Linear(1024, 64*H*W//(4*4), key=layer_keys[1]),
            eqx.nn.PReLU(init_alpha=0.),
            lambda x: x.reshape((64, H//4, W//4)),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(64, 16, kernel_size, padding="SAME", key=layer_keys[2]),
            eqx.nn.PReLU(init_alpha=0.),
            Upsample2D(factor=2),
            eqx.nn.ConvTranspose2d(16, C, kernel_size, padding="SAME", key=layer_keys[3]),
            jax.nn.sigmoid
        ]

    def __call__(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        return x



class FuncContextParams(eqx.Module):
    params: list
    img_size: list
    kernel_size: list
    latent_dim: int
    context_size: int

    ctx_utils: any

    def __init__(self, nb_envs, key=None):

        keys = jax.random.split(key, num=nb_envs)

        # all_contexts = [Decoder(img_size=[32, 32, 3], kernel_size=[3, 3], latent_dim=18, key=keys[i]) for i in range(nb_envs)]

        ## Load the decoders from 240101-193230-VAE/decoder.eqx
        all_contexts = [eqx.tree_deserialise_leaves("runs/240101-193230-VAE/decoder.eqx", Decoder(img_size=[32, 32, 3], kernel_size=[3, 3], latent_dim=18, key=keys[i])) for i in range(nb_envs)]

        self.img_size = all_contexts[0].img_size
        self.kernel_size = all_contexts[0].kernel_size
        self.latent_dim = all_contexts[0].latent_dim

        ex_params, ex_static = eqx.partition(all_contexts[0], eqx.is_array)
        ex_ravel, ex_shapes, ex_treedef = flatten_pytree(ex_params)
        # self.ctx_utils = (ex_shapes, ex_treedef, ex_static)
        non_empty_shapes = []
        for shape in ex_shapes:
            if shape == ():
                non_empty_shapes.append((1,))
            else:
                non_empty_shapes.append(shape)
        self.ctx_utils = (non_empty_shapes, ex_treedef, ex_static)

        all_params_1D = [flatten_pytree(eqx.filter(context, eqx.is_array))[0] for context in all_contexts]
        self.context_size = all_params_1D[0].shape[0]

        self.params = jnp.stack(all_params_1D, axis=0)
        # self.params = jnp.zeros_like(jnp.stack(all_params_1D, axis=0))


    def __call__(self, z):
        def unravel_and_call(ctx, z):
            context = jax.flatten_util.unravel_pytree(ctx, self.treedef)
            return context(z)
        return jax.vmap(unravel_and_call)(self.params, z)
