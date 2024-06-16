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

            Y_hat, _ = jax.vmap(model, in_axes=(None, None, 0))(X, ctx, ctx_pool)
            Y_new = jnp.broadcast_to(Y, Y_hat.shape)

            return env_loss_fn(model, ctx, Y_new, Y_hat)

        def loss_fn(model, contexts, batch, weightings, key):
            losses, (_, terms1, terms2) = jax.vmap(env_loss_fn_, in_axes=(None, 0, 0, None, None))(model, batch, contexts.params, contexts.params, key)
            return jnp.sum(losses*weightings), (None, terms1, terms2)

        self.loss_fn = loss_fn

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
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]

            if self.taylor_order==0:
                return vf(ctx)

            elif self.taylor_order==1:
                return vf(ctx_) + 1.0*gradvf(ctx_)

            elif self.taylor_order==2:
                return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

            else:
                raise NotImplementedError("Higher order terms are not implemented yet.")

        ys = eqx.filter_vmap(point_predict)(xs)

        return ys, None
