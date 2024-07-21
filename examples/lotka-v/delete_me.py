#%%


import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

import equinox as eqx
import diffrax
import jax
import jax.numpy as jnp
jax.print_environment_info()

data_size=2
key = jax.random.PRNGKey(0)

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self):
        self.mlp = eqx.nn.MLP(
            in_size=data_size+1,
            out_size=data_size,
            width_size=4,
            depth=2,
            activation=jax.nn.softplus,
            key=key,
        )

    def __call__(self, t, y, args):
        alpha = args[0]
        y = jnp.concatenate([y, alpha])
        return self.mlp(y)


class NeuralODE(eqx.Module):
    func: Func

    def __init__(self):
        self.func = Func()

    def __call__(self, ts, y0, alpha):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            args=(alpha,),
            adjoint=diffrax.DirectAdjoint(),
            max_steps=256*16*16,
            # adjoint=diffrax.RecursiveCheckpointAdjoint(),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

def loss_fn(model, alpha):
    ts = jnp.linspace(0, 1, 100)
    y0 = jnp.zeros(data_size)
    return jnp.mean(model(ts, y0, alpha) ** 2)


def inner_step(model, alpha):
    alpha_grad = eqx.filter_grad(lambda alpha, model: loss_fn(model, alpha))(alpha, model)
    return jnp.mean(alpha_grad)

def outer_step(model, alpha):
    model_grad = eqx.filter_grad(inner_step)(model, alpha)
    return model_grad




model = NeuralODE()
alpha = jnp.array([1.])

outer_step(model, alpha)

