import jax
import jax.numpy as jnp
from flax import nnx

from typing import Callable

class FwdMLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        latent: int,
        out_features: int,
        n_blocks: int,
        means: jnp.ndarray,
        stds: jnp.ndarray,
        activation: Callable,
        rngs
    ):
        self.means = nnx.Variable(means)
        self.stds = nnx.Variable(stds)

        self.act = activation
        self.head = nnx.Linear(in_features, latent, rngs=rngs)
        self.blocks = nnx.List([nnx.Linear(latent, latent, rngs=rngs) for _ in range(n_blocks)])
        self.tail = nnx.Linear(latent, out_features, rngs=rngs)
    
    def __call__(self, x):
        x = (x - self.means) / self.stds
        x = self.head(x)
        for block in self.blocks:
            x = block(x)
            x = self.act(x)
        x = self.tail(x)
        return x

class ResBlock(nnx.Module):
    def __init__(
        self, 
        activation: Callable,
        features: int,
        rngs,
    ):
        self.lin_l = nnx.Linear(features, features, rngs=rngs)
        self.lin_r = nnx.Linear(features, features, rngs=rngs)
        self.act = activation
        
    def __call__(self, x):
        x_id = x
        x = self.lin_l(x)
        x = self.act(x)
        x = self.lin_r(x)
        x += x_id
        return x

class ResMLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        latent: int,
        out_features: int,
        n_blocks: int,
        means: jnp.ndarray,
        stds: jnp.ndarray,
        activation: Callable,
        rngs
    ):
        self.means = nnx.Variable(means)
        self.stds = nnx.Variable(stds)

        self.head = nnx.Linear(in_features, latent, rngs=rngs)
        self.blocks = nnx.List([ResBlock(activation, latent, rngs=rngs) for _ in range(n_blocks)])
        self.tail = nnx.Linear(latent, out_features, rngs=rngs)
    
    def __call__(self, x):
        x = (x - self.means) / self.stds
        x = self.head(x)
        for block in self.blocks:
            x = block(x)
        x = self.tail(x)
        return x
