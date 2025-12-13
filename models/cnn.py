import jax
import jax.numpy as jnp
from flax import nnx

from typing import Callable, List, Tuple

class FwdCNN(nnx.Module):
    def __init__(
        self,
        in_features: int,
        latent: int,
        kernel_size: int,
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
        self.blocks = nnx.List([nnx.Conv(latent, latent, (kernel_size, kernel_size), padding='CIRCULAR', rngs=rngs) for _ in range(n_blocks)])
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
        in_features: int,
        out_features: int,
        kernel_size: List[int], 
        rngs,
    ):
        self.conv = nnx.Conv(in_features, out_features, kernel_size, padding='CIRCULAR', rngs=rngs)
        self.lin_1 = nnx.Linear(out_features, out_features, rngs=rngs)
        self.act = activation
        self.lin_2 = nnx.Linear(out_features, out_features, rngs=rngs)
        
    def __call__(self, x):
        x_id = x
        x = self.conv(x)
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)
        x += x_id
        return x

class ResCNN(nnx.Module):
    def __init__(
        self,
        in_features: int,
        latent: int,
        kernel_size: int,
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
        self.blocks = nnx.List([ResBlock(activation, latent, latent, (kernel_size, kernel_size), rngs=rngs) for _ in range(n_blocks)])
        self.tail = nnx.Linear(latent, out_features, rngs=rngs)
    
    def __call__(self, x):
        x = (x - self.means) / self.stds
        x = self.head(x)
        for block in self.blocks:
            x = block(x)
        x = self.tail(x)
        return x
