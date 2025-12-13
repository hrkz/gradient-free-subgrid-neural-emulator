import h5py
import numpy as np
import jax
import jax.numpy as jnp

from typing import Callable, Optional, Tuple

from models.time_solver import OdeScheme

class L96:
    def __init__(
        self,
        b: float,
        c: float,
        h: float,
        n_k: int,
        n_j: int,
    ):
        self.b = b
        self.c = c
        self.h = h

        self.n_k = n_k
        self.n_j = n_j

    def save(self, filename: str, time: float, x_k: np.ndarray, y_j: np.ndarray):
        with h5py.File(filename, 'w') as f:
            f.attrs['b'] = self.b
            f.attrs['c'] = self.c
            f.attrs['h'] = self.h
            
            f.attrs['n_k'] = self.n_k
            f.attrs['n_j'] = self.n_j
            
            f.attrs['time'] = time
            
            f.create_dataset('x_k',
                             data=np.array(x_k))
            f.create_dataset('y_j',
                             data=np.array(y_j))

    def load(filename: str):
        with h5py.File(filename, 'r') as f:
            eq = L96(
                b=f.attrs['b'].item(),
                c=f.attrs['c'].item(),
                h=f.attrs['h'].item(), 
                n_k=f.attrs['n_k'].item(),
                n_j=f.attrs['n_j'].item()
            )
            
            return (
                eq,
                f.attrs['time'].item(),
                np.array(f['x_k']),
                np.array(f['y_j'])
            )

    def __repr__(self):
        return '''Two-time scale Lorenz 96 system: 
        Parameters: b={}, c={}, h={}
        Truncation: n_x={}, n_j={}'''.format(
            self.b,
            self.c,
            self.h,
            self.n_k,
            self.n_j
        )

def dynamical_solver(
    eq: L96,
    solver: OdeScheme,
    source: Callable,
) -> Callable:
    # Explicit term
    def __explicit__(
        x_k: jnp.ndarray,
        y_j: jnp.ndarray,
        source: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:    
        hcb = (eq.h * eq.c) / eq.b
        
        x_dot = -jnp.roll(x_k, -1) * (jnp.roll(x_k, -2) - jnp.roll(x_k, 1)) - x_k + source - hcb * jnp.sum(y_j, axis=0)  
        y_dot = -eq.c * eq.b * jnp.roll(y_j, 1) * (jnp.roll(y_j, 2) - jnp.roll(y_j, -1)) - eq.c * y_j + hcb * x_k

        return (
            x_dot,
            y_dot
        )

    return solver(
        source,
        __explicit__,
    )

def dynamical_solver_single(
    eq: L96,
    solver: OdeScheme,
    source: Callable,
) -> Callable:
    # Explicit term
    def __explicit__(
        x_k: jnp.ndarray,
        y_j: jnp.ndarray,
        source: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        hcb = (eq.h * eq.c) / eq.b
        
        x_dot = -jnp.roll(x_k, -1) * (jnp.roll(x_k, -2) - jnp.roll(x_k, 1)) - x_k + source 

        return (
            x_dot,
            0
        )

    return solver(
        source,
        __explicit__,
    )
