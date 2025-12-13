import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from typing import Optional, Tuple

def spectral_pad(
    f_s: jnp.ndarray,
    n_kx: int,
    n_ky: int,
    n_px: int = None,
    n_py: int = None
) -> jnp.ndarray:
    n_px = n_kx if n_px is None else n_px
    n_py = n_ky if n_py is None else n_py
    return jnp.zeros((n_py, n_px), dtype=jnp.complex128) \
        .at[:n_ky // 2, :n_kx].set(
            f_s[:n_ky // 2, :n_kx]
        ) \
        .at[-n_ky // 2:, :n_kx].set(
            f_s[-n_ky // 2:, :n_kx]
        )

def into_s_pad(f_g: jnp.ndarray, n_kx: int, n_ky: int) -> jnp.ndarray:
    """Transform *2D* grid values into Fourier coefficients, with dealiasing."""
    f_s = spectral_pad(into_s(f_g), n_kx, n_ky, n_kx, n_ky)
    return f_s

def from_s_pad(f_s: jnp.ndarray, n_x: int, n_y: int) -> jnp.ndarray:
    """Transform back Fourier coefficients on the *2D* grid, with 3/2 dealiasing."""
    n_ky, n_kx = f_s.shape
    f_s = spectral_pad(f_s, n_kx, n_ky, n_x//2 + 1, n_y)
    f_g = jnp.fft.irfft2(f_s, norm='forward', s=(n_y, n_x))
    return f_g

def into_s(f_g: jnp.ndarray) -> jnp.ndarray:
    """Transform *2D* grid values into Fourier coefficients."""
    return jnp.fft.rfft2(f_g, norm='forward')

def from_s(f_s: jnp.ndarray) -> jnp.ndarray:
    """Transform back Fourier coefficients on the *2D* grid."""
    return jnp.fft.irfft2(f_s, norm='forward')
