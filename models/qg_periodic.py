import h5py
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jns
import jax.random as jnr

from typing import Callable, Optional, Tuple

from models.time_solver import ImexScheme
from utils import (
    into_s_pad, 
    from_s_pad,
    spectral_pad,
)

class QgPeriodic:
    def __init__(
        self,
        nu: float,
        mu: float,
        beta: float,
        n_kx: int,
        n_ky: int,
    ):
        self.nu = nu
        self.mu = mu
        self.beta = beta

        self.n_kx = n_kx
        self.n_ky = n_ky
        self.n_x = int((self.n_kx - 1) * 2)
        self.n_y = int(self.n_ky)
        
        self.n_x_pad = int((self.n_kx - 1) * 3)
        self.n_y_pad = int(self.n_ky * 3 / 2)
        
        self.x = np.linspace(0, 2 * np.pi, self.n_x, endpoint=False)
        self.y = np.linspace(0, 2 * np.pi, self.n_y, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.n_ks = 2 * (self.n_kx - 1)
        self.k_x = np.fft.rfftfreq(self.n_ks, 1 / self.n_ks).reshape((1, -1))
        self.k_y = np.fft. fftfreq(self.n_ky, 1 / self.n_ky).reshape((-1, 1))
        self.lap = self.k_x**2 + self.k_y**2

        with np.errstate(divide='ignore'):
            self.ilap = 1 / self.lap
            self.ilap[0, 0] = 0

    def save(self, filename: str, time: float, om_s: jnp.ndarray):
        with h5py.File(filename, 'w') as f:
            f.attrs['nu'] = self.nu
            f.attrs['mu'] = self.mu
            f.attrs['beta'] = self.beta
            
            f.attrs['n_kx'] = self.n_kx
            f.attrs['n_ky'] = self.n_ky

            f.attrs['time'] = time
            
            f.create_dataset('om_s',
                             data=np.array(om_s))

    def load(filename: str):
        with h5py.File(filename, 'r') as f:
            eq = QgPeriodic(
                nu=f.attrs['nu'].item(),
                mu=f.attrs['mu'].item(),
                beta=f.attrs['beta'].item(), 
                n_kx=f.attrs['n_kx'].item(),
                n_ky=f.attrs['n_ky'].item()
            )
            
            return (
                eq,
                f.attrs['time'].item(),
                np.array(f['om_s']),
            )

    def __repr__(self):
        return '''Doubly-periodic quasi-geostrophic system: 
        Parameters: nu={}, mu={}, beta={}
        Truncation: n_x={}, n_y={}'''.format(
            self.nu,
            self.mu,
            self.beta,
            self.n_x,
            self.n_y
        )

def dynamical_solver(
    eq: QgPeriodic,
    solver: ImexScheme,
    source: Callable,
) -> Callable:
    # Courant–Friedrichs–Lewy condition (number).
    def __cfl__(
        ux_g: jnp.ndarray,
        uy_g: jnp.ndarray
    ) -> float:
        ux_max = jnp.sqrt(jnp.max(ux_g**2))
        uy_max = jnp.sqrt(jnp.max(uy_g**2))

        dx = (2 * np.pi) / eq.n_x
        dy = (2 * np.pi) / eq.n_y
        return jnp.min(
            jnp.array([jnp.finfo(jnp.float64).max, dx / ux_max, dy / uy_max])
        )

    # Implicit term
    def __implicit__(
        om_s: jnp.ndarray
    ) -> jnp.ndarray:
        imp_om = -eq.mu - eq.nu * eq.lap + 1j * eq.beta * eq.k_x * eq.ilap
        return om_s * imp_om

    # Explicit term
    def __explicit__(
        om_s: jnp.ndarray,
        source: Callable,
        t: float
    ) -> Tuple[jnp.ndarray, float]:
        ps_s = -eq.ilap * om_s
        ux_s = -1j * eq.k_y * ps_s
        uy_s =  1j * eq.k_x * ps_s
        
        om_g = from_s_pad(om_s, eq.n_x_pad, eq.n_y_pad)
        ux_g = from_s_pad(ux_s, eq.n_x_pad, eq.n_y_pad)
        uy_g = from_s_pad(uy_s, eq.n_x_pad, eq.n_y_pad)
        
        c = __cfl__(ux_g, uy_g)

        uxom_g = ux_g * om_g
        uyom_g = uy_g * om_g
    
        uxom_s = into_s_pad(uxom_g, eq.n_kx, eq.n_ky)
        uyom_s = into_s_pad(uyom_g, eq.n_kx, eq.n_ky)
        exp_om = -1j * eq.k_x * uxom_s - 1j * eq.k_y * uyom_s + source(om_s, t)
        return (
            exp_om,
            c
        )

    # Implicit solve
    def __solve__(
        eq_system: jnp.ndarray,
        rhs_om: jnp.ndarray
    ) -> jnp.ndarray:
        return rhs_om / eq_system
    system = 1 - solver.coef() * __implicit__(jnp.array(1))
    return solver(
        eq,
        source,
        system,
        __implicit__,
        __explicit__,
        __solve__,
    )

def divergence(
    eq: QgPeriodic,
    F_x: jnp.ndarray,
    F_y: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the divergence of F."""
    return 1j * eq.k_x * F_x + 1j * eq.k_y * F_y

def subgrid_fluxes(
    eq: QgPeriodic,
    eq_coarse: QgPeriodic,
    om_s: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the subgrid fluxes between eq grid and eq_coarse grid."""
    ps_s = -eq.ilap * om_s
    ux_s = -1j * eq.k_y * ps_s
    uy_s =  1j * eq.k_x * ps_s
    
    om_g = from_s_pad(om_s, eq.n_x_pad, eq.n_y_pad)
    ux_g = from_s_pad(ux_s, eq.n_x_pad, eq.n_y_pad)
    uy_g = from_s_pad(uy_s, eq.n_x_pad, eq.n_y_pad)

    uxom_s = into_s_pad(ux_g * om_g, eq.n_kx, eq.n_ky)
    uyom_s = into_s_pad(uy_g * om_g, eq.n_kx, eq.n_ky)

    om_c = spectral_pad(om_s, eq_coarse.n_kx, eq_coarse.n_ky)
    ps_c = -eq_coarse.ilap * om_c
    ux_c = -1j * eq_coarse.k_y * ps_c
    uy_c =  1j * eq_coarse.k_x * ps_c
    
    om_gc = from_s_pad(om_c, eq_coarse.n_x_pad, eq_coarse.n_y_pad)
    ux_gc = from_s_pad(ux_c, eq_coarse.n_x_pad, eq_coarse.n_y_pad)
    uy_gc = from_s_pad(uy_c, eq_coarse.n_x_pad, eq_coarse.n_y_pad)

    uxom_c = into_s_pad(ux_gc * om_gc, eq_coarse.n_kx, eq_coarse.n_ky)
    uyom_c = into_s_pad(uy_gc * om_gc, eq_coarse.n_kx, eq_coarse.n_ky)

    return (
        uxom_c - spectral_pad(uxom_s, eq_coarse.n_kx, eq_coarse.n_ky),
        uyom_c - spectral_pad(uyom_s, eq_coarse.n_kx, eq_coarse.n_ky)
    )

def average(
    f_s: jnp.ndarray,
) -> float:
    """Compute the average of f_s over the domain."""
    sum_k = jnp.sum(f_s[:, 0]) + jnp.sum(f_s[:, -1]) + jnp.sum(2 * f_s[:, 1:-1])
    return sum_k

def kinetic_energy(
    ux_s: jnp.ndarray,
    uy_s: jnp.ndarray
) -> float:
    """Compute the kinetic energy."""
    return 0.5 * (
        average(ux_s.real**2 + ux_s.imag**2) + 
        average(uy_s.real**2 + uy_s.imag**2)
    )

def enstrophy(
    om_s: jnp.ndarray
) -> float:
    """Compute the enstrophy."""
    return 0.5 * average(om_s.real**2 + om_s.imag**2)

def turnover_time(
    om_s: jnp.ndarray
) -> float:
    """Compute the turnover time."""
    return 2 * np.pi * np.sqrt(1 / enstrophy(om_s))
    
def iso_spectrum(
    eq: QgPeriodic,
    f_s: jnp.ndarray,
    avg: bool = True
) -> Tuple[jnp.ndarray, int, jnp.ndarray]:
    """Compute the isotropic spectrum."""
    k_max = min(eq.n_kx, eq.n_ky)
    dk = jnp.sqrt(2)
    kr = jnp.arange(k_max) * dk
    bin_max = jnp.flatnonzero(kr < k_max + dk / 2)[-1]

    f_s = jnp.concatenate(
        [
            jnp.expand_dims(f_s[...,  0] / 2, -1),
            f_s[..., 1:-1],
            jnp.expand_dims(f_s[..., -1] / 2, -1),
        ],
        axis=-1,
    )
    
    bins = jnp.floor(jnp.sqrt(eq.lap) / dk).astype(jnp.uint32)
    f_bin = jax.ops.segment_sum(
        f_s.ravel(),
        bins.ravel(),
        num_segments=kr.size,
        indices_are_sorted=False,
        unique_indices=False,
    )
    
    return (
        kr + dk / 2, 
        bin_max,
        f_bin
    )
