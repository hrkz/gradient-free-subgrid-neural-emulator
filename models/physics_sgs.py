import jax
import jax.numpy as jnp

from typing import Callable

from models.qg_periodic import (
    QgPeriodic,
)
from utils import (
    into_s_pad, 
    from_s_pad,
)

def strain_rate_sgs(
    eq: QgPeriodic,
    nu_model: Callable,
    om_s: jnp.ndarray,
) -> jnp.ndarray:
    ps_s = -eq.ilap * om_s
    ux_s = -1j * eq.k_y * ps_s
    uy_s =  1j * eq.k_x * ps_s

    S_xx = from_s_pad(1j * eq.k_x * ux_s, eq.n_x, eq.n_y)
    S_yy = from_s_pad(1j * eq.k_y * uy_s, eq.n_x, eq.n_y)
    S_xy = 0.5 * from_s_pad(1j * eq.k_y * ux_s + 1j * eq.k_x * uy_s, eq.n_x, eq.n_y)

    nu_e = nu_model(om_s, S_xx, S_yy, S_xy)
    
    nu_S_xx = into_s_pad(nu_e * S_xx, eq.n_kx, eq.n_ky)
    nu_S_xy = into_s_pad(nu_e * S_xy, eq.n_kx, eq.n_ky)
    nu_S_yy = into_s_pad(nu_e * S_yy, eq.n_kx, eq.n_ky)
    tau_x = 2 * (1j * eq.k_x * nu_S_xx + 1j * eq.k_y * nu_S_xy)
    tau_y = 2 * (1j * eq.k_x * nu_S_xy + 1j * eq.k_y * nu_S_yy)

    return (
        -1j * eq.k_y * tau_x + 1j * eq.k_x * tau_y
    )

def smagorinsky(
    eq: QgPeriodic,
    C_S: float
) -> Callable[jnp.ndarray, jnp.ndarray]:
    """A Smagorinsky model with constant coefficient."""
    def __impl__(om_s, S_xx, S_yy, S_xy):
        delta = (2 * jnp.pi) / max(eq.n_x, eq.n_y)
        return (C_S * delta)**2 * jnp.sqrt(2 * (S_xx**2 + S_yy**2 + 2 * S_xy**2))
    return jax.tree_util.Partial(
        strain_rate_sgs, eq, __impl__
    )

def leith(
    eq: QgPeriodic,
    C_L: float
) -> Callable[jnp.ndarray, jnp.ndarray]:
    """A Leith model with constant coefficient."""
    def  __impl__(om_s, S_xx, S_yy, S_xy):
        om_x = from_s_pad(1j * eq.k_x * om_s, eq.n_x, eq.n_y)
        om_y = from_s_pad(1j * eq.k_y * om_s, eq.n_x, eq.n_y)
        delta = (2 * jnp.pi) / max(eq.n_x, eq.n_y)
        return (C_L * delta)**3 * jnp.sqrt(om_x**2 + om_y**2)
    return jax.tree_util.Partial(
        strain_rate_sgs, eq, __impl__
    )
