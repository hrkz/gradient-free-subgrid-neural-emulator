import argparse
import tqdm
import os

import matplotlib.pyplot as plt

plt.rcParams.update({
  'mathtext.fontset': 'cm'
})

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jnr

jax.config.update(
  'jax_enable_x64', True
)

import models.time_solver as stepper
from models.qg_periodic import (
    QgPeriodic, 
    dynamical_solver,
    kinetic_energy,
    enstrophy
)
from utils import (
    into_s,
)

def main(args: argparse.Namespace) -> None:
    print(args)
    key_seed = 42
    key = jnr.key(key_seed)
    
    eq = QgPeriodic(
        nu=args.nu,
        mu=args.mu,
        beta=args.beta,
        n_kx=args.n_kx,
        n_ky=args.n_ky
    )
    print(eq)

    om_s = 1e-3 * np.exp(2 * np.pi * 1j * jnr.normal(key, (eq.n_ky, eq.n_kx)))
    
    iso_K = np.sqrt(eq.lap)

    forcing_spectrum = np.full_like(iso_K, args.sigma)
    forcing_spectrum[iso_K < args.k_f - 1] = 0
    forcing_spectrum[iso_K > args.k_f + 1] = 0
    forcing_spectrum[iso_K == 0] = 0
    forcing_det = into_s(np.cos(args.k_f * eq.X) + np.cos(args.k_f * eq.Y))

    def source(om_s, t):
        f_s = forcing_spectrum * forcing_det
        return f_s
    
    dt = args.dt
    solver = jax.jit(dynamical_solver(
        eq,
        stepper.BPR353(dt),
        source
    ))

    time = 0
    iters = int(args.T / dt)
    logs = 1000
    logs_freq = int(iters / logs)

    time_t = []
    eke_t = []
    ens_t = []
    
    print('Running dynamical solver...')
    pbar = tqdm.tqdm(range(iters), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:
        c, om_s = solver(om_s, time)
        time += dt
        if i % logs_freq == 0:
            time_t.append(time)
    
            ps_s = -eq.ilap * om_s
            ux_s = -1j * eq.k_y * ps_s
            uy_s =  1j * eq.k_x * ps_s
    
            eke_t.append(
                kinetic_energy(ux_s, uy_s)
            )
            ens_t.append(
                enstrophy(om_s)
            )
        if not np.isfinite(c):
            print('Solver crashed with cfl =',c)
            exit(1)

    print('Saving snapshot...')
    data_path = os.path.join(os.path.join(os.getcwd(), 'data'), args.name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    eq.save(
        os.path.join(data_path, 'spinup.h5'),
        args.T, 
        om_s,
    )
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(7.0, 4.0), dpi=120)

    axs[0].plot(time_t, eke_t, color='k')
    axs[0].set_xlabel(r'$t$', fontsize=15)
    axs[0].set_ylabel(r'$E(t)$', fontsize=15)
    axs[0].tick_params(reset=True, axis='both', which='both', direction='in')
    
    axs[1].plot(time_t, ens_t, color='k')
    axs[1].set_xlabel(r'$t$', fontsize=15)
    axs[1].set_ylabel(r'$Z(t)$', fontsize=15)
    axs[1].tick_params(reset=True, axis='both', which='both', direction='in')
    
    fig.tight_layout()
    fig.savefig(
        os.path.join(data_path, 'spinup_integrals.pdf')
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python spinup.py',
        description='Integrate a periodic QG system configuration to time T (spinup) and save a snapshot for dataset generation.'
    )
    
    parser.add_argument('-n', '--name', type=str, help='Name of the configuration', required=True)
    parser.add_argument('-nu', type=float, help='Kinematic viscosity', required=True)
    parser.add_argument('-mu', type=float, help='Linear drag coefficient', required=True)
    parser.add_argument('-beta', type=float, help='Beta plane coefficient', required=True)

    parser.add_argument('-sigma', type=float, help='Forcing amplitude', required=True)
    parser.add_argument('-k_f', type=float, help='Forcing wavenumber', required=True)

    parser.add_argument('-n_kx', type=int, help='Number of Fourier coefficients (x-direction)', required=True)
    parser.add_argument('-n_ky', type=int, help='Number of Fourier coefficients (y-direction)', required=True) 

    parser.add_argument('-dt', type=float, help='Discrete (fixed) time step', required=True)
    parser.add_argument('-T', type=float, help='Final time of the integration', required=True)
    
    args = parser.parse_args()
    main(args)
