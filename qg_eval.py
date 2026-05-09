import argparse
import tqdm
import h5py
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jnr

from flax import nnx
import orbax.checkpoint as ocp

jax.config.update(
    'jax_enable_x64', True
)

from typing import Callable, Optional

import models.time_solver as stepper
from models.qg_periodic import (
    QgPeriodic, 
    dynamical_solver,
    divergence,
    subgrid_fluxes,
)
from models.cnn import (
    FwdCNN,
)
from utils import (
    into_s,
    from_s,
    spectral_pad,
)

def main(args: argparse.Namespace) -> None:
    print(args)
    data_path = os.path.join(os.path.join(os.getcwd(), 'data'), args.name)

    with h5py.File(os.path.join(data_path, 'datasets.h5'), 'r') as f:
        dt = f.attrs['dt']
        ratio = f.attrs['ratio']
    
        sigma = f.attrs['sigma']
        k_f = f.attrs['k_f']
    eq, t0, om_s = QgPeriodic.load(os.path.join(data_path, 'snapshot.h5'))
    print(eq)
    
    eq_coarse = QgPeriodic(
        nu=eq.nu,
        mu=eq.mu,
        beta=eq.beta,
        n_kx=int((eq.n_kx - 1) / ratio + 1),
        n_ky=int(eq.n_ky / ratio)
    )
    
    iso_K = np.sqrt(eq.lap)
    forcing_spectrum = np.full_like(iso_K, sigma)
    forcing_spectrum[iso_K < k_f - 1] = 0
    forcing_spectrum[iso_K > k_f + 1] = 0
    forcing_spectrum[iso_K == 0] = 0
    forcing_det = into_s(np.cos(k_f * eq.X) + np.cos(k_f * eq.Y))

    def source(om_s, t):
        f_s = forcing_spectrum * forcing_det
        return f_s
    def compute_tau(om_s):
        tau_x, tau_y = subgrid_fluxes(eq, eq_coarse, om_s)
        return divergence(
            eq_coarse, tau_x, tau_y
        )
    
    # DNS
    dns_file = os.path.join(args.save_path, 'logs_dns.h5')
    if not os.path.isfile(dns_file):
        with h5py.File(dns_file, 'w') as f:
            solver = jax.jit(dynamical_solver(
                eq,
                stepper.BPR353(dt),
                source
            ))

            print('Integrating DNS...')
            run_system(
                eq,
                solver,
                compute_tau=compute_tau,
                om_s=om_s,
                dt=dt,
                t0=t0,
                T=args.T,
                n_logs=args.n_logs,
                file=f
            )

    # SGS models
    
    iso_K_coarse = np.sqrt(eq_coarse.lap)
    forcing_spectrum_coarse = np.full_like(iso_K_coarse, sigma)
    forcing_spectrum_coarse[iso_K_coarse < k_f - 1] = 0
    forcing_spectrum_coarse[iso_K_coarse > k_f + 1] = 0
    forcing_spectrum_coarse[iso_K_coarse == 0] = 0
    forcing_det_coarse = into_s(np.cos(k_f * eq_coarse.X) + np.cos(k_f * eq_coarse.Y))

    def source_coarse(om_s, t):
        f_s = forcing_spectrum_coarse * forcing_det_coarse
        return f_s
    
    nop_file = os.path.join(args.save_path, 'logs_nop.h5')
    if not os.path.isfile(nop_file):
        with h5py.File(nop_file, 'w') as f:
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                stepper.BPR353(dt * ratio),
                source_coarse
            ))

            print('Integrating coarse-resolution (no model)...')
            run_system(
                eq_coarse,
                solver,
                compute_tau=None,
                om_s=spectral_pad(om_s, eq_coarse.n_kx, eq_coarse.n_ky),
                dt=dt * ratio,
                t0=t0,
                T=args.T,
                n_logs=args.n_logs,
                file=f
            )

    abstract_model = nnx.eval_shape(lambda: FwdCNN(
        in_features=1,
        latent=args.ml_model_latent,
        kernel_size=5,
        out_features=1,
        n_blocks=args.ml_model_blocks,
        means=0,
        stds=1,
        activation=nnx.relu,
        rngs=nnx.Rngs(42)
    ))
    
    graph, abstract_state = nnx.split(abstract_model)

    checkpoint_path = os.path.join(data_path, 'off_checkpoint/')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    state = checkpointer.restore(checkpoint_path, abstract_state)
    eq_off = nnx.merge(graph, state)

    def source_off(om_s, t):
        f_s = forcing_spectrum_coarse * forcing_det_coarse
        tau_s = into_s(eq_off(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
        return f_s + tau_s
    def compute_tau_off(om_s):
        return into_s(eq_off(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())

    off_file = os.path.join(args.save_path, 'logs_off.h5')
    if not os.path.isfile(off_file):
        with h5py.File(off_file, 'w') as f:
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                stepper.BPR353(dt * ratio),
                source_off
            ))

            print('Integrating with offline model correction...')
            run_system(
                eq_coarse,
                solver,
                compute_tau=compute_tau_off,
                om_s=spectral_pad(om_s, eq_coarse.n_kx, eq_coarse.n_ky),
                dt=dt * ratio,
                t0=t0,
                T=args.T,
                n_logs=args.n_logs,
                file=f
            )
    
    checkpoint_path = os.path.join(data_path, 'state_ref_ek_checkpoint/')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    state = checkpointer.restore(checkpoint_path, abstract_state)
    eq_ref = nnx.merge(graph, state)

    def source_ref(om_s, t):
        f_s = forcing_spectrum_coarse * forcing_det_coarse
        tau_s = into_s(eq_ref(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
        return f_s + tau_s
    def compute_tau_ref(om_s):
        return into_s(eq_ref(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())

    ref_file = os.path.join(args.save_path, 'logs_state_ref_ek.h5')
    if not os.path.isfile(ref_file):
        with h5py.File(ref_file, 'w') as f:
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                stepper.BPR353(dt * ratio),
                source_ref
            ))

            print('Integrating with reference (state) online model correction...')
            run_system(
                eq_coarse,
                solver,
                compute_tau=compute_tau_ref,
                om_s=spectral_pad(om_s, eq_coarse.n_kx, eq_coarse.n_ky),
                dt=dt * ratio,
                t0=t0,
                T=args.T,
                n_logs=args.n_logs,
                file=f
            )

    checkpoint_path = os.path.join(data_path, 'subgrid_ref_mse_checkpoint/')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    state = checkpointer.restore(checkpoint_path, abstract_state)
    eq_ref = nnx.merge(graph, state)

    def source_ref(om_s, t):
        f_s = forcing_spectrum_coarse * forcing_det_coarse
        tau_s = into_s(eq_ref(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
        return f_s + tau_s
    def compute_tau_ref(om_s):
        return into_s(eq_ref(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())

    ref_file = os.path.join(args.save_path, 'logs_subgrid_ref_mse.h5')
    if not os.path.isfile(ref_file):
        with h5py.File(ref_file, 'w') as f:
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                stepper.BPR353(dt * ratio),
                source_ref
            ))

            print('Integrating with reference (subgrid) online model correction...')
            run_system(
                eq_coarse,
                solver,
                compute_tau=compute_tau_ref,
                om_s=spectral_pad(om_s, eq_coarse.n_kx, eq_coarse.n_ky),
                dt=dt * ratio,
                t0=t0,
                T=args.T,
                n_logs=args.n_logs,
                file=f
            )

    checkpoint_path = os.path.join(data_path, 'state_small_ek_checkpoint/')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    state = checkpointer.restore(checkpoint_path, abstract_state)
    eq_state_small = nnx.merge(graph, state)

    def source_state_small(om_s, t):
        f_s = forcing_spectrum_coarse * forcing_det_coarse
        tau_s = into_s(eq_state_small(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
        return f_s + tau_s
    def compute_tau_state_small(om_s):
        return into_s(eq_state_small(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())

    state_small_file = os.path.join(args.save_path, 'logs_state_small_ek.h5')
    if not os.path.isfile(state_small_file):
        with h5py.File(state_small_file, 'w') as f:
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                stepper.BPR353(dt * ratio),
                source_state_small
            ))

            print('Integrating with emulator-based (state loss, small emulator) model correction...')
            run_system(
                eq_coarse,
                solver,
                compute_tau=compute_tau_state_small,
                om_s=spectral_pad(om_s, eq_coarse.n_kx, eq_coarse.n_ky),
                dt=dt * ratio,
                t0=t0,
                T=args.T,
                n_logs=args.n_logs,
                file=f
            )

    checkpoint_path = os.path.join(data_path, 'subgrid_small_mse_checkpoint/')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    state = checkpointer.restore(checkpoint_path, abstract_state)
    eq_subgrid_small = nnx.merge(graph, state)

    def source_subgrid_small(om_s, t):
        f_s = forcing_spectrum_coarse * forcing_det_coarse
        tau_s = into_s(eq_subgrid_small(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
        return f_s + tau_s
    def compute_tau_subgrid_small(om_s):
        return into_s(eq_subgrid_small(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())

    subgrid_small_file = os.path.join(args.save_path, 'logs_subgrid_small_mse.h5')
    if not os.path.isfile(subgrid_small_file):
        with h5py.File(subgrid_small_file, 'w') as f:
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                stepper.BPR353(dt * ratio),
                source_subgrid_small
            ))

            print('Integrating with emulator-based (subgrid loss, small emulator) model correction...')
            run_system(
                eq_coarse,
                solver,
                compute_tau=compute_tau_subgrid_small,
                om_s=spectral_pad(om_s, eq_coarse.n_kx, eq_coarse.n_ky),
                dt=dt * ratio,
                t0=t0,
                T=args.T,
                n_logs=args.n_logs,
                file=f
            )

    checkpoint_path = os.path.join(data_path, 'state_large_ek_checkpoint/')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    state = checkpointer.restore(checkpoint_path, abstract_state)
    eq_state_large = nnx.merge(graph, state)

    def source_state_large(om_s, t):
        f_s = forcing_spectrum_coarse * forcing_det_coarse
        tau_s = into_s(eq_state_large(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
        return f_s + tau_s
    def compute_tau_state_large(om_s):
        return into_s(eq_state_large(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())

    state_large_file = os.path.join(args.save_path, 'logs_state_large_ek.h5')
    if not os.path.isfile(state_large_file):
        with h5py.File(state_large_file, 'w') as f:
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                stepper.BPR353(dt * ratio),
                source_state_large
            ))

            print('Integrating with emulator-based (state loss, large emulator) model correction...')
            run_system(
                eq_coarse,
                solver,
                compute_tau=compute_tau_state_large,
                om_s=spectral_pad(om_s, eq_coarse.n_kx, eq_coarse.n_ky),
                dt=dt * ratio,
                t0=t0,
                T=args.T,
                n_logs=args.n_logs,
                file=f
            )

    checkpoint_path = os.path.join(data_path, 'subgrid_large_mse_checkpoint/')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    state = checkpointer.restore(checkpoint_path, abstract_state)
    eq_subgrid_large = nnx.merge(graph, state)

    def source_subgrid_large(om_s, t):
        f_s = forcing_spectrum_coarse * forcing_det_coarse
        tau_s = into_s(eq_subgrid_large(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
        return f_s + tau_s
    def compute_tau_subgrid_large(om_s):
        return into_s(eq_subgrid_large(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())

    subgrid_large_file = os.path.join(args.save_path, 'logs_subgrid_large_mse.h5')
    if not os.path.isfile(subgrid_large_file):
        with h5py.File(subgrid_large_file, 'w') as f:
            solver = jax.jit(dynamical_solver(
                eq_coarse,
                stepper.BPR353(dt * ratio),
                source_subgrid_large
            ))

            print('Integrating with emulator-based (subgrid loss, large emulator) model correction...')
            run_system(
                eq_coarse,
                solver,
                compute_tau=compute_tau_subgrid_large,
                om_s=spectral_pad(om_s, eq_coarse.n_kx, eq_coarse.n_ky),
                dt=dt * ratio,
                t0=t0,
                T=args.T,
                n_logs=args.n_logs,
                file=f
            )

def run_system(
    eq: QgPeriodic,
    solver: Callable,
    compute_tau: Optional[Callable],
    om_s: jnp.ndarray,
    dt: float,
    t0: float,
    T: float,
    n_logs: int,
    file
):
    time = t0
    iters = int(T / dt)
    logs_freq = int(iters / n_logs)
    sample_digits = len(str(int(iters / logs_freq)))

    time_t = []
    pbar = tqdm.tqdm(range(iters), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:
        c, om_s = solver(om_s, time)
        time += dt
        if not np.isfinite(c):
            print('Solver crashed with cfl =',c)
            break
        if i % logs_freq == 0:
            time_t.append(time)

            file.create_dataset('om_s_' + str(i // logs_freq).zfill(sample_digits), 
                                data=np.array(om_s))
            if compute_tau != None:
                tau = compute_tau(om_s)
                file.create_dataset('tau_s_' + str(i // logs_freq).zfill(sample_digits), 
                                    data=np.array(tau))
            
            pbar.set_postfix(
                cfl=format(dt / c, ".2f"),
            )
    file.attrs['digits'] = sample_digits
    file.create_dataset('time',
                        data=np.array(time_t))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python eval.py',
        description='Integrate reference DNS and multiple SGS models for the QG system and save states.'
    )
    
    parser.add_argument('-n', '--name', type=str, help='Name of the configuration', required=True)
    parser.add_argument('--save_path', type=str, help='File path for saving the field samples', required=True)

    parser.add_argument('-n_logs', type=int, help='Number of saved snapshots', required=True)
    parser.add_argument('-T', type=float, help='Final time of the integration', required=True)

    parser.add_argument('-ml_model_blocks', type=int, help='Number of CNNNext blocks for the SGS model correction', required=True)
    parser.add_argument('-ml_model_latent', type=int, help='Size of the latent space for the SGS model correction', required=True)
    
    args = parser.parse_args()
    main(args)
