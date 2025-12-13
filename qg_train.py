import shutil
import argparse
import tqdm
import h5py
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jnr

jax.config.update(
  'jax_enable_x64', True
)

from flax import nnx
import optax
import orbax.checkpoint as ocp

from typing import Callable, Optional

import models.time_solver as stepper
from models.qg_periodic import (
    QgPeriodic, 
    dynamical_solver,
    divergence,
    average,
    kinetic_energy,
    enstrophy,
)
from models.cnn import (
    FwdCNN,
    ResCNN,
)
from utils import (
    into_s,
    from_s
)

def main(args: argparse.Namespace) -> None:
    key = jnr.key(42)
    rngs = nnx.Rngs(key)
    
    data_path = os.path.join(os.path.join(os.getcwd(), 'data'), args.name)
    with h5py.File(os.path.join(data_path, 'datasets.h5'), 'r') as f:
        dt = f.attrs['dt']
        ratio = f.attrs['ratio']
        n_steps = f.attrs['n_steps']
        n_trajs = f.attrs['n_trajs']

        sigma = f.attrs['sigma']
        k_f = f.attrs['k_f']

        om_c_ref = np.array(f['om_c_ref'])
        tau_x_ref = np.array(f['tau_x_ref'])
        tau_y_ref = np.array(f['tau_y_ref'])
        om_c_emu = np.array(f['om_c_emu'])
    eq, _, _ = QgPeriodic.load(
        os.path.join(data_path, 'snapshot.h5')
    )

    eq_coarse = QgPeriodic(
        nu=eq.nu,
        mu=eq.mu,
        beta=eq.beta,
        n_kx=int((eq.n_kx - 1) / ratio + 1),
        n_ky=int(eq.n_ky / ratio)
    )
    print(eq_coarse)

    iso_K = np.sqrt(eq_coarse.lap)
    
    forcing_spectrum = np.full_like(iso_K, sigma)
    forcing_spectrum[iso_K < k_f - 1] = 0
    forcing_spectrum[iso_K > k_f + 1] = 0
    forcing_spectrum[iso_K == 0] = 0

    forcing_det = into_s(np.cos(k_f * eq_coarse.X) + np.cos(k_f * eq_coarse.Y))

    def kinetic_energy_loss(om_hat_c, om_c):
        ps_c = -eq_coarse.ilap * om_c
        ux_c = -1j * eq_coarse.k_y * ps_c
        uy_c =  1j * eq_coarse.k_x * ps_c

        ps_hat_c = -eq_coarse.ilap * om_hat_c
        ux_hat_c = -1j * eq_coarse.k_y * ps_hat_c
        uy_hat_c =  1j * eq_coarse.k_x * ps_hat_c

        return 2 * kinetic_energy(
            ux_hat_c - ux_c,
            uy_hat_c - uy_c
        )

    def physical_loss(y_hat_c, y_c):
        return jnp.mean(jnp.square(from_s(y_hat_c) - from_s(y_c)))

    # Get "physical" emulator data
    om_emu = np.zeros((n_trajs, n_steps, eq_coarse.n_y, eq_coarse.n_x))
    for traj in range(n_trajs):
        for step in range(n_steps):
            om_emu[traj, step] = from_s(om_c_emu[traj, step])

    om_c_emu_mean = np.mean(om_emu)
    om_c_emu_std = np.std(om_emu)

    with np.printoptions(precision=4):
        print('Emulator dataset statistics: om_c = {} ± σ({})'.format(
            om_c_emu_mean, 
            om_c_emu_std
        ))

    def emu_flow_loss(
        eq_emu: nnx.Module, 
        om_c_0: jnp.ndarray, 
        om_c_traj: jnp.ndarray
    ) -> float:
        def __source__(om_c, t):
            f_s = forcing_spectrum * forcing_det
            return f_s

        # The explicit term in the PDE is now based on the emulator.
        def __explicit__(om_c: jnp.ndarray, source: Callable, t: float):
            om_c_dt = into_s(eq_emu(jnp.expand_dims(from_s(om_c), (0,-1))).squeeze())
            c = 0
            return (
                om_c_dt + source(om_c, t),
                c
            )
        def __implicit__(om_c: jnp.ndarray):
            return 0
        def __solve__(eq_system: jnp.ndarray, rhs_om: jnp.ndarray) -> jnp.ndarray:
            return rhs_om
            
        # Uses differentiable dynamical solver
        solver = stepper.BPR353(ratio * dt)
        step_fn = solver(
            eq_coarse,
            __source__,
            None,
            __implicit__,
            __explicit__,
            __solve__,
        )
        
        def __loop__(
            om_c: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            _, om_c = step_fn(om_c, 0)
            return (
                jnp.array(om_c), 
                jnp.array(om_c)
            )
    
        # Loop over time steps and accumulate states \bar{\omega}(t)
        _, om_hat_c = jax.lax.scan(__loop__, om_c_0, length=om_c_traj.shape[0])
        loss = jnp.mean(jax.vmap(physical_loss)(om_hat_c, om_c_traj))
        return loss
    
    @nnx.jit 
    def emu_train_step(eq_emu, optimizer, traj_batch: jnp.ndarray):
        om_c_0, om_c_traj = traj_batch
        loss, grads = nnx.value_and_grad(emu_flow_loss)(eq_emu, om_c_0, om_c_traj)
        optimizer.update(eq_emu, grads)
        return loss

    eq_emu_small = ResCNN(
        in_features=1,
        latent=args.emu_latent_small,
        kernel_size=args.emu_kernel_small,
        out_features=1,
        n_blocks=args.emu_blocks_small,
        means=jnp.array(om_c_emu_mean), 
        stds=jnp.array(om_c_emu_std),
        activation=nnx.relu,
        rngs=rngs
    )
    print('Emulator (small) architecture')
    print(eq_emu_small)

    online_train_loop(
        key,
        optimizer=nnx.Optimizer(eq_emu_small, optax.adamw(args.emu_lr), wrt=nnx.Param),
        step_fn=emu_train_step,
        dataset=om_c_emu,
        model=eq_emu_small,
        epochs=args.emu_epochs,
        curriculum_epochs=0,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='emu_small'
    )

    eq_emu_large = ResCNN(
        in_features=1,
        latent=args.emu_latent_large,
        kernel_size=args.emu_kernel_large,
        out_features=1,
        n_blocks=args.emu_blocks_large,
        means=jnp.array(om_c_emu_mean), 
        stds=jnp.array(om_c_emu_std),
        activation=nnx.relu,
        rngs=rngs
    )
    print('Emulator (large) architecture')
    print(eq_emu_large)

    online_train_loop(
        key,
        optimizer=nnx.Optimizer(eq_emu_large, optax.adamw(args.emu_lr), wrt=nnx.Param),
        step_fn=emu_train_step,
        dataset=om_c_emu,
        model=eq_emu_large,
        epochs=args.emu_epochs,
        curriculum_epochs=0,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='emu_large'
    )

    # Get "physical" reference data
    om_ref = np.zeros((n_trajs, n_steps, eq_coarse.n_y, eq_coarse.n_x))
    for traj in range(n_trajs):
        for step in range(n_steps):
            om_ref[traj, step] = from_s(om_c_ref[traj, step])

    om_c_ref_mean = np.mean(om_ref)
    om_c_ref_std = np.std(om_ref)

    with np.printoptions(precision=4):
        print('Reference online model correction dataset statistics: om_c = {} ± σ({})'.format(
            om_c_ref_mean, 
            om_c_ref_std
        ))

    eq_off = FwdCNN(
        in_features=1,
        latent=args.sgs_latent,
        kernel_size=5,
        out_features=1,
        n_blocks=args.sgs_blocks,
        means=jnp.array(om_c_ref_mean),
        stds=jnp.array(om_c_ref_std),
        activation=nnx.relu,
        rngs=rngs
    )
    print('Offline model correction architecture')
    print(eq_off)

    offline_steps_epochs = args.offline_epochs * n_trajs
    offline_schedule = optax.cosine_decay_schedule(args.offline_lr, offline_steps_epochs, alpha=0.1)

    def off_loss(
        eq_off: nnx.Module, 
        om_c: jnp.ndarray, 
        tau_c: jnp.ndarray
    ) -> float:
        tau_hat_c = into_s(eq_off(jnp.expand_dims(from_s(om_c), -1)).squeeze())
        loss = jnp.mean(jax.vmap(physical_loss)(tau_hat_c, tau_c))
        return loss

    @nnx.jit 
    def off_train_step(eq_off, optimizer, batch_data: jnp.ndarray):
        om_c, tau_c = batch_data
        loss, grads = nnx.value_and_grad(off_loss)(eq_off, om_c, tau_c)
        optimizer.update(eq_off, grads)
        return loss

    # Get "spectral" reference data div tau
    tau_ref = np.zeros((n_trajs, n_steps, eq_coarse.n_ky, eq_coarse.n_kx), dtype=np.complex128)
    for traj in range(n_trajs):
        for step in range(n_steps):
            tau_ref[traj, step] = divergence(
                eq_coarse, 
                tau_x_ref[traj, step], 
                tau_y_ref[traj, step]
            )

    offline_train_loop(
        key,
        optimizer=nnx.Optimizer(eq_off, optax.adamw(offline_schedule), wrt=nnx.Param),
        step_fn=off_train_step,
        dataset=(
            np.reshape(om_c_ref, (-1, eq_coarse.n_ky, eq_coarse.n_kx)), 
            np.reshape(tau_ref,  (-1, eq_coarse.n_ky, eq_coarse.n_kx))
        ),
        model=eq_off,
        epochs=args.offline_epochs,
        batch_size=n_steps,
        data_path=data_path,
        name='off'
    )

    eq_state_ref = FwdCNN(
        in_features=1,
        latent=args.sgs_latent,
        kernel_size=5,
        out_features=1,
        n_blocks=args.sgs_blocks,
        means=jnp.array(om_c_ref_mean),
        stds=jnp.array(om_c_ref_std),
        activation=nnx.relu,
        rngs=rngs
    )
    print('Reference (state) online model correction architecture')
    print(eq_state_ref)

    state_steps_epochs = args.state_epochs * n_trajs
    state_schedule = optax.cosine_decay_schedule(args.state_lr, state_steps_epochs, alpha=0.01)

    def state_ref_flow_loss(
        eq_ref: nnx.Module, 
        om_c_0: jnp.ndarray, 
        om_c_traj: jnp.ndarray
    ) -> float:
        # The source of QG the system now also includes the CNN correction.
        def __source__(
            om_s: jnp.ndarray,
            _t: float, 
        ) -> jnp.ndarray:
            f_s = forcing_spectrum * forcing_det
            tau_s = into_s(eq_ref(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            return f_s + tau_s
    
        # Uses differentiable dynamical solver
        solver = dynamical_solver(
            eq_coarse,
            stepper.BPR353(ratio * dt),
            __source__
        )
        
        def __loop__(
            om_c: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            _, om_c = solver(om_c, 0)
            return (
                jnp.array(om_c), 
                jnp.array(om_c)
            )
    
        # Loop over time steps and accumulate states \bar{\omega}(t)
        _, om_hat_c = jax.lax.scan(__loop__, om_c_0, length=om_c_traj.shape[0])
        loss = jnp.mean(jax.vmap(kinetic_energy_loss)(om_hat_c, om_c_traj))
        return loss
    
    @nnx.jit 
    def state_ref_train_step(eq_ref, optimizer, traj_batch: jnp.ndarray):
        om_c_0, om_c_traj = traj_batch
        loss, grads = nnx.value_and_grad(state_ref_flow_loss)(eq_ref, om_c_0, om_c_traj)
        optimizer.update(eq_ref, grads)
        return loss

    online_train_loop(
        key,
        optimizer=nnx.Optimizer(eq_state_ref, optax.adamw(state_schedule), wrt=nnx.Param),
        step_fn=state_ref_train_step,
        dataset=om_c_ref,
        model=eq_state_ref,
        epochs=args.state_epochs,
        curriculum_epochs=0,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='state_ref_ek'
    )

    sgs_c_ref = np.zeros((n_trajs, n_steps, eq_coarse.n_ky, eq_coarse.n_kx), dtype=np.complex128)
    sgs_c_ref[:, 0 ] = om_c_ref[:, 0 ]
    sgs_c_ref[:, 1:] =  tau_ref[:, 1:]

    eq_subgrid_ref = FwdCNN(
        in_features=1,
        latent=args.sgs_latent,
        kernel_size=5,
        out_features=1,
        n_blocks=args.sgs_blocks,
        means=jnp.array(om_c_ref_mean),
        stds=jnp.array(om_c_ref_std),
        activation=nnx.relu,
        rngs=rngs
    )
    #print('Reference (subgrid) online model correction architecture')
    #print(eq_subgrid_ref)

    subgrid_steps_epochs = args.subgrid_epochs * n_trajs
    subgrid_schedule = optax.cosine_decay_schedule(args.subgrid_lr, subgrid_steps_epochs, alpha=0.01)

    def subgrid_ref_flow_loss(
        eq_ref: nnx.Module, 
        om_s_0: jnp.ndarray, 
        tau_s_traj: jnp.ndarray
    ) -> float:
        # The source of QG the system now also includes the CNN correction.
        def __source__(
            om_s: jnp.ndarray,
            _t: float, 
        ) -> jnp.ndarray:
            f_s = forcing_spectrum * forcing_det
            tau_s = into_s(eq_ref(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            return f_s + tau_s
    
        # Uses differentiable dynamical solver
        solver = dynamical_solver(
            eq_coarse,
            stepper.BPR353(ratio * dt),
            __source__
        )
        
        def __loop__(
            om_s: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            _, om_s = solver(om_s, 0)
            tau_s = into_s(eq_ref(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            return (
                jnp.array(om_s), 
                jnp.array(tau_s)
            )
    
        # Loop over time steps and accumulate subgrid terms \tau(t)
        _, tau_hat_s = jax.lax.scan(__loop__, om_s_0, length=tau_s_traj.shape[0])
        loss = jnp.mean(jax.vmap(physical_loss)(tau_hat_s, tau_s_traj))
        return loss
    
    @nnx.jit 
    def subgrid_ref_train_step(eq_ref, optimizer, traj_batch: jnp.ndarray):
        om_s_0, tau_s_traj = traj_batch
        loss, grads = nnx.value_and_grad(subgrid_ref_flow_loss)(eq_ref, om_s_0, tau_s_traj)
        optimizer.update(eq_ref, grads)
        return loss

    online_train_loop(
        key,
        optimizer=nnx.Optimizer(eq_subgrid_ref, optax.adamw(subgrid_schedule), wrt=nnx.Param),
        step_fn=subgrid_ref_train_step,
        dataset=sgs_c_ref,
        model=eq_subgrid_ref,
        epochs=args.subgrid_epochs,
        curriculum_epochs=0,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='subgrid_ref_mse'
    )

    abstract_model = nnx.eval_shape(lambda: ResCNN(
        in_features=1,
        latent=args.emu_latent_small,
        kernel_size=args.emu_kernel_small,
        out_features=1,
        n_blocks=args.emu_blocks_small,
        means=jnp.array(om_c_emu_mean), 
        stds=jnp.array(om_c_emu_std),
        activation=nnx.relu,
        rngs=nnx.Rngs(42)
    ))

    graph, abstract_state = nnx.split(abstract_model)

    checkpoint_path = os.path.join(data_path, 'emu_small_checkpoint/')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    import warnings
    with warnings.catch_warnings(action='ignore'):
        state = checkpointer.restore(checkpoint_path, abstract_state)
    eq_emu_small = nnx.merge(graph, state)

    eq_state_small = FwdCNN(
        in_features=1,
        latent=args.sgs_latent,
        kernel_size=5,
        out_features=1,
        n_blocks=args.sgs_blocks,
        means=jnp.array(om_c_ref_mean),
        stds=jnp.array(om_c_ref_std),
        activation=nnx.relu,
        rngs=rngs
    )
    #print('Emulator-based model architecture (state loss, small emulator)')
    #print(eq_state_small)

    def state_small_flow_loss(
        eq_state: nnx.Module, 
        om_s_0: jnp.ndarray, 
        om_s_traj: jnp.ndarray
    ) -> float:
        def __source__(om_s, t):
            f_s = forcing_spectrum * forcing_det
            tau_s = into_s(eq_state(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            return f_s + tau_s

        # The explicit term in the PDE is now based on the emulator.
        def __explicit__(om_s: jnp.ndarray, source: Callable, t: float):
            om_s_dt = into_s(eq_emu_small(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            c = 0
            return (
                om_s_dt + source(om_s, t),
                c
            )
        def __implicit__(om_s: jnp.ndarray):
            return 0
        def __solve__(eq_system: jnp.ndarray, rhs_om: jnp.ndarray) -> jnp.ndarray:
            return rhs_om
    
        # Uses differentiable dynamical solver
        solver = stepper.BPR353(ratio * dt)(
            eq_coarse,
            __source__,
            None,
            __implicit__,
            __explicit__,
            __solve__,
        )
        
        def __loop__(
            om_s: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            _, om_s = solver(om_s, 0)
            return (
                jnp.array(om_s), 
                jnp.array(om_s)
            )
    
        # Loop over time steps and accumulate states \bar{\omega}(t)
        _, om_hat_s = jax.lax.scan(__loop__, om_s_0, length=om_s_traj.shape[0])
        loss = jnp.mean(jax.vmap(kinetic_energy_loss)(om_hat_s, om_s_traj))
        return loss

    @nnx.jit 
    def state_small_train_step(eq_state_small, optimizer, traj_batch: jnp.ndarray):
        om_s_0, om_s_traj = traj_batch
        loss, grads = nnx.value_and_grad(state_small_flow_loss)(eq_state_small, om_s_0, om_s_traj)
        optimizer.update(eq_state_small, grads)
        return loss

    online_train_loop(
        key,
        optimizer=nnx.Optimizer(eq_state_small, optax.adamw(state_schedule), wrt=nnx.Param),
        step_fn=state_small_train_step,
        dataset=om_c_ref,
        model=eq_state_small,
        epochs=args.state_epochs,
        curriculum_epochs=0,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='state_small_ek'
    )

    eq_subgrid_small = FwdCNN(
        in_features=1,
        latent=args.sgs_latent,
        kernel_size=5,
        out_features=1,
        n_blocks=args.sgs_blocks,
        means=jnp.array(om_c_ref_mean),
        stds=jnp.array(om_c_ref_std),
        activation=nnx.relu,
        rngs=rngs
    )
    #print('Emulator-based model architecture (subgrid loss, small emulator)')
    #print(eq_subgrid_small)

    def subgrid_small_flow_loss(
        eq_subgrid: nnx.Module, 
        om_s_0: jnp.ndarray, 
        tau_s_traj: jnp.ndarray
    ) -> float:
        def __source__(om_s, t):
            f_s = forcing_spectrum * forcing_det
            tau_s = into_s(eq_subgrid(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            return f_s + tau_s

        # The explicit term in the PDE is now based on the emulator.
        def __explicit__(om_s: jnp.ndarray, source: Callable, t: float):
            om_s_dt = into_s(eq_emu_small(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            c = 0
            return (
                om_s_dt + source(om_s, t),
                c
            )
        def __implicit__(om_s: jnp.ndarray):
            return 0
        def __solve__(eq_system: jnp.ndarray, rhs_om: jnp.ndarray) -> jnp.ndarray:
            return rhs_om
    
        # Uses differentiable dynamical solver
        solver = stepper.BPR353(ratio * dt)(
            eq_coarse,
            __source__,
            None,
            __implicit__,
            __explicit__,
            __solve__,
        )
        
        def __loop__(
            om_s: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            _, om_s = solver(om_s, 0)
            tau_s = into_s(eq_subgrid(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            return (
                jnp.array(om_s), 
                jnp.array(tau_s)
            )
    
        # Loop over time steps and accumulate states \bar{\omega}(t)
        _, tau_hat_s = jax.lax.scan(__loop__, om_s_0, length=tau_s_traj.shape[0])
        loss = jnp.mean(jax.vmap(physical_loss)(tau_hat_s, tau_s_traj))
        return loss

    @nnx.jit 
    def subgrid_small_train_step(eq_subgrid_small, optimizer, traj_batch: jnp.ndarray):
        om_s_0, tau_s_traj = traj_batch
        loss, grads = nnx.value_and_grad(subgrid_small_flow_loss)(eq_subgrid_small, om_s_0, tau_s_traj)
        optimizer.update(eq_subgrid_small, grads)
        return loss
            
    online_train_loop(
        key,
        optimizer=nnx.Optimizer(eq_subgrid_small, optax.adamw(subgrid_schedule), wrt=nnx.Param),
        step_fn=subgrid_small_train_step,
        dataset=sgs_c_ref,
        model=eq_subgrid_small,
        epochs=args.subgrid_epochs,
        curriculum_epochs=0,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='subgrid_small_mse'
    )

    abstract_model = nnx.eval_shape(lambda: ResCNN(
        in_features=1,
        latent=args.emu_latent_large,
        kernel_size=args.emu_kernel_large,
        out_features=1,
        n_blocks=args.emu_blocks_large,
        means=jnp.array(om_c_emu_mean), 
        stds=jnp.array(om_c_emu_std),
        activation=nnx.relu,
        rngs=nnx.Rngs(42)
    ))

    graph, abstract_state = nnx.split(abstract_model)

    checkpoint_path = os.path.join(data_path, 'emu_large_checkpoint/')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    import warnings
    with warnings.catch_warnings(action='ignore'):
        state = checkpointer.restore(checkpoint_path, abstract_state)
    eq_emu_large = nnx.merge(graph, state)

    eq_state_large = FwdCNN(
        in_features=1,
        latent=args.sgs_latent,
        kernel_size=5,
        out_features=1,
        n_blocks=args.sgs_blocks,
        means=jnp.array(om_c_ref_mean),
        stds=jnp.array(om_c_ref_std),
        activation=nnx.relu,
        rngs=rngs
    )
    #print('Emulator-based model architecture (state loss, large emulator)')
    #print(eq_state_large)

    def state_large_flow_loss(
        eq_state: nnx.Module, 
        om_s_0: jnp.ndarray, 
        om_s_traj: jnp.ndarray
    ) -> float:
        def __source__(om_s, t):
            f_s = forcing_spectrum * forcing_det
            tau_s = into_s(eq_state(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            return f_s + tau_s

        # The explicit term in the PDE is now based on the emulator.
        def __explicit__(om_s: jnp.ndarray, source: Callable, t: float):
            om_s_dt = into_s(eq_emu_large(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            c = 0
            return (
                om_s_dt + source(om_s, t),
                c
            )
        def __implicit__(om_s: jnp.ndarray):
            return 0
        def __solve__(eq_system: jnp.ndarray, rhs_om: jnp.ndarray) -> jnp.ndarray:
            return rhs_om
    
        # Uses differentiable dynamical solver
        solver = stepper.BPR353(ratio * dt)(
            eq_coarse,
            __source__,
            None,
            __implicit__,
            __explicit__,
            __solve__,
        )
        
        def __loop__(
            om_s: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            _, om_s = solver(om_s, 0)
            return (
                jnp.array(om_s), 
                jnp.array(om_s)
            )
    
        # Loop over time steps and accumulate states \bar{\omega}(t)
        _, om_hat_s = jax.lax.scan(__loop__, om_s_0, length=om_s_traj.shape[0])
        loss = jnp.mean(jax.vmap(kinetic_energy_loss)(om_hat_s, om_s_traj))
        return loss

    @nnx.jit 
    def state_large_train_step(eq_state_large, optimizer, traj_batch: jnp.ndarray):
        om_s_0, om_s_traj = traj_batch
        loss, grads = nnx.value_and_grad(state_large_flow_loss)(eq_state_large, om_s_0, om_s_traj)
        optimizer.update(eq_state_large, grads)
        return loss

    online_train_loop(
        key,
        optimizer=nnx.Optimizer(eq_state_large, optax.adamw(state_schedule), wrt=nnx.Param),
        step_fn=state_large_train_step,
        dataset=om_c_ref,
        model=eq_state_large,
        epochs=args.state_epochs,
        curriculum_epochs=0,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='state_large_ek'
    )

    eq_subgrid_large = FwdCNN(
        in_features=1,
        latent=args.sgs_latent,
        kernel_size=5,
        out_features=1,
        n_blocks=args.sgs_blocks,
        means=jnp.array(om_c_ref_mean),
        stds=jnp.array(om_c_ref_std),
        activation=nnx.relu,
        rngs=rngs
    )
    #print('Emulator-based model architecture (subgrid loss, large emulator)')
    #print(eq_subgrid_large)

    def subgrid_large_flow_loss(
        eq_subgrid: nnx.Module, 
        om_s_0: jnp.ndarray, 
        tau_s_traj: jnp.ndarray
    ) -> float:
        def __source__(om_s, t):
            f_s = forcing_spectrum * forcing_det
            tau_s = into_s(eq_subgrid(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            return f_s + tau_s

        # The explicit term in the PDE is now based on the emulator.
        def __explicit__(om_s: jnp.ndarray, source: Callable, t: float):
            om_s_dt = into_s(eq_emu_large(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            c = 0
            return (
                om_s_dt + source(om_s, t),
                c
            )
        def __implicit__(om_s: jnp.ndarray):
            return 0
        def __solve__(eq_system: jnp.ndarray, rhs_om: jnp.ndarray) -> jnp.ndarray:
            return rhs_om
    
        # Uses differentiable dynamical solver
        solver = stepper.BPR353(ratio * dt)(
            eq_coarse,
            __source__,
            None,
            __implicit__,
            __explicit__,
            __solve__,
        )
        
        def __loop__(
            om_s: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            _, om_s = solver(om_s, 0)
            tau_s = into_s(eq_subgrid(jnp.expand_dims(from_s(om_s), (0,-1))).squeeze())
            return (
                jnp.array(om_s), 
                jnp.array(tau_s)
            )
    
        # Loop over time steps and accumulate states \bar{\omega}(t)
        _, tau_hat_s = jax.lax.scan(__loop__, om_s_0, length=tau_s_traj.shape[0])
        loss = jnp.mean(jax.vmap(physical_loss)(tau_hat_s, tau_s_traj))
        return loss

    @nnx.jit 
    def subgrid_large_train_step(eq_subgrid_large, optimizer, traj_batch: jnp.ndarray):
        om_s_0, tau_s_traj = traj_batch
        loss, grads = nnx.value_and_grad(subgrid_large_flow_loss)(eq_subgrid_large, om_s_0, tau_s_traj)
        optimizer.update(eq_subgrid_large, grads)
        return loss
            
    online_train_loop(
        key,
        optimizer=nnx.Optimizer(eq_subgrid_large, optax.adamw(subgrid_schedule), wrt=nnx.Param),
        step_fn=subgrid_large_train_step,
        dataset=sgs_c_ref,
        model=eq_subgrid_large,
        epochs=args.subgrid_epochs,
        curriculum_epochs=0,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='subgrid_large_mse'
    )

def online_train_loop(
    key,
    optimizer,
    step_fn: Callable,
    dataset: np.ndarray,
    model: nnx.Module, 
    epochs: int,
    curriculum_epochs: int,
    n_steps: int,
    n_trajs: int,
    data_path: str,
    name: str
):
    train_loss = []
    print('Training `' + name + '`...')
    pbar = tqdm.tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:
        # linear schedule
        cur_steps = n_steps if i >= curriculum_epochs else int(1 + max(1, i * n_steps / curriculum_epochs))
        key, subkey = jnr.split(key)
        data_sample = online_batch_gen(
            dataset, 
            n_trajs, 
            cur_steps,
            shuffle=True, 
            key=subkey
        )
    
        t_loss = 0.0
        for traj in range(n_trajs):
            traj_batch = next(data_sample)
            loss = step_fn(model, optimizer, traj_batch)
            t_loss += loss / (cur_steps - 1)
    
            pbar.set_postfix(
                sub_traj=traj,
                n_steps=(cur_steps - 1),
                loss=t_loss / (traj + 1),
                refresh=False
            )
        train_loss.append(t_loss / n_trajs)
    np.savez(os.path.join(data_path, name + '_loss.npz'), loss=train_loss)
    print('Saving `' + name + '` parameters...')
    _, state = nnx.split(model)
    checkpoint_path = os.path.join(data_path, name + '_checkpoint/')
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(
        checkpoint_path,
        state
    )
    return key
        
# Online (continuous) batch generator
def online_batch_gen(
    f: np.ndarray, 
    sub_trajs: int,
    n_steps: int,
    shuffle=True,  
    key=None
):
    """Generate trajectories (batch of continous samples) from a given dataset."""
    idx = np.arange(sub_trajs)
    if shuffle:
        idx = jnr.permutation(
            key, 
            idx,
            independent=True
        )
    
    for batch in range(sub_trajs):
        curr_idx = idx[batch]
        batch_inputs = f[curr_idx, 0] # starting state of the trajectory: y(t = 0)
        batch_target = f[curr_idx, 1:n_steps] # remaining states of the trajectory: y(t), t > 0
    
        yield (
            jnp.array(batch_inputs), 
            jnp.array(batch_target)
        )

def offline_train_loop(
    key,
    optimizer,
    step_fn: Callable,
    dataset: (np.ndarray, np.ndarray),
    model: nnx.Module, 
    epochs: int,
    batch_size: int,
    data_path: str,
    name: str
):
    train_loss = []
    print('Training `' + name + '`...')
    pbar = tqdm.tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:
        key, subkey = jnr.split(key)
        inputs, targets = dataset
        data_sample = offline_batch_gen(
            inputs,
            targets,
            batch_size, 
            shuffle=True, 
            key=subkey
        )

        n_batches = next(data_sample)
        t_loss = 0.0
        for batch in range(n_batches):
            batch_data = next(data_sample)
            loss = step_fn(model, optimizer, batch_data)
            t_loss += loss / (batch_size - 1)
    
            pbar.set_postfix(
                batch=batch,
                loss=t_loss / (batch + 1),
                refresh=False
            )
        train_loss.append(t_loss / n_batches)
    np.savez(os.path.join(data_path, name + '_loss.npz'), loss=train_loss)
    print('Saving `' + name + '` parameters...')
    _, state = nnx.split(model)
    checkpoint_path = os.path.join(data_path, name + '_checkpoint/')
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(
        checkpoint_path,
        state
    )
    return key

# Offline (classical) batch generator
def offline_batch_gen(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle=True,
    key=None
):
    """Generate a batch from a given dataset."""
    data_size = inputs.shape[0]
    idx = np.arange(data_size)
    if shuffle:
        idx = jnr.permutation(
            key, 
            idx,
            independent=True
        )

    n_batches = int(data_size / batch_size)
    yield n_batches
    
    for batch in range(n_batches):
        curr_idx = idx[batch * batch_size:(batch + 1) * batch_size]
        yield (
            jnp.array(inputs [curr_idx]),
            jnp.array(targets[curr_idx])
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python qg_train.py',
        description='Full tranining pipeline for the QG neural emulation and subgrid modeling'
    )
    
    parser.add_argument('-n', '--name', type=str, help='Name of the configuration', required=True)

    parser.add_argument('-emu_blocks_small', type=int, help='Number of MLP blocks for the (small) emulator', required=True)
    parser.add_argument('-emu_kernel_small', type=int, help='Kernel size for the (small) emulator', required=True)
    parser.add_argument('-emu_latent_small', type=int, help='Size of the latent space for the (small) emulator', required=True)
    parser.add_argument('-emu_blocks_large', type=int, help='Number of MLP blocks for the (large) emulator', required=True)
    parser.add_argument('-emu_kernel_large', type=int, help='Kernel size for the (large) emulator', required=True)
    parser.add_argument('-emu_latent_large', type=int, help='Size of the latent space for the (large) emulator', required=True)
    parser.add_argument('-emu_epochs', type=int, help='Number of trainig epochs for the emulator', required=True)
    parser.add_argument('-emu_lr', type=float, help='Learning rate for the emulator training', required=True)

    parser.add_argument('-sgs_blocks', type=int, help='Number of CNNNext blocks for the SGS model correction', required=True)
    parser.add_argument('-sgs_latent', type=int, help='Size of the latent space for the SGS model correction', required=True)
    
    parser.add_argument('-state_epochs', type=int, help='Number of trainig epochs for the offline SGS model correction', required=True)
    parser.add_argument('-state_lr', type=float, help='Learning rate for the offline SGS model correction training', required=True)
    
    parser.add_argument('-subgrid_epochs', type=int, help='Number of trainig epochs for the online SGS model correction', required=True)
    parser.add_argument('-subgrid_lr', type=float, help='Learning rate for the online SGS model correction training', required=True)

    parser.add_argument('-offline_epochs', type=int, help='Number of trainig epochs for the online SGS model correction', required=True)
    parser.add_argument('-offline_lr', type=float, help='Learning rate for the online SGS model correction training', required=True)
    
    args = parser.parse_args()
    main(args)
