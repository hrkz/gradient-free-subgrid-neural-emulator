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
from models.l96 import (
    L96, 
    dynamical_solver,
    dynamical_solver_single
)
from models.mlp import (
    FwdMLP, 
    ResMLP,
)

def main(args: argparse.Namespace) -> None:
    key = jnr.key(42)
    rngs = nnx.Rngs(key)
    
    data_path = os.path.join(os.path.join(os.getcwd(), 'data'), args.name)
    with h5py.File(os.path.join(data_path, 'datasets.h5'), 'r') as f:
        dt = f.attrs['dt']
        n_steps = f.attrs['n_steps']
        n_trajs = f.attrs['n_trajs']
        source_val = f.attrs['source_val']

        x_k_ref = np.array(f['x_k_ref'])
        y_j_ref = np.array(f['y_j_ref'])
        x_k_emu = np.array(f['x_k_emu'])
    eq, time, x_k, y_j = L96.load(
        os.path.join(data_path, 'snapshot.h5')
    )
    print(eq)

    x_k_emu_means = np.mean(x_k_emu, axis=(0,1))
    x_k_emu_stds = np.std(x_k_emu, axis=(0,1))

    with np.printoptions(precision=4):
        print('Emulator dataset statistics: x_k = {} ± σ({})'.format(
            x_k_emu_means, 
            x_k_emu_stds
        ))

    eq_emu = ResMLP(
        in_features=eq.n_k,
        latent=args.emu_latent,
        out_features=eq.n_k,
        n_blocks=args.emu_blocks,
        means=jnp.array(x_k_emu_means), 
        stds=jnp.array(x_k_emu_stds),
        activation=nnx.relu,
        rngs=rngs
    )
    print('Emulator architecture')
    print(eq_emu)

    def emu_flow_loss(
        eq_emu: nnx.Module, 
        x_k_0: jnp.ndarray, 
        x_k_traj: jnp.ndarray
    ) -> float:
        def __source__(
            x_k: jnp.ndarray,
            _t: float, 
        ) -> jnp.ndarray:
            return source_val

        # The explicit term in the ODE is now based on the emulator.
        def __explicit__(
            x_k,
            _y_j,
            source
        ):
            x_dot = eq_emu(x_k) + source    
            return (
                x_dot,
                0
            )
    
        # Uses differentiable dynamical solver
        solver = stepper.RK4(eq.n_j * dt)(
            __source__, 
            __explicit__
        )
        
        def __loop__(
            x_k: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            x_k, _ = solver(x_k, 0, 0)
            return (
                jnp.array(x_k), 
                jnp.stack(x_k, axis=-1)
            )
    
        # Loop over time steps and accumulate states \bar{X_k}(t)
        _, x_k_hat = jax.lax.scan(__loop__, x_k_0, length=n_steps - 1)
        loss = jnp.mean(jnp.square(x_k_hat - x_k_traj))
        return loss
    
    @nnx.jit 
    def emu_train_step(eq_emu, optimizer, traj_batch: jnp.ndarray):
        x_k_0, x_k_traj = traj_batch
        loss, grads = nnx.value_and_grad(emu_flow_loss)(eq_emu, x_k_0, x_k_traj)
        optimizer.update(eq_emu, grads)
        return loss

    key = train_loop(
        key,
        optimizer=nnx.Optimizer(eq_emu, optax.adamw(args.emu_lr), wrt=nnx.Param),
        step_fn=emu_train_step,
        dataset=x_k_emu,
        model=eq_emu,
        epochs=args.emu_epochs,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='emu'
    )

    x_k_ref_means = np.mean(x_k_ref, axis=(0,1))
    x_k_ref_stds = np.std(x_k_ref, axis=(0,1))

    with np.printoptions(precision=4):
        print('Reference (SGS model correction) dataset statistics: x_k = {} ± σ({})'.format(
            x_k_ref_means, 
            x_k_ref_stds
        ))

    eq_ref = FwdMLP(
        in_features=eq.n_k,
        latent=args.sgs_latent,
        out_features=eq.n_k,
        n_blocks=args.sgs_blocks,
        means=jnp.array(x_k_ref_means), 
        stds=jnp.array(x_k_ref_stds),
        activation=nnx.relu,
        rngs=rngs
    )
    print('Reference (SGS model correction) architecture')
    print(eq_ref)

    # "Single-timescale" model, with N_j = 0
    eq_single = L96(
        b=eq.b,
        c=eq.c,
        h=eq.h,
        n_k=eq.n_k,
        n_j=0
    )

    def ref_flow_loss(
        eq_ref: nnx.Module, 
        x_k_0: jnp.ndarray, 
        x_k_traj: jnp.ndarray
    ) -> float:
        # The source of L96 the system now also includes the MLP correction.
        def __source__(
            x_k: jnp.ndarray,
            _t: float, 
        ) -> jnp.ndarray:
            tau = eq_ref(x_k)
            return source_val + tau
    
        # Uses differentiable dynamical solver
        solver = dynamical_solver_single(
            eq_single,
            stepper.RK4(eq.n_j * dt),
            __source__
        )
        
        def __loop__(
            x_k: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            x_k, _ = solver(x_k, 0, 0)
            return (
                jnp.array(x_k), 
                jnp.array(x_k)
            )
    
        # Loop over time steps and accumulate states \bar{X_k}(t)
        _, x_k_hat = jax.lax.scan(__loop__, x_k_0, length=n_steps - 1)
        loss = jnp.mean(jnp.square(x_k_hat - x_k_traj))
        return loss
    
    @nnx.jit 
    def ref_train_step(eq_ref, optimizer, traj_batch: jnp.ndarray):
        x_k_0, x_k_traj = traj_batch
        loss, grads = nnx.value_and_grad(ref_flow_loss)(eq_ref, x_k_0, x_k_traj)
        optimizer.update(eq_ref, grads)
        return loss

    key = train_loop(
        key,
        optimizer=nnx.Optimizer(eq_ref, optax.adamw(args.sgs_lr), wrt=nnx.Param),
        step_fn=ref_train_step,
        dataset=x_k_ref,
        model=eq_ref,
        epochs=args.sgs_epochs,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='ref'
    )

    eq_state = FwdMLP(
        in_features=eq.n_k,
        latent=args.sgs_latent,
        out_features=eq.n_k,
        n_blocks=args.sgs_blocks,
        means=jnp.array(x_k_ref_means), 
        stds=jnp.array(x_k_ref_stds),
        activation=nnx.relu,
        rngs=rngs
    )
    print('Emulator-based model architecture (state loss)')
    #print(eq_state)

    abstract_model = nnx.eval_shape(lambda: ResMLP(
        in_features=eq.n_k,
        latent=args.emu_latent,
        out_features=eq.n_k,
        n_blocks=args.emu_blocks,
        means=jnp.array(x_k_emu_means), 
        stds=jnp.array(x_k_emu_stds),
        activation=nnx.relu,
        rngs=nnx.Rngs(42)
    ))

    graph, abstract_state = nnx.split(abstract_model)

    checkpoint_path = os.path.join(data_path, 'emu_checkpoint/')
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    import warnings
    with warnings.catch_warnings(action='ignore'):
        state = checkpointer.restore(checkpoint_path, abstract_state)
    eq_emu = nnx.merge(graph, state)

    def state_flow_loss(
        eq_state: nnx.Module, 
        x_k_0: jnp.ndarray, 
        x_k_traj: jnp.ndarray
    ) -> float:
        def __source__(
            x_k: jnp.ndarray,
            _t: float, 
        ) -> jnp.ndarray:
            tau = eq_state(x_k)
            return source_val + tau

        # The explicit term in the ODE is still only based on the emulator.
        def __explicit__(
            x_k,
            _y_j,
            source
        ):
            x_dot = eq_emu(x_k) + source    
            return (
                x_dot,
                0
            )
    
        # Uses differentiable dynamical solver
        solver = stepper.RK4(eq.n_j * dt)(
            __source__, 
            __explicit__
        )
        
        def __loop__(
            x_k: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            x_k, _ = solver(x_k, 0, 0)
            return (
                jnp.array(x_k), 
                jnp.stack(x_k, axis=-1)
            )
    
        # Loop over time steps and accumulate states \bar{X_k}(t)
        _, x_k_hat = jax.lax.scan(__loop__, x_k_0, length=n_steps - 1)
        loss = jnp.mean(jnp.square(x_k_hat - x_k_traj))
        return loss
    
    @nnx.jit 
    def state_train_step(eq_state, optimizer, traj_batch: jnp.ndarray):
        x_k_0, x_k_traj = traj_batch
        loss, grads = nnx.value_and_grad(state_flow_loss)(eq_state, x_k_0, x_k_traj)
        optimizer.update(eq_state, grads)
        return loss

    key = train_loop(
        key,
        optimizer=nnx.Optimizer(eq_state, optax.adamw(args.sgs_lr), wrt=nnx.Param),
        step_fn=state_train_step,
        dataset=x_k_ref,
        model=eq_state,
        epochs=args.sgs_epochs,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='state'
    )

    eq_subgrid = FwdMLP(
        in_features=eq.n_k,
        latent=args.sgs_latent,
        out_features=eq.n_k,
        n_blocks=args.sgs_blocks,
        means=jnp.array(x_k_ref_means), 
        stds=jnp.array(x_k_ref_stds),
        activation=nnx.relu,
        rngs=rngs
    )
    print('Emulator-based SGS model correction architecture (subgrid loss)')
    #print(eq_subgrid)

    def subgrid_flow_loss(
        eq_subgrid: nnx.Module, 
        x_k_0: jnp.ndarray, 
        y_j_traj: jnp.ndarray
    ) -> float:
        def __source__(
            x_k: jnp.ndarray,
            _t: float, 
        ) -> jnp.ndarray:
            tau = eq_subgrid(x_k)
            return source_val + tau

        # The explicit term in the ODE is still only based on the emulator.
        def __explicit__(
            x_k,
            _y_j,
            source
        ):
            x_dot = eq_emu(x_k) + source    
            return (
                x_dot,
                0
            )
    
        # Uses differentiable dynamical solver
        solver = stepper.RK4(eq.n_j * dt)(
            __source__, 
            __explicit__
        )
        
        def __loop__(
            x_k: jnp.ndarray, 
            cur_step: int
        ) -> jnp.ndarray:
            x_k, _ = solver(x_k, 0, 0)
            y_j = eq_subgrid(x_k)
            return (
                jnp.array(x_k), 
                jnp.stack(y_j, axis=-1)
            )
    
        # Loop over time steps and accumulate states \bar{X_k}(t)
        _, y_j_hat = jax.lax.scan(__loop__, x_k_0, length=n_steps - 1)
        loss = jnp.mean(jnp.square(y_j_hat - y_j_traj))
        return loss
    
    @nnx.jit 
    def subgrid_train_step(eq_subgrid, optimizer, traj_batch: jnp.ndarray):
        x_k_0, y_j_traj = traj_batch
        loss, grads = nnx.value_and_grad(subgrid_flow_loss)(eq_subgrid, x_k_0, y_j_traj)
        optimizer.update(eq_subgrid, grads)
        return loss

    subgrid_ref = np.copy(x_k_ref)
    subgrid_ref[:, 1:] = -((eq.h * eq.c) / eq.b) * np.sum(y_j_ref[:, 1:], axis=2)

    key = train_loop(
        key,
        optimizer=nnx.Optimizer(eq_subgrid, optax.adamw(args.sgs_lr), wrt=nnx.Param),
        step_fn=subgrid_train_step,
        dataset=subgrid_ref,
        model=eq_subgrid,
        epochs=args.sgs_epochs,
        n_steps=n_steps,
        n_trajs=n_trajs,
        data_path=data_path,
        name='subgrid'
    )

def train_loop(
    key,
    optimizer,
    step_fn: Callable,
    dataset: np.ndarray,
    model: nnx.Module, 
    epochs: int,
    n_steps: int,
    n_trajs: int,
    data_path: str,
    name: str
):
    train_loss = []
    print('Training `' + name + '`...')
    pbar = tqdm.tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i in pbar:
        key, subkey = jnr.split(key)
        data_sample = batch_gen(
            dataset, 
            n_trajs, 
            shuffle=True, 
            key=subkey
        )
    
        t_loss = 0.0
        for traj in range(n_trajs):
            traj_batch = next(data_sample)
            loss = step_fn(model, optimizer, traj_batch)
            t_loss += loss / (n_steps - 1)
    
            pbar.set_postfix(
                sub_traj=traj,
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

# Batch generator
def batch_gen(
    f: np.ndarray, 
    sub_trajs: int,
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
        batch_target = f[curr_idx, 1:] # remaining states of the trajectory: y(t), t > 0
    
        yield (
            jnp.array(batch_inputs), 
            jnp.array(batch_target)
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python l96_train.py',
        description='Full tranining pipeline for the L96 neural emulation and subgrid modeling'
    )
    
    parser.add_argument('-n', '--name', type=str, help='Name of the configuration', required=True)

    parser.add_argument('-emu_blocks', type=int, help='Number of MLP blocks for the emulator', required=True)
    parser.add_argument('-emu_latent', type=int, help='Size of the latent space for the emulator', required=True)
    parser.add_argument('-emu_epochs', type=int, help='Number of trainig epochs for the emulator', required=True)
    parser.add_argument('-emu_lr', type=float, help='Learning rate for the emulator training', required=True)

    parser.add_argument('-sgs_blocks', type=int, help='Number of MLP blocks for the SGS model correction', required=True)
    parser.add_argument('-sgs_latent', type=int, help='Size of the latent space for the SGS model correction', required=True)
    parser.add_argument('-sgs_epochs', type=int, help='Number of trainig epochs for the SGS model correction', required=True)
    parser.add_argument('-sgs_lr', type=float, help='Learning rate for the SGS model correction training', required=True)
    
    args = parser.parse_args()
    main(args)
