<p align="center">
  <img src="https://github.com/hrkz/gradient-free-subgrid-neural-emulator/blob/main/assets/repo-abstract.png" alt="Repository Abstract" width="300"/>
</p>

> This repository contains a JAX implementation for the paper ["Gradient-free online learning of subgrid-scale dynamics with neural emulators"](https://arxiv.org/abs/2310.19385) submitted to the Journal of Advances in Modeling Earth Systems (JAMES). It can be used to reproduce results presented in the manuscript.

## 📦 Getting started

To setup and run the Python scripts and notebooks, we use [uv](https://docs.astral.sh/uv/) to manage the package dependencies in a custom environment

1. **Create and activate the environment**

```bash
cd gradient-free-subgrid-neural-emulator
uv init
```

2. **Install the required packages**

```bash
uv add -r requirements.txt
```

Note: you need the access to a GPU device since the default requirement packages are based on the CUDA version of JAX. Running the code on CPU is posible, but modification of the `requirements.txt` file is necessary.

## 🚀 Reproducing results

Below are the steps used to produce the results and figures from the paper. These steps have been tested with a NVIDIA H100 but a device with 16GB of (V)RAM should be enough to run at the considered numerical resolutions.

### 🌀 Demonstration: two-timescales Lorenz-96

The first step for the demonstration is to launch the `docs/l96_data.ipynb` notebook. Here, the default parameters are set to the ones used in the paper. Running the cells until the `get_dataset_stats`, we obtain the number of steps $N_t$ for each trajectory, corresponding to 10% of the decorrelation time $t_c = 6$ of the system. The following cells will generate the datasets for both training the neural emulator and the subgrid-scale model.

#### Training the models

Now that the datasets have been saved in `data/l96/` (hence `-n l96`), we can launch the training script that sequentially train the neural emulator and then use it to train the SGS model. Following the learning setup described in Section 4, we run the following command:

```bash
uv run l96_train.py -n l96 \
    -emu_blocks 3 -emu_latent 64 -emu_epochs 20000 -emu_lr 1e-5 \
    -sgs_blocks 3 -sgs_latent 16 -sgs_epochs 5000 -sgs_lr 2e-4
```

Once finished, the training checkpoint are saved in `data/l96/` and we can use the model parameters for evaluation.

#### Evaluating the models

Finally, we want to evaluate the trained models and their performance and launch the `docs/l96_eval.ipynb` notebook. Simulations for the two-timescale system and the SGS models are run for $300 t_c$, equivalent to 600000 and 30000 time steps, respectively. The following cells can be used to compute and visualise the metrics described in the paper.

### 🌊 Application: quasi-geostrophic dynamics

In the paper, we describe a quasi-geostrophic configuration in a forced-dissiped setup, and we perform learning and inference on the steady-state dynamics.
To spinup the simulation and generate a snapshot, we run the following command (parameters are those used in the paper):

```bash
uv run qg_spinup.py -n qg-det -nu 1e-5 -mu 2e-2 -beta 30 -sigma 10.0 -k_f 15 -n_kx 1025 -n_ky 2048 -dt 2e-4 -T 1000
```

Upon finishing, the script saves a `snapshot.h5` file under the folder `data/qg-det`. Note that we did not evaluate the capabilities of our approach
on a variety of configurations (change in Coriolis parameter $\beta$, viscosity $\nu$, bottom drag $\mu$ or forcing $\mathcal{F}$), but it should work fine in theory.

#### Generating the coarse-grained dataset and traning the models

We can now launch the notebook `docs/qg_data.ipynb` notebook. The first cells load the simulation state from the spinup file and run short simulations in order to evaluate some statitics, including the turnover time $t_L$ and the decorrelation time $t_c$. Providing the statistics to the `get_dataset_stats` function, we are given with the dataset parameters (number of samples from $N_\text{traj}$ trajectories of $N_t$ steps). The remaining cells show the effect of coarse-graining and generate the SGS and emulator datasets, respectively.

The procedure to train the models is similar to the one used for the L96 demonstration. We run the following command using the learning parameters described in Section 5:

```bash
uv run qg_train.py -n qg-det \
    -emu_blocks_small 8 -emu_kernel_small 5 -emu_latent_small 32 \
    -emu_blocks_large 8 -emu_kernel_large 7 -emu_latent_large 64 \
    -emu_epochs 500 -emu_lr 1e-4 \
    -sgs_blocks 5 -sgs_latent 64 \
    -state_epochs 500 -state_lr 1e-4 \
    -subgrid_epochs 500 -subgrid_lr 1e-3 \
    -offline_epochs 500 -offline_lr 1e-3
```

Once finished, the training checkpoint are saved in `data/qg-det/` and we can use the model parameters for evaluation.

### Evaluating the models

Finally, we want to evaluate the trained models and baselines the reference DNS and some other baselines. We run the evaluation script for $t = 240$, corresponding to almost 300 turnovers, and save 2500 samples. Note that depending on the number of saved samples, the data generated by the evaluation can be large, hence, the script allows for generating data in a separate directory `my_path`:

```bash
uv run qg_eval.py -n qg-det --save_path 'my_path' -n_logs 2500 -T 240 -ml_model_blocks 5 -ml_model_latent 64
```

We can now compute and visualise some metrics in the `docs/qg_metrics.ipynb` notebook. Note that the similarity metrics should only be computed for models that have not crashed in during the evaluation script.

## 📖 Citing

Still a preprint.
