# Differentiable MPC — Identifiability Experiments

Course project investigating dynamics identifiability via Differentiable MPC on the pendulum system.
Built on top of [Amos et al., NeurIPS 2018](https://arxiv.org/abs/1810.13400) and their [mpc.pytorch](https://github.com/locuslab/mpc.pytorch) library (bundled in `mpc_pytorch_lib/`).

## Setup

```bash
source setup.sh
```

Creates a virtual environment in `.venv/`, installs all dependencies, and sets `PYTHONPATH`.

## Running the experiments

Open and run `experiments.ipynb` from the repo root.

The notebook reproduces the `mpc.dx` dynamics identifiability experiment end-to-end:
- Generates expert data from a pendulum MPC with true parameters $(g=10, m=1, l=1)$
- Runs 8 random initializations in parallel, each minimizing imitation loss via differentiable iLQR
- Plots imitation loss, parameter trajectories, identifiability scatter, and loss vs parameter MSE comparison
- Includes an unrestricted initialization ablation

Results are saved to `results/conservative/` and `results/unrestricted/`.