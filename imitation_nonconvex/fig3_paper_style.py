#!/usr/bin/env python3
"""
fig3_paper_style.py — Reproduce Figure 3 of the Differentiable MPC paper,
adapted for the pendulum mpc.dx experiment.

The paper's Figure 3 (Section 5.2, LQR case) shows two panels:
  Left:  Imitation Loss  = ||u* - û||²   vs iteration
  Right: Model Loss      = MSE(θ*, θ̂)   vs iteration   (parameter distance)

For pendulum the individual parameters (g, m, l) are NOT identifiable — only
the functional ratios g/l and 1/(m*l²) matter. So we produce three panels:

  Panel 1 — Imitation Loss      (same as paper, linear scale)
  Panel 2 — Dynamics MSE        (correct model loss: ||f(x,u;θ̂) - f(x,u;θ*)||²
                                  averaged over a fixed test grid)
  Panel 3 — Parameter MSE       (what the paper literally shows for LQR;
                                  misleading here due to identifiability, but
                                  included for direct comparison)

The 2-panel figure fig3_paper_fig3_match.png replicates the paper's exact layout
(Imitation Loss | Dynamics MSE) on linear axes.

Usage:
    python3 fig3_paper_style.py [--out_dir fig3_out] [--nowait]

    # Point at the unrestricted run:
    python3 fig3_paper_style.py --out_dir fig3_out_unrestricted

Outputs (written to the same out_dir):
    fig3_paper_match.png     — 2-panel Figure 3 replica (imitation + dynamics MSE)
    fig3_paper_full.png      — 3-panel extended version (+ parameter MSE)
"""

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_DIR       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DIR, '..'))
_MPC_LIB   = os.path.join(_REPO_ROOT, 'mpc_pytorch_lib')
if _MPC_LIB not in sys.path:
    sys.path.insert(0, _MPC_LIB)

from mpc.env_dx.pendulum import PendulumDx

TRUE_G, TRUE_M, TRUE_L = 10.0, 1.0, 1.0


# ── Test grid for dynamics MSE ────────────────────────────────────────────────

def build_test_grid(n=500, seed=999):
    """
    Sample n random (x, u) pairs uniformly from a wide region of state-space.
    Returns (test_x, test_u) as (n, 3) and (n, 1) tensors.
    """
    torch.manual_seed(seed)
    th    = torch.FloatTensor(n).uniform_(-np.pi, np.pi)
    thdot = torch.FloatTensor(n).uniform_(-5.0,   5.0)
    test_x = torch.stack([torch.cos(th), torch.sin(th), thdot], dim=1)  # (n, 3)
    test_u = torch.FloatTensor(n, 1).uniform_(-2.0, 2.0)                 # (n, 1)
    return test_x, test_u


# ── Model loss computation ─────────────────────────────────────────────────────

def compute_dynamics_mse_batch(g_traj, m_traj, l_traj, true_dx, test_x, test_u):
    """
    Compute dynamics MSE at every iteration for one seed.

    Returns a numpy array of shape (n_iter,).

    Rather than calling PendulumDx per iteration (slow), we vectorise by
    observing that the pendulum dynamics depend only on two ratios:
        α = g/l      (controls the gravity term)
        β = 1/(m*l²) (controls the torque term)

    newdth = dth + dt * (3α/2 * sin_th + 3β * u)   [note: sign matches code]

    True values: α* = 10/1 = 10,  β* = 1/(1*1) = 1
    """
    with torch.no_grad():
        dt = true_dx.dt  # 0.05

        cos_th = test_x[:, 0]
        sin_th = test_x[:, 1]
        dth    = test_x[:, 2]
        u_cl   = torch.clamp(test_u[:, 0], -2.0, 2.0)

        # Vectorised over iterations — compute α and β for each step
        g_arr = np.asarray(g_traj, dtype=np.float32)
        m_arr = np.asarray(m_traj, dtype=np.float32)
        l_arr = np.asarray(l_traj, dtype=np.float32)
        alpha = g_arr / l_arr          # shape (n_iter,)
        beta  = 1.0 / (m_arr * l_arr**2)

        # True alpha, beta
        alpha_true = TRUE_G / TRUE_L          # 10.0
        beta_true  = 1.0 / (TRUE_M * TRUE_L**2)  # 1.0

        # Residuals: d_newdth = dt * 3 * [(α-α*)/2 * sin_th  +  (β-β*) * u]
        # Shape broadcasting: (n_iter, 1) × (n,) → (n_iter, n)
        sin_th_np = sin_th.numpy()   # (n,)
        u_np      = u_cl.numpy()     # (n,)

        d_alpha = (alpha - alpha_true)[:, None]   # (n_iter, 1)
        d_beta  = (beta  - beta_true )[:, None]   # (n_iter, 1)

        # Difference in angular acceleration (×dt)
        # From forward: newdth = dth + dt*(-3g/(2l)*(-sin_th) + 3u/(ml²))
        #                       = dth + dt*(3α/2 * sin_th + 3β * u)
        d_dth = dt * (1.5 * d_alpha * sin_th_np + 3.0 * d_beta * u_np)   # (n_iter, n)

        # Difference in new angle (approximated as d_dth * dt, since newth = th + newdth*dt)
        th_np = np.arctan2(sin_th.numpy(), cos_th.numpy())
        d_th  = d_dth * dt   # (n_iter, n)

        # Difference in (cos_newth, sin_newth, newdth)
        newth_true_np = (th_np + (dth.numpy() + dt*(1.5*alpha_true*sin_th_np
                                                     + 3.0*beta_true*u_np)) * dt)
        newth_hat = newth_true_np[None, :] + d_th   # (n_iter, n)

        d_cos  = np.cos(newth_hat) - np.cos(newth_true_np)[None, :]
        d_sin  = np.sin(newth_hat) - np.sin(newth_true_np)[None, :]
        # d_dth is already the difference in newdth

        mse = (d_cos**2 + d_sin**2 + d_dth**2).mean(axis=1)  # (n_iter,)
    return mse


def compute_param_mse_series(g_traj, m_traj, l_traj):
    """Return parameter MSE at every iteration: ((g-g*)² + (m-m*)² + (l-l*)²) / 3."""
    g_arr = np.asarray(g_traj, dtype=np.float64)
    m_arr = np.asarray(m_traj, dtype=np.float64)
    l_arr = np.asarray(l_traj, dtype=np.float64)
    return ((g_arr - TRUE_G)**2 + (m_arr - TRUE_M)**2 + (l_arr - TRUE_L)**2) / 3.0


# ── Plotting ───────────────────────────────────────────────────────────────────

def make_paper_plots(all_logs, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    n_seeds = len(all_logs)
    cmap    = plt.cm.tab10
    colors  = [cmap(i / max(n_seeds - 1, 1)) for i in range(n_seeds)]

    n_iter  = len(all_logs[0]['im_loss'])
    iters   = np.arange(n_iter)

    true_dx = PendulumDx()
    test_x, test_u = build_test_grid()

    print('Computing model losses...')
    dyn_mse_all   = []
    param_mse_all = []
    for s, log in enumerate(all_logs):
        print(f'  seed {s}...', end=' ', flush=True)
        dyn_mse_all.append(
            compute_dynamics_mse_batch(log['g'], log['m'], log['l'],
                                       true_dx, test_x, test_u)
        )
        param_mse_all.append(
            compute_param_mse_series(log['g'], log['m'], log['l'])
        )
        print('done')

    # ── Figure 3 exact match: 2 panels (Imitation Loss | Parameter MSE) ────────
    # This directly replicates Figure 3 of the paper:
    #   Left:  Imitation Loss   (paper: 0–1.2, linear)
    #   Right: Model Loss       (paper: MSE(θ*, θ̂), linear)
    # For pendulum: Model Loss = ((ĝ-10)² + (m̂-1)² + (l̂-1)²) / 3
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for s, log in enumerate(all_logs):
        ax.semilogy(iters, log['im_loss'], color=colors[s], alpha=0.85,
                    linewidth=1.5, label=f'seed {s}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Imitation Loss  (log scale)', fontsize=12)
    ax.set_title('Imitation Loss', fontsize=13)
    ax.set_xlim(0, n_iter - 1)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    for s in range(n_seeds):
        ax.semilogy(iters, param_mse_all[s], color=colors[s], alpha=0.85,
                    linewidth=1.5, label=f'seed {s}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(r'Model Loss   $\mathrm{MSE}(\theta^*, \hat\theta)$  (log scale)', fontsize=11)
    ax.set_title(r'Model Loss  $= \frac{(\hat{g}-g^*)^2+(\hat{m}-m^*)^2+(\hat{l}-l^*)^2}{3}$',
                 fontsize=12)
    ax.set_xlim(0, n_iter - 1)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('Figure 3 replica — Pendulum mpc.dx   '
                 '(paper: LQR, θ={A,B};  here: pendulum, θ={g,m,l})',
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, 'fig3_paper_match.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p}')

    # ── Figure 3 with dynamics MSE on right (more principled for pendulum) ─────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for s, log in enumerate(all_logs):
        ax.semilogy(iters, log['im_loss'], color=colors[s], alpha=0.85,
                    linewidth=1.5, label=f'seed {s}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Imitation Loss  (log scale)', fontsize=12)
    ax.set_title('Imitation Loss', fontsize=13)
    ax.set_xlim(0, n_iter - 1)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    for s in range(n_seeds):
        ax.semilogy(iters, dyn_mse_all[s], color=colors[s], alpha=0.85,
                    linewidth=1.5, label=f'seed {s}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(r'Dynamics MSE   $\|f(x,u;\hat\theta)-f(x,u;\theta^*)\|^2$  (log)',
                  fontsize=11)
    ax.set_title('Model Loss  (dynamics prediction error, 500 test pairs)',
                 fontsize=12)
    ax.set_xlim(0, n_iter - 1)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('Figure 3 variant — Pendulum mpc.dx   '
                 'Dynamics MSE → 0 when g/l and 1/ml² converge (even if g,m,l do not)',
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, 'fig3_dyn_mse.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p}')

    # ── Full 3-panel: + Parameter MSE ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    ax = axes[0]
    for s, log in enumerate(all_logs):
        ax.semilogy(iters, log['im_loss'], color=colors[s], alpha=0.85,
                    linewidth=1.5, label=f'seed {s}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Imitation Loss  (log)', fontsize=11)
    ax.set_title('Imitation Loss', fontsize=12)
    ax.set_xlim(0, n_iter - 1)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    for s in range(n_seeds):
        ax.semilogy(iters, dyn_mse_all[s], color=colors[s], alpha=0.85,
                    linewidth=1.5, label=f'seed {s}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel(r'$\|f(x,u;\hat\theta)-f(x,u;\theta^*)\|^2$  (log)', fontsize=11)
    ax.set_title('Dynamics MSE  (correct model loss)', fontsize=12)
    ax.set_xlim(0, n_iter - 1)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[2]
    for s in range(n_seeds):
        ax.semilogy(iters, param_mse_all[s], color=colors[s], alpha=0.85,
                    linewidth=1.5, label=f'seed {s}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel(r'$\frac{(\hat g-g^*)^2+(\hat m-m^*)^2+(\hat l-l^*)^2}{3}$  (log)',
                  fontsize=11)
    ax.set_title('Parameter MSE  (paper\'s metric, misleading here)', fontsize=12)
    ax.set_xlim(0, n_iter - 1)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    note = ('Note: Individual (g,m,l) are not identifiable from imitation loss alone — '
            'only g/l and 1/(m·l²) are. Dynamics MSE → 0 whenever both ratios converge, '
            'even if raw parameters differ from true values.')
    fig.suptitle('Figure 3 extended — Pendulum mpc.dx\n' + note, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, 'fig3_paper_full.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p}')

    # ── Print final summary ────────────────────────────────────────────────────
    print(f'\n{"Seed":>4}  {"Im.loss":>10}  {"Dyn.MSE":>10}  {"Param.MSE":>10}  '
          f'{"g/l":>8}  {"1/ml²":>8}')
    for s, log in enumerate(all_logs):
        print(f'{s:>4}  {log["im_loss"][-1]:>10.4e}  '
              f'{dyn_mse_all[s][-1]:>10.4e}  '
              f'{param_mse_all[s][-1]:>10.4e}  '
              f'{log["g_over_l"][-1]:>8.4f}  '
              f'{log["inv_ml2"][-1]:>8.4f}')
    print(f'{"true":>4}  {"—":>10}  {"0.0000":>10}  {"0.0000":>10}  '
          f'{TRUE_G/TRUE_L:>8.4f}  {1/(TRUE_M*TRUE_L**2):>8.4f}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out_dir', type=str,
                   default=os.path.join(_DIR, 'fig3_out'),
                   help='Directory containing seed_N.pt checkpoints (default: fig3_out)')
    p.add_argument('--n_seeds', type=int, default=8)
    p.add_argument('--nowait', action='store_true',
                   help='Plot with whatever seeds are present; do not wait for missing ones')
    return p.parse_args()


if __name__ == '__main__':
    cfg = parse_args()

    present = [s for s in range(cfg.n_seeds)
               if os.path.exists(os.path.join(cfg.out_dir, f'seed_{s}.pt'))]
    missing = [s for s in range(cfg.n_seeds) if s not in present]
    if missing and not cfg.nowait:
        print(f'Missing seeds: {missing}  (use --nowait to plot anyway)')
        sys.exit(1)
    if not present:
        print('No seed files found.')
        sys.exit(1)

    print(f'Loading {len(present)} seeds from {cfg.out_dir}/')
    all_logs = []
    for s in sorted(present):
        log = torch.load(os.path.join(cfg.out_dir, f'seed_{s}.pt'))
        all_logs.append(log)

    make_paper_plots(all_logs, cfg.out_dir)
    print(f'\nDone. Plots saved in: {cfg.out_dir}/')
