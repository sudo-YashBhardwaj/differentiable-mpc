#!/usr/bin/env python3
"""
fig3_identifiability.py — standalone Figure 3 reproduction.

Demonstrates the identifiability degeneracy of pendulum system identification
via differentiable MPC: 8 random (g, m, l) initializations all converge to
the same functional ratios g/l and 1/(m*l^2), even though the individual
parameters differ from the true values.

Usage (from repo root):
    source setup.sh
    python imitation_nonconvex/fig3_identifiability.py [--n_iter 2000] [--n_seeds 8]

Outputs (in imitation_nonconvex/fig3_out/):
    plot1_loss.png      — imitation loss curves for all seeds
    plot2_params.png    — individual g, m, l trajectories
    plot3_scatter.png   — (g/l, 1/ml^2) scatter — the identifiability plot
    plot4_table.png     — per-seed outcome table
    logs.pt             — raw log tensors (for custom post-processing)
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure mpc library is importable regardless of how this script is invoked
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_MPC_LIB   = os.path.join(_REPO_ROOT, 'mpc_pytorch_lib')
if _MPC_LIB not in sys.path:
    sys.path.insert(0, _MPC_LIB)

from mpc import mpc as mpc_module
from mpc.mpc import GradMethods, QuadCost
from mpc.env_dx.pendulum import PendulumDx


# ── Defaults (all overridable via CLI) ────────────────────────────────────────
DEFAULTS = dict(
    n_seeds     = 8,
    n_train     = 10,      # expert trajectories
    n_iter      = 2000,    # gradient steps per seed
    T           = 20,      # MPC horizon
    lqr_iter    = 500,     # iLQR iterations inside MPC
    lr          = 1e-2,    # RMSprop learning rate
    rms_alpha   = 0.5,     # RMSprop alpha
    warmstart_reset = 50,  # reset warmstart every N iters
    data_seed   = 42,      # seed for expert data generation
    out_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig3_out'),
)

# True pendulum params (g, m, l) — from PendulumDx.__init__
TRUE_G, TRUE_M, TRUE_L = 10.0, 1.0, 1.0
TRUE_G_OVER_L   = TRUE_G / TRUE_L          # 10.0
TRUE_INV_ML2    = 1.0 / (TRUE_M * TRUE_L**2)  # 1.0

# Random init ranges — centred away from truth so gradients must travel
G_RANGE = (5.0,  20.0)
M_RANGE = (0.3,   4.0)
L_RANGE = (0.3,   1.5)


# ── Expert data ───────────────────────────────────────────────────────────────

def make_expert_data(true_dx, cfg):
    """
    Generate n_train expert (xinit, u*) pairs using the true dynamics.
    Initial states: theta ~ Uniform(-pi/2, pi/2), thetadot ~ Uniform(-1, 1).
    Returns:
        xinit    (n_train, 3)
        u_expert (n_train, T, 1)   — shape matches how loss is computed
    """
    torch.manual_seed(cfg.data_seed)
    N = cfg.n_train

    th     = torch.FloatTensor(N).uniform_(-np.pi / 2, np.pi / 2)
    thdot  = torch.FloatTensor(N).uniform_(-1.0, 1.0)
    xinit  = torch.stack([torch.cos(th), torch.sin(th), thdot], dim=1)  # (N, 3)

    q_vec, p_vec = true_dx.get_true_obj()
    Q = torch.diag(q_vec).unsqueeze(0).unsqueeze(0).repeat(cfg.T, N, 1, 1)  # (T, N, 4, 4)
    p = p_vec.unsqueeze(0).repeat(cfg.T, N, 1)                               # (T, N, 4)

    # No torch.no_grad() here: AUTO_DIFF linearises via autograd internally.
    # We detach the result so no gradient flows through the expert trajectory.
    _, u_mpc, _ = mpc_module.MPC(
        true_dx.n_state, true_dx.n_ctrl, cfg.T,
        u_lower=true_dx.lower, u_upper=true_dx.upper,
        lqr_iter=cfg.lqr_iter, verbose=0,
        exit_unconverged=False, detach_unconverged=True,
        linesearch_decay=true_dx.linesearch_decay,
        max_linesearch_iter=true_dx.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF, eps=true_dx.mpc_eps,
    )(xinit, QuadCost(Q, p), true_dx)

    u_expert = u_mpc.transpose(0, 1).detach()   # (N, T, 1)
    return xinit, u_expert


# ── Learner MPC call ──────────────────────────────────────────────────────────

def run_learner_mpc(learn_dx, xinit, q_vec, p_vec, cfg, u_init=None):
    """
    Run MPC with the learner's dynamics. Gradient flows through env_params.
    Returns u_pred (n_train, T, 1).
    """
    N = xinit.shape[0]
    Q = torch.diag(q_vec).unsqueeze(0).unsqueeze(0).repeat(cfg.T, N, 1, 1)
    p = p_vec.unsqueeze(0).repeat(cfg.T, N, 1)

    # u_init must be (T, N, 1) for MPC
    u_init_mpc = u_init.transpose(0, 1) if u_init is not None else None

    _, u_mpc, _ = mpc_module.MPC(
        learn_dx.n_state, learn_dx.n_ctrl, cfg.T,
        u_lower=learn_dx.lower, u_upper=learn_dx.upper,
        u_init=u_init_mpc,
        lqr_iter=cfg.lqr_iter, verbose=0,
        exit_unconverged=False, detach_unconverged=True,
        linesearch_decay=learn_dx.linesearch_decay,
        max_linesearch_iter=learn_dx.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF, eps=learn_dx.mpc_eps,
    )(xinit, QuadCost(Q, p), learn_dx)

    return u_mpc.transpose(0, 1)   # (N, T, 1)


# ── Training loop for one seed ────────────────────────────────────────────────

def train_one_seed(seed, xinit, u_expert, true_dx, cfg):
    """
    Randomly initialise (g, m, l) and minimise imitation loss by
    backpropagating through the differentiable MPC solver.

    Returns a dict of logged time-series.
    """
    torch.manual_seed(seed)

    g_init = float(torch.FloatTensor(1).uniform_(*G_RANGE))
    m_init = float(torch.FloatTensor(1).uniform_(*M_RANGE))
    l_init = float(torch.FloatTensor(1).uniform_(*L_RANGE))

    env_params = torch.tensor([g_init, m_init, l_init], dtype=torch.float32,
                               requires_grad=True)

    opt = optim.RMSprop(
        [{'params': env_params, 'lr': cfg.lr, 'alpha': cfg.rms_alpha}]
    )

    q_vec, p_vec = true_dx.get_true_obj()

    # Warmstart for iLQR (kept across iterations, reset periodically)
    warmstart = torch.zeros(cfg.n_train, cfg.T, true_dx.n_ctrl)   # (N, T, 1)

    log = dict(
        im_loss=[], g=[], m=[], l=[], g_over_l=[], inv_ml2=[],
        g_init=g_init, m_init=m_init, l_init=l_init,
    )

    print(f'  Seed {seed}: init g={g_init:.3f} m={m_init:.3f} l={l_init:.3f} '
          f'| g/l={g_init/l_init:.3f}  1/(ml^2)={1/(m_init*l_init**2):.3f}')

    for i in range(cfg.n_iter):
        if i % cfg.warmstart_reset == 0:
            warmstart = warmstart.detach().zero_()

        # Build dynamics from current params — new object each step so that
        # PendulumDx.params IS env_params and gradient flows correctly
        learn_dx = PendulumDx(params=env_params)

        u_pred = run_learner_mpc(learn_dx, xinit, q_vec, p_vec, cfg,
                                 u_init=warmstart)          # (N, T, 1)
        warmstart = u_pred.detach()

        im_loss = (u_expert - u_pred).pow(2).mean()

        g, m, l = env_params.detach().tolist()
        log['im_loss'].append(im_loss.item())
        log['g'].append(g)
        log['m'].append(m)
        log['l'].append(l)
        log['g_over_l'].append(g / l)
        log['inv_ml2'].append(1.0 / (m * l**2))

        if i % 200 == 0 or i == cfg.n_iter - 1:
            print(f'    iter {i:4d}  loss={im_loss.item():.5f}  '
                  f'g={g:.3f} m={m:.3f} l={l:.3f}  '
                  f'g/l={g/l:.3f} (true=10)  '
                  f'1/(ml^2)={1/(m*l**2):.3f} (true=1)')

        opt.zero_grad()
        im_loss.backward()
        opt.step()

        # Enforce physicality: all params must stay positive
        with torch.no_grad():
            env_params.clamp_(min=1e-3)

    return log


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plots(all_logs, cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    iters  = np.arange(cfg.n_iter)
    cmap   = plt.cm.tab10
    colors = [cmap(i / max(cfg.n_seeds - 1, 1)) for i in range(cfg.n_seeds)]

    # ── Plot 1: Imitation loss ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for s, log in enumerate(all_logs):
        ax.semilogy(iters, log['im_loss'], color=colors[s], alpha=0.85,
                    linewidth=1.4, label=f'seed {s}')
    ax.set_xlabel('Training iteration', fontsize=11)
    ax.set_ylabel('Imitation loss  ||u* - u||^2  (log)', fontsize=11)
    ax.set_title('Plot 1 — Imitation loss: 8 random initialisations', fontsize=12)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    p = os.path.join(cfg.out_dir, 'plot1_loss.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'Saved {p}')

    # ── Plot 2: Individual parameter trajectories ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    param_cfg = [
        ('g',  TRUE_G,  r'Gravity  $\hat{g}$',    'g'),
        ('m',  TRUE_M,  r'Mass  $\hat{m}$',         'm'),
        ('l',  TRUE_L,  r'Length  $\hat{l}$',        'l'),
    ]
    for ax, (key, true_val, ylabel, sym) in zip(axes, param_cfg):
        for s, log in enumerate(all_logs):
            ax.plot(iters, log[key], color=colors[s], alpha=0.8, linewidth=1.2)
        ax.axhline(true_val, color='black', linestyle='--', linewidth=2.0,
                   label=f'true {sym} = {true_val}')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{ylabel}  (true = {true_val})', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Plot 2 — Individual parameters', fontsize=12)
    fig.tight_layout()
    p = os.path.join(cfg.out_dir, 'plot2_params.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'Saved {p}')

    # ── Plot 3: Identifiability scatter ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full trajectory in ratio space
    ax = axes[0]
    for s, log in enumerate(all_logs):
        ax.plot(log['g_over_l'], log['inv_ml2'],
                color=colors[s], alpha=0.35, linewidth=0.9)
        ax.scatter(log['g_over_l'][-1], log['inv_ml2'][-1],
                   color=colors[s], s=70, zorder=5)
    ax.scatter(TRUE_G_OVER_L, TRUE_INV_ML2, marker='*', s=500,
               color='black', zorder=10, label=f'True  (g/l={TRUE_G_OVER_L}, 1/ml²={TRUE_INV_ML2})')
    ax.set_xlabel(r'Learned  $\hat{g}/\hat{l}$', fontsize=11)
    ax.set_ylabel(r'Learned  $1/(\hat{m}\hat{l}^2)$', fontsize=11)
    ax.set_title('Trajectories in functional-ratio space', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: zoom around true value
    ax = axes[1]
    for s, log in enumerate(all_logs):
        ax.scatter(log['g_over_l'][-1], log['inv_ml2'][-1],
                   color=colors[s], s=100, zorder=5,
                   label=(f's{s}: ({log["g_over_l"][-1]:.2f}, '
                          f'{log["inv_ml2"][-1]:.2f})'))
    ax.scatter(TRUE_G_OVER_L, TRUE_INV_ML2, marker='*', s=500,
               color='black', zorder=10, label=f'True ({TRUE_G_OVER_L}, {TRUE_INV_ML2})')
    ax.set_xlabel(r'$\hat{g}/\hat{l}$  (final)', fontsize=11)
    ax.set_ylabel(r'$1/(\hat{m}\hat{l}^2)$  (final)', fontsize=11)
    ax.set_title('Final ratio values — do seeds cluster near truth?', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Plot 3 — The identifiability plot: '
                 'different (g,m,l) → same functional ratios', fontsize=12)
    fig.tight_layout()
    p = os.path.join(cfg.out_dir, 'plot3_scatter.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'Saved {p}')

    # ── Plot 4: Summary table ─────────────────────────────────────────────────
    col_labels = [
        'Seed',
        'Init g', 'Init m', 'Init l',
        'Final g', 'Final m', 'Final l',
        'g/l (true=10)', '1/ml^2 (true=1)',
        'Final loss', 'Verdict',
    ]
    rows = []
    for s, log in enumerate(all_logs):
        fg, fm, fl = log['g'][-1], log['m'][-1], log['l'][-1]
        gr  = log['g_over_l'][-1]
        ir  = log['inv_ml2'][-1]
        fl_ = log['im_loss'][-1]

        ratio_ok = abs(gr - TRUE_G_OVER_L)  < 1.0    # within 10%
        inv_ok   = abs(ir - TRUE_INV_ML2)   < 0.15
        loss_ok  = fl_ < 5e-3

        if loss_ok and ratio_ok and inv_ok:
            verdict = 'Converged'
        elif loss_ok:
            verdict = 'Compensated'   # zero loss, wrong ratios
        elif ratio_ok and inv_ok:
            verdict = 'Right ratios'  # correct physics, not fully converged
        else:
            verdict = 'Not converged'

        rows.append([
            str(s),
            f'{log["g_init"]:.2f}', f'{log["m_init"]:.2f}', f'{log["l_init"]:.2f}',
            f'{fg:.3f}',            f'{fm:.3f}',             f'{fl:.3f}',
            f'{gr:.3f}',            f'{ir:.3f}',
            f'{fl_:.2e}',           verdict,
        ])

    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    # Colour header row
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#4C72B0')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    ax.set_title('Plot 4 — Per-seed outcomes', fontsize=12, fontweight='bold', pad=14)
    fig.tight_layout()
    p = os.path.join(cfg.out_dir, 'plot4_table.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p}')

    # Save raw logs
    torch.save(all_logs, os.path.join(cfg.out_dir, 'logs.pt'))
    print(f'Saved logs.pt')


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n_seeds',          type=int,   default=DEFAULTS['n_seeds'])
    p.add_argument('--n_train',          type=int,   default=DEFAULTS['n_train'])
    p.add_argument('--n_iter',           type=int,   default=DEFAULTS['n_iter'])
    p.add_argument('--T',                type=int,   default=DEFAULTS['T'])
    p.add_argument('--lqr_iter',         type=int,   default=DEFAULTS['lqr_iter'])
    p.add_argument('--lr',               type=float, default=DEFAULTS['lr'])
    p.add_argument('--rms_alpha',        type=float, default=DEFAULTS['rms_alpha'])
    p.add_argument('--warmstart_reset',  type=int,   default=DEFAULTS['warmstart_reset'])
    p.add_argument('--data_seed',        type=int,   default=DEFAULTS['data_seed'])
    p.add_argument('--out_dir',          type=str,   default=DEFAULTS['out_dir'])
    # Single-seed mode: run exactly this seed and save seed_N.pt, then exit.
    # Used for parallel launches (one process per seed).
    p.add_argument('--single_seed',      type=int,   default=None,
                   help='Run only this seed index and save seed_N.pt, then exit.')
    return p.parse_args()


if __name__ == '__main__':
    cfg = parse_args()

    os.makedirs(cfg.out_dir, exist_ok=True)
    true_dx = PendulumDx()

    print(f'Generating expert data (data_seed={cfg.data_seed}, n_train={cfg.n_train})...')
    xinit, u_expert = make_expert_data(true_dx, cfg)

    # ── Single-seed mode (used for parallel launches) ──────────────────────────
    if cfg.single_seed is not None:
        seed = cfg.single_seed
        ckpt_path = os.path.join(cfg.out_dir, f'seed_{seed}.pt')
        if os.path.exists(ckpt_path):
            print(f'seed_{seed}.pt already exists, skipping.')
            sys.exit(0)
        print(f'Running seed {seed}...')
        log = train_one_seed(seed, xinit, u_expert, true_dx, cfg)
        torch.save(log, ckpt_path)
        print(f'Saved {ckpt_path}')
        sys.exit(0)

    # ── Sequential mode (original behaviour) ──────────────────────────────────
    print('=' * 60)
    print('Figure 3 — Identifiability experiment')
    print(f'  True params:  g={TRUE_G}  m={TRUE_M}  l={TRUE_L}')
    print(f'  True ratios:  g/l={TRUE_G_OVER_L}   1/(ml^2)={TRUE_INV_ML2}')
    print(f'  Seeds: {cfg.n_seeds}  |  n_train: {cfg.n_train}  |  '
          f'n_iter: {cfg.n_iter}  |  T: {cfg.T}')
    print('=' * 60)

    all_logs = []
    for seed in range(cfg.n_seeds):
        ckpt_path = os.path.join(cfg.out_dir, f'seed_{seed}.pt')

        if os.path.exists(ckpt_path):
            log = torch.load(ckpt_path)
            all_logs.append(log)
            print(f'\n--- Seed {seed} --- LOADED from checkpoint '
                  f'({len(log["im_loss"])} iters, '
                  f'g/l={log["g_over_l"][-1]:.3f}, '
                  f'1/(ml^2)={log["inv_ml2"][-1]:.3f})')
            continue

        print(f'\n--- Seed {seed} / {cfg.n_seeds - 1} ---')
        log = train_one_seed(seed, xinit, u_expert, true_dx, cfg)
        all_logs.append(log)
        torch.save(log, ckpt_path)
        print(f'  Checkpoint saved: {ckpt_path}')

    print('\n' + '=' * 60)
    print('Final ratio summary')
    print(f'  {"Seed":>4}  {"g/l":>8}  {"1/(ml^2)":>10}  {"loss":>10}')
    for s, log in enumerate(all_logs):
        print(f'  {s:>4}  {log["g_over_l"][-1]:>8.4f}  '
              f'{log["inv_ml2"][-1]:>10.4f}  {log["im_loss"][-1]:>10.2e}')
    print(f'  {"true":>4}  {TRUE_G_OVER_L:>8.4f}  {TRUE_INV_ML2:>10.4f}')
    print('=' * 60)

    print('\nGenerating plots...')
    make_plots(all_logs, cfg)
    print(f'\nDone. All outputs in: {cfg.out_dir}/')
