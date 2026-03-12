#!/usr/bin/env python3
"""
fig3_mpc_cost.py — Section 5.3 cost-parameter learning via differentiable MPC.

The learner knows the true pendulum dynamics (g=10, m=1, l=1) but must
recover the cost function (quadratic weights Q and linear term p encoding
the goal state) by differentiating imitation loss through the MPC solver.

Cost parameterisation (following il_exp.py):
    q    = sigmoid(q_logit)        shape (4,)  — diagonal of C, values in (0,1)
    p    = sqrt(q) * p_raw         shape (4,)  — linear term c

MPC cost per step: 0.5 * z^T diag(q) z + p^T z   where z = [x; u]
Completing the square: effective target for component i is
    z_i* = -p_i / q_i = -(sqrt(q_i)*p_raw_i) / q_i = -p_raw_i / sqrt(q_i)

    ⇒  decoded goal state = –p_raw[:3] / sqrt(q[:3])

Identifiable quantities (invariant to overall cost scale):
    goal_hat  = –p_raw[:3]/sqrt(q[:3])  → should → [1., 0., 0.]  (upright)
    q_ratio_02 = q[0]/q[2]              → should → 10.0  (cos_th vs dth weight)
    q_ratio_01 = q[0]/q[1]             → should → 1.0   (cos_th vs sin_th)

Contrast with fig3_identifiability.py (dynamics identifiability):
    Dynamics: individual (g, m, l) are NOT recovered; only g/l and 1/(ml²) are.
    Cost:     the decoded goal state and weight ratios ARE expected to converge
              uniquely, because the cost landscape has fewer degenerate directions.

Usage (from repo root):
    source setup.sh
    # Run all 8 seeds sequentially (resumes from checkpoints if present):
    python imitation_nonconvex/fig3_mpc_cost.py

    # Run one seed, checkpoint, and exit (use for parallel launches):
    OMP_NUM_THREADS=1 python imitation_nonconvex/fig3_mpc_cost.py \\
        --single_seed 0 --out_dir imitation_nonconvex/fig3_cost_out

Outputs (in imitation_nonconvex/fig3_cost_out/):
    seed_N.pt           — per-seed checkpoint
    plot1_loss.png      — imitation loss curves for all seeds
    plot2_goal.png      — decoded goal-state component trajectories
    plot3_scatter.png   — (decoded goal[0], q_ratio_02) identifiability scatter
    plot4_table.png     — per-seed outcome table
    logs.pt             — merged raw logs
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_MPC_LIB   = os.path.join(_REPO_ROOT, 'mpc_pytorch_lib')
if _MPC_LIB not in sys.path:
    sys.path.insert(0, _MPC_LIB)

from mpc import mpc as mpc_module
from mpc.mpc import GradMethods, QuadCost
from mpc.env_dx.pendulum import PendulumDx


# ── True cost constants (from PendulumDx) ────────────────────────────────────
# goal_state   = [cos(0), sin(0), dth=0] = [1., 0., 0.]  (upright pendulum)
# goal_weights = [1., 1., 0.1]           (cos_th, sin_th, angular velocity)
# ctrl_penalty = 0.001
# → q_true = [1., 1., 0.1, 0.001]       (3 state + 1 ctrl)
# → p_true = [-sqrt(1)*1, -sqrt(1)*0, -sqrt(0.1)*0, 0] = [-1., 0., 0., 0.]
# Under the parameterisation p = sqrt(q)*p_raw, the effective target is:
#   goal_decoded_i = -p_raw_i / sqrt(q_i)
# The true cost (q=[1,1,0.1,0.001], p=[-1,0,0,0]) gives:
#   goal_decoded = [-(-1)/sqrt(1), 0/sqrt(1), 0/sqrt(0.1)] = [1., 0., 0.]  ✓
TRUE_GOAL      = [1., 0., 0.]          # upright goal state
TRUE_Q_RATIO_01 = 1.0                  # q[0]/q[1] = 1/1
TRUE_Q_RATIO_02 = 10.0                 # q[0]/q[2] = 1/0.1

# Random init ranges — spread wide so gradients must travel
Q_LOGIT_RANGE = (-3.0, 3.0)   # sigmoid(-3)≈0.05, sigmoid(3)≈0.95
P_RAW_RANGE   = (-3.0, 3.0)   # decoded goal = -p_raw/sqrt(q), ranges widely

DEFAULTS = dict(
    n_seeds         = 8,
    n_train         = 10,      # expert trajectories
    n_iter          = 2000,    # gradient steps per seed
    T               = 20,      # MPC horizon
    lqr_iter        = 500,     # iLQR iterations inside MPC
    lr              = 1e-2,    # RMSprop learning rate
    rms_alpha       = 0.5,     # RMSprop alpha
    warmstart_reset = 50,      # reset iLQR warmstart every N iters
    data_seed       = 42,      # seed for expert data generation
    out_dir         = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'fig3_cost_out'),
)


# ── Expert data ───────────────────────────────────────────────────────────────

def make_expert_data(true_dx, cfg):
    """
    Generate n_train expert (xinit, u*) pairs using true dynamics + true cost.
    Initial states: theta ~ Uniform(-pi/2, pi/2), thetadot ~ Uniform(-1, 1).
    Returns:
        xinit    (n_train, 3)
        u_expert (n_train, T, 1)
    """
    torch.manual_seed(cfg.data_seed)
    N = cfg.n_train

    th    = torch.FloatTensor(N).uniform_(-np.pi / 2, np.pi / 2)
    thdot = torch.FloatTensor(N).uniform_(-1.0, 1.0)
    xinit = torch.stack([torch.cos(th), torch.sin(th), thdot], dim=1)  # (N, 3)

    q_vec, p_vec = true_dx.get_true_obj()
    Q = torch.diag(q_vec).unsqueeze(0).unsqueeze(0).repeat(cfg.T, N, 1, 1)
    p = p_vec.unsqueeze(0).repeat(cfg.T, N, 1)

    # No torch.no_grad(): AUTO_DIFF linearises dynamics via autograd internally.
    _, u_mpc, _ = mpc_module.MPC(
        true_dx.n_state, true_dx.n_ctrl, cfg.T,
        u_lower=true_dx.lower, u_upper=true_dx.upper,
        lqr_iter=cfg.lqr_iter, verbose=0,
        exit_unconverged=False, detach_unconverged=True,
        linesearch_decay=true_dx.linesearch_decay,
        max_linesearch_iter=true_dx.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF, eps=true_dx.mpc_eps,
    )(xinit, QuadCost(Q, p), true_dx)

    return xinit, u_mpc.transpose(0, 1).detach()   # (N, T, 1)


# ── Learner MPC with learned cost ─────────────────────────────────────────────

def run_learner_mpc(true_dx, xinit, q_logit, p_raw, cfg, u_init=None):
    """
    Run MPC with true dynamics and learned cost parameters.
    Gradient flows back through q_logit and p_raw.
    Returns u_pred (n_train, T, 1).
    """
    N = xinit.shape[0]

    q_vec = torch.sigmoid(q_logit)         # (4,) ∈ (0,1)
    p_vec = q_vec.sqrt() * p_raw           # (4,)

    Q = torch.diag(q_vec).unsqueeze(0).unsqueeze(0).repeat(cfg.T, N, 1, 1)
    p = p_vec.unsqueeze(0).repeat(cfg.T, N, 1)

    u_init_mpc = u_init.transpose(0, 1) if u_init is not None else None

    _, u_mpc, _ = mpc_module.MPC(
        true_dx.n_state, true_dx.n_ctrl, cfg.T,
        u_lower=true_dx.lower, u_upper=true_dx.upper,
        u_init=u_init_mpc,
        lqr_iter=cfg.lqr_iter, verbose=0,
        exit_unconverged=False, detach_unconverged=True,
        linesearch_decay=true_dx.linesearch_decay,
        max_linesearch_iter=true_dx.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF, eps=true_dx.mpc_eps,
    )(xinit, QuadCost(Q, p), true_dx)

    return u_mpc.transpose(0, 1)   # (N, T, 1)


# ── Training loop for one seed ────────────────────────────────────────────────

def train_one_seed(seed, xinit, u_expert, true_dx, cfg):
    """
    Randomly initialise cost parameters (q_logit, p_raw) and minimise
    imitation loss by backpropagating through the differentiable MPC solver.
    True dynamics are fixed throughout.

    Returns a dict of logged time-series.
    """
    torch.manual_seed(seed)

    q_logit_init = torch.FloatTensor(4).uniform_(*Q_LOGIT_RANGE)
    p_raw_init   = torch.FloatTensor(4).uniform_(*P_RAW_RANGE)

    q_logit = q_logit_init.clone().requires_grad_(True)
    p_raw   = p_raw_init.clone().requires_grad_(True)

    opt = optim.RMSprop(
        [{'params': [q_logit, p_raw], 'lr': cfg.lr, 'alpha': cfg.rms_alpha}]
    )

    warmstart = torch.zeros(cfg.n_train, cfg.T, true_dx.n_ctrl)   # (N, T, 1)

    with torch.no_grad():
        q0  = torch.sigmoid(q_logit_init)
        g0  = (-p_raw_init[:3] / q0[:3].sqrt()).tolist()   # decoded goal = -p_raw/sqrt(q)
        wr0 = (q0[0] / q0[2]).item()

    log = dict(
        im_loss=[],
        goal=[],         # decoded goal state = –p_raw[:3]/sqrt(q[:3])  shape (n_iter, 3)
        q=[],            # sigmoid(q_logit)                              shape (n_iter, 4)
        p_raw=[],        # raw linear param                              shape (n_iter, 4)
        q_ratio_01=[],   # q[0]/q[1]  → true = 1.0
        q_ratio_02=[],   # q[0]/q[2]  → true = 10.0
        q_logit_init=q_logit_init.tolist(),
        p_raw_init=p_raw_init.tolist(),
        goal_init=g0,
        q_ratio_02_init=wr0,
    )

    print(f'  Seed {seed}: init goal=[{g0[0]:.2f},{g0[1]:.2f},{g0[2]:.2f}]  '
          f'q_ratio_02={wr0:.3f} (true={TRUE_Q_RATIO_02})')

    for i in range(cfg.n_iter):
        if i % cfg.warmstart_reset == 0:
            warmstart = warmstart.detach().zero_()

        u_pred    = run_learner_mpc(true_dx, xinit, q_logit, p_raw, cfg,
                                    u_init=warmstart)
        warmstart = u_pred.detach()

        im_loss = (u_expert - u_pred).pow(2).mean()

        with torch.no_grad():
            q_cur    = torch.sigmoid(q_logit)
            pr_cur   = p_raw.detach()
            goal_cur = (-pr_cur[:3] / q_cur[:3].sqrt()).tolist()   # -p_raw/sqrt(q)
            wr01 = (q_cur[0] / q_cur[1]).item()
            wr02 = (q_cur[0] / q_cur[2]).item()

        log['im_loss'].append(im_loss.item())
        log['goal'].append(goal_cur)
        log['q'].append(q_cur.tolist())
        log['p_raw'].append(pr_cur.tolist())
        log['q_ratio_01'].append(wr01)
        log['q_ratio_02'].append(wr02)

        if i % 200 == 0 or i == cfg.n_iter - 1:
            print(f'    iter {i:4d}  loss={im_loss.item():.5f}  '
                  f'goal=[{goal_cur[0]:.3f},{goal_cur[1]:.3f},{goal_cur[2]:.3f}]  '
                  f'q_ratio_02={wr02:.3f} (true=10)  '
                  f'q_ratio_01={wr01:.3f} (true=1)')

        opt.zero_grad()
        im_loss.backward()
        opt.step()

    return log


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plots(all_logs, cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    iters  = np.arange(cfg.n_iter)
    cmap   = plt.cm.tab10
    n      = len(all_logs)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # ── Plot 1: Imitation loss ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for s, log in enumerate(all_logs):
        ax.semilogy(iters, log['im_loss'], color=colors[s], alpha=0.85,
                    linewidth=1.4, label=f'seed {s}')
    ax.set_xlabel('Training iteration', fontsize=11)
    ax.set_ylabel('Imitation loss  ||u* – û||²  (log)', fontsize=11)
    ax.set_title('Plot 1 — Imitation loss: 8 random cost initialisations', fontsize=12)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    p = os.path.join(cfg.out_dir, 'plot1_loss.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'Saved {p}')

    # ── Plot 2: Decoded goal state over iterations ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    goal_cfg = [
        (0, TRUE_GOAL[0], r'Goal $\hat{x}_1$ = $\cos\theta$  (true = 1)'),
        (1, TRUE_GOAL[1], r'Goal $\hat{x}_2$ = $\sin\theta$  (true = 0)'),
        (2, TRUE_GOAL[2], r'Goal $\hat{x}_3$ = $\dot\theta$  (true = 0)'),
    ]
    for ax, (idx, true_val, ylabel) in zip(axes, goal_cfg):
        for s, log in enumerate(all_logs):
            traj = [step[idx] for step in log['goal']]
            ax.plot(iters, traj, color=colors[s], alpha=0.8, linewidth=1.2)
        ax.axhline(true_val, color='black', linestyle='--', linewidth=2.0,
                   label=f'true = {true_val}')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(r'Plot 2 — Decoded goal state  ($-p_{raw}/\sqrt{q}$)  over training', fontsize=12)
    fig.tight_layout()
    p = os.path.join(cfg.out_dir, 'plot2_goal.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'Saved {p}')

    # ── Plot 3: Identifiability scatter ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full trajectory in (goal[0], q_ratio_02) space
    ax = axes[0]
    for s, log in enumerate(all_logs):
        goal0_traj = [step[0] for step in log['goal']]
        ax.plot(goal0_traj, log['q_ratio_02'],
                color=colors[s], alpha=0.35, linewidth=0.9)
        ax.scatter(goal0_traj[-1], log['q_ratio_02'][-1],
                   color=colors[s], s=70, zorder=5)
    ax.scatter(TRUE_GOAL[0], TRUE_Q_RATIO_02, marker='*', s=500,
               color='black', zorder=10,
               label=f'True  (goal₀={TRUE_GOAL[0]}, q₀/q₂={TRUE_Q_RATIO_02})')
    ax.set_xlabel(r'Decoded goal$_0$  = $-p_{raw,0}/\sqrt{q_0}$  (cos $\theta$ target)', fontsize=11)
    ax.set_ylabel(r'Weight ratio  $q_0 / q_2$  (cos $\theta$ vs $\dot\theta$)', fontsize=11)
    ax.set_title('Trajectories in cost-interpretation space', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: final values only — do seeds cluster?
    ax = axes[1]
    for s, log in enumerate(all_logs):
        goal0_f = log['goal'][-1][0]
        wr02_f  = log['q_ratio_02'][-1]
        ax.scatter(goal0_f, wr02_f, color=colors[s], s=120, zorder=5,
                   label=f's{s}: ({goal0_f:.2f}, {wr02_f:.2f})')
    ax.scatter(TRUE_GOAL[0], TRUE_Q_RATIO_02, marker='*', s=500,
               color='black', zorder=10,
               label=f'True ({TRUE_GOAL[0]}, {TRUE_Q_RATIO_02})')
    ax.set_xlabel(r'Decoded goal$_0$  (final)', fontsize=11)
    ax.set_ylabel(r'$q_0 / q_2$  (final)', fontsize=11)
    ax.set_title('Final cost parameters — do seeds cluster near truth?', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Plot 3 — Cost identifiability scatter\n'
                 'Contrast with dynamics: do all seeds find the SAME cost?',
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(cfg.out_dir, 'plot3_scatter.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'Saved {p}')

    # ── Plot 4: Summary table ─────────────────────────────────────────────────
    col_labels = [
        'Seed',
        'Init goal₀', 'Init q₀/q₂',
        'Final goal₀', 'Final goal₁', 'Final goal₂',
        'q₀/q₁ (→1)', 'q₀/q₂ (→10)',
        'Final loss', 'Verdict',
    ]
    rows = []
    for s, log in enumerate(all_logs):
        gf  = log['goal'][-1]
        wr01 = log['q_ratio_01'][-1]
        wr02 = log['q_ratio_02'][-1]
        lf   = log['im_loss'][-1]

        goal_ok  = abs(gf[0] - TRUE_GOAL[0]) < 0.2 and abs(gf[1]) < 0.3 and abs(gf[2]) < 0.3
        wr02_ok  = abs(wr02 - TRUE_Q_RATIO_02) < 2.0
        wr01_ok  = abs(wr01 - TRUE_Q_RATIO_01) < 0.3
        loss_ok  = lf < 5e-3

        if loss_ok and goal_ok and wr02_ok:
            verdict = 'Recovered'
        elif loss_ok:
            verdict = 'Compensated'   # near-zero loss, wrong cost params
        elif goal_ok and wr02_ok:
            verdict = 'Right cost'    # correct params, not converged
        else:
            verdict = 'Not converged'

        rows.append([
            str(s),
            f'{log["goal_init"][0]:.2f}',
            f'{log["q_ratio_02_init"]:.2f}',
            f'{gf[0]:.3f}', f'{gf[1]:.3f}', f'{gf[2]:.3f}',
            f'{wr01:.3f}',  f'{wr02:.3f}',
            f'{lf:.2e}',    verdict,
        ])

    fig, ax = plt.subplots(figsize=(17, 3.5))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#2c7bb6')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    ax.set_title('Plot 4 — Per-seed cost-learning outcomes', fontsize=12,
                 fontweight='bold', pad=14)
    fig.tight_layout()
    p = os.path.join(cfg.out_dir, 'plot4_table.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p}')

    torch.save(all_logs, os.path.join(cfg.out_dir, 'logs.pt'))
    print('Saved logs.pt')


# ── CLI ───────────────────────────────────────────────────────────────────────

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
    p.add_argument('--single_seed',      type=int,   default=None,
                   help='Run only this seed index, save seed_N.pt, then exit. '
                        'Used for parallel launches.')
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = parse_args()

    os.makedirs(cfg.out_dir, exist_ok=True)
    true_dx = PendulumDx()   # true physics, fixed throughout

    print(f'Generating expert data (data_seed={cfg.data_seed}, n_train={cfg.n_train})...')
    xinit, u_expert = make_expert_data(true_dx, cfg)

    # ── Single-seed mode ──────────────────────────────────────────────────────
    if cfg.single_seed is not None:
        seed      = cfg.single_seed
        ckpt_path = os.path.join(cfg.out_dir, f'seed_{seed}.pt')
        if os.path.exists(ckpt_path):
            print(f'seed_{seed}.pt already exists — skipping.')
            sys.exit(0)
        print(f'Running seed {seed}...')
        log = train_one_seed(seed, xinit, u_expert, true_dx, cfg)
        torch.save(log, ckpt_path)
        print(f'Saved {ckpt_path}')
        sys.exit(0)

    # ── Sequential mode (runs all seeds, resumes from checkpoints) ────────────
    print('=' * 60)
    print('Fig 3 (cost variant) — cost identifiability experiment')
    print(f'  True goal state:   {TRUE_GOAL}')
    print(f'  True q₀/q₁:  {TRUE_Q_RATIO_01}   True q₀/q₂:  {TRUE_Q_RATIO_02}')
    print(f'  Seeds: {cfg.n_seeds}  |  n_train: {cfg.n_train}  |  '
          f'n_iter: {cfg.n_iter}  |  T: {cfg.T}')
    print('=' * 60)

    all_logs = []
    for seed in range(cfg.n_seeds):
        ckpt_path = os.path.join(cfg.out_dir, f'seed_{seed}.pt')

        if os.path.exists(ckpt_path):
            log = torch.load(ckpt_path)
            # Recompute decoded goal using the correct formula: -p_raw/sqrt(q)
            # (old checkpoints stored -p_raw which is missing the /sqrt(q) factor)
            if log['p_raw'] and log['q']:
                log['goal'] = [
                    (-torch.tensor(pr[:3]) / torch.tensor(q[:3]).sqrt()).tolist()
                    for pr, q in zip(log['p_raw'], log['q'])
                ]
                log['goal_init'] = (
                    -torch.tensor(log['p_raw_init'][:3]) /
                     torch.tensor(log['q_logit_init'][:3]).sigmoid().sqrt()
                ).tolist()
            all_logs.append(log)
            gf   = log['goal'][-1]
            wr02 = log['q_ratio_02'][-1]
            print(f'\n--- Seed {seed} --- LOADED  '
                  f'goal=[{gf[0]:.3f},{gf[1]:.3f},{gf[2]:.3f}]  '
                  f'q₀/q₂={wr02:.3f}  loss={log["im_loss"][-1]:.2e}')
            continue

        print(f'\n--- Seed {seed} / {cfg.n_seeds - 1} ---')
        log = train_one_seed(seed, xinit, u_expert, true_dx, cfg)
        all_logs.append(log)
        torch.save(log, ckpt_path)
        print(f'  Checkpoint saved: {ckpt_path}')

    print('\n' + '=' * 60)
    print('Final summary')
    print(f'  {"Seed":>4}  {"goal₀":>7}  {"goal₁":>7}  {"goal₂":>7}  '
          f'{"q₀/q₂":>8}  {"q₀/q₁":>8}  {"loss":>10}')
    for s, log in enumerate(all_logs):
        gf  = log['goal'][-1]
        print(f'  {s:>4}  {gf[0]:>7.3f}  {gf[1]:>7.3f}  {gf[2]:>7.3f}  '
              f'{log["q_ratio_02"][-1]:>8.3f}  {log["q_ratio_01"][-1]:>8.3f}  '
              f'{log["im_loss"][-1]:>10.2e}')
    print(f'  {"true":>4}  '
          f'{TRUE_GOAL[0]:>7.3f}  {TRUE_GOAL[1]:>7.3f}  {TRUE_GOAL[2]:>7.3f}  '
          f'{TRUE_Q_RATIO_02:>8.3f}  {TRUE_Q_RATIO_01:>8.3f}')
    print('=' * 60)

    print('\nGenerating plots...')
    make_plots(all_logs, cfg)
    print(f'\nDone. All outputs in: {cfg.out_dir}/')
