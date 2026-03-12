#!/usr/bin/env python3
"""
fig3_merge_and_plot.py — wait for all seed_N.pt checkpoints, then plot.

Usage:
    python3 fig3_merge_and_plot.py [--out_dir fig3_out] [--n_seeds 8] [--poll 30]

The script polls for seed_N.pt files until all n_seeds are present,
then calls make_plots() from fig3_identifiability and saves the four figures.
Run this in a separate terminal while the parallel seed jobs are running.
"""

import argparse
import os
import sys
import time

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DIR, '..'))
_MPC_LIB   = os.path.join(_REPO_ROOT, 'mpc_pytorch_lib')
if _MPC_LIB not in sys.path:
    sys.path.insert(0, _MPC_LIB)

# Import plot function and constants from the main script
sys.path.insert(0, _DIR)
from fig3_identifiability import make_plots, TRUE_G_OVER_L, TRUE_INV_ML2, DEFAULTS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out_dir',  type=str, default=DEFAULTS['out_dir'])
    p.add_argument('--n_seeds',  type=int, default=DEFAULTS['n_seeds'])
    p.add_argument('--n_iter',   type=int, default=DEFAULTS['n_iter'])
    p.add_argument('--n_train',  type=int, default=DEFAULTS['n_train'])
    p.add_argument('--data_seed',type=int, default=DEFAULTS['data_seed'])
    p.add_argument('--T',        type=int, default=DEFAULTS['T'])
    p.add_argument('--lqr_iter', type=int, default=DEFAULTS['lqr_iter'])
    p.add_argument('--lr',       type=float, default=DEFAULTS['lr'])
    p.add_argument('--rms_alpha',type=float, default=DEFAULTS['rms_alpha'])
    p.add_argument('--warmstart_reset', type=int, default=DEFAULTS['warmstart_reset'])
    p.add_argument('--poll',     type=int, default=30,
                   help='Seconds between polls while waiting for seeds (default: 30)')
    p.add_argument('--nowait',   action='store_true',
                   help='Plot immediately with whatever seeds are present, do not wait')
    return p.parse_args()


def main():
    cfg = parse_args()

    print(f'Watching {cfg.out_dir} for seed_N.pt files (n_seeds={cfg.n_seeds})')

    while True:
        present   = [s for s in range(cfg.n_seeds)
                     if os.path.exists(os.path.join(cfg.out_dir, f'seed_{s}.pt'))]
        missing   = [s for s in range(cfg.n_seeds) if s not in present]

        print(f'  Done: {sorted(present)}   Missing: {sorted(missing)}')

        if not missing or cfg.nowait:
            break

        print(f'  Waiting {cfg.poll}s ...')
        time.sleep(cfg.poll)

    if not present:
        print('No seed files found. Nothing to plot.')
        sys.exit(1)

    # Load logs
    all_logs = []
    for s in sorted(present):
        path = os.path.join(cfg.out_dir, f'seed_{s}.pt')
        log  = torch.load(path)
        all_logs.append(log)
        print(f'  Loaded seed {s}: {len(log["im_loss"])} iters  '
              f'g/l={log["g_over_l"][-1]:.3f}  '
              f'1/(ml^2)={log["inv_ml2"][-1]:.3f}  '
              f'loss={log["im_loss"][-1]:.2e}')

    # Summary
    print(f'\n{"Seed":>4}  {"g/l":>8}  {"1/(ml^2)":>10}  {"loss":>10}')
    for s, log in zip(sorted(present), all_logs):
        print(f'{s:>4}  {log["g_over_l"][-1]:>8.4f}  '
              f'{log["inv_ml2"][-1]:>10.4f}  {log["im_loss"][-1]:>10.2e}')
    print(f'{"true":>4}  {TRUE_G_OVER_L:>8.4f}  {TRUE_INV_ML2:>10.4f}')

    # Merge and save combined logs.pt
    torch.save(all_logs, os.path.join(cfg.out_dir, 'logs.pt'))

    print('\nGenerating plots...')
    make_plots(all_logs, cfg)
    print(f'\nDone. Plots saved in: {cfg.out_dir}/')


if __name__ == '__main__':
    main()
