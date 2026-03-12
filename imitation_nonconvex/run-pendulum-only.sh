#!/bin/bash
# Run Figure-4-style experiments for pendulum only (no cartpole).
# Saves: best.pkl (when val improves), train_losses.csv, val_test_losses.csv,
#        dx_hist.csv (if learn_dx), cost_hist.csv (if learn_cost).
# Requires: data/pendulum.pkl (create with: python make_dataset.py --env_name pendulum)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/../setup.sh"

N_EPOCH=10000
DATA="./data/pendulum.pkl"

args_all_modes() {
    echo --n_epoch $N_EPOCH --mode nn --no-cuda --data $DATA $*
    echo --n_epoch $N_EPOCH --mode sysid --no-cuda --data $DATA $*
    echo --n_epoch $N_EPOCH --mode empc --no-cuda --learn_cost --data $DATA $*
    echo --n_epoch $N_EPOCH --mode empc --no-cuda --learn_dx --data $DATA $*
    echo --n_epoch $N_EPOCH --mode empc --no-cuda --learn_cost --learn_dx --data $DATA $*
}

args_all_sizes() {
    SEED=$1
    args_all_modes --seed $SEED --n_train 10
    args_all_modes --seed $SEED --n_train 50
    args_all_modes --seed $SEED --n_train 100
}

args_all_seeds() {
    for SEED in {0..3}; do
        args_all_sizes $SEED
    done
}

run_single() {
    ./il_exp.py "$@"
}
export -f run_single

if [ ! -f "$DATA" ]; then
    echo "Missing $DATA. Create with: python make_dataset.py --env_name pendulum"
    exit 1
fi

export OMP_NUM_THREADS=1
MAX_PROCS=8
# Each line is one job; parallel splits on spaces and passes as args to run_single
args_all_seeds | parallel --no-notice --max-procs $MAX_PROCS run_single
wait
echo "Done. Results in work/il.pendulum.<mode>.n_train=<N>.<seed>/"
