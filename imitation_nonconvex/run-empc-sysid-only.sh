#!/bin/bash
# Run only empc and sysid jobs (no nn). Use this to "finish" Figure 4 after NN runs
# have already completed, or to re-run empc/sysid after fixing disk quota / checkpoint errors.
# Overwrites existing work/il.<env>.empc.* and work/il.<env>.sysid.* dirs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "${SCRIPT_DIR}/../setup.sh"

N_EPOCH=10000

args_empc_sysid_only() {
    echo --n_epoch $N_EPOCH --mode sysid --no-cuda $*
    echo --n_epoch $N_EPOCH --mode empc --no-cuda --learn_cost $*
    echo --n_epoch $N_EPOCH --mode empc --no-cuda --learn_dx $*
    echo --n_epoch $N_EPOCH --mode empc --no-cuda --learn_cost --learn_dx $*
}

args_all_sizes() {
    DATA=$1
    SEED=$2
    args_empc_sysid_only --data $DATA --seed $SEED --n_train 10
    args_empc_sysid_only --data $DATA --seed $SEED --n_train 50
    args_empc_sysid_only --data $DATA --seed $SEED --n_train 100
}

args_all_seeds() {
    DATA=$1
    for SEED in {0..3}; do
        args_all_sizes $DATA $SEED
    done
}

run_single() {
    prefix=$1
    jobid=$2
    shift 2
    mkdir -p logs
    ./il_exp.py $* >> "logs/${prefix}_empc_sysid_job_${jobid}.log" 2>&1
}
export -f run_single

export OMP_NUM_THREADS=1

MAX_PROCS=16
args_all_seeds ./data/pendulum.pkl | parallel --no-notice --max-procs $MAX_PROCS run_single pendulum {#} {} &
args_all_seeds ./data/cartpole.pkl | parallel --no-notice --max-procs $MAX_PROCS run_single cartpole {#} {} &
wait
echo "Done. Results in work/il.<env>.<mode>.n_train=<N>/<seed>/"
