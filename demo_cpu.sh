#!bin/sh

export KMP_AFFINITY=compact,1,0,granularity=fine
export OMP_NUM_THREADS=56
export KMP_BLOCKTIME=1

ARGS=""
if [ "$1" == "prof" ]; then
    ARGS="$ARGS -prof"
fi
python train.py -data data/demo -save_model demo-model -train_steps 150 $ARGS
