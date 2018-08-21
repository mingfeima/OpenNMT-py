ARGS=""
if [ "$1" == "prof" ]; then
    ARGS="$ARGS -prof"
fi
python train.py -data data/demo -save_model demo-model -train_steps 150 $ARGS -gpuid 0
