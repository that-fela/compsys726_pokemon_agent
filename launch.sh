#!/bin/bash

experiment=$1

file_name="train.py"
if [ "$experiment" -eq 1 ]; then
    file_name="experiment.py"
fi

cd ~/compsys726/gymnasium_envrionments/scripts && \
    python $file_name run \
    --gym pyboy \
    --domain pokemon \
    --number_eval_episodes 1 \
    --number_steps_per_evaluation 500 \
    --task brock TD3 \
    --seed 420 \
    # --emulation_speed 20 \
