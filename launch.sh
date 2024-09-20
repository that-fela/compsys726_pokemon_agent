#!/bin/bash

cd ~/compsys726/gymnasium_envrionments/scripts && \
    python train.py run \
    --gym pyboy \
    --domain pokemon \
    --number_eval_episodes 2 \
    --number_steps_per_evaluation 500 \
    --task brock TD3 \
    --seed 420 \
    # --headless 1
    # --emulation_speed 20 \
