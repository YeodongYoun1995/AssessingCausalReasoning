#!/bin/bash

# Bash script to test llama2 with all hyperparameter settings

echo "Starting tests for llama2"

# Define model
models=("llama2")

# Define hyperparameter settings
num_beams_values=(1 3)
temperatures=(1.0 0.7)

# Define categories
categories=("counterfactual" "cause-effect" "scm")

for model in "${models[@]}"; do
    for num_beams in "${num_beams_values[@]}"; do
        for temperature in "${temperatures[@]}"; do
            for category in "${categories[@]}"; do
                echo "Testing model: $model, num_beams: $num_beams, temperature: $temperature, category: $category"
                python main.py --model $model --category $category --num_beams $num_beams --temperature $temperature --gpu 0
            done
        done
    done
done

echo "Finished tests for llama2"