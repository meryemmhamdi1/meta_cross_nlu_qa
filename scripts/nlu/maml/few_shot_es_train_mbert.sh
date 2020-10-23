#!/bin/bash

data_path="" # replace by the root directory for the NLU dataset
results_path="" #  replace by the root directory for the Results directory
pre_trained_model_name="" # replace by binary pytorch file hosting pre-trained model

python main.py --use-slots --train-langs en --dev-langs es --test-langs es th --data-dir $data_path --out-dir $results_path \
       --pre-trained-model-name $pre_trained_model_name