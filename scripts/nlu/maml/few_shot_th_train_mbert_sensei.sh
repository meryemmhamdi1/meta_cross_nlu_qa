#!/bin/bash
user_space="/trainman-mount/trainman-storage-7b0a9747-2045-40cf-9b8a-354a7e7219c2/"
data_path=$user_space"Datasets/mixed-language-training/data/nlu/nlu_data/"
results_path=$user_space"Results/meta_zsl_nlu/FacebookData_TUNE_BERT/"
pre_trained_model_name=$user_space"Results/meta_zsl_nlu/FacebookData/train_en-test_es,th/pytorch_model.bin"

python main.py --use-slots --train-langs en --dev-langs th --test-langs th es --data-dir $data_path --out-dir $results_path \
       --pre-trained-model-name $pre_trained_model_name