#!/bin/bash
user_space="/trainman-mount/trainman-storage-7b0a9747-2045-40cf-9b8a-354a7e7219c2/"
data_path=$user_space"Datasets/mixed-language-training/data/nlu/nlu_data/"
results_path=$user_space"Results/meta_zsl_nlu/FacebookData_PROTO_HYBRID_TH/"
pre_trained_model_name=$user_space"Results/meta_zsl_nlu/FacebookData_BERT_TUNE/train_en-test_es,th/pytorch_model.bin"

python -m torch.distributed.launch proto_main.py --use-slots --train-langs en --dev-langs th --test-langs es th \
       --data-dir $data_path --out-dir $results_path --epoch 10 --batch-sz 500 \
       --use-pretrained-model --pre-trained-model-name $pre_trained_model_name