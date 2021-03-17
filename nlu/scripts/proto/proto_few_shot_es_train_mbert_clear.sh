#!/bin/bash
data_path="/nas/clear/users/meryem/Datasets/NLU/CrossLingualNLU/facebook_nlu/"
results_path="/nas/clear/users/meryem/Results/meta_zsl_nlu/"
pre_trained_model_name=$results_path"MAML/train_en-test_es,th/l2l/kspt_6-qqry_6/few_shot_es/use_adapt/pytorch_model.bin"

python proto_main_l2l.py --use-slots --train-langs en --dev-langs es --test-langs en es th --k-spt 6 --q-qry 3 \
       --data-dir $data_path --out-dir $results_path --batch-sz 100 --use-pretrained-model \
       --pre-trained-model-name $pre_trained_model_name