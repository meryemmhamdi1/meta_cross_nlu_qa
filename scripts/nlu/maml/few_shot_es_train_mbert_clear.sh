#!/bin/bash
data_path="/nas/clear/users/meryem/Datasets/NLU/CrossLingualNLU/facebook_nlu/"
results_path="/nas/clear/users/meryem/Results/meta_zsl_nlu/"
pre_trained_model_name=$results_path"MAML/train_en-test_es,th/l2l/kspt_6-qqry_6/few_shot_es/use_adapt/pytorch_model.bin"

python main.py --use-slots --train-langs en --dev-langs es --test-langs es th --use-adapt --k-spt 6 --q-qry 6 \
       --data-dir $data_path --out-dir $results_path --batch-sz 2500 --pre-train-steps 2000 --local_rank 0 \
       --use-pretrained-model --pre-trained-model-name $pre_trained_model_name