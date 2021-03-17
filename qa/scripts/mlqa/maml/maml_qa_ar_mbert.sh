#!/bin/bash
data_dir="/nas/clear/users/meryem/Datasets/QA/MLQA_V1"
results_path="/nas/clear/users/meryem/Results/meta_zsl_qa/MLQA"
pre_trained_model_name=$results_path"/MAML_QRY/train_en-test_ar,de,es,hi,vi,zh/l2l/kspt_6-qqry_6/en_train_set/few_shot_ar/use_adapt/checkpoint-5550/"

python main_new_data.py --use-slots --train-langs en --dev-langs ar --test-langs ar de es hi vi zh \
       --do-lower-case --adam-lr 3e-5 --max-seq-length 384  --doc-stride 128 --save-steps 50 --gradient-accumulation-steps 4 \
       --use-adapt --k-spt 6 --q-qry 6 --data-dir $data_dir --out-dir $results_path --batch-sz 2500 --warmup-steps 500 \
       --pre-train-steps 5000 --local_rank -1 --use-pretrained-model --pre-trained-model-name $pre_trained_model_name