#!/bin/bash
data_path=""
results_path=""
pre_trained_model_name=$results_path""
model="BertBaseMultilingualCased"

for LANG in es th
do
python proto_main_l2l.py --use-slots --train-langs en --dev-langs $LANG --test-langs en es th --k-spt 6 --q-qry 3 \
       --data-dir $data_path --out-dir $results_path --batch-sz 100 --trans-model $model --use-pretrained-model \
       --pre-trained-model-name $pre_trained_model_name --lang $LANG

done