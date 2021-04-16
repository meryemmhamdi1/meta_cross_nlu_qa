#!/bin/bash
data_path=""
results_path=""
pre_trained_model_name=""
model="BertBaseMultilingualCased"

for PORTION in "0.1" "0.3" "0.5" "0.7"
do
    for SEED in 42 35 119 40
    do
        python main.py --option "META" --use-slots --train-langs en --dev-langs th --test-langs en es th --use-adapt --use-back --use-non-overlap \
               --k-spt 6 --q-qry 6 --data-dir $data_path --out-dir $results_path --batch-sz 2500 --pre-train-steps 2000 --local_rank 0 \
               --use-pretrained-model "pre_trained" --pre-trained-model-name $pre_trained_model_name --seed $SEED --portions $PORTION
    done
done
