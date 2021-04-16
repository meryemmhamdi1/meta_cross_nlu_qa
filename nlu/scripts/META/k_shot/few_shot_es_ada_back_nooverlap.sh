#!/bin/bash
data_path=""
results_path=""
pre_trained_model_name=""
model="BertBaseMultilingualCased"

for SHOT in 1 3 6 9
do
    for SEED in 42 35 119 40
    do
        python main.py --option "META" --use-slots --train-langs en --dev-langs es --test-langs en es th --use-adapt --use-back --use-non-overlap \
               --k-spt $SHOT --q-qry $SHOT --data-dir $data_path --out-dir $results_path --batch-sz 2500 --pre-train-steps 2000 --local_rank 0 \
               --use-pretrained-model "pre_trained" --pre-trained-model-name $pre_trained_model_name --seed $SEED
    done
done
