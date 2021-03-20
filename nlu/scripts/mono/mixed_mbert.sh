#!/bin/bash
data_path=""
results_path=""
pre_trained_model_name=""
model="BertBaseMultilingualCased"

echo "FINE TUNING MIXED WITH ENGLISH"
for LANG in es th
do
    echo "FINE-TUNING on "$LANG
    for SEED in 42 35 119 40
    do
        python main.py --option "MONO" --use-slots --train-langs en --dev-langs en $LANG --test-langs es th --use-adapt --k-spt 2 --q-qry 2 \
           --data-dir $data_path --out-dir $results_path --batch-sz 2500 --pre-train-steps 5000 --local_rank 0 --seed $SEED \
           --trans-model $model --adam-lr 1e-3 --portions "1,1"
    done
done