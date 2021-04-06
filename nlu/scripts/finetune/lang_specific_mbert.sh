#!/bin/bash
data_path=""
results_path=""
pre_trained_model_name=""
model="BertBaseMultilingualCased"

echo "FINE TUNING ON EACH LANGUAGE INDEPENDENTLY"
for LANG in es th
do
    echo "FINE-TUNING on "$LANG
    for SEED in 42 35 119 40
    do
        python main.py --option "FT"  --use-slots --train-langs en --dev-langs $LANG --test-langs en es th \
               --k-spt 6 --q-qry 6 --data-dir $data_path --out-dir $results_path --batch-sz 2500 --pre-train-steps 2000 \
               --local_rank 0 --use-pretrained-model --pre-trained-model-name $pre_trained_model_name \
               --trans-model $model --adam-lr 1e-3 --seed $SEED --portions "1,1"
    done
done
