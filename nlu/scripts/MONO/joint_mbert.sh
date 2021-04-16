#!/bin/bash
data_path=""
results_path=""
pre_trained_model_name=""
model="BertBaseMultilingualCased"

echo "MONOLINGUAL TRAINING (without any pre-trained model on ENGLISH) JOINTLY ON ALL LANGUAGES"
for SEED in 42 35 119 40
do
    python main.py --option "MONO" --use-slots --train-langs en --dev-langs es th --test-langs es th --use-adapt --k-spt 2 --q-qry 2 \
       --data-dir $data_path --out-dir $results_path --batch-sz 2500 --pre-train-steps 5000 --local_rank 0 --seed $SEED \
       --trans-model $model --adam-lr 1e-3 --portions "1,1"
done
