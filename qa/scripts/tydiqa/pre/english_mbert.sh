#!/bin/bash
data_dir=""
results_path="" ##en fi id ko ru sw te
model="BertBaseMultilingualCased"

for SEED in 40 42 44 163
do
    python main_pre_train.py --use-slots --train-langs en --dev-langs en --test-langs ar bn en fi id ru sw te \
           --do-lower-case --adam-lr 3e-5 --max-seq-length 384  --doc-stride 128 --save-steps 50 --gradient-accumulation-steps 4 \
           --use-adapt --k-spt 6 --q-qry 6 --data-dir $data_dir --out-dir $results_path --batch-sz 2500 --warmup-steps 500 \
           --pre-train-steps 5000 --local_rank -1 --seed $SEED --trans-model $model
done
