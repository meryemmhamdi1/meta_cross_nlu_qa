#!/bin/bash
data_path="" # Path to the root directory of TyDiQA dataset
results_path="" # Path to the root results directory
pre_trained_model_name="" # Path to the model pre-trained on English
model="BertBaseMultilingualCased"

echo "FINE TUNING ON EACH LANGUAGE INDEPENDENTLY + ENGLISH"
for LANG in ar de es hi vi zh
do
    echo "FINE-TUNING on"$LANG
    for SEED in 42 35 119 40
    do
        python main_fine_tune.py --use-slots --train-langs en --dev-langs en $LANG --test-langs ar bn en fi id ko ru sw te \
           --do-lower-case --adam-lr 3e-5 --max-seq-length 384  --doc-stride 128 --save-steps 50 --gradient-accumulation-steps 4 \
           --use-adapt --k-spt 6 --q-qry 6 --data-dir $data_path --out-dir $results_path --batch-sz 2500 --warmup-steps 500 \
           --pre-train-steps 5000 --local_rank -1 --trans-model $model --seed $SEED

    done
done