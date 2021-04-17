#!/bin/bash
data_path="" # Path to the root directory of TyDiQA dataset
results_path="" # Path to the root results directory
pre_trained_model_name="" # Path to the model pre-trained on English
model="BertBaseMultilingualCased"

echo "FINE-TUNING ON EACH LANGUAGE INDEPENDENTLY"
for LANG in ar bn fi id ko ru sw te
do
    echo "FINE-TUNE on "$LANG
    for SEED in 40 42 44 163
    do
        python main_fine_tune.py --train-langs en \
                                 --dev-langs $LANG \
                                 --test-langs ar bn en fi id ko ru sw te \
                                 --do-lower-case \
                                 --adam-lr 3e-5 \
                                 --max-seq-length 384 \
                                 --doc-stride 128 \
                                 --save-steps 50 \
                                 --gradient-accumulation-steps 4 \
                                 --data-dir $data_path \
                                 --out-dir $results_path \
                                 --warmup-steps 500 \
                                 --pre-train-steps 5000 \
                                 --local_rank -1 \
                                 --trans-model $model \
                                 --use-pretrained-model \
                                 --pre-trained-model-name $pre_trained_model_name \
                                 --seed $SEED \
                                 --option "FT"

    done
done