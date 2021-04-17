#!/bin/bash
home_dir=""
data_dir=$home_dir""
results_path=$home_dir""
pre_trained_model_name=$results_path""
cache_dir=""

echo "FEW-SHOT LEARNING ON ALL LANGUAGES JOINTLY"
for SEED in 442 44 40 163
do
    for LANG in ar bn fi id ru sw te
    do
        python main_trans_ada_no_acc.py --use-adapt \
                                        --train-langs en \
                                        --dev-langs $LANG \
                                        --test-langs ar bn en fi id ko ru sw te \
                                        --do-lower-case \
                                        --adam-lr 3e-5 \
                                        --max-seq-length 384 \
                                        --doc-stride 128 \
                                        --save-steps 50 \
                                        --gradient-accumulation-steps 4 \
                                        --k-spt 6 --q-qry 6 \
                                        --data-dir $data_dir \
                                        --out-dir $results_path \
                                        --batch-sz 2500 \
                                        --warmup-steps 500 \
                                        --pre-train-steps 5000 \
                                        --local_rank -1 \
                                        --use-pretrained-model \
                                        --pre-trained-model-name $pre_trained_model_name \
                                        --cache_dir $cache_dir \
                                        --seed $SEED
    done
done