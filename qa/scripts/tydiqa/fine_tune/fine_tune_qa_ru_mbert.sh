#!/bin/bash
data_dir="/nas/clear/users/meryem/Datasets/QA/tydiqa"
results_path="/nas/clear/users/meryem/Results/meta_zsl_qa/tydiqa" ##en fi id ko ru sw te
pre_trained_model_name=$results_path"/MAML/train_en-test_ar,bn,en,fi,id,ko,ru,sw,te/l2l/kspt_2-qqry_2/en_train_set/few_shot_ar/use_adapt/checkpoint-2000/"

python main_fine_tune.py --use-slots --train-langs en --dev-langs ru --test-langs ar bn en fi id ko ru sw te  \
       --do-lower-case --adam-lr 3e-5 --max-seq-length 384  --doc-stride 128 --save-steps 50 --gradient-accumulation-steps 4 \
       --use-adapt --k-spt 6 --q-qry 6 --data-dir $data_dir --out-dir $results_path --batch-sz 2500 --warmup-steps 500 \
       --pre-train-steps 5000 --local_rank -1 --use-pretrained-model --pre-trained-model-name $pre_trained_model_name