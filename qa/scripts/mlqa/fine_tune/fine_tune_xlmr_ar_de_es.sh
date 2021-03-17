#!/usr/bin/env bash
data_dir="/nas/clear/users/meryem/Datasets/QA/MLQA_V1"
results_path="/nas/clear/users/meryem/Results/meta_zsl_qa/MLQA"
pre_trained_model_name="/nas/clear/users/meryem/Results/meta_zsl_qa/MLQA/MAML_QRY/XLMRoberta_basetrain_en-test_ar,de,es,hi,vi,zh/l2l/kspt_6-qqry_6/en_train_set/few_shot_ar/use_adapt/"

python main_fine_tune.py --use-slots --train-langs en --dev-langs ar --test-langs ar de es hi vi zh \
       --do-lower-case --adam-lr 3e-5 --max-seq-length 384  --doc-stride 128 --save-steps 50 --gradient-accumulation-steps 4 \
       --use-adapt --k-spt 3 --q-qry 3 --data-dir $data_dir --out-dir $results_path --batch-sz 2500 --warmup-steps 500 \
       --pre-train-steps 5000 --local_rank -1 --use-pretrained-model --pre-trained-model-name $pre_trained_model_name \
       --trans-model "XLMRoberta_base" --model-type "xlm-roberta"

python main_fine_tune.py --use-slots --train-langs en --dev-langs de --test-langs ar de es hi vi zh \
       --do-lower-case --adam-lr 3e-5 --max-seq-length 384  --doc-stride 128 --save-steps 50 --gradient-accumulation-steps 4 \
       --use-adapt --k-spt 3 --q-qry 3 --data-dir $data_dir --out-dir $results_path --batch-sz 2500 --warmup-steps 500 \
       --pre-train-steps 5000 --local_rank -1 --use-pretrained-model --pre-trained-model-name $pre_trained_model_name \
       --trans-model "XLMRoberta_base" --model-type "xlm-roberta"

python main_fine_tune.py --use-slots --train-langs en --dev-langs es --test-langs ar de es hi vi zh \
       --do-lower-case --adam-lr 3e-5 --max-seq-length 384  --doc-stride 128 --save-steps 50 --gradient-accumulation-steps 4 \
       --use-adapt --k-spt 3 --q-qry 3 --data-dir $data_dir --out-dir $results_path --batch-sz 2500 --warmup-steps 500 \
       --pre-train-steps 5000 --local_rank -1 --use-pretrained-model --pre-trained-model-name $pre_trained_model_name \
       --trans-model "XLMRoberta_base" --model-type "xlm-roberta"