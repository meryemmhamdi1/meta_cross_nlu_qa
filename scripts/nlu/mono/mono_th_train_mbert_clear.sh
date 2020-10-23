#!/bin/bash
data_path="/nas/clear/users/meryem/Datasets/NLU/CrossLingualNLU/facebook_nlu/"
results_path="/nas/clear/users/meryem/Results/meta_zsl_nlu/"

python main_mono.py --use-slots --train-langs en --dev-langs th --test-langs es th --use-adapt --k-spt 2 --q-qry 2 \
       --data-dir $data_path --out-dir $results_path --batch-sz 2500 --pre-train-steps 5000 --local_rank 0