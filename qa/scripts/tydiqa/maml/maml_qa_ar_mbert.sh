#!/bin/bash
#SBATCH --partition=isi
#SBATCH --mem=40g
#SBATCH --time=23:59:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:2
#SBATCH --output=R-%x.out.%j
#SBATCH --error=R-%x.err.%j
#SBATCH --export=NONE

#conda activate xtreme

home_dir="/home1/mmhamdi/" #/nas/clear/users/meryem/
data_dir=$home_dir"Datasets/QA/tydiqa"
task="tydiqa"
results_path=$home_dir"Results/meta_zsl_qa/tydiqa" ##en fi id ko ru sw te
pre_trained_model_name=$results_path"/MAML_QRY/BertBaseMultilingualCased/train_en-test_ar,bn,en,fi,id,ko,ru,sw,te/l2l/kspt_12-qqry_12/en_train_set/few_shot_ar/use_adapt/"

python main_new_data.py --use-slots --train-langs en --dev-langs ar --test-langs ar bn en fi id ko ru sw te \
       --do-lower-case --adam-lr 3e-5 --max-seq-length 384  --doc-stride 128 --save-steps 50 --gradient-accumulation-steps 4 \
       --use-adapt --k-spt 12 --q-qry 12 --data-dir $data_dir --out-dir $results_path --batch-sz 2500 --warmup-steps 500 \
       --pre-train-steps 5000 --local_rank -1 --cache_dir "/home1/mmhamdi/Models/mbert/" --task $task \
       --use-pretrained-model --pre-trained-model-name $pre_trained_model_name
