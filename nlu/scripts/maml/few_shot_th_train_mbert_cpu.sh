#!/bin/bash
data_path="/Users/mhamdi/AdobeOneDrive/Project/Datasets/NLU/multilingual_task_oriented_dialog_slotfilling/"
results_path="/Users/mhamdi/AdobeOneDrive/Project/Results/NLUBaselines/cpu-outputs/meta_zsl_nlu/FacebookData/THAI/"
pre_trained_model_name="/Users/mhamdi/AdobeOneDrive/Project/Results/NLUBaselines/cpu-outputs/meta_zsl_nlu/FacebookData/THAI/pytorch_model.bin"
python main.py --use-slots --train-langs en --dev-langs th --test-langs th es --data-dir $data_path --out-dir $results_path \
       --pre-trained-model-name $pre_trained_model_name