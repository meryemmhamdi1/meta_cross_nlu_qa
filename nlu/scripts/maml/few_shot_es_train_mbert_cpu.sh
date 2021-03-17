#!/bin/bash
data_path="/Users/mhamdi/AdobeOneDrive/Project/Datasets/NLU/multilingual_task_oriented_dialog_slotfilling/"
results_path="/Users/mhamdi/AdobeOneDrive/Project/Results/NLUBaselines/cpu-outputs/meta_zsl_nlu/FacebookData/"
pre_trained_model_name="/Users/mhamdi/AdobeOneDrive/Project/Results/NLUBaselines/Sensei-outputs/FacebookData/train_en-test_en,es,th/pytorch_model.bin"
python main.py --use-slots --train-langs en --dev-langs es --test-langs es --data-dir $data_path --out-dir $results_path \
       --pre-trained-model-name $pre_trained_model_name
