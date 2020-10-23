# Meta-Transfer Learning for Cross-lingual Natural Language Understanding

## Meryem M'hamdi, Doo Soo Kim, Trung Bui, Franck Dernoncourt

This is the implementation for cross-lingual Transfer Learning NLU based on different flavours of meta-learning.


#### Abstract:

Meta-learning has been shown helpful for several tasks especially in computer vision with tasks like ImageNet 
and Omniglot and on some NLP tasks but hasn’t gained much attention especially in enhancing cross-lingual transfer. 
Unlike direct transfer learning which applies to new languages without adaptation and multi-tasking or joint training 
which can still overfit to high-resourced languages and requires measures for balancing the different tasks and longer 
training to reach stability, meta-learning is better fitted to enhance the generalizability to scenarios where we don’t
have enough training instances in particular low-resourced languages. We have implemented a meta-learning based extension
 to cross-lingual base model for joint intent detection and slot filling and will use it to analyze the performance 
 trends in zero-shot transfer learning on typologically variant NLU datasets. 

#### Requirements:

conda create --name "zsl_nlu" python=3.6

conda activate zsl_nlu

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

pip install pytorch_transformers pytorch_pretrained_bert scikit-learn transformers torchviz tensorboard

If installing pytorch with cuda in Sensei, add links to cuda binaries by setting LD_LIBRARY_PATH in ~/.bashrc.

#### Preparing/Loading the dataset:
This code works for both [public Facebook NLU dataset](https://github.com/zliucr/mixed-language-training/tree/master/data/nlu/nlu_data) 
and [Jarvis Intent dataset](https://git.corp.adobe.com/mhamdi/jarvis-multilingual/tree/meryem/data).Download them and
point --data-dir flag in pre_train_base.py and main.py towards their root directory containing the splits per language. 
If working with Facebook NLU dataset use tsv as --data-format otherwise use json. The preprocessor will automatically
know how to handle each dataset type.

#### Running the code:

1) Initializing the parameters \theta_{0}:
    * Pre-training the joint NLU Transformer model:
        * Offline: 
        ```
        python pre_train_base.py --train --train-langs en --test-langs en es th --use-slots --data-format "tsv"
                                 --trans-model "BertBaseMultilingualCased" --data-dir "Facebook-NLU-Data/"
                                 --out-dir "out" --pre-train-steps 2000 --batch-size 32 --adam-lr 4e-5
                                 --adam-eps 1e-08 
        ```
        * As a part of the whole meta-learning pipeline:
        run main.py which automatically calls pre_train_base functionalities for training and evaluation and to that 
        effect run main.py without setting --use-pretrained-model flag
        
    * Use of saved pre-trained model for NLU:
    In this case, add --use-pretrained-model flag to main.py and provide the path to the binary pytorch file 
    --pre-trained-model-name (see step 2 below)  
    
    
2) Training meta-learning MAML:
    * Few-shot learning on Thai, Zero-shot on Spanish:
    ```
    python main.py --train --train-langs en --dev-langs th --test-langs en es th --use-slots --data-format "tsv" 
                   --trans-model "BertBaseMultilingualCased" --data-dir "Facebook-NLU-Data/" --out-dir "out"
                   --pre-train-steps 2000 --batch-size 32 --adam-lr 4e-5 --adam-eps 1e-08 --n-way 11 --k-spt 5 
                   --q-qry 5 --k-tune 5 --batch-sz 10000 --epoch 100 --n-task 4 --n-up-train-step 5 --n-up-test-step 5 
                   --alpha-lr 1e-2 --beta-lr 1e-3 --gamma-lr 1e-3     
    ```
   
    * Zero-shot learning on Thai, Few-shot on Spanish:
    ```
    python main.py --train --train-langs en --test-langs en es th --use-slots --trans-model "BertBaseMultilingualCased" 
                   --data-dir "Facebook-NLU-Data/" --out-dir "out" --data-format "tsv" --pre-train-steps 2000 
                   --batch-size 32 --adam-lr 4e-5 --adam-eps 1e-08 --n-way 11 --k-spt 5 --q-qry 5 --k-tune 5 
                   --batch-sz 10000 --epoch 100 --n-task 4 --n-up-train-step 5 --n-up-test-step 5 --alpha-lr 1e-2 
                   --beta-lr 1e-3 --gamma-lr 1e-3          
    ```

4) Training meta-learning Hybrid of MAML/Prototypical:

COMING SOON

5) Training meta-learning with auto-encoder alignment loss:

COMING SOON


 #### Replicating experiments:

Please refer to scripts folder

 #### Reported Results:
 
 Coming soon (will be in the form of visualizations in the notebooks folder and tensorboard runs folder)
 In the agenda:
 
 * Quantitative Analysis:
     * Comparisons with:
        * Direct transfer learning (transformer alone)
        * Joint or multi-tasking training 
        * State of the art like mixed-training or latent variable model
     * Ablation studies:
        * Per language
        * Per model component
        * Per loss component
        
 * Qualitative Analysis:
    * Visualization of learned alignments and how they impact the performance
    * Visualization of what amount of data leads to parameter changes and how it impacts the performance 
    * Fine-grained visualization of which layers are impacted by the meta-training
    * Which language help each other
 

 