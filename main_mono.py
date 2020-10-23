from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import argparse
import os
import gc

from data_utils import Dataset
from meta_reader import MetaDataset
from pre_train_base import *
from base_model_l2l import TransformerNLU
from meta_learner_l2l import *
from transformers_config import *

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_paths(train_langs, dev_langs, test_langs, use_few_shot, data_dir, data_format):
    """
    ## 1.2. Load train/dev/test splits from train/dev/test languages => train_set
    ### 1.2.1. Train split of train lang used in the base model
    ### 1.2.2. Dev split of train/dev/test languages used in meta-learning
        #### a. Dev split of train lang used in support set of meta-training => spt_set
        #### b. Dev split of dev lang used in query set of meta-training => qry_set
        #### c. Dev split of test lang used in meta-adaptation in few-shot case => tune_set
        ### 1.2.3. Test split of all train/dev/test languages used in testing the adaptation of the new model => test_set
    :param train_langs:
    :param dev_langs:
    :param test_langs:
    :param use_few_shot:
    :param data_dir:
    :param data_format:
    :return:
    """
    train_paths, spt_paths, qry_paths, tune_paths, test_paths = {}, {}, {}, {}, {}
    train_ext, tune_ext, test_ext = "train", "eval", "test"
    for lang in train_langs:
        train_paths.update({lang: os.path.join(os.path.join(data_dir, lang), train_ext+"-"+lang+"."+data_format)})
        spt_paths.update({lang: os.path.join(os.path.join(data_dir, lang), tune_ext+"-"+lang+"."+data_format)})

    for lang in dev_langs:
        qry_paths.update({lang: os.path.join(os.path.join(data_dir, lang), tune_ext+"-"+lang+"."+data_format)})

    if use_few_shot:
        for lang in test_langs:
            tune_paths.update({lang: os.path.join(os.path.join(data_dir, lang), tune_ext+"-"+lang+"."+data_format)})
    else:
        for lang in dev_langs:
            tune_paths.update({lang: os.path.join(os.path.join(data_dir, lang), tune_ext+"-"+lang+"."+data_format)})

    for lang in test_langs:
        test_paths.update({lang: os.path.join(os.path.join(data_dir, lang), test_ext+"-"+lang+"."+data_format)})

    return train_paths, spt_paths, qry_paths, tune_paths, test_paths


def run(use_adapt, use_dpp, trans_model, data_dir, train_langs, dev_langs, test_langs, use_few_shot, data_format,
        use_slots, data_config, use_pretrained_model, pre_train_config, opt_config, pre_trained_model_path, out_dir):
    """
    Training/Fine tuning using meta-training and testing/analyzing the setup against query

    ## 1.4. Constructing meta-learning dataset
    ## Step 1: meta-training: spt_lang is the support language (source language in zero-shot learning (ZSL)
    ###                       tune_lang is the target language in few-shot or is different in ZSL

    :return:
    """

    out_dir = os.path.join(out_dir, "FINE_TUNE/without_en/train_"+",".join(train_langs)+"-test_"+",".join(test_langs) +
                           "/few_shot_"+",".join(dev_langs)+"/")

    writer = SummaryWriter(os.path.join(out_dir, 'runs'))
    tokenizer = MODELS_dict[trans_model][1].from_pretrained(MODELS_dict[trans_model][0],
                                                            do_lower_case=True,
                                                            do_basic_tokenize=False)

    print("Preparing base dataset")
    train_paths, spt_paths, qry_paths, tune_paths, test_paths = prepare_paths(train_langs, dev_langs, test_langs,
                                                                              use_few_shot, data_dir, data_format)

    dataset = Dataset(tokenizer, data_format, use_slots, train_paths, spt_paths, qry_paths, tune_paths, test_paths)

    print("Initializing Base Transformer NLU model")
    model_trans = MODELS_dict[trans_model][2].from_pretrained(MODELS_dict[trans_model][0],
                                                              num_labels=len(dataset.intent_types))

    model = TransformerNLU(model_trans,
                           len(dataset.intent_types),
                           use_slots=use_slots,
                           num_slots=len(dataset.slot_types))
    if torch.cuda.device_count() > 0:
        model.cuda()

    print("Fine tuning ...")

    pre_train_from_scratch(model, device, dataset, dataset.qry, pre_train_config, use_slots, test_langs, writer, out_dir)
    for lang in test_langs:
        if use_slots:
            test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1, test_slot_prec, test_slot_rec, \
            test_slot_f1 = nlu_evaluation(model, dataset, lang, dataset.test[lang].size, use_slots, device)

            print('Test on {} | Intent Accuracy = {:.4f} Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f} '
                  '| Slot  Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f}'
                  .format(lang, test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1,
                          test_slot_prec, test_slot_rec, test_slot_f1))
        else:
            test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1 \
                = nlu_evaluation(model, dataset, lang, dataset.test[lang].size, use_slots, device)

            print('Test on {} | Intent Accuracy = {:.4f} Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f} '
                  .format(lang, test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1))

    return dataset, model


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-langs", help="train languages list", nargs="+", default=[])
    parser.add_argument("--dev-langs", help="dev languages list", nargs="+", default=[])
    parser.add_argument("--test-langs", help="test languages list", nargs="+", default=[])
    parser.add_argument('--use-few-shot', help='If true, use test languages in the meta-adaptation stage',
                        action='store_true')  # zero-shot by default

    parser.add_argument('--use-slots', help='If true, optimize for slot filling loss too', action='store_true')

    parser.add_argument("--trans-model", help="name of transformer model", default="BertBaseMultilingualCased")
    parser.add_argument('--data-dir', help='Path of data',  default="")

    parser.add_argument('--out-dir', help='Path of output data', default="")
    parser.add_argument('--pre-trained-model-name', help='Path of output pre-trained model binary', default="")
    parser.add_argument('--data-format', help='Whether it is tsv or json', default="tsv")

    ## Pre-training hyperparameters
    parser.add_argument('--pre-train-steps', help='the number of iterations if pre-training is done from scratch',
                        type=int, default=2000)
    parser.add_argument('--batch-size', help="batch size in the pre-training process", type=int, default=32)
    parser.add_argument('--adam-lr', help="learning rate of adam optimizer when training base model from scratch",
                        type=float, default=4e-5)
    parser.add_argument('--adam-eps', help="epsilon of adam optimizer when training base model from scratch",
                        type=float, default= 1e-08)
    parser.add_argument('--use-pretrained-model', help='If true, use pre-trained NLU model', action='store_true')


    ## Meta-learning Dataset Hyperparameters (Have to run using grid search to analyze learning curves)
    parser.add_argument('--n-way', help='Number of classes for each task in the meta-learning', type=int, default=11)
    parser.add_argument('--k-spt', help='Number of support examples per class', type=int, default=4)
    parser.add_argument('--q-qry', help='Number of query examples per class', type=int, default=4)
    parser.add_argument('--k-tune', help='Number of query examples per class', type=int, default=2)
    parser.add_argument('--batch-sz', help='Number of iterations', type=int, default=1000)
    parser.add_argument('--seed', help="Random seed for initialization", type=int, default=42)

    ## Meta-learning optimization Hyperparameters (tunable => hyperparameter optimization search or some automatic tool)
    parser.add_argument('--epoch', help='Number of epochs', type=int, default=10)  # Early stopping

    parser.add_argument('--n-task', help='Number of tasks', type=int, default=4)

    parser.add_argument('--n-up-train-step', help='Number of update steps in the meta-training stage', type=int,
                        default=5)

    parser.add_argument('--n-up-test-step', help="Number of update steps in the meta-update stage", type=int,
                        default=10)

    parser.add_argument('--alpha-lr', help='Learning rate during the meta-training stage (inner loop)', type=int,
                        default=1e-2)

    parser.add_argument('--beta-lr', help='Learning rate during the meta-update stage (outer loop)', type=int,
                        default=1e-3)

    parser.add_argument('--gamma-lr', help='Learning rate during the meta-update in the adaptation', type=int,
                        default=1e-3)

    parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--use-dpp', help='Whether to use DPP or not', action='store_true')
    parser.add_argument('--use-adapt', help='Whether to use meta-adaptation', action='store_true')

    args = parser.parse_args()
    set_seed(args)

    #torch.cuda.set_device(args.local_rank)
    #torch.distributed.init_process_group(backend='nccl', init_method='env://')
    #cudnn.benchmark = True

    pre_train_config = {"pre_train_steps": args.pre_train_steps, "batch_size": args.batch_size,
                        "adam_lr": args.adam_lr, "adam_eps": args.adam_eps}

    data_config = {"n_way": args.n_way, "k_spt": args.k_spt, "q_qry": args.q_qry, "batch_sz": args.batch_sz}

    opt_config = {"epoch": args.epoch, "n_task": args.n_task, "n_up_train_step": args.n_up_train_step,
                  "n_up_test_step": args.n_up_test_step, "alpha_lr": args.alpha_lr, "beta_lr": args.beta_lr,
                  "gamma_lr": args.gamma_lr}

    dataset, model = run(args.use_adapt, args.use_dpp, args.trans_model, args.data_dir, args.train_langs, args.dev_langs,
                         args.test_langs, args.use_few_shot, args.data_format, args.use_slots, data_config,
                         args.use_pretrained_model, pre_train_config, opt_config,
                         os.path.join(args.out_dir, args.pre_trained_model_name), args.out_dir)




