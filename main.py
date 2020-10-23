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
        spt_paths.update({lang: os.path.join(os.path.join(data_dir, lang), train_ext+"-"+lang+"."+data_format)})

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


def run(freeze_bert, use_adapt, use_dpp, trans_model, data_dir, train_langs, dev_langs, test_langs, use_few_shot, data_format,
        use_slots, data_config, use_pretrained_model, pre_train_config, opt_config, pre_trained_model_path, out_dir):
    """
    Training/Fine tuning using meta-training and testing/analyzing the setup against query

    ## 1.4. Constructing meta-learning dataset
    ## Step 1: meta-training: spt_lang is the support language (source language in zero-shot learning (ZSL)
    ###                       tune_lang is the target language in few-shot or is different in ZSL

    :return:
    """
    if use_adapt:
        flag_adapt = "use_adapt/"
    else:
        flag_adapt = "no_adapt/"

    if len(freeze_bert) == 0:
        freeze_bert_flag = ""
    else:
        freeze_bert_flag = "freeze_bert_" + ",".join(freeze_bert)

    out_dir = os.path.join(out_dir, "MAML/train_"+",".join(train_langs)+"-test_"+",".join(test_langs) + "/l2l"
                                    + "/kspt_" + str(data_config["k_spt"]) + "-qqry_" + str(data_config["q_qry"])
                                    + "/en_train_set/" + freeze_bert_flag + "/few_shot_"+",".join(dev_langs)+"/"
                                    + flag_adapt)

    writer = SummaryWriter(os.path.join(out_dir, 'runs'))
    tokenizer = MODELS_dict[trans_model][1].from_pretrained(MODELS_dict[trans_model][0],
                                                            do_lower_case=True,
                                                            do_basic_tokenize=False)

    print("Saving in out_dir:", out_dir)

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

    # 2. Loading a pre-trained base-model if it exists
    if use_pretrained_model:
        print("Loading the already pretrained model")
        pretrained_model = torch.load(pre_trained_model_path)
        model.load_state_dict(pretrained_model)

    else:
        print("Pre-training from scratch")
        pre_train_from_scratch(model, device, dataset, pre_train_config, use_slots, test_langs, writer, out_dir)

    print("Evaluation on test set at the end of pre-training...")
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

    print("Preparing Meta dataset for training")
    meta_dataset = MetaDataset(dataset, data_config)
    meta_train_dataset = meta_dataset.auxi_train_batches

    print("Preparing Meta dataset for tuning")
    meta_tune_dataset = meta_dataset.auxi_tune_batches

    # 3. Initialize meta-learner
    print("Initializing Meta-learner model")
    meta_learner = MetaLearner(opt_config, model, use_slots, dataset.intent_types, dataset.slot_types,
                               device, dataset, freeze_bert)

    if torch.cuda.device_count() > 0:
        meta_learner.cuda()
        device_ids = list(range(torch.cuda.device_count()))
        if use_dpp:
            meta_learner = DDP(meta_learner, device_ids=device_ids, output_device=0, broadcast_buffers=False,
                               find_unused_parameters=True)

    # 4. Meta-training
    number_steps = 0
    opt = torch.optim.Adam(model.parameters(),  lr=opt_config["alpha_lr"])
    for epoch in tqdm(range(opt_config["epoch"])):
        gc.collect()
        print("Number of iterations:", data_config["batch_sz"]//opt_config["n_task"])
        for i in range(data_config["batch_sz"]//opt_config["n_task"]):
            opt.zero_grad()
            batch = meta_dataset.next_batch(meta_train_dataset, opt_config["n_task"], i)

            batch = tuple(t.cuda() for t in batch)

            inp_ids_spt_all, tok_typ_ids_spt_all, att_masks_spt_all, len_spt_all, int_l_spt_all, slot_l_spt_all, \
            inp_ids_qry_all, tok_typ_ids_qry_all, att_masks_qry_all, len_qry_all, int_l_qry_all, slot_l_qry_all = batch

            batch_tune = meta_dataset.next_batch(meta_tune_dataset, opt_config["n_task"], i)
            batch_tune = tuple(t.cuda() for t in batch_tune)

            inp_ids_spt_tune, tok_typ_ids_spt_tune, att_masks_spt_tune, len_spt_tune, int_l_spt_tune, \
            slot_l_spt_tune, inp_ids_qry_tune, tok_typ_ids_qry_tune, att_masks_qry_tune, len_qry_tune, \
            int_l_qry_tune, slot_l_qry_tune = batch_tune

            maml, meta_train_error, meta_train_accuracy, meta_tune_error, meta_tune_accuracy \
                = meta_learner(use_adapt, opt, inp_ids_spt_all,  tok_typ_ids_spt_all, att_masks_spt_all, len_spt_all,
                                       int_l_spt_all, slot_l_spt_all, inp_ids_qry_all, tok_typ_ids_qry_all,
                                       att_masks_qry_all, len_qry_all, int_l_qry_all, slot_l_qry_all,
                                       ##
                                       inp_ids_spt_tune, tok_typ_ids_spt_tune, att_masks_spt_tune, len_spt_tune,
                                       int_l_spt_tune, slot_l_spt_tune, inp_ids_qry_tune, tok_typ_ids_qry_tune,
                                       att_masks_qry_tune, len_qry_tune, int_l_qry_tune, slot_l_qry_tune)

            # Print some metrics
            meta_train_error_avg = meta_train_error / opt_config["n_task"]
            meta_train_acc_avg = meta_train_accuracy / opt_config["n_task"]
            meta_tune_error_avg = meta_tune_error / opt_config["n_task"]
            meta_tune_acc_avg = meta_tune_accuracy / opt_config["n_task"]

            print('\n')
            print('Iteration', i)
            print('Meta Train Error', meta_train_error_avg)
            print('Meta Train Intent Accuracy', meta_train_acc_avg)
            print('Meta Tune Error', meta_tune_error_avg)
            print('Meta Tune Intent Accuracy', meta_tune_acc_avg)

            writer.add_scalar("META_train_error", meta_train_error_avg, number_steps)
            writer.add_scalar("META_train_intent_acc", meta_train_acc_avg, number_steps)
            writer.add_scalar("META_tune_error", meta_tune_error_avg, number_steps)
            writer.add_scalar("META_tune_intent_acc", meta_tune_acc_avg, number_steps)

            if i % 10 == 0:
                # 5. Testing
                print("Testing at Epoch {}, Step {} |".format(epoch, i))
                if use_dpp:
                    metrics = meta_learner.module.zero_shot_test(test_langs, dataset)
                else:
                    metrics = meta_learner.zero_shot_test(test_langs, dataset)

                for lang in test_langs:
                    string_eval = "\tTest on {} "
                    tuple_l = [lang]
                    for metric in metrics[lang]:
                        string_eval += "| " + metric + " = {:.4f} "
                        tuple_l.append(metrics[lang][metric])
                        writer.add_scalar('META-Test-'+metric+"-"+lang, metrics[lang][metric], number_steps)
                    print(string_eval.format(*tuple_l))

            number_steps += 1

    # 5. Testing
    print("Results on Test Set at the end of META-TRAINING:\n")
    if use_dpp:
        metrics = meta_learner.module.zero_shot_test(test_langs, dataset)
    else:
        metrics = meta_learner.zero_shot_test(test_langs, dataset)

    for lang in test_langs:
        tuple_l = []
        string_eval = "End of Training Test on {} | "
        tuple_l.append(lang)
        for metric in metrics:
            string_eval += metric + " = {:.4f} |"
            tuple_l.append(metrics[metric][lang])
            writer.add_scalar('META-Test-'+metric+"-"+lang, metrics[lang][metric], number_steps)
        print(string_eval.format(*tuple_l))

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
    parser.add_argument('--freeze-bert', help='Whether to use meta-adaptation', nargs="+", default=[])

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

    dataset, model = run(args.freeze_bert, args.use_adapt, args.use_dpp, args.trans_model, args.data_dir,
                         args.train_langs, args.dev_langs, args.test_langs, args.use_few_shot, args.data_format,
                         args.use_slots, data_config, args.use_pretrained_model, pre_train_config, opt_config,
                         os.path.join(args.out_dir, args.pre_trained_model_name), args.out_dir)




