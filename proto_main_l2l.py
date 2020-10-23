from sklearn.decomposition import PCA
import argparse
import random
import os
import gc

from data_utils import Dataset
from proto.meta_reader import MetaDataset
from proto.base_model_l2l import PrototypicalTransformerNLU
from proto.meta_learner_l2l import *
from transformers_config import *

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch import optim
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pca = PCA(n_components=300)


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
        qry_paths.update({lang: os.path.join(os.path.join(data_dir, lang), train_ext+"-"+lang+"."+data_format)})

    if use_few_shot:
        for lang in test_langs:
            tune_paths.update({lang: os.path.join(os.path.join(data_dir, lang), tune_ext+"-"+lang+"."+data_format)})
    else:
        for lang in dev_langs:
            tune_paths.update({lang: os.path.join(os.path.join(data_dir, lang), tune_ext+"-"+lang+"."+data_format)})

    for lang in test_langs:
        test_paths.update({lang: os.path.join(os.path.join(data_dir, lang), test_ext+"-"+lang+"."+data_format)})

    return train_paths, spt_paths, qry_paths, tune_paths, test_paths


def compute_prototypes(dataset, data_size, data_part, model, use_slots,  plot_ext):
    intent_proto_dict = {i: [] for i in range(len(dataset.intent_types))}
    for i in range(0, len(dataset.intent_types)):
        intent_proto_dict.update({i: []})

    X_i, Y_i, T_i = [], [], []

    if use_slots:
        slot_proto_dict = {i: [] for i in range(len(dataset.slot_types))}

        X_s, Y_s, T_s = [], [], []

    for _ in tqdm(range(100)):
        (input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts), text \
            = dataset.next_batch(1, data_part)

        input_ids = input_ids.cuda()

        true_intent = intent_labels.squeeze().item()

        lm_output_1 = model.trans_model(input_ids)[0][0]  # average over all layers

        intent_proto_dict[true_intent].append(lm_output_1[0])
        X_i.append(lm_output_1[0].cpu().detach().numpy().tolist())
        Y_i.append(dataset.intent_types[true_intent])
        T_i.append("Point")

        if use_slots:
            for j in range(1, len(lm_output_1)):
                slot_label = slot_labels[0][j].item()
                slot_proto_dict[slot_label].append(lm_output_1[j])
                X_s.append(lm_output_1[j].cpu().detach().numpy().tolist())
                slot_type = dataset.slot_types[slot_label]
                if "-" in slot_type:
                    slot_type = slot_type.split("-")[1]
                Y_s.append(slot_type)
                T_s.append("Point")

    intent_mean_list = []
    counts_intent_list = []

    for intent in intent_proto_dict:
        embed_list = intent_proto_dict[intent]
        if len(embed_list) > 0:
            counts_intent_list.append(len(embed_list))
        else:
            counts_intent_list.append(1)
        if len(embed_list) > 0:
            torch_tensors = torch.stack(embed_list)
            intent_proto_mean = torch.mean(torch_tensors, dim=0)
        else:
            intent_proto_mean = torch.rand(768, requires_grad=True).cuda()
        intent_mean_list.append(intent_proto_mean)
        X_i.append(intent_proto_mean.cpu().detach().numpy().tolist())
        Y_i.append(dataset.intent_types[intent])
        T_i.append("Centroid")

    #intent_pca = pca.fit_transform(X_i)


    #intent_prototypes = torch.DoubleTensor(intent_pca).to(torch.double).cuda()
    intent_prototypes = torch.stack(intent_mean_list).cuda()
    counts_intents = torch.IntTensor(counts_intent_list).cuda()
    #plot_tsne(X_i, Y_i, T_i , "intents_proto_"+plot_ext+".png")

    proto = {"intent": [intent_prototypes, counts_intents]}

    if use_slots:
        slot_mean_list = []
        counts_slot_list = []
        for slot in slot_proto_dict:
            embed_list = slot_proto_dict[slot]
            if len(embed_list) > 0:
                counts_slot_list.append(len(embed_list))
            else:
                counts_slot_list.append(1)
            if len(embed_list)>0:
                torch_tensors = torch.stack(embed_list)
                slot_proto_mean = torch.mean(torch_tensors, dim=0)
            else:
                slot_proto_mean = torch.rand(768, requires_grad=True).cuda()
            slot_mean_list.append(slot_proto_mean)
            X_s.append(slot_proto_mean.cpu().detach().numpy().tolist())
            slot_type = dataset.slot_types[slot]
            if "-" in slot_type:
                slot_type = slot_type.split("-")[1]
            Y_s.append(slot_type)
            T_s.append("Centroid")

        #slot_pca = pca.fit_transform(X_s)
        #slot_prototypes = torch.DoubleTensor(slot_pca).to(torch.double).cuda()
        slot_prototypes = torch.stack(slot_mean_list).cuda()
        counts_slots = torch.IntTensor(counts_slot_list).cuda()

        #plot_tsne(X_s, Y_s, T_s ,"slots_proto_"+plot_ext+".png")

        proto.update({"slot": [slot_prototypes, counts_slots]})

    return proto


def eval(dataset, meta_learner, test_langs, ext, proto):
    """
    Cross-lingual Evaluation
    :param dataset:
    :param meta_learner:
    :param test_langs:
    :param ext:
    :param proto:
    :return:
    """
    metrics = meta_learner.zero_shot_test(test_langs, dataset, ext, proto)

    print("metrics:", metrics)


def run(use_adapt, use_dpp, trans_model, use_pretrained_model, pre_trained_model_path, data_dir, train_langs, dev_langs,
        test_langs, use_few_shot, data_format, use_slots, data_config, opt_config, out_dir, use_aae, use_triplet_loss):
    """
    Training/Fine tuning using meta-training and testing/analyzing the setup against query

    ## 1.4. Constructing meta-learning dataset
    ## Step 1: meta-training: spt_lang is the support language (source language in zero-shot learning (ZSL)
    ###                       tune_lang is the target language in few-shot or is different in ZSL

    ## Step 2: meta-adaptation (fine tuning)
    # 2. Loading a pre-trained base-model if it exists
    :return:
    """

    if use_adapt:
        flag_adapt = "use_adapt/"
    else:
        flag_adapt = "no_adapt/"

    if use_aae:
        flag_aae = "use_aae/"
    else:
        flag_aae = "no_aae/"

    out_dir = os.path.join(out_dir, "HYMP/train_"+",".join(train_langs)+"-test_"+",".join(test_langs) + "/l2l"
                                    + "/kspt_" + str(data_config["k_spt"]) + "-qqry_" + str(data_config["q_qry"])
                                    + "/few_shot_"+",".join(dev_langs)+"/" +flag_adapt+flag_aae)

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

    model = PrototypicalTransformerNLU(trans_model=model_trans,
                                       num_intents=len(dataset.intent_types),
                                       intent_types=dataset.intent_types,
                                       slot_types=dataset.slot_types,
                                       use_slots=use_slots,
                                       num_slots=len(dataset.slot_types),
                                       use_aae=use_aae)

    if torch.cuda.device_count() > 0:
        model.cuda()

    print("Preparing Meta dataset for training and tuning ...")
    meta_dataset = MetaDataset(dataset, data_config)
    meta_train_dataset = meta_dataset.auxi_train_batches

    meta_tune_dataset = meta_dataset.auxi_tune_batches

    # 3. Initialize meta-learner
    meta_learner = MetaLearner(opt_config, model, use_slots, dataset.intent_types, dataset.slot_types, device,
                               dataset)

    if torch.cuda.device_count() > 0:
        meta_learner.cuda()
        device_ids = list(range(torch.cuda.device_count()))
        if use_dpp:
            meta_learner = DDP(meta_learner, device_ids=device_ids, output_device=0, broadcast_buffers=False,
                               find_unused_parameters=True)

    if use_pretrained_model:
        print("Use pre-trained BERT embeddings ...")
        base_model = torch.load(pre_trained_model_path)
        model_dict = model.state_dict()
        base_dict = {k: v for k, v in base_model.items() if k in model_dict}

        for k, v in base_dict.items():
            model_dict.update({k: v})

        model.load_state_dict(model_dict)

    model.cuda()

    mode = "after"
    prototypes = compute_prototypes(dataset, dataset.train_size, dataset.train, model, use_slots, mode)

    if use_triplet_loss:
        all_train_set_batch = dataset.next_batch_list(dataset.train_size, dataset.train)
        print("Refinement of Embeddings on Cross-lingual Dataset using Triplet Loss")
        all_qry_set_batch = dataset.next_batch_list(dataset.qry_size, dataset.qry)
        max_seq = max(all_train_set_batch[4], all_qry_set_batch[4])
        # Constructing Anchor/positive/negative examples
        list_intent_classes = list(all_train_set_batch[0][0].keys()) + list(all_qry_set_batch[0][0].keys())
        triplet_optim = optim.SGD(model.trans_model.parameters(), lr=0.01, momentum=0.9)
        input_ids = {}
        for intent_class in list_intent_classes:
            if intent_class not in input_ids:
                input_ids.update({intent_class: []})
            if intent_class in all_train_set_batch[0][0]:
                input_ids[intent_class].extend(all_train_set_batch[0][0][intent_class])
            if intent_class in all_qry_set_batch[0][0]:
                input_ids[intent_class].extend(all_qry_set_batch[0][0][intent_class])

        for i, intent in tqdm(enumerate(list_intent_classes)):
            input_ids_intent = input_ids[intent]
            # Triplet Intent
            for _ in tqdm(range(0, 100)): # refinement steps per intent class
                pos_indices = random.sample(list(range(len(input_ids_intent))), 11)

                # Anchor Example
                all_inp_ids_anchor = torch.LongTensor(input_ids_intent[pos_indices[0]] +
                                                      [0] * (max_seq - len(input_ids_intent[pos_indices[0]]))).unsqueeze(0)\
                                                      .cuda()

                # Positive Examples
                inp_ids_pos = []
                for pos_index in pos_indices[1:]:
                    pos_eg = input_ids_intent[pos_index]
                    inp_ids_pos.append(torch.LongTensor(pos_eg + [0] * (max_seq - len(pos_eg))))

                all_inp_ids_pos = torch.stack(inp_ids_pos, dim=0).cuda()

                # Negative Examples
                neg_classes = [item for item in input_ids.keys() if item!=intent]
                all_inp_ids_neg = []
                for _ in range(10):
                    neg_eg = input_ids[random.choice(neg_classes)]
                    neg_index = random.sample(list(range(len(neg_eg))), 1)[0]
                    all_inp_ids_neg.append(torch.LongTensor(neg_eg[neg_index] + [0] * (max_seq - len(neg_eg[neg_index]))))

                all_inp_ids_neg = torch.stack(all_inp_ids_neg, dim=0).cuda()

                model.trans_model.train()
                lm_output_anch = model.trans_model(all_inp_ids_anchor)[0] # anchor embeddings
                lm_output_pos = model.trans_model(all_inp_ids_pos)[0] # positive embeddings
                lm_output_neg = model.trans_model(all_inp_ids_neg)[0] # neg embeddings

                distance_positive = (lm_output_anch - lm_output_pos).pow(2).sum(1)
                distance_negative = (lm_output_anch - lm_output_neg).pow(2).sum(1)
                losses = F.relu(distance_positive - distance_negative + 10)
                triplet_intent_loss = losses.mean()
                # Triplet Slots
                triplet_loss = triplet_intent_loss #+ triplet_slot_loss

                triplet_optim.zero_grad()
                triplet_loss.backward()
                triplet_optim.step()

        print("After Triplet Loss refinement")
        prototypes = compute_prototypes(dataset, dataset.train_size, dataset.train, model, use_slots, mode)
        eval(dataset, meta_learner, test_langs, mode, prototypes)

    number_steps = 0
    opt = torch.optim.Adam(model.parameters(),  lr=opt_config["alpha_lr"])
    # 4. Meta-training
    print("META_LEARNING ... ")
    for epoch in tqdm(range(opt_config["epoch"])):
        gc.collect()
        num_iterations = data_config["batch_sz"]//opt_config["n_task"]
        for i in range(num_iterations):
            opt.zero_grad()

            # meta_train_batch
            batch = meta_dataset.next_batch(meta_train_dataset, opt_config["n_task"], i)

            batch = tuple(t.cuda() for t in batch)

            inp_ids_spt, tok_typ_ids_spt, att_masks_spt, len_spt, int_l_spt, slot_l_spt, inp_ids_spt_2, \
            tok_typ_ids_spt_2, att_masks_spt_2, len_spt_2, int_l_spt_2, slot_l_spt_2, inp_ids_qry, tok_typ_ids_qry,\
            att_masks_qry, len_qry, int_l_qry, slot_l_qry = batch


            # meta_tune_batch
            batch_tune = meta_dataset.next_batch(meta_tune_dataset, opt_config["n_task"], i)

            batch_tune = tuple(t.cuda() for t in batch_tune)

            inp_ids_spt_tune, tok_typ_ids_spt_tune, att_masks_spt_tune, len_spt_tune, int_l_spt_tune, slot_l_spt_tune, \
            inp_ids_spt_2_tune, tok_typ_ids_spt_2_tune,  att_masks_spt_2_tune, len_spt_2_tune, int_l_spt_2_tune, \
            slot_l_spt_2_tune, inp_ids_qry_tune, tok_typ_ids_qry_tune, att_masks_qry_tune, len_qry_tune, int_l_qry_tune, \
            slot_l_qry_tune = batch_tune

            maml, meta_train_error, meta_train_accuracy, meta_tune_error, meta_tune_accuracy, prototypes \
                = meta_learner(use_adapt, opt, inp_ids_spt, tok_typ_ids_spt, att_masks_spt, len_spt, int_l_spt, slot_l_spt,
                               inp_ids_spt_2, tok_typ_ids_spt_2, att_masks_spt_2, len_spt_2, int_l_spt_2, slot_l_spt_2,
                               inp_ids_qry, tok_typ_ids_qry, att_masks_qry, len_qry, int_l_qry, slot_l_qry,
                               ##
                               inp_ids_spt_tune, tok_typ_ids_spt_tune, att_masks_spt_tune, len_spt_tune, int_l_spt_tune, slot_l_spt_tune,
                               inp_ids_spt_2_tune, tok_typ_ids_spt_2_tune,  att_masks_spt_2_tune, len_spt_2_tune, int_l_spt_2_tune,
                               slot_l_spt_2_tune, inp_ids_qry_tune, tok_typ_ids_qry_tune, att_masks_qry_tune, len_qry_tune, int_l_qry_tune,
                               slot_l_qry_tune, prototypes)

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
                # Testing
                print("Testing at Epoch {}, Step {} |")
                metrics = meta_learner.zero_shot_test(test_langs, dataset, ext="few_shot", proto=prototypes)

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
    metrics = meta_learner.zero_shot_test(test_langs, dataset, ext="few_shot", proto=prototypes)

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
                        action='store_true') # zero-shot by default
    parser.add_argument("--use-triplet-loss", help="If True, refine using triplet loss", action='store_true')

    parser.add_argument('--use-slots', help='If true, optimize for slot filling loss too', action='store_true')

    parser.add_argument("--trans-model", help="name of transformer model", default="BertBaseMultilingualCased")
    parser.add_argument('--data-dir', help='Path of data',  default="")

    parser.add_argument('--out-dir', help='Path of output data', default="")
    parser.add_argument('--data-format', help='Whether it is tsv or json', default="tsv")

    parser.add_argument('--use-pretrained-model', help='If true, use pre-trained NLU model', action='store_true')
    parser.add_argument('--use-aae', help='If true, use AAE alignment', action='store_true')
    parser.add_argument('--pre-trained-model-name', help='Path of output pre-trained model binary', default="")
    parser.add_argument('--seed', help="Random seed for initialization", type=int, default=42)

    ## Meta-learning Dataset Hyperparameters (Have to run using grid search to analyze learning curves)
    parser.add_argument('--n-way', help='Number of classes for each task in the meta-learning', type=int, default=12)
    parser.add_argument('--k-spt', help='Number of support examples per class', type=int, default=12)
    parser.add_argument('--q-qry', help='Number of query examples per class', type=int, default=6)
    parser.add_argument('--batch-sz', help='Number of iterations', type=int, default=10000)

    ## Meta-learning optimization Hyperparameters (tunable => hyperparameter optimization search or some automatic tool)
    parser.add_argument('--epoch', help='Number of epochs', type=int, default=10)  # Early stopping

    parser.add_argument('--n-task', help='Number of tasks', type=int, default=2)

    parser.add_argument('--n-up-train-step', help='Number of update steps in the meta-training stage', type=int,
                        default=3)

    parser.add_argument('--n-up-test-step', help="Number of update steps in the meta-update stage", type=int,
                        default=10)

    parser.add_argument('--alpha-lr', help='Learning rate during the meta-training stage (inner loop)', type=int,
                        default=0.1)

    parser.add_argument('--beta-lr', help='Learning rate during the meta-update stage (outer loop)', type=int,
                        default=0.01)

    parser.add_argument('--gamma-lr', help='Learning rate during the meta-update in the adaptation', type=int,
                        default=1e-3)

    parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--use-adapt', help='Whether to use meta-adaptation', action='store_true')
    parser.add_argument('--use-dpp', help='Whether to use DPP or not', action='store_true')

    args = parser.parse_args()
    set_seed(args)

    data_config = {"n_way": args.n_way, "k_spt": args.k_spt, "q_qry": args.q_qry, "batch_sz": args.batch_sz}

    opt_config = {"epoch": args.epoch, "n_task": args.n_task, "n_up_train_step": args.n_up_train_step,
                  "n_up_test_step": args.n_up_test_step, "alpha_lr": args.alpha_lr, "beta_lr": args.beta_lr,
                  "gamma_lr": args.gamma_lr}

    dataset, model = run(args.use_adapt, args.use_dpp, args.trans_model, args.use_pretrained_model, args.pre_trained_model_name,
                         args.data_dir, args.train_langs, args.dev_langs, args.test_langs, args.use_few_shot,
                         args.data_format, args.use_slots, data_config, opt_config, args.out_dir, args.use_aae,
                         args.use_triplet_loss)
