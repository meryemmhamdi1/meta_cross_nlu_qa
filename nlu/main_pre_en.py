from pytorch_transformers import AdamW, WarmupLinearSchedule
import numpy as np
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
import torch
import os
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse
from transformers_config import *
from base_model import *


def nlu_evaluation(model, dataset, lang, nb_examples, use_slots):
    model.eval()

    intent_corrects = 0
    intents_true = []
    intents_pred = []

    slots_true = []
    slots_pred = []

    for _ in range(nb_examples):
        (input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts), text \
            = dataset.next_batch(1, dataset.test, [lang])

        input_ids = input_ids.cuda()
        lengths = lengths.cuda()
        intent_labels = intent_labels.cuda()
        slot_labels = slot_labels.cuda()

        with torch.no_grad():
            if use_slots:
                intent_logits, slot_logits, intent_loss, slot_loss = model(input_ids=input_ids,
                                                                           intent_labels=intent_labels,
                                                                           slot_labels=slot_labels)

                """ Slot Golden Truth/Predictions """
                true_slot = slot_labels[0]
                pred_slot = list(slot_logits.cpu().squeeze().max(-1)[1].numpy())

                true_slot_l = [dataset.slot_types[s] for s in true_slot]
                pred_slot_l = [dataset.slot_types[s] for s in pred_slot]

                true_slot_no_x = []
                pred_slot_no_x = []

                for i, slot in enumerate(true_slot_l):
                    if slot != "X":
                        true_slot_no_x.append(true_slot_l[i])
                        pred_slot_no_x.append(pred_slot_l[i])

                slots_true.extend(true_slot_no_x)
                slots_pred.extend(pred_slot_no_x)


            else:
                intent_logits, intent_loss = model(input_ids=input_ids,
                                                   intent_labels=intent_labels)

        """ Intent Golden Truth/Predictions """
        true_intent = intent_labels.squeeze().item()
        pred_intent = intent_logits.squeeze().max(0)[1]

        intent_corrects += int(pred_intent == true_intent)

        masked_text = ' '.join(dataset.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist()))
        intents_true.append(true_intent)
        intents_pred.append(pred_intent.item())

    intent_accuracy = float(intent_corrects) / nb_examples
    intent_prec = precision_score(intents_true, intents_pred, average="macro")
    intent_rec = recall_score(intents_true, intents_pred, average="macro")
    intent_f1 = f1_score(intents_true, intents_pred, average="macro")

    if use_slots:
        slot_prec = precision_score(slots_true, slots_pred, average="macro")
        slot_rec = recall_score(slots_true, slots_pred, average="macro")
        slot_f1 = f1_score(slots_true, slots_pred, average="macro")

        return intent_accuracy, intent_prec, intent_rec, intent_f1, slot_prec, slot_rec, slot_f1

    return intent_accuracy, intent_prec, intent_rec, intent_f1


def pre_train_from_scratch(model, dataset, dataset_type, dev_langs, pre_train_config, use_slots, test_langs, writer,
                           out_dir):

    print('Preparing optimizer ...')
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=pre_train_config["adam_lr"],
                      eps=pre_train_config["adam_eps"])

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0,
                                     t_total=2000)

    model.zero_grad()

    # Start Training
    print('Starting pre-training .....')
    best_test_intent_f1 = 0
    no_impr = 0
    for i in range(pre_train_config["pre_train_steps"]):
        optimizer.zero_grad()

        batch, _ = dataset.next_batch(pre_train_config["batch_size"],
                                      dataset_type,
                                      dev_langs)

        input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts = batch

        input_ids = input_ids.cuda()
        lengths = lengths.cuda()
        token_type_ids = token_type_ids.cuda()
        attention_mask = attention_mask.cuda()
        intent_labels = intent_labels.cuda()
        slot_labels = slot_labels.cuda()

        if use_slots:
            logits_intents, logits_slots, intent_loss, slot_loss = model(input_ids,
                                                                         intent_labels=intent_labels,
                                                                         slot_labels=slot_labels)

            loss = intent_loss + slot_loss

            writer.add_scalar('train_slot_loss', slot_loss.mean(), i)
        else:
            logits_intents, intent_loss = model(input_ids,
                                                intent_labels=intent_labels)
            loss = intent_loss

        loss = loss.mean()
        loss.backward()

        writer.add_scalar('train_intent_loss', intent_loss.mean(), i)

        optimizer.step()
        scheduler.step()

        if i > 0 and i % 10 == 0:
            if use_slots:
                print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(i, intent_loss.mean(),
                                                                                   slot_loss.mean()))
            else:
                print('Iter {} | Intent Loss = {:.4f} '.format(i, intent_loss.mean()))

        if i == 0:
            print("Saving the base model")
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(out_dir, WEIGHTS_NAME)
            # output_config_file = os.path.join(output_dir, CONFIG_NAME)
            output_config_file = os.path.join(out_dir, "bert_config.json")
            torch.save(model_to_save.state_dict(), str(output_model_file))
            model_to_save.config.to_json_file(output_config_file)
            dataset.tokenizer.save_vocabulary(out_dir)

        if i > 0 and i % 1000 == 0:
            print("Evaluation on test set ...")
            test_intent_f1_l = []
            for lang in test_langs:
                if use_slots:
                    test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1, test_slot_prec, test_slot_rec,\
                    test_slot_f1 = nlu_evaluation(model,
                                                  dataset,
                                                  lang,
                                                  dataset.test[lang].size,
                                                  use_slots)

                    writer.add_scalar('test_slot_prec_'+lang, test_slot_prec, i)
                    writer.add_scalar('test_slot_rec_'+lang, test_slot_rec, i)
                    writer.add_scalar('test_slot_f1_'+lang, test_slot_f1, i)

                    print('Test on {} | Intent Accuracy = {:.4f} Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f} '
                          '| Slot  Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f}'
                          .format(lang, test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1,
                                  test_slot_prec, test_slot_rec, test_slot_f1))
                else:
                    test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1 \
                        = nlu_evaluation(model,
                                         dataset,
                                         lang,
                                         dataset.test[lang].size,
                                         use_slots)

                    print('Test on {} | Intent Accuracy = {:.4f} Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f} '
                          .format(lang, test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1))

                test_intent_f1_l.append(test_intent_f1)

                writer.add_scalar('test_intent_acc_'+lang, test_intent_acc, i)
                writer.add_scalar('test_intent_prec_'+lang, test_intent_prec, i)
                writer.add_scalar('test_intent_rec_'+lang, test_intent_rec, i)
                writer.add_scalar('test_intent_f1_'+lang, test_intent_f1, i)

            test_intent_f1_avg = np.mean(test_intent_f1_l)
            if test_intent_f1_avg > best_test_intent_f1:
                print("Saving the best model based on cross-lingual validation intent accuracy to ...",
                      output_model_file)

                no_impr = 0
                best_test_intent_f1 = test_intent_f1_avg

                # Save the trained model and configuration
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(out_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(out_dir, CONFIG_NAME)
                torch.save(model_to_save.state_dict(), str(output_model_file))
                model_to_save.config.to_json_file(output_config_file)
                dataset.tokenizer.save_vocabulary(out_dir)

            else:
                no_impr += 1

    return model


def fine_tune_from_scratch(model, dataset, meta_dataset, pre_train_config, use_slots, test_langs, dev_langs, writer,
                           out_dir):

    print('Preparing optimizer ...')
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=pre_train_config["adam_lr"],
                      eps=pre_train_config["adam_eps"])

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0,
                                     t_total=2000)

    model.zero_grad()

    meta_train_dataset = meta_dataset.auxi_train_batches
    # Start Training
    print('Starting pre-training .....')
    best_dev = 0
    no_impr = 0
    for k in range(10):
        for i in range(1000//4):
            optimizer.zero_grad()
            all_batch = meta_dataset.next_batch(meta_train_dataset, 4, i)

            all_batch = tuple(t.cuda() for t in all_batch)

            inp_ids_spt_all, tok_typ_ids_spt_all, att_masks_spt_all, len_spt_all, int_l_spt_all, slot_l_spt_all, \
                inp_ids_qry_all, tok_typ_ids_qry_all, att_masks_qry_all, len_qry_all, int_l_qry_all, slot_l_qry_all = \
                all_batch

            for j in range(4):
                l = k*(10**2)+i*(10**1)+j
                input_ids = inp_ids_qry_all[j]
                lengths = len_qry_all[j]
                token_type_ids = tok_typ_ids_qry_all[j]
                attention_mask = att_masks_qry_all[j]
                intent_labels = int_l_qry_all[j]
                slot_labels = slot_l_qry_all[j]

                if use_slots:
                    logits_intents, logits_slots, intent_loss, slot_loss = model(input_ids,
                                                                                 intent_labels=intent_labels,
                                                                                 slot_labels=slot_labels)
                    loss = intent_loss + slot_loss

                    writer.add_scalar('train_slot_loss', slot_loss.mean(), l)
                else:
                    logits_intents, intent_loss = model(input_ids,
                                                        intent_labels=intent_labels)
                    loss = intent_loss

                loss = loss.mean()
                loss.backward()

                writer.add_scalar('train_intent_loss', intent_loss.mean(), l)

                optimizer.step()
                scheduler.step()

                if i*j > 0 and i*j % 10 == 0:
                    if use_slots:
                        print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(i, intent_loss.mean(),
                                                                                           slot_loss.mean()))
                    else:
                        print('Iter {} | Intent Loss = {:.4f} '.format(i, intent_loss.mean()))

                if i*j == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(out_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(out_dir, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), str(output_model_file))
                    model_to_save.config.to_json_file(output_config_file)
                    dataset.tokenizer.save_vocabulary(out_dir)

                if i*j > 0 and i*j % 100 == 0:
                    print("Evaluation on DEV set ...")
                    dev_avg = 0
                    for lang in dev_langs:
                        if use_slots:
                            dev_intent_acc, dev_intent_prec, dev_intent_rec, dev_intent_f1, dev_slot_prec, \
                            dev_slot_rec, dev_slot_f1 = nlu_evaluation(model,
                                                                       dataset,
                                                                       lang,
                                                                       dataset.val[lang].size,
                                                                       use_slots)

                            writer.add_scalar('dev_slot_prec_'+lang, dev_slot_prec, l)
                            writer.add_scalar('dev_slot_rec_'+lang, dev_slot_rec, l)
                            writer.add_scalar('dev_slot_f1_'+lang, dev_slot_f1, l)

                            print('Val on {} | Intent Acc = {:.4f} Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f} '
                                  '| Slot  Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f}'
                                  .format(lang, dev_intent_acc, dev_intent_prec, dev_intent_rec, dev_intent_f1,
                                          dev_slot_prec, dev_slot_rec, dev_slot_f1))

                            dev_avg += dev_slot_prec + dev_slot_rec + dev_slot_f1
                        else:
                            dev_intent_acc, dev_intent_prec, dev_intent_rec, dev_intent_f1 \
                                = nlu_evaluation(model, dataset, lang, dataset.val[lang].size, use_slots)

                            print('Val on {} | Intent Acc = {:.4f} Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f} '
                                  .format(lang, dev_intent_acc, dev_intent_prec, dev_intent_rec, dev_intent_f1))

                        dev_avg += dev_intent_acc + dev_intent_prec + dev_intent_rec + dev_intent_f1

                        writer.add_scalar('dev_intent_acc_'+lang, dev_intent_acc, l)
                        writer.add_scalar('dev_intent_prec_'+lang, dev_intent_prec, l)
                        writer.add_scalar('dev_intent_rec_'+lang, dev_intent_rec, l)
                        writer.add_scalar('dev_intent_f1_'+lang, dev_intent_f1, l)

                    print("Evaluation on TEST set ...")
                    for lang in test_langs:
                        if use_slots:
                            test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1, test_slot_prec, \
                                test_slot_rec, test_slot_f1 = nlu_evaluation(model,
                                                                             dataset,
                                                                             lang,
                                                                             dataset.test[lang].size,
                                                                             use_slots)

                            writer.add_scalar('test_slot_prec_'+lang, test_slot_prec, l)
                            writer.add_scalar('test_slot_rec_'+lang, test_slot_rec, l)
                            writer.add_scalar('test_slot_f1_'+lang, test_slot_f1, l)

                            print('Test on {} | Intent Acc = {:.4f} Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f} '
                                  '| Slot  Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f}'
                                  .format(lang, test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1,
                                          test_slot_prec, test_slot_rec, test_slot_f1))
                        else:
                            test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1 \
                                = nlu_evaluation(model, dataset, lang, dataset.test[lang].size, use_slots)

                            print('Test on {} | Intent Acc = {:.4f} Precision = {:.4f} Recall = {:.4f} and F1 = {:.4f} '
                                  .format(lang, test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1))

                        writer.add_scalar('test_intent_acc_'+lang, test_intent_acc, l)
                        writer.add_scalar('test_intent_prec_'+lang, test_intent_prec, l)
                        writer.add_scalar('test_intent_rec_'+lang, test_intent_rec, l)
                        writer.add_scalar('test_intent_f1_'+lang, test_intent_f1, l)

                    if dev_avg > best_dev:
                        print("Saving the best model based on cross-lingual validation intent accuracy to ...",
                              output_model_file)
                        no_impr = 0
                        best_dev = dev_avg

                        # Save the trained model and configuration
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(out_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(out_dir, CONFIG_NAME)
                        torch.save(model_to_save.state_dict(), str(output_model_file))
                        model_to_save.config.to_json_file(output_config_file)
                        dataset.tokenizer.save_vocabulary(out_dir)

                    else:
                        no_impr += 1

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-langs", help="train languages list", nargs="+", default=[])
    parser.add_argument("--dev-langs", help="dev languages list", nargs="+", default=[])
    parser.add_argument("--test-langs", help="test languages list", nargs="+", default=[])
    parser.add_argument('--use-few-shot', help='If true, use test languages in the meta-adaptation stage',
                        action='store_true')# zero-shot by default

    parser.add_argument('--use-slots', help='If true, optimize for slot filling loss too', action='store_true')

    parser.add_argument("--trans-model", help="name of transformer model", default="BertBaseMultilingualCased")
    parser.add_argument('--data-dir', help='Path of input data',  default="")

    parser.add_argument('--out-dir', help='Path of output data', default="")
    parser.add_argument('--pre-trained-model-name', help='Path of output pre-trained model binary', default="")
    parser.add_argument('--data-format', help='Whether it is tsv or json', default="tsv")

    ## Pre-training hyperparameters
    parser.add_argument('--pre-train-steps', help='the number of iterations', type=int, default=2000)
    parser.add_argument('--batch-size', help="batch size in the pre-training process", type=int, default=32)
    parser.add_argument('--adam-lr', help="learning rate of adam optimizer when training base model from scratch",
                        type=float, default=4e-5)
    parser.add_argument('--adam-eps', help="epsilon of adam optimizer when training base model from scratch",
                        type=float, default= 1e-08)

    args = parser.parse_args()
    print("Training base model from scratch ...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = args.out_dir
    train_langs = args.train_langs
    dev_langs = args.dev_langs
    test_langs = args.test_langs
    trans_model = args.trans_model
    data_dir = args.data_dir
    data_format = args.data_format
    use_few_shot = args.use_few_shot
    use_slots = args.use_slots

    out_dir = os.path.join(out_dir, "train_"+",".join(train_langs)+"-test_"+",".join(test_langs))

    writer = SummaryWriter(os.path.join(out_dir, 'runs'))
    # 1. Loading dataset and choice of tokenizer
    """ 1.1. Prepare tokenizer """
    tokenizer = MODELS_dict[trans_model][1].from_pretrained(MODELS_dict[trans_model][0],
                                                            do_lower_case=True,
                                                            do_basic_tokenize=False)

    print("Preparing base dataset")
    """ 1.2. Load train/dev/test splits from train/dev/test languages => train_set """
    train_paths = {}
    spt_paths = {}
    for lang in train_langs:
        """ 1.2.1. Train split of train lang used in the base model """
        train_paths.update({lang: os.path.join(os.path.join(data_dir, lang), "train-"+lang+"."+data_format)})
        """ 1.2.2. Dev split of train/dev/test languages used in meta-learning """
        """ --> a. Dev split of train lang used in support set of meta-training => spt_set """
        spt_paths.update({lang: os.path.join(os.path.join(data_dir, lang), "eval-"+lang+"."+data_format)})

    qry_paths = {}
    for lang in dev_langs:
        """ --> b. Dev split of dev lang used in query set of meta-training => qry_set """
        qry_paths.update({lang: os.path.join(os.path.join(data_dir, lang), "eval-"+lang+"."+data_format)})

    tune_paths = {}
    """ --> c. Dev split of test lang used in meta-adaptation in few-shot case => tune_set """
    if use_few_shot:
        for lang in test_langs:
            tune_paths.update({lang: os.path.join(os.path.join(data_dir, lang), "test-"+lang+"."+data_format)})
    else:
        for lang in dev_langs:
            tune_paths.update({lang: os.path.join(os.path.join(data_dir, lang), "eval-"+lang+"."+data_format)})

    """1.2.3. Test split of all train/dev/test languages used in testing the adaptation of the new model => test_set"""
    test_paths = {}
    for lang in test_langs:
        test_paths.update({lang: os.path.join(os.path.join(data_dir, lang), "test-"+lang+"."+data_format)})

    """ 1.3. Loading Word Piece Processed Dataset where batches are not padded or converted to tensors yet """
    dataset = Dataset(tokenizer, data_format, use_slots, train_paths, spt_paths, qry_paths, tune_paths, test_paths)

    # 2. Loading a pre-trained base-model if it exists
    print("Initializing Base Transformer NLU model")
    model_trans = MODELS_dict[trans_model][2].from_pretrained(MODELS_dict[trans_model][0],
                                                              num_labels=len(dataset.intent_types))

    model = TransformerNLU(model_trans,
                           len(dataset.intent_types),
                           use_slots=use_slots,
                           num_slots=len(dataset.slot_types))

    print("Pre-training from scratch")
    pre_train_config = {"pre_train_steps": args.pre_train_steps, "batch_size": args.batch_size,
                        "adam_lr": args.adam_lr, "adam_eps": args.adam_eps}

    pre_train_from_scratch(model, dataset, pre_train_config, use_slots, test_langs, writer, out_dir)

