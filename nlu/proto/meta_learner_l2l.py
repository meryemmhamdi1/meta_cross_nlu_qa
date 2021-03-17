from torch import nn
import torch
from torch import optim
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torchviz import make_dot
from copy import deepcopy
from tqdm import tqdm
from tsne_plot import *
import learn2learn as l2l


def accuracy(logits, targets):
    intent_corrects = 0
    for j in range(len(logits)):
        true_intent = targets[j].squeeze().item()
        pred_intent = logits[j].squeeze().max(0)[1]
        intent_corrects += int(pred_intent == true_intent)
    return intent_corrects / targets.size(0)


class MetaLearner(nn.Module):
    def __init__(self, opti_config, base_model, use_slots, intent_types, slot_types, device, dataset):
        super(MetaLearner, self).__init__()
        self.n_up_train_step = opti_config["n_up_train_step"]
        self.n_up_test_step = opti_config["n_up_test_step"]
        self.alpha_lr = opti_config["alpha_lr"]
        self.beta_lr = opti_config["beta_lr"]
        self.gamma_lr = opti_config["gamma_lr"]
        self.use_slots = use_slots
        self.slot_types = slot_types
        self.intent_types = intent_types
        self.dataset = dataset

        self.base_model = base_model
        self.initial_parameters = base_model.parameters()
        self.inner_optim = optim.SGD(self.base_model.parameters(), lr=self.alpha_lr, momentum=0.9)
        self.outer_optim = optim.SGD(self.base_model.parameters(), lr=self.beta_lr, momentum=0.9)
        self.device = device

    def forward(self, use_adapt, opt, inp_ids_spt, tok_typ_ids_spt, att_masks_spt, len_spt, int_l_spt, slot_l_spt,
                inp_ids_spt_2, tok_typ_ids_spt_2, att_masks_spt_2, len_spt_2, int_l_spt_2, slot_l_spt_2,
                inp_ids_qry, tok_typ_ids_qry, att_masks_qry, len_qry, int_l_qry, slot_l_qry,

                inp_ids_spt_tune, tok_typ_ids_spt_tune, att_masks_spt_tune, len_spt_tune, int_l_spt_tune, slot_l_spt_tune,
                inp_ids_spt_2_tune, tok_typ_ids_spt_2_tune,  att_masks_spt_2_tune, len_spt_2_tune, int_l_spt_2_tune,
                slot_l_spt_2_tune, inp_ids_qry_tune, tok_typ_ids_qry_tune, att_masks_qry_tune, len_qry_tune,
                int_l_qry_tune, slot_l_qry_tune, proto):

        n_tasks = inp_ids_qry.size(0)

        maml = l2l.algorithms.MAML(self.base_model, lr=0.001, first_order=True)

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_tune_error = 0.0
        meta_tune_accuracy = 0.0

        for i in tqdm(range(n_tasks)):
            learner = maml.clone()
            shape_qry = inp_ids_qry[i].shape[0]
            shape_qry_tune = inp_ids_qry_tune[i].shape[0]

            ### Update steps
            for _ in tqdm(range(0, self.n_up_train_step)):
                logits, losses, proto = learner(input_ids_1=inp_ids_spt[i],
                                                input_ids_2=inp_ids_spt_2[i],
                                                intent_labels_1=int_l_spt[i],
                                                intent_labels_2=int_l_spt_2[i],
                                                slot_labels_1=slot_l_spt[i],
                                                slot_labels_2=slot_l_spt_2[i],
                                                proto=proto)

                loss = losses["total_loss"]

                # Taking backward pass over total_loss of support 2 part
                learner.adapt(loss, allow_nograd=True, allow_unused=True)

            ## Applying the model to the query set after the update
            logits, losses, proto = learner(input_ids_1=inp_ids_spt[i][:shape_qry],
                                            input_ids_2=inp_ids_qry[i],
                                            intent_labels_1=int_l_spt[i][:shape_qry],
                                            intent_labels_2=int_l_qry[i],
                                            slot_labels_1=slot_l_spt[i][:shape_qry],
                                            slot_labels_2=slot_l_qry[i],
                                            proto=proto)

            loss = losses["total_loss"]

            if i == n_tasks - 1:
                loss.backward()
            else:
                loss.backward(retain_graph=True)

            meta_train_error += loss.item()
            meta_train_accuracy += accuracy(logits["intent"], int_l_qry[i])

            if use_adapt:
                learner = maml.clone()
                for _ in range(0, self.n_up_test_step):
                    logits, losses, proto = learner(input_ids_1=inp_ids_spt_tune[i][:shape_qry_tune],
                                                    input_ids_2=inp_ids_spt_2_tune[i],
                                                    intent_labels_1=int_l_spt_tune[i][:shape_qry_tune],
                                                    intent_labels_2=int_l_spt_2_tune[i],
                                                    slot_labels_1=slot_l_spt_tune[i][:shape_qry_tune],
                                                    slot_labels_2=slot_l_spt_2_tune[i],
                                                    proto=proto)

                    learner.adapt(loss, allow_nograd=True, allow_unused=True, retain_graph=True)

                logits, losses, proto = learner(input_ids_1=inp_ids_spt_tune[i][:shape_qry_tune],
                                                input_ids_2=inp_ids_qry_tune[i],
                                                intent_labels_1=int_l_spt_tune[i][:shape_qry_tune],
                                                intent_labels_2=int_l_qry_tune[i],
                                                slot_labels_1=slot_l_spt_tune[i][:shape_qry_tune],
                                                slot_labels_2=slot_l_qry_tune[i],
                                                proto=proto)

                loss = losses["total_loss"]

                meta_tune_error += loss.item()
                meta_tune_accuracy += accuracy(logits["intent"], int_l_qry_tune[i])


        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            if p.grad is not None:
                p.grad.mul_(1.0 / n_tasks)
        opt.step()

        return maml, meta_train_error, meta_train_accuracy, meta_tune_error, meta_tune_accuracy, proto

    def zero_shot_test(self, test_langs, dataset, ext, proto):
        """
        Testing the method on test split in the non-meta-learning setup and comparing before and after the meta-training
        :return:
        """

        X_i_all, Y_i_all, lang_i = [], [], []
        X_s_all, Y_s_all, lang_s = [], [], []

        metrics = {}

        for lang in test_langs:
            print("Evaluating for Lang:", lang)
            metrics_sub = {}
            self.base_model.eval()

            intent_corrects = 0

            intents_true, intents_pred = [], []
            slots_true, slots_pred = [], []

            X_i, Y_i, T_i = [], [], []
            X_s, Y_s, T_s = [], [], []

            for _ in tqdm(range(dataset.test_size[lang])):
                (input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts), text \
                    = dataset.next_batch(1, dataset.test[lang])
                batch = input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels = batch

                with torch.no_grad():
                    logits, embed = self.base_model.test(input_ids=input_ids, proto=proto)
                    if self.use_slots:
                        # Slot Golden Truth/Predictions
                        true_slot = slot_labels[0][1:]
                        pred_slot = list(logits["slot"].cpu().squeeze().max(-1)[1].numpy())

                        true_slot_l = [dataset.slot_types[s] for s in true_slot]
                        pred_slot_l = [dataset.slot_types[s] for s in pred_slot]

                        true_slot_no_x, pred_slot_no_x = [], []

                        for i, slot in enumerate(true_slot_l):
                            if slot not in [0, dataset.slot_types.index('X')]:
                                true_slot_no_x.append(true_slot_l[i])
                                pred_slot_no_x.append(pred_slot_l[i])
                                X_s.append(embed["slot"][i])
                                slot_type = pred_slot_l[i]
                                if "-" in slot_type:
                                    slot_type = slot_type.split("-")[1]
                                Y_s.append(slot_type)
                                T_s.append("Point")

                        slots_true.extend(true_slot_no_x)
                        slots_pred.extend(pred_slot_no_x)


                    # Intent Golden Truth/Predictions
                    true_intent = intent_labels.squeeze().item()
                    pred_intent = logits["intent"].squeeze().max(0)[1]
                    X_i.append(embed["intent"])
                    Y_i.append(self.intent_types[pred_intent])
                    T_i.append("Point")

                    intent_corrects += int(pred_intent == true_intent)

                    masked_text = ' '.join(dataset.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist()))
                    intents_true.append(true_intent)
                    intents_pred.append(pred_intent.item())

            metrics_sub.update({"intent_accuracy":float(intent_corrects) / dataset.test_size[lang],
                               "intent_prec": precision_score(intents_true, intents_pred, average="macro"),
                               "intent_rec": recall_score(intents_true, intents_pred, average="macro"),
                               "intent_f1": f1_score(intents_true, intents_pred, average="macro")})

            #plot_tsne(X_i, Y_i, T_i, "intents_"+lang+"_"+ext+".png")

            X_i_all.extend(X_i)
            Y_i_all.extend(Y_i)
            lang_i.extend([lang]*len(X_i))

            if self.use_slots:
                metrics_sub.update({"slot_prec": precision_score(slots_true, slots_pred, average="macro"),
                                    "slot_rec": recall_score(slots_true, slots_pred, average="macro"),
                                    "slot_f1": f1_score(slots_true, slots_pred, average="macro")})

                #plot_tsne(X_s, Y_s, T_s ,"slots_"+lang+"_"+ext+".png")
                X_s_all.extend(X_s)
                Y_s_all.extend(Y_s)
                lang_s.extend([lang]*len(X_s))

            metrics.update({lang:metrics_sub})

        # Plot for cross-lingual aggregation

        #plot_tsne(X_i_all, Y_i_all, lang_i ,"intents_all_langs_"+ext+".png")

        #if self.use_slots:
            #plot_tsne(X_s_all, Y_s_all, lang_s ,"slots_all_langs_"+ext+".png")

        return metrics
