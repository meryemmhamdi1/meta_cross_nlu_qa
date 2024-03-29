from torch import optim
from torch import nn
import torch

from sklearn.metrics import f1_score, precision_score, recall_score
from copy import deepcopy
from tqdm import tqdm
import learn2learn as l2l


def accuracy(logits, targets):
    intent_corrects = 0
    for j in range(len(logits)):
        true_intent = targets[j].squeeze().item()
        pred_intent = logits[j].squeeze().max(0)[1]
        intent_corrects += int(pred_intent == true_intent)
    return intent_corrects / targets.size(0)


class MetaLearner(nn.Module):
    """
    Accumulates gradients in the outer loop over tasks
    """
    def __init__(self, opti_config, base_model, use_slots, intent_types, slot_types, device, dataset, freeze_bert, use_freeze_bert, use_freeze_linear):
        super(MetaLearner, self).__init__()
        self.n_up_train_step = opti_config["n_up_train_step"]
        self.n_up_test_step = opti_config["n_up_test_step"]
        self.alpha_lr = opti_config["alpha_lr"]
        self.beta_lr = opti_config["beta_lr"]
        self.gamma_lr = opti_config["gamma_lr"]
        self.use_slots = use_slots
        self.slot_types = slot_types

        self.base_model = base_model
        self.initial_parameters = base_model.parameters()
        self.inner_optim = optim.Adam(self.base_model.parameters(), lr=self.alpha_lr)
        self.tune_optim = optim.Adam(self.base_model.parameters(), lr=self.gamma_lr)
        self.device = device
        self.use_freeze_bert = use_freeze_bert
        self.use_freeze_linear = use_freeze_linear
        self.freeze_bert_params = []
        if 13 in freeze_bert:
            self.freeze_bert_params += ["trans_model.embeddings.word_embeddings.weight",
                                        "trans_model.embeddings.position_embeddings.weight",
                                        "trans_model.embeddings.token_type_embeddings.weight",
                                        "trans_model.embeddings.LayerNorm.weight",
                                        "trans_model.embeddings.LayerNorm.bias"]
        if 12 in freeze_bert:
            self.freeze_bert_params += ["trans_model.pooler.dense.weight",
                                        "trans_model.pooler.dense.bias"]
        freeze_bert_layers = []
        for i in freeze_bert:
            if i not in [12, 13]:
                freeze_bert_layers.append(i)
        self.freeze_bert_params += ["trans_model.encoder.layer."+str(i)+".attention.self."+j+"."+k
                                    for i in freeze_bert_layers for j in ["query", "key"] for k in ["weight", "bias"]]+ \
                                   [["trans_model.encoder.layer."+str(i)+".attention.output."+j+"."+k for i in freeze_bert_layers
                                     for j in ["dense", "LayerNorm"] for k in ["weight", "bias"]]] + \
                                   [["trans_model.encoder.layer."+str(i)+"."+j+".dense."+k for i in freeze_bert_layers
                                     for j in ["intermediate", "output"] for k in ["weight", "bias"] ]] + \
                                   [["trans_model.encoder.layer."+str(i)+".output.LayerNorm."+j for i in freeze_bert_layers
                                     for j in ["weight", "bias"] ]] + ['pooler.dense.weight', 'pooler.dense.bias']

    def forward(self, use_adapt, use_back, use_spt_back, use_ada_independent, opt, inp_ids_spt, tok_typ_ids_spt, att_masks_spt, len_spt, int_l_spt, slot_l_spt,
                inp_ids_qry, tok_typ_ids_qry, att_masks_qry, len_qry, int_l_qry, slot_l_qry,
                ##
                inp_ids_spt_tune, tok_typ_ids_spt_tune, att_masks_spt_tune, len_spt_tune,
                int_l_spt_tune, slot_l_spt_tune, inp_ids_qry_tune, tok_typ_ids_qry_tune,
                att_masks_qry_tune, len_qry_tune, int_l_qry_tune, slot_l_qry_tune):

        n_tasks = inp_ids_qry.size(0)

        maml = l2l.algorithms.MAML(self.base_model, lr=self.alpha_lr, first_order=True)

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_tune_error = 0.0
        meta_tune_accuracy = 0.0

        for name, param in self.base_model.named_parameters():
            if self.use_freeze_bert:
                if "trans_model" in name:
                    param.requires_grad = False

            if self.use_freeze_linear:
                if "trans_model" not in name:
                    param.requires_grad = False

        loss_qry_all = 0.0
        for i in range(n_tasks):
            learner = maml.clone()

            for _ in range(0, self.n_up_train_step):
                if self.use_slots:
                    logits_intents, logits_slots, intent_loss, slot_loss = learner(input_ids=inp_ids_spt[i],
                                                                                   intent_labels=int_l_spt[i],
                                                                                   slot_labels=slot_l_spt[i])

                    loss = intent_loss + slot_loss

                else:
                    logits_intents, intent_loss = learner(input_ids=inp_ids_spt[i],
                                                          intent_labels=int_l_spt[i])
                    loss = intent_loss

                learner.adapt(loss, allow_nograd=True, allow_unused=True)

            # On the query data
            if self.use_slots:
                logits_intents, logits_slots, intent_loss, slot_loss = learner(input_ids=inp_ids_qry[i],
                                                                               intent_labels=int_l_qry[i],
                                                                               slot_labels=slot_l_qry[i])

                loss_qry = intent_loss + slot_loss

            else:
                logits_intents, intent_loss = learner(input_ids=inp_ids_qry[i],
                                                      intent_labels=int_l_qry[i])
                loss_qry = intent_loss

            loss_qry_all += loss_qry
            meta_train_error += loss_qry.item()
            meta_train_accuracy += accuracy(logits_intents, int_l_qry[i])

            """ Uses the adaptation within the meta-train stage one task at a time"""
            if use_adapt and not use_ada_independent:

                for _ in range(0, self.n_up_test_step):
                    if self.use_slots:
                        logits_intents, logits_slots, intent_loss, slot_loss = learner(input_ids=inp_ids_spt_tune[i],
                                                                                       intent_labels=int_l_spt_tune[i],
                                                                                       slot_labels=slot_l_spt_tune[i])

                        loss = intent_loss + slot_loss

                    else:
                        logits_intents, intent_loss = learner(input_ids=inp_ids_spt_tune[i],
                                                              intent_labels=int_l_spt_tune[i])
                        loss = intent_loss

                    learner.adapt(loss, allow_nograd=True, allow_unused=True)

                    if use_spt_back:
                        loss.backward(retain_graph=True)

                if self.use_slots:
                    logits_intents, logits_slots, intent_loss, slot_loss = learner(input_ids=inp_ids_qry_tune[i],
                                                                                   intent_labels=int_l_qry_tune[i],
                                                                                   slot_labels=slot_l_qry_tune[i])

                    loss_qry = intent_loss + slot_loss

                else:
                    logits_intents, intent_loss = learner(input_ids=inp_ids_qry_tune[i],
                                                          intent_labels=int_l_qry_tune[i])
                    loss_qry = intent_loss

                meta_tune_error += loss_qry.item()
                meta_tune_accuracy += accuracy(logits_intents, int_l_qry_tune[i])

                loss_qry.backward()

        # Average the accumulated gradients and optimize
        loss_qry_all = loss_qry_all / n_tasks
        for p in maml.parameters():
            if p.grad is not None:
                p.grad.mul_(1.0 / n_tasks)
        loss_qry_all.backward()
        opt.step()

        """ Uses the adaptation at the end of the meta-train stage"""
        loss_qry_tune_all = 0.0
        if use_adapt and use_ada_independent:
            for i in range(n_tasks):
                learner = maml.clone()

                for _ in range(0, self.n_up_test_step):
                    if self.use_slots:
                        logits_intents, logits_slots, intent_loss, slot_loss = learner(input_ids=inp_ids_spt_tune[i],
                                                                                       intent_labels=int_l_spt_tune[i],
                                                                                       slot_labels=slot_l_spt_tune[i])

                        loss = intent_loss + slot_loss

                    else:
                        logits_intents, intent_loss = learner(input_ids=inp_ids_spt_tune[i],
                                                              intent_labels=int_l_spt_tune[i])
                        loss = intent_loss

                    learner.adapt(loss, allow_nograd=True, allow_unused=True)
                    if use_spt_back:
                        loss.backward(retain_graph=True)

                # On the query data
                if self.use_slots:
                    logits_intents, logits_slots, intent_loss, slot_loss = learner(input_ids=inp_ids_qry_tune[i],
                                                                                   intent_labels=int_l_qry_tune[i],
                                                                                   slot_labels=slot_l_qry_tune[i])

                    loss_qry = intent_loss + slot_loss

                else:
                    logits_intents, intent_loss = learner(input_ids=inp_ids_qry_tune[i],
                                                          intent_labels=int_l_qry_tune[i])
                    loss_qry = intent_loss

                loss_qry_tune_all += loss_qry

                meta_tune_error += loss_qry.item()
                meta_tune_accuracy += accuracy(logits_intents, int_l_qry[i])

        # Average the accumulated gradients and optimize
        loss_qry_tune_all = loss_qry_tune_all / n_tasks
        for p in maml.parameters():
            if p.grad is not None:
                p.grad.mul_(1.0 / n_tasks)
        loss_qry_tune_all.backward()
        opt.step()

        return maml, meta_train_error, meta_train_accuracy, meta_tune_error, meta_tune_accuracy

    def zero_shot_test(self, test_langs, dataset):
        """
        Testing the method on test split in the non-meta-learning setup and comparing before and after the meta-training
        :return:
        """

        metrics = {}

        for lang in test_langs:
            metrics_sub = {}
            self.base_model.eval()

            intent_corrects = 0
            intents_true, intents_pred, slots_true, slots_pred = [], [], [], []

            for _ in range(dataset.test_size[lang]):
                (input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts), text \
                    = dataset.next_batch(1, dataset.test[lang], dev_langs=[])
                batch = input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels = batch

                with torch.no_grad():
                    if self.use_slots:
                        intent_logits, slot_logits, intent_loss, slot_loss \
                            = self.base_model(input_ids=input_ids,
                                              intent_labels=intent_labels,
                                              slot_labels=slot_labels)

                        # Slot Golden Truth/Predictions
                        true_slot = slot_labels[0]
                        pred_slot = list(slot_logits.cpu().squeeze().max(-1)[1].numpy())

                        true_slot_l = [dataset.slot_types[s] for s in true_slot]
                        pred_slot_l = [dataset.slot_types[s] for s in pred_slot]

                        true_slot_no_x, pred_slot_no_x = [], []

                        for i, slot in enumerate(true_slot_l):
                            if slot != "X":
                                true_slot_no_x.append(true_slot_l[i])
                                pred_slot_no_x.append(pred_slot_l[i])

                        slots_true.extend(true_slot_no_x)
                        slots_pred.extend(pred_slot_no_x)

                    else:
                        intent_logits, intent_loss = self.base_model(input_ids=input_ids,
                                                                     intent_labels=intent_labels)

                    # Intent Golden Truth/Predictions
                    true_intent = intent_labels.squeeze().item()
                    pred_intent = intent_logits.squeeze().max(0)[1]

                    intent_corrects += int(pred_intent == true_intent)

                    masked_text = ' '.join(dataset.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist()))
                    intents_true.append(true_intent)
                    intents_pred.append(pred_intent.item())

            metrics_sub.update({"intent_accuracy": float(intent_corrects) / dataset.test_size[lang],
                                "intent_prec": precision_score(intents_true, intents_pred, average="macro"),
                                "intent_rec": recall_score(intents_true, intents_pred, average="macro"),
                                "intent_f1": f1_score(intents_true, intents_pred, average="macro")})

            if self.use_slots:
                metrics_sub.update({"slot_prec": precision_score(slots_true, slots_pred, average="macro"),
                                    "slot_rec": recall_score(slots_true, slots_pred, average="macro"),
                                    "slot_f1": f1_score(slots_true, slots_pred, average="macro")})

            metrics.update({lang: metrics_sub})

        return metrics

    def val_eval(self, dev_langs, dataset):
        """
        Validating the method on dev split
        :return:
        """

        metrics = {}
        for lang in dev_langs:
            metrics_sub = {}
            self.base_model.eval()

            intent_corrects = 0
            intents_true, intents_pred, slots_true, slots_pred = [], [], [], []

            for _ in range(dataset.val_size[lang]):
                (input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts), text \
                    = dataset.next_batch(1, dataset.val[lang], dev_langs=[])
                batch = input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels = batch

                with torch.no_grad():
                    if self.use_slots:
                        intent_logits, slot_logits, intent_loss, slot_loss \
                            = self.base_model(input_ids=input_ids,
                                              intent_labels=intent_labels,
                                              slot_labels=slot_labels)

                        # Slot Golden Truth/Predictions
                        true_slot = slot_labels[0]
                        pred_slot = list(slot_logits.cpu().squeeze().max(-1)[1].numpy())

                        true_slot_l = [dataset.slot_types[s] for s in true_slot]
                        pred_slot_l = [dataset.slot_types[s] for s in pred_slot]

                        true_slot_no_x, pred_slot_no_x = [], []

                        for i, slot in enumerate(true_slot_l):
                            if slot != "X":
                                true_slot_no_x.append(true_slot_l[i])
                                pred_slot_no_x.append(pred_slot_l[i])

                        slots_true.extend(true_slot_no_x)
                        slots_pred.extend(pred_slot_no_x)

                    else:
                        intent_logits, intent_loss = self.base_model(input_ids=input_ids,
                                                                     intent_labels=intent_labels)

                    # Intent Golden Truth/Predictions
                    true_intent = intent_labels.squeeze().item()
                    pred_intent = intent_logits.squeeze().max(0)[1]

                    intent_corrects += int(pred_intent == true_intent)

                    masked_text = ' '.join(dataset.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist()))
                    intents_true.append(true_intent)
                    intents_pred.append(pred_intent.item())

            metrics_sub.update({"intent_accuracy": float(intent_corrects) / dataset.test_size[lang],
                                "intent_prec": precision_score(intents_true, intents_pred, average="macro"),
                                "intent_rec": recall_score(intents_true, intents_pred, average="macro"),
                                "intent_f1": f1_score(intents_true, intents_pred, average="macro")})

            if self.use_slots:
                metrics_sub.update({"slot_prec": precision_score(slots_true, slots_pred, average="macro"),
                                    "slot_rec": recall_score(slots_true, slots_pred, average="macro"),
                                    "slot_f1": f1_score(slots_true, slots_pred, average="macro")})

            metrics.update({lang: metrics_sub})

        return metrics
