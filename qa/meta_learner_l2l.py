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
    def __init__(self, opti_config, base_model, device, freeze_bert, use_adapt):
        super(MetaLearner, self).__init__()
        self.n_up_train_step = opti_config["n_up_train_step"]
        self.n_up_test_step = opti_config["n_up_test_step"]
        self.alpha_lr = opti_config["alpha_lr"]
        self.beta_lr = opti_config["beta_lr"]
        self.gamma_lr = opti_config["gamma_lr"]
        self.use_adapt = use_adapt

        self.base_model = base_model
        self.initial_parameters = base_model.parameters()
        self.inner_optim = optim.Adam(self.base_model.parameters(), lr=self.alpha_lr)
        self.tune_optim = optim.Adam(self.base_model.parameters(), lr=self.gamma_lr)
        self.device = device
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
        self.freeze_bert_params += ["trans_model.encoder.layer."+str(i)+".attention.self."+j+"."+k for i in freeze_bert_layers
                                   for j in ["query", "key"] for k in ["weight", "bias"]] + \
                                  [["trans_model.encoder.layer."+str(i)+".attention.output."+j+"."+k for i in freeze_bert_layers
                                    for j in ["dense", "LayerNorm"] for k in ["weight", "bias"] ]] + \
                                  [["trans_model.encoder.layer."+str(i)+"."+j+".dense."+k for i in freeze_bert_layers
                                    for j in ["intermediate", "output"] for k in ["weight", "bias"] ]] + \
                                  [["trans_model.encoder.layer."+str(i)+".output.LayerNorm."+j for i in freeze_bert_layers
                                    for j in ["weight", "bias"] ]] + ['pooler.dense.weight', 'pooler.dense.bias']

    def forward(self, opt, spt_scheduler, qry_scheduler, spt_batch, qry_batch, tune_batch):

        maml = l2l.algorithms.MAML(self.base_model, lr=0.001, first_order=True)

        meta_train_error = 0.0
        #meta_train_accuracy = 0.0
        meta_tune_error = 0.0
        #meta_tune_accuracy = 0.0
        n_tasks = 1

        #for name, param in self.base_model.named_parameters():
        #    if name in self.freeze_bert_params:
        #        param.requires_grad = False

        self.base_model.train()

        for _ in range(n_tasks):
            learner = maml.clone()

            for _ in range(0, self.n_up_train_step):
                outputs = learner(**spt_batch)
                loss = outputs[0]

                loss = loss.mean()

                loss.backward()
                opt.step()
                spt_scheduler.step()

                #learner.adapt(loss, allow_nograd=True, allow_unused=True)

            # On the query data
            loss_qry = learner(**qry_batch)[0]

            loss_qry.backward()
            qry_scheduler.step()

            meta_train_error += loss_qry.item()

            if self.use_adapt:
                learner = maml.clone()
                for _ in range(0, self.n_up_test_step):
                    tune_outputs = learner(**tune_batch)
                    tune_loss = tune_outputs[0]

                    learner.adapt(tune_loss, allow_nograd=True, allow_unused=True)

                meta_tune_error += tune_loss.item()

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            if p.grad is not None:
                p.grad.mul_(1.0 / n_tasks)
        opt.step()
        qry_scheduler.step()
        return maml, meta_train_error, meta_tune_error #, meta_train_accuracy, meta_tune_error, meta_tune_accuracy

    def forward_debug(self, opt, spt_scheduler, qry_scheduler, spt_batch, qry_batch, tune_batch):

        meta_train_error = 0.0
        #meta_train_accuracy = 0.0
        meta_tune_error = 0.0
        #meta_tune_accuracy = 0.0
        n_tasks = 1
        self.n_up_train_step = 1

        #for name, param in self.base_model.named_parameters():
        #    if name in self.freeze_bert_params:
        #        param.requires_grad = False

        self.base_model.train()

        for _ in range(n_tasks):

            for _ in range(0, self.n_up_train_step):
                outputs = self.base_model(**spt_batch)
                loss = outputs[0]

                loss = loss.mean()

                loss.backward()
                opt.step()
                qry_scheduler.step()
                self.base_model.zero_grad()

        return None, meta_train_error, meta_tune_error #, meta_train_accuracy, meta_tune_error, meta_tune_accuracy

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
                    = dataset.next_batch(1, dataset.test[lang])
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
