from torch import nn
import torch
from torch import optim
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torchviz import make_dot
from copy import deepcopy
from tqdm import tqdm
from tsne_plot import *


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

    def forward(self, inp_ids_spt,  int_l_spt, slot_l_spt, inp_ids_spt_2,  int_l_spt_2, slot_l_spt_2,
                inp_ids_qry, int_l_qry, slot_l_qry):

        n_tasks = inp_ids_qry.size(0)
        qry_size = inp_ids_qry.size(1)

        losses_qry = [0 for _ in range(self.n_up_train_step+1)]
        logits_all = {}

        # Iterate over batches of tasks
        fast_model = deepcopy(self.base_model)
        #fast_model.cuda()
        inner_optim = optim.Adam(fast_model.parameters(), lr=self.alpha_lr)
        for i in tqdm(range(n_tasks)):
            shape_qry = inp_ids_qry[i].shape[0]

            ### Applying model ASIS on query set before updates
            fast_model.eval()
            with torch.no_grad():
                logits, loss = fast_model(input_ids_1=inp_ids_spt[i][:shape_qry],
                                          input_ids_2=inp_ids_qry[i],
                                          intent_labels_1=int_l_spt[i][:shape_qry],
                                          intent_labels_2=int_l_qry[i],
                                          slot_labels_1=slot_l_spt[i][:shape_qry],
                                          slot_labels_2=slot_l_qry[i])

                logits_sub = {key: [] for key in logits}
                for key, value in logits.items():
                    logits_sub[key].append(value)

                losses_qry[0] += loss["total_loss"]

            ### Update steps
            print("Update steps")
            for k in tqdm(range(1, self.n_up_train_step+1)):
                fast_model.train()
                logits, loss = fast_model(input_ids_1=inp_ids_spt[i],
                                          input_ids_2=inp_ids_spt_2[i],
                                          intent_labels_1=int_l_spt[i],
                                          intent_labels_2=int_l_spt_2[i],
                                          slot_labels_1=slot_l_spt[i],
                                          slot_labels_2=slot_l_spt_2[i])

                # Taking the gradient with respect to the base model parameters
                loss = loss["total_loss"]
                inner_optim.zero_grad()

                loss.backward(retain_graph=True)
                inner_optim.step()

                ## Applying the model to the query set after the update
                logits, loss = fast_model(input_ids_1=inp_ids_spt[i][:shape_qry],
                                          input_ids_2=inp_ids_qry[i],
                                          intent_labels_1=int_l_spt[i][:shape_qry],
                                          intent_labels_2=int_l_qry[i],
                                          slot_labels_1=slot_l_spt[i][:shape_qry],
                                          slot_labels_2=slot_l_qry[i])

                for key, value in logits.items():
                    logits_sub[key].append(value)

                losses_qry[k] += loss["total_loss"]

                true_intents = int_l_qry[i]
                pred_intents = logits["intent"].squeeze().max(1)[1]

                intent_corrects = torch.mean(torch.eq(pred_intents, true_intents).float())
                print("intent_corrects:", intent_corrects)

                true_slots = slot_l_qry[i][:, 1:]
                pred_slots = logits["slot"].squeeze().max(-1)[1]

                print("true_slots:", true_slots)
                print("pred_slots:", pred_slots)

                true_slot_no_x = []
                pred_slot_no_x = []

                for im, batch in enumerate(true_slots):
                    for jm, slot in enumerate(batch):
                        if slot.item() not in [0, self.slot_types.index('X')]:
                            true_slot_no_x.append(true_slots[im][jm].item())
                            pred_slot_no_x.append(pred_slots[im][jm].item())

                print("F1 score:", f1_score(true_slot_no_x, pred_slot_no_x, average="macro"))

            for k, v in logits_sub.items():
                if k not in logits_all:
                    logits_all.update({k: []})
                logits_all[k].append(torch.stack(v))

        self.base_model.intent_proto = fast_model.intent_proto
        self.base_model.slot_proto = fast_model.slot_proto
        self.base_model.intent_count = fast_model.intent_count
        self.base_model.slot_count = fast_model.slot_count

        del fast_model
        torch.cuda.empty_cache()

        # Outer loop Updates
        loss_q = losses_qry[-1]/n_tasks

        ## Either do adam or something conventional
        self.outer_optim.zero_grad()
        loss_q.backward()
        self.outer_optim.step()

        logits = {k: torch.stack(logits_all[k]) for k, v in logits_all.items()}

        #metrics = self.zero_shot_test(list(self.dataset.test.keys()), self.dataset, "few_shot")

        #print("At the end INTERNAL CHECK IN ")
        #print(metrics)
        return loss_q, logits

    def zero_shot_test(self, test_langs, dataset, ext):
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

            #for _ in tqdm(range(dataset.test_size[lang])):
            for _ in tqdm(range(100)):
                (input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts), text \
                    = dataset.next_batch(1, dataset.test[lang])
                batch = input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels = batch

                with torch.no_grad():
                    logits, embed = self.base_model.test(input_ids=input_ids)
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
