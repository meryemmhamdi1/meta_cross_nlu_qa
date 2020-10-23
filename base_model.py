import torch.nn as nn
import torch
from crf import *

SLOT_PAD = 0


class TransformerNLU(nn.Module):
    def __init__(self, trans_model, num_intents, use_slots=False, num_slots=0):
        super(TransformerNLU, self).__init__()
        self.num_intents = num_intents
        self.config = trans_model.config
        self.trans_model = trans_model
        self.dropout = nn.Dropout(0.3)#trans_model.config.hidden_dropout_prob)
        self.use_slots = use_slots

        intent_w = nn.Parameter(torch.zeros(num_intents, trans_model.config.hidden_size), requires_grad=True)
        nn.init.xavier_normal_(intent_w)
        intent_b = nn.Parameter(torch.zeros(num_intents,), requires_grad=True)
        self.intent_classifier = nn.Linear(trans_model.config.hidden_size, num_intents)
        self.intent_classifier.weight.data = intent_w
        self.intent_classifier.bias.data = intent_b

        self.intent_criterion = torch.nn.CrossEntropyLoss()

        self.params = {'intent_w': self.intent_classifier.weight,
                       'intent_b': self.intent_classifier.bias}

        if use_slots:
            slot_w = nn.Parameter(torch.zeros(num_slots, trans_model.config.hidden_size), requires_grad=True)
            nn.init.xavier_normal_(slot_w)
            slot_b = nn.Parameter(torch.zeros(num_slots,), requires_grad=True)

            self.slot_classifier = nn.Linear(trans_model.config.hidden_size, num_slots)
            self.slot_classifier.weight.data = slot_w
            self.slot_classifier.bias.data = slot_b

            self.params.update({'slot_w': self.slot_classifier.weight,
                                'slot_b': self.slot_classifier.bias})

            self.crf_layer = CRF(num_slots)
            self.params.update({"transitions": self.crf_layer.params["transitions"],
                                "start_transitions": self.crf_layer.params["start_transitions"],
                                "stop_transitions": self.crf_layer.params["stop_transitions"]})

    def forward(self, input_ids, intent_labels=None, slot_labels=None, params=None, freeze_bert=False):
        if params is not None:
            self.params = params
        else:
            params = self.params

        if self.training and not freeze_bert:
            self.trans_model.train()
            lm_output = self.trans_model(input_ids)
        else:
            self.trans_model.eval()
            with torch.no_grad():
                lm_output = self.trans_model(input_ids)

        cls_token = lm_output[0][:, 0, :]
        pooled_output = self.dropout(cls_token)

        self.intent_classifier.weight = params["intent_w"]
        self.intent_classifier.bias = params["intent_b"]

        logits_intents = self.intent_classifier(pooled_output)
        intent_loss = self.intent_criterion(logits_intents, intent_labels)

        if self.use_slots:
            self.slot_classifier.weight = params["slot_w"]
            self.slot_classifier.bias = params["slot_b"]

            logits_slots = self.slot_classifier(lm_output[0])
            slot_loss = self.crf_layer.loss(logits_slots, slot_labels, params)

        if intent_labels is not None:
            if slot_labels is None:
                return logits_intents, intent_loss

            return logits_intents, logits_slots, intent_loss, slot_loss
        else:
            if self.use_slots:
                return intent_loss, slot_loss

            return intent_loss

    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: output of SlotPredictor (bsz, seq_len, num_slot)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction

    def make_mask(self, lengths):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        mask = torch.LongTensor(bsz, max_len).fill_(1)
        for i in range(bsz):
            length = lengths[i]
            mask[i, length:max_len] = 0
        return mask

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
        for i in range(bsz):
            y_i = y[i]
            padded_y[i, 0:] = torch.LongTensor(y_i)

        return padded_y
