import csv
import json

from io import open
from torch import LongTensor
import re
import random

# detect pattern
# detect <TIME>
pattern_time1 = re.compile(r"[0-9]+[ap]")
pattern_time2 = re.compile(r"[0-9]+[;.h][0-9]+")
pattern_time3 = re.compile(r"[ap][.][am]")
pattern_time4 = range(2000, 2020)
# pattern_time5: token.isdigit() and len(token) == 3

pattern_time_th1 = re.compile(r"[\u0E00-\u0E7F]+[0-9]+")
pattern_time_th2 = re.compile(r"[0-9]+[.]*[0-9]*[\u0E00-\u0E7F]+")
pattern_time_th3 = re.compile(r"[0-9]+[.][0-9]+")

# detect <LAST>
pattern_last1 = re.compile(r"[0-9]+min")
pattern_last2 = re.compile(r"[0-9]+h")
pattern_last3 = re.compile(r"[0-9]+sec")

# detect <DATE>
pattern_date1 = re.compile(r"[0-9]+st")
pattern_date2 = re.compile(r"[0-9]+nd")
pattern_date3 = re.compile(r"[0-9]+rd")
pattern_date4 = re.compile(r"[0-9]+th")

remove_list = ["'s", "'ll", "'ve", "'d", "'m"]

class AugmentedList:
    def __init__(self, items, shuffle_between_epoch=False):
        self.items = items
        self.cur_idx = 0
        self.shuffle_between_epoch = shuffle_between_epoch
        if shuffle_between_epoch:
            random.shuffle(self.items)

    def next_items(self, batch_size):
        if self.cur_idx == 0 and self.shuffle_between_epoch:
            random.shuffle(self.items)
        items = self.items
        start_idx = self.cur_idx
        end_idx = start_idx + batch_size
        if end_idx <= self.size:
            self.cur_idx = end_idx % self.size
            return items[start_idx : end_idx]
        else:
            first_part = items[start_idx : self.size]
            remain_size = batch_size - (self.size - start_idx)
            second_part = items[0 : remain_size]
            self.cur_idx = remain_size
            returned_batch = [item for item in first_part + second_part]
            if self.shuffle_between_epoch:
                random.shuffle(self.items)
            return returned_batch
    @property
    def size(self):
        return len(self.items)

def clean_text(token_list, lang):
    """
    Applying the same pre-processing on NLU as in the latest AAAI 2020 publication
    taken from https://github.com/zliucr/mixed-language-training
    :param token_list:
    :param lang:
    :return:
    """
    token_list_clean = []
    for token in token_list:
        new_token = token
        # detect <TIME>
        if lang != "th" and ( bool(re.match(pattern_time1, token)) or bool(re.match(pattern_time2, token))
                              or bool(re.match(pattern_time3, token)) or token in pattern_time4 or (token.isdigit()
                                                                                                    and len(token)==3)):
            new_token = "<TIME>"
            token_list_clean.append(new_token)
            continue
        if lang == "th" and ( bool(re.match(pattern_time_th1, token)) or bool(re.match(pattern_time_th2, token))
                              or bool(re.match(pattern_time_th3, token)) ):
            new_token = "<TIME>"
            token_list_clean.append(new_token)
            continue
        # detect <LAST>
        if lang == "en" and ( bool(re.match(pattern_last1, token)) or bool(re.match(pattern_last2, token))
                              or bool(re.match(pattern_last3, token)) ):
            new_token = "<LAST>"
            token_list_clean.append(new_token)
            continue
        # detect <DATE>
        if lang == "en" and ( bool(re.match(pattern_date1, token)) or bool(re.match(pattern_date2, token))
                              or bool(re.match(pattern_date3, token)) or bool(re.match(pattern_date4, token))):
            new_token = "<DATE>"
            token_list_clean.append(new_token)
            continue
        # detect <LOCATION>
        if lang != "th" and ( token.isdigit() and len(token)==5 ):
            new_token = "<LOCATION>"
            token_list_clean.append(new_token)
            continue
        # detect <NUMBER>
        if token.isdigit():
            new_token = "<NUMBER>"
            token_list_clean.append(new_token)
            continue
        if lang == "en" and ("n't" in token):
            new_token = "not"
            token_list_clean.append(new_token)
            continue
        if lang == "en":
            for item in remove_list:
                if item in token:
                    new_token = token.replace(item, "")
                    break

        token_list_clean.append(new_token)

    assert len(token_list_clean) == len(token_list)

    return token_list_clean

def _parse_tsv(data_path, tokenizer, lang, intent_set=[], slot_set=["O", "X"]):
    """
    Taken from https://github.com/zliucr/mixed-language-training
    Input:
        data_path: the path of data
        intent_set: set of intent (empty if it is train data)
        slot_set: set of slot type (empty if it is train data)
    Output:
        data_tsv: {"text": [[token1, token2, ...], ...], "slot": [[slot_type1, slot_type2, ...], ...],
                  "intent": [intent_type, ...]}
        intent_set: set of intent
        slot_set: set of slot type
    """
    slot_type_list = ["alarm", "datetime", "location", "reminder", "weather"]
    process_egs = []
    with open(data_path) as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for i, line in enumerate(reader):
            intent = line[0]
            if intent not in intent_set:
                intent_set.append(intent)

            slot_splits = line[1].split(",")
            slot_line = []
            slot_flag = True
            if line[1] != '':
                for item in slot_splits:
                    item_splits = item.split(":")
                    assert len(item_splits) == 3
                    slot_item = {"start": item_splits[0], "end": item_splits[1], "slot": item_splits[2]}
                    flag = False
                    for slot_type in slot_type_list:
                        if slot_type in slot_item["slot"]:
                            flag = True

                    if flag == False:
                        slot_flag = False
                        break
                    slot_line.append(slot_item)

            if slot_flag == False:
                # slot flag not correct
                continue

            token_part = json.loads(line[4])
            tokens = clean_text(token_part["tokenizations"][0]["tokens"], lang)
            tokenSpans = token_part["tokenizations"][0]["tokenSpans"]

            slots = []
            for tokenspan in tokenSpans:
                nolabel = True
                for slot_item in slot_line:
                    start = tokenspan["start"]
                    # if int(start) >= int(slot_item["start"]) and int(start) < int(slot_item["end"]):
                    if int(start) == int(slot_item["start"]):
                        nolabel = False
                        slot_ = "B-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                    if int(start) > int(slot_item["start"]) and int(start) < int(slot_item["end"]):
                        nolabel = False
                        slot_ = "I-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                if nolabel == True: slots.append("O")

            assert len(slots) == len(tokens)

            sub_tokens = ['[CLS]']
            sub_slots = ['X']
            for j, token in enumerate(tokens):
                sub_sub_tokens = tokenizer.tokenize(token)
                sub_tokens += sub_sub_tokens
                for k, sub_token in enumerate(sub_sub_tokens):
                    if k == 0:
                        sub_slots.append(slots[j])
                    else:
                        sub_slots.append('X')

            sub_tokens += ['[SEP']
            sub_slots.append('X')
            assert len(sub_slots) == len(sub_tokens)

            process_egs.append((' '.join(tokens), sub_tokens, intent, sub_slots))

    return process_egs, intent_set, slot_set

def _parse_json(data_path, tokenizer, intent_set=[]):
    process_egs = []
    with open(data_path) as fp:
        for entry in json.load(fp):
            intent = entry['intent']
            if intent not in intent_set:
                intent_set.append(intent)
            words = entry['text'].lower().strip().split(' ')
            if len(words) >= 3 and words[-2].endswith('?'):
                words[-2] = words[-2][:-1]
            tokenized_words = ['[CLS]'] + tokenizer.tokenize(' '.join(words)) + ['[SEP]']
            process_egs.append((''.join(words), list(tokenized_words),  intent))
    return process_egs, intent_set

class Dataset:
    """  """
    def __init__(self, tokenizer, data_format, use_slots, train_fpaths, spt_paths, qry_paths, tune_paths, test_paths):
        self.tokenizer = tokenizer
        self.use_slots = use_slots
        self.data_format = data_format
        self.intent_types = []
        self.slot_types = ["O", "X"]

        # Train set
        train_set = self.read_split(train_fpaths)
        self.train_size = len(train_set)
        self.train = AugmentedList(train_set, shuffle_between_epoch=True)

        # Support set
        spt_set = self.read_split(spt_paths)
        self.spt_size = len(spt_set)
        self.spt = AugmentedList(spt_set, shuffle_between_epoch=True)

        # Query set
        qry_set = self.read_split(qry_paths)
        self.qry_size = len(qry_set)
        self.qry = AugmentedList(qry_set, shuffle_between_epoch=True)

        # Tune set
        tune_set = self.read_split(tune_paths)
        self.tune_size = len(tune_set)
        self.tune = AugmentedList(tune_set, shuffle_between_epoch=True)

        self.test = {}
        self.test_size = {}
        for lang in test_paths:
            test_set = self.read_split({lang: test_paths[lang]})
            self.test_size.update({lang: len(test_set)})
            self.test.update({lang: AugmentedList(test_set, shuffle_between_epoch=True)})

        self.intent_types.sort()
        if use_slots:
            self.slot_types.sort()

    def read_split(self, fpaths):
        """

        :param fpaths:
        :return:
        """

        intent_set = self.intent_types
        slot_set = self.slot_types
        process_egs = []
        for lang in fpaths:
            if self.data_format == "tsv":
                process_egs, intent_set, slot_set = _parse_tsv(fpaths[lang], self.tokenizer, lang, intent_set, slot_set)
            else:
                process_egs, intent_set = _parse_json(fpaths[lang], self.tokenizer, intent_set)

        self.intent_types = intent_set
        if self.use_slots:
            self.slot_types = slot_set

        return process_egs

    def next_batch(self, batch_size, data_split):
        """
        Usual next batch mechanism for pre-training base model
        :param batch_size:
        :param data_split: train or test
        :return:
        """
        examples = data_split.next_items(batch_size)

        max_sent_len = 0
        input_ids, lengths, intent_labels, slot_labels, token_type_ids, attention_mask, input_texts \
            = [], [], [], [], [], [], []

        for example in examples:
            input_texts.append(example[0])

            cur_input_ids = self.tokenizer.convert_tokens_to_ids(example[1])
            assert len(cur_input_ids) == len(example[1])
            input_ids.append(cur_input_ids)

            max_sent_len = max(max_sent_len, len(example[1]))

            lengths.append(len(cur_input_ids))

            intent_labels.append(self.intent_types.index(example[2]))

            if self.use_slots:
                assert len(cur_input_ids) == len(example[3])
                slot_labels_sub = []
                for slot in example[3]:
                    slot_labels_sub.append(self.slot_types.index(slot))
                slot_labels.append(slot_labels_sub)

        # Padding
        for i in range(batch_size):
            input_ids[i] += [0] * (max_sent_len - len(input_ids[i]))

            token_type_ids.append([1 for x in input_ids[i]])
            attention_mask.append([int(x > 0) for x in input_ids[i]])
            if self.use_slots:
                slot_labels[i] +=  [0] * (max_sent_len - len(slot_labels[i]))

        # Convert to LongTensors
        slot_labels = LongTensor(slot_labels)
        input_ids = LongTensor(input_ids)
        lengths = LongTensor(lengths)
        intent_labels = LongTensor(intent_labels)
        token_type_ids = LongTensor(token_type_ids)
        attention_mask = LongTensor(attention_mask)

        return (input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts), examples

    def next_batch_list(self, batch_size, data_split):
        """
        Next batch for meta-learning purpose without padding and returning input_texts, input_ids, lengths, slot_labels
        as dictionaries of intents
        :param batch_size:
        :param data_split: train, spt, qry, tune or test
        :return:
        """
        examples = data_split.next_items(batch_size)

        input_texts, input_ids, lengths, slot_labels = {}, {}, {}, {}
        intent_set = []
        slot_set = []
        max_seq = 0

        for example in examples:
            intent = self.intent_types.index(example[2])
            if intent not in intent_set:
                intent_set.append(intent)

            # input_texts
            if intent not in input_texts:
                input_texts.update({intent: []})
            input_texts[intent].append(example[0])

            # input_ids
            cur_input_ids = self.tokenizer.convert_tokens_to_ids(example[1])
            assert len(cur_input_ids) == len(example[1])
            if intent not in input_ids:
                input_ids.update({intent: []})
            input_ids[intent].append(cur_input_ids)

            # lengths
            if intent not in lengths:
                lengths.update({intent: []})
            lengths[intent].append(len(cur_input_ids))
            max_seq = max(max_seq, len(cur_input_ids))

            if self.use_slots:
                assert len(cur_input_ids) == len(example[3])
                slot_labels_sub = []
                for slot in example[3]:
                    slot_labels_sub.append(self.slot_types.index(slot))

                for slot in slot_labels_sub:
                    if slot not in slot_set:
                        slot_set.append(slot)

                if intent not in slot_labels:
                    slot_labels.update({intent: []})
                slot_labels[intent].append(slot_labels_sub)

        return (input_ids, lengths, slot_labels, input_texts), examples, intent_set, slot_set, max_seq

    def next_batch_slot_list(self, batch_size, data_split):
        """
        Next batch for meta-learning purpose without padding and returning input_texts, input_ids, lengths, slot_labels
        as dictionaries of intents
        :param batch_size:
        :param data_split: train, spt, qry, tune or test
        :return:
        """
        examples = data_split.next_items(batch_size)

        input_texts, input_ids, lengths, slot_labels = {}, {}, {}, {}
        intent_set = []
        slot_set = []
        max_seq = 0

        for example in examples:
            intent = self.intent_types.index(example[2])
            if intent not in intent_set:
                intent_set.append(intent)

            # input_texts
            if intent not in input_texts:
                input_texts.update({intent: []})
            input_texts[intent].append(example[0])

            # input_ids
            cur_input_ids = self.tokenizer.convert_tokens_to_ids(example[1])
            assert len(cur_input_ids) == len(example[1])
            if intent not in input_ids:
                input_ids.update({intent: []})
            input_ids[intent].append(cur_input_ids)

            # lengths
            if intent not in lengths:
                lengths.update({intent: []})
            lengths[intent].append(len(cur_input_ids))
            max_seq = max(max_seq, len(cur_input_ids))

            if self.use_slots:
                assert len(cur_input_ids) == len(example[3])
                slot_labels_sub = []
                for slot in example[3]:
                    slot_labels_sub.append(self.slot_types.index(slot))

                for slot in slot_labels_sub:
                    if slot not in slot_set:
                        slot_set.append(slot)

                if intent not in slot_labels:
                    slot_labels.update({intent: []})
                slot_labels[intent].append(slot_labels_sub)

        return (input_ids, lengths, slot_labels, input_texts), examples, intent_set, slot_set, max_seq