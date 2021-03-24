from torch.utils.data import Dataset
import random
from torch import LongTensor
from tqdm import tqdm


class MetaDataset(Dataset):
    def __init__(self, dataset, data_config):
        self.n_way = data_config["n_way"]
        self.k_spt = data_config["k_spt"] # number of utterances per class for support set
        self.q_qry = data_config["q_qry"] # number of utterances per class for query set
        self.batch_sz = data_config["batch_sz"] # size of meta-dataset instances

        self.dataset = dataset
        self.train_set = dataset.train
        self.spt_set = dataset.spt
        self.qry_set = dataset.qry
        self.tune_set = dataset.tune
        self.test_set = dataset.test

        # Create batches of auxiliary instances
        self.auxi_train_batches = self.create_auxi_meta_train_batches()
        self.auxi_tune_batches = self.create_auxi_meta_adapt_batches()

    def create_auxi_meta_train_batches(self):
        """
        Based on the development set of each language type, we perform the splitting
        :param ori_lang: language of the support set
        :param x_lang: language of the query set
        :return: x_spt, y_spt, x_qry, y_qry
          batch_sz x distinct well shuffled splits consisting of
          [1,...k_spt, k_pt+1,....k_pt+q_qry] and their labels
        """
        all_spt_set_batch = self.dataset.next_batch_list(self.dataset.spt_size, self.spt_set)
        all_qry_set_batch = {}
        for lang in self.qry_set:
            all_qry_set_batch.update({lang: self.dataset.next_batch_list(self.dataset.qry_size[lang], self.qry_set[lang])})

        inp_txt_spt_all, inp_ids_spt_all, tok_typ_ids_spt_all, att_masks_spt_all, len_spt_all, int_l_spt_all, \
        slot_l_spt_all, inp_txt_qry_all, inp_ids_qry_all, tok_typ_ids_qry_all, att_masks_qry_all, len_qry_all, \
        int_l_qry_all, slot_l_qry_all = [], [], [], [], [], [], [], [], [], [], [], [], [], []

        distinct_classes = set(all_spt_set_batch[2]) #list(set(all_spt_set_batch[2]) & set(all_qry_set_batch[2]))
        for lang in all_qry_set_batch:
            distinct_classes = distinct_classes & set(all_qry_set_batch[lang][2])

        distinct_classes = list(distinct_classes)
        for _ in tqdm(range(self.batch_sz)):
            classes_set = random.sample(distinct_classes,  k=len(distinct_classes))

            inp_txt_spt_b, inp_ids_spt_b, tok_typ_ids_spt_b, att_masks_spt_b, len_spt_b, int_l_spt_b, slot_l_spt_b, \
            inp_txt_qry_b, inp_ids_qry_b, tok_typ_ids_qry_b, att_masks_qry_b, len_qry_b, int_l_qry_b, slot_l_qry_b \
                = [], [], [], [], [], [], [], [], [], [], [], [], [], []

            inp_ids_spt_b_pad = []
            inp_ids_qry_b_pad = []
            slot_spt_pad = []
            slot_qry_pad = []

            for intent in classes_set:
                # 1. Choose at random k_spt instances from the support set for that intent

                all_inp_ids_spt = all_spt_set_batch[0][0][intent]
                all_len_spt = all_spt_set_batch[0][1][intent]
                if len(all_spt_set_batch[0][2]) > 0:
                    all_slot_l_spt = all_spt_set_batch[0][2][intent]
                else:
                    # When slots are not provided (intent classification only), use dummy values
                    all_slot_l_spt = [[0]*all_len_spt[j] for j in range(len(all_len_spt))]

                all_inp_txt_spt = all_spt_set_batch[0][3][intent]

                c = list(zip(all_inp_txt_spt, all_inp_ids_spt, all_len_spt, all_slot_l_spt))
                random.shuffle(c)

                all_inp_txt_spt, all_inp_ids_spt, all_len_spt, all_slot_l_spt = zip(*c)

                inp_txt_spt_b.extend(all_inp_txt_spt[:self.k_spt])
                inp_ids_spt_b.extend(all_inp_ids_spt[:self.k_spt])
                len_spt_b.extend(all_len_spt[:self.k_spt])
                int_l_spt_b.extend(len(all_inp_ids_spt[:self.k_spt])*[intent])
                slot_l_spt_b.extend(all_slot_l_spt[:self.k_spt])

                # 2. Choose at random q_qry instances from the query set for that intent
                for lang in all_qry_set_batch:
                    all_inp_ids_qry = all_qry_set_batch[lang][0][0][intent]
                    all_len_qry = all_qry_set_batch[lang][0][1][intent]
                    if len(all_qry_set_batch[lang][0][2]) > 0:
                        all_slot_l_qry = all_qry_set_batch[lang][0][2][intent]
                    else:
                        all_slot_l_qry = [[0]*all_len_qry[lang][j] for j in range(len(all_len_qry))]

                    all_inp_txt_qry = all_qry_set_batch[lang][0][3][intent]

                    c = list(zip(all_inp_txt_qry, all_inp_ids_qry, all_len_qry, all_slot_l_qry))
                    random.shuffle(c)

                    all_inp_txt_qry, all_inp_ids_qry, all_len_qry, all_slot_l_qry = zip(*c)

                    inp_txt_qry_b.extend(all_inp_txt_qry[:self.q_qry])
                    inp_ids_qry_b.extend(all_inp_ids_qry[:self.q_qry])
                    len_qry_b.extend(all_len_qry[:self.q_qry])
                    int_l_qry_b.extend(len(all_inp_ids_qry[:self.q_qry])*[intent])
                    slot_l_qry_b.extend(all_slot_l_qry[:self.q_qry])

            ## Padding
            for i, input_ids in enumerate(inp_ids_spt_b):
                inp_ids_spt_b_pad.append(input_ids + [0]*(all_spt_set_batch[4] - len(input_ids)))
                tok_typ_ids_spt_b.append([1 for _ in range(all_spt_set_batch[4])])
                att_masks_spt_b.append([int(x > 0) for x in range(all_spt_set_batch[4])])
                slot_spt_pad.append(slot_l_spt_b[i] + [0]*(all_spt_set_batch[4] - len(slot_l_spt_b[i])))


            max_seq = 0
            for lang in all_qry_set_batch:
                if all_qry_set_batch[lang][4] > max_seq:
                    max_seq = all_qry_set_batch[lang][4]
            for i, input_ids in enumerate(inp_ids_qry_b):
                inp_ids_qry_b_pad.append(input_ids + [0]*(max_seq - len(input_ids)))
                tok_typ_ids_qry_b.append([1 for _ in range(max_seq)])
                att_masks_qry_b.append([int(x > 0) for x in range(max_seq)])
                slot_qry_pad.append(slot_l_qry_b[i] + [0]*(max_seq - len(slot_l_qry_b[i])))

            inp_txt_spt_all.append(inp_txt_spt_b)
            inp_ids_spt_all.append(inp_ids_spt_b_pad)
            tok_typ_ids_spt_all.append(tok_typ_ids_spt_b)
            att_masks_spt_all.append(att_masks_spt_b)
            len_spt_all.append(len_spt_b)
            int_l_spt_all.append(int_l_spt_b)
            slot_l_spt_all.append(slot_spt_pad)

            inp_txt_qry_all.append(inp_txt_qry_b)
            inp_ids_qry_all.append(inp_ids_qry_b_pad)
            tok_typ_ids_qry_all.append(tok_typ_ids_qry_b)
            att_masks_qry_all.append(att_masks_qry_b)
            len_qry_all.append(len_qry_b)
            int_l_qry_all.append(int_l_qry_b)
            slot_l_qry_all.append(slot_qry_pad)

        return LongTensor(inp_ids_spt_all), LongTensor(tok_typ_ids_spt_all), LongTensor(att_masks_spt_all), \
               LongTensor(len_spt_all), LongTensor(int_l_spt_all), LongTensor(slot_l_spt_all), \
               LongTensor(inp_ids_qry_all), LongTensor(tok_typ_ids_qry_all), LongTensor(att_masks_qry_all), \
               LongTensor(len_qry_all), LongTensor(int_l_qry_all), LongTensor(slot_l_qry_all)

    def create_auxi_meta_adapt_batches(self):
        """
        Based on the development set of each language type, we perform the splitting
        :param ori_lang: language of the support set
        :param x_lang: language of the query set
        :return: x_spt, y_spt, x_qry, y_qry
          batch_sz x distinct well shuffled splits consisting of
          [1,...k_spt, k_pt+1,....k_pt+q_qry] and their labels

        """

        all_set_batch = {}
        for lang in self.tune_set:
            all_set_batch.update({lang: self.dataset.next_batch_list(self.dataset.tune_size[lang], self.tune_set[lang])})

        inp_txt_spt_tune, inp_ids_spt_tune, tok_typ_ids_spt_tune, att_masks_spt_tune, len_spt_tune, int_l_spt_tune, \
        slot_l_spt_tune, inp_txt_qry_tune, inp_ids_qry_tune, tok_typ_ids_qry_tune, att_masks_qry_tune, len_qry_tune, \
        int_l_qry_tune, slot_l_qry_tune = [], [], [], [], [], [], [], [], [], [], [], [], [], []

        max_seq = 0
        for lang in all_set_batch:
            if all_set_batch[lang][4] > max_seq:
                max_seq = all_set_batch[lang][4]

        for _ in tqdm(range(self.batch_sz)):
            inp_txt_spt_b, inp_ids_spt_b, tok_typ_ids_spt_b, att_masks_spt_b, len_spt_b, int_l_spt_b, slot_l_spt_b, \
            inp_txt_qry_b, inp_ids_qry_b, tok_typ_ids_qry_b, att_masks_qry_b, len_qry_b, int_l_qry_b, slot_l_qry_b = \
                [], [], [], [], [], [], [], [], [], [], [], [], [], []

            inp_ids_spt_b_pad = []
            inp_ids_qry_b_pad = []
            slot_spt_pad = []
            slot_qry_pad = []
            for lang in all_set_batch:
                distinct_classes = all_set_batch[lang][2]
                classes_set = random.sample(distinct_classes,  k=len(distinct_classes))

                for intent in classes_set:
                    # 1. Choose at random k_spt and q_qry instances from the tune set for that intent
                    all_inp_ids = all_set_batch[lang][0][0][intent]
                    all_len = all_set_batch[lang][0][1][intent]
                    if len(all_set_batch[lang][0][2]) > 0:
                        all_slot_l = all_set_batch[lang][0][2][intent]
                    else:
                        all_slot_l = [[0]*all_len[j] for j in range(len(all_len))]

                    all_inp_txt = all_set_batch[lang][0][3][intent]

                    c = list(zip(all_inp_txt, all_inp_ids, all_len, all_slot_l))
                    random.shuffle(c)

                    all_inp_txt, all_inp_ids, all_len, all_slot_l = zip(*c)

                    inp_txt_spt_b.extend(all_inp_txt[:self.k_spt])
                    inp_ids_spt_b.extend(all_inp_ids[:self.k_spt])
                    len_spt_b.extend(all_len[:self.k_spt])
                    int_l_spt_b.extend(len(all_inp_ids[:self.k_spt])*[intent])
                    slot_l_spt_b.extend(all_slot_l[:self.k_spt])

                    inp_txt_qry_b.extend(all_inp_txt[self.k_spt:self.k_spt+self.q_qry])
                    inp_ids_qry_b.extend(all_inp_ids[self.k_spt:self.k_spt+self.q_qry])
                    len_qry_b.extend(all_len[self.k_spt:self.k_spt+self.q_qry])
                    int_l_qry_b.extend(len(all_inp_ids[self.k_spt:self.k_spt+self.q_qry])*[intent])
                    slot_l_qry_b.extend(all_slot_l[self.k_spt:self.k_spt+self.q_qry])


            ## Padding
            for i, input_ids in enumerate(inp_ids_spt_b):
                inp_ids_spt_b_pad.append(input_ids + [0]*(max_seq - len(input_ids)))
                tok_typ_ids_spt_b.append([1 for _ in range(max_seq)])
                att_masks_spt_b.append([int(x > 0) for x in range(max_seq)])
                slot_spt_pad.append(slot_l_spt_b[i] + [0]*(max_seq - len(slot_l_spt_b[i])))

            for i, input_ids in enumerate(inp_ids_qry_b):
                inp_ids_qry_b_pad.append(input_ids + [0]*(max_seq - len(input_ids)))
                tok_typ_ids_qry_b.append([1 for _ in range(max_seq)])
                att_masks_qry_b.append([int(x > 0) for x in range(max_seq)])
                slot_qry_pad.append(slot_l_qry_b[i] + [0]*(max_seq - len(slot_l_qry_b[i])))

            inp_txt_spt_tune.append(inp_txt_spt_b)
            inp_ids_spt_tune.append(inp_ids_spt_b_pad)
            tok_typ_ids_spt_tune.append(tok_typ_ids_spt_b)
            att_masks_spt_tune.append(att_masks_spt_b)
            len_spt_tune.append(len_spt_b)
            int_l_spt_tune.append(int_l_spt_b)
            slot_l_spt_tune.append(slot_spt_pad)

            inp_txt_qry_tune.append(inp_txt_qry_b)
            inp_ids_qry_tune.append(inp_ids_qry_b_pad)
            tok_typ_ids_qry_tune.append(tok_typ_ids_qry_b)
            att_masks_qry_tune.append(att_masks_qry_b)
            len_qry_tune.append(len_qry_b)
            int_l_qry_tune.append(int_l_qry_b)
            slot_l_qry_tune.append(slot_qry_pad)

        return LongTensor(inp_ids_spt_tune), LongTensor(tok_typ_ids_spt_tune), LongTensor(att_masks_spt_tune), \
               LongTensor(len_spt_tune), LongTensor(int_l_spt_tune), LongTensor(slot_l_spt_tune), \
               LongTensor(inp_ids_qry_tune), LongTensor(tok_typ_ids_qry_tune), LongTensor(att_masks_qry_tune), \
               LongTensor(len_qry_tune), LongTensor(int_l_qry_tune), LongTensor(slot_l_qry_tune)

    def next_batch(self, all_batches, batch_size, i):
        inp_ids_spt_all, tok_typ_ids_spt_all, att_masks_spt_all, len_spt_all, int_l_spt_all, slot_l_spt_all, \
        inp_ids_qry_all, tok_typ_ids_qry_all, att_masks_qry_all, len_qry_all, int_l_qry_all, slot_l_qry_all \
            = all_batches

        inp_ids_spt_batch = inp_ids_spt_all[batch_size*i:batch_size*(i+1)]
        tok_typ_ids_spt_batch = tok_typ_ids_spt_all[batch_size*i:batch_size*(i+1)]
        att_masks_spt_batch = att_masks_spt_all[batch_size*i:batch_size*(i+1)]
        len_spt_batch = len_spt_all[batch_size*i:batch_size*(i+1)]
        int_l_spt_batch = int_l_spt_all[batch_size*i:batch_size*(i+1)]
        slot_l_spt_batch = slot_l_spt_all[batch_size*i:batch_size*(i+1)]

        inp_ids_qry_batch = inp_ids_qry_all[batch_size*i:batch_size*(i+1)]
        tok_typ_ids_qry_batch = tok_typ_ids_qry_all[batch_size*i:batch_size*(i+1)]
        att_masks_qry_batch = att_masks_qry_all[batch_size*i:batch_size*(i+1)]
        len_qry_batch = len_qry_all[batch_size*i:batch_size*(i+1)]
        int_l_qry_batch = int_l_qry_all[batch_size*i:batch_size*(i+1)]
        slot_l_qry_batch = slot_l_qry_all[batch_size*i:batch_size*(i+1)]

        return inp_ids_spt_batch, tok_typ_ids_spt_batch, att_masks_spt_batch, len_spt_batch, int_l_spt_batch, \
               slot_l_spt_batch, inp_ids_qry_batch, tok_typ_ids_qry_batch, att_masks_qry_batch, \
               len_qry_batch, int_l_qry_batch, slot_l_qry_batch

    def __getitem__(self, index):
        batch = set()
        for el in self.auxi_batches:
            batch.add(el[index])

        return batch

    def __len__(self):
        return self.batch_sz
