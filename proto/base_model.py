import torch.nn as nn
import torch
import torch.nn.functional as F

SLOT_PAD = 0
EPS = 1e-15


class PrototypicalTransformerNLU(nn.Module):
    def __init__(self, trans_model, num_intents, intent_types, slot_types, use_slots=False, num_slots=0,
                 use_aae=False):
        super(PrototypicalTransformerNLU, self).__init__()
        self.num_intents = num_intents
        self.num_slots = num_slots
        self.config = trans_model.config
        self.trans_model = trans_model
        self.dropout = nn.Dropout(0.3)#trans_model.config.hidden_dropout_prob)
        self.use_slots = use_slots
        self.intent_types = intent_types
        self.slot_types = slot_types
        self.hidden_sz = trans_model.config.hidden_size
        self.encoder_dim = 300
        self.discr_dim = 100
        self.use_aae = use_aae

        # Encoder (could experiments with more layers later)
        encoder_w = nn.Parameter(torch.zeros(self.encoder_dim, self.hidden_sz), requires_grad=True)
        nn.init.xavier_normal_(encoder_w)
        #encoder_b = nn.Parameter(torch.zeros(self.encoder_dim,), requires_grad=True)
        self.encoder = nn.Linear(self.hidden_sz, self.encoder_dim) # bert hidden size x encoder dim
        self.encoder.weight.data = encoder_w
        #self.encoder.bias.data = encoder_b

        identity_encoder_w = nn.Parameter(torch.eye(self.encoder_dim, self.hidden_sz), requires_grad=False)
        self.identity_encoder = nn.Linear(self.hidden_sz, self.encoder_dim)
        self.identity_encoder.weight.data = identity_encoder_w

        # Decoder (using Transpose of encoder w since we are using tied matrices)
        self.decoder = nn.Linear(self.encoder_dim, self.hidden_sz)
        self.decoder.weight.data = encoder_w.t()

        # Discriminator
        ## Linear discriminator layer
        dis_linear_w = nn.Parameter(torch.zeros(1, self.encoder_dim), requires_grad=True)
        nn.init.xavier_normal_(dis_linear_w)
        self.discriminator = nn.Linear(self.encoder_dim, 1)
        self.discriminator.weight.data = dis_linear_w

        self.cos = nn.CosineSimilarity(dim=0)

        self.soft = nn.Softmax()

        self.cross_entropy = torch.nn.CrossEntropyLoss()

        # Similarities parameters for pad, O and X labels
        self.b_pad = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b_O = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b_X = nn.Parameter(torch.zeros(1), requires_grad=True)

        #self.initial_prototypes = initial_prototypes
        #self.intent_proto = intent_proto
        #self.slot_proto = slot_proto
        #self.intent_count = intent_counts
        #self.slot_count = slot_counts

    def forward(self, input_ids_1, input_ids_2, intent_labels_1=None, intent_labels_2=None, slot_labels_1=None,
                slot_labels_2=None, proto=None, params=None):
        """
        :param input_ids_1:
        :param input_ids_2:
        :param intent_labels_1:
        :param slot_labels_1:
        :return:
        """
        #print("BEFORE: intent_proto: ", proto["intent"][0].shape)

        #intent_proto = self.encoder(proto["intent"][0])
        intent_proto = proto["intent"][0]
        #print("AFTER: intent_proto: ", intent_proto.shape)
        intent_count = proto["intent"][1]

        if self.use_slots:
            slot_proto = self.encoder(proto["slot"][0])
            slot_count = proto["slot"][1]

        """ Embeddings """
        if self.training:
            #self.trans_model.train()
            self.trans_model.eval()
            with torch.no_grad():
                lm_output_1 = self.trans_model(input_ids_1) # source embeddings space
                lm_output_2 = self.trans_model(input_ids_2) # target embeddings space
        else:
            self.trans_model.eval()
            with torch.no_grad():
                lm_output_1 = self.trans_model(input_ids_1) # source embeddings space
                lm_output_2 = self.trans_model(input_ids_2) # target embeddings space

        lm_output_1 = lm_output_1[0] # average over all layers (batch_size x max_seq x embed_dim)
        lm_output_2 = lm_output_2[0] # average over all layers # X_{2} (batch_size x max_seq x embed_dim)
        lm_output_1 = self.dropout(lm_output_1)
        lm_output_2 = self.dropout(lm_output_2)

        lm_output_1 = self.encoder(lm_output_1)
        lm_output_2 = self.encoder(lm_output_2)

        losses = {}
        if self.use_aae:
            #Encoder to be applied to non-English Source embeddings space only
            encoded_2 = self.encoder(lm_output_2)  # Z_{2}

            #Decoder
            decoded_2 = self.decoder(encoded_2) # X_{R2}

            ## 1. Binary Cross entropy as reconstruction loss
            reconstruction_loss = torch.nn.L1Loss()(decoded_2+EPS, lm_output_2+EPS)
            reconstruction_loss_no = torch.nn.L1Loss()(decoded_2, lm_output_2)

            #Discriminator
            ## Discriminator on Real Target distribution
            identity_encoded_1 = self.identity_encoder(lm_output_1)
            z_real_gauss = identity_encoded_1.add(torch.autograd.Variable(torch.randn(lm_output_1.size()[0],
                                                                                      lm_output_1.size()[1],
                                                               self.encoder_dim) * 5.).cuda())

            y_real = torch.autograd.Variable(torch.ones(lm_output_1.size()[0]*lm_output_1.size()[1])).cuda()
            discriminator_trg = nn.Sigmoid()(self.discriminator(z_real_gauss))
            dis_real_loss = torch.nn.BCELoss()(discriminator_trg.view(-1), y_real)

            y_fake = torch.autograd.Variable(torch.zeros(lm_output_2.size()[0]*lm_output_2.size()[1])).cuda()
            discriminator_src = nn.Sigmoid()(self.discriminator(encoded_2))
            dis_fake_loss = torch.nn.BCELoss()(discriminator_src.view(-1), y_fake)
            generator_loss = - torch.mean(torch.log(discriminator_src+EPS))

            discrimination_loss = dis_real_loss + dis_fake_loss
            losses.update({"reconstruction_loss": reconstruction_loss})
            losses.update({"generator_loss": generator_loss})
            losses.update({"discrimination_loss": discrimination_loss})

        else:
            decoded_2 = lm_output_2

        # Compute train prototypes over target support set (English)
        intents_embeddings = {i: [intent_proto[i]] * intent_count[i] for i in range(self.num_intents) }
        if self.use_slots:
            slots_embeddings = {i: [slot_proto[i]] * slot_count[i] for i in range(self.num_slots)}

        for i, batch in enumerate(lm_output_1):
            intent_class = intent_labels_1[i].item()
            intents_embeddings[intent_class].append(batch[0])

            if self.use_slots:
                for j in range(1, len(batch)):
                    slot_class = slot_labels_1[i][j].item()
                    slots_embeddings[slot_class].append(batch[j])

        intents_embed_list = []
        intents_counts = []
        for intent in intents_embeddings:
            intent_stacked = torch.stack(intents_embeddings[intent])
            intents_stack = torch.mean(intent_stacked, dim=0)
            intents_embed_list.append(intents_stack)
            intents_counts.append(intents_stack.shape[0])

        intent_proto = torch.stack(intents_embed_list)
        intent_count = torch.IntTensor(intents_counts)
        proto = {"intent": [intent_proto, intent_count]}

        if self.use_slots:
            slots_embed_list = []
            slots_counts = []
            for slot in slots_embeddings:
                slots_embed_list.append(torch.mean(torch.stack(slots_embeddings[slot]), dim=0))
                slots_counts.append(len(slots_embeddings[slot]))

            slot_proto = torch.stack(slots_embed_list)
            slot_count = torch.IntTensor(slots_counts)
            proto.update({"slot": [slot_proto, slot_count]})

        #self.initial_prototypes = proto
        #print("self.initial_prototypes:", self.initial_prototypes)

        # Apply the learned prototypes to label sequences of words in the source support set compute similarity scores
        ## Apart from labels X, O in slots and padding for which we assign constant similarities

        intent_losses = []
        intent_logits = []
        if self.use_slots:
            slot_losses = []
            slot_logits = []

        for m in range(0, len(decoded_2)):
            intent_sim = []
            for it, intent_pro in enumerate(intent_proto):
                intent_sim.append(self.cos(intent_pro, decoded_2[m][0]))

            intents_prob = self.soft(torch.stack(intent_sim)).unsqueeze(0)
            intent_logits.append(intents_prob)
            intent_loss = self.cross_entropy(intents_prob, intent_labels_2[m].unsqueeze(0))
            intent_losses.append(intent_loss)

            #print("intents_prob:", intents_prob, " intent_labels_2[m]:", intent_labels_2[m].unsqueeze(0), " intent_loss:", intent_loss)

            if self.use_slots:
                slot_logits_sub = []
                for j in range(1, len(decoded_2[m])):
                    slot_true = slot_labels_2[m][j]
                    slot_sim = []
                    for slot, slot_pro in enumerate(slot_proto):
                        if slot == self.slot_types.index('O'):
                            slot_sim.append(torch.tensor(self.b_O.item()).cuda())
                        elif slot == self.slot_types.index('X'):
                            slot_sim.append(torch.tensor(self.b_X.item()).cuda())
                        else:
                            slot_sim.append(self.cos(slot_pro, decoded_2[m][j]))

                    slots_prob = self.soft(torch.stack(slot_sim)).unsqueeze(0)
                    slot_logits_sub.append(slots_prob)
                    slot_loss = self.cross_entropy(slots_prob, slot_true.unsqueeze(0))
                    slot_losses.append(slot_loss)
                    #print("slots_prob:", slots_prob, " slot_true:", slot_true.unsqueeze(0), " slot_loss:", slot_loss)

                slot_logits.append(torch.stack(slot_logits_sub))

        ## 4. Cross entropy for word classification
        intent_loss = torch.mean(torch.stack(intent_losses))
        cross_entropy_loss = intent_loss
        losses.update({"intent_loss": intent_loss})

        intent_logits = torch.stack(intent_logits)
        logits = {"intent": intent_logits}
        if self.use_slots:
            slot_logits = torch.stack(slot_logits)
            logits.update({"slot": slot_logits})
            slot_loss = torch.mean(torch.stack(slot_losses))
            cross_entropy_loss += slot_loss
            losses.update({"slot_loss": slot_loss})

        total_loss = sum(losses.values())
        losses.update({"total_loss": total_loss})
        for k,v in losses.items():
            print(k, v.item())
        print(losses)
        #proto["intent"][1] = proto["intent"][1].cuda()

        return logits, losses, proto
        #return losses, proto

    def test(self, input_ids, proto):
        self.trans_model.eval()
        with torch.no_grad():
            lm_output_1 = self.trans_model(input_ids)

        lm_output_1 = self.encoder(lm_output_1[0]) # average over all layers

        if self.use_aae:
            encoded_2 = self.encoder(lm_output_1)  # Z_{2}
            lm_output_1 = self.decoder(encoded_2) # X_{R2}

        intent_sim = [self.cos(intent, lm_output_1[0][0]) for intent in proto["intent"][0]]

        logits = {"intent": self.soft(torch.stack(intent_sim))}
        embed = {"intent": lm_output_1[0][0].cpu().numpy().tolist()}

        if self.use_slots:
            slot_range = range(1, len(lm_output_1[0]))
            slot_sim = [torch.stack([self.cos(slot, lm_output_1[0][i]) for slot in proto["slot"][0]]) for i in slot_range]
            slot_embed = [lm_output_1[0][i].cpu().numpy().tolist() for i in slot_range]

            logits.update({"slot": self.soft(torch.stack(slot_sim))})
            embed.update({"slot": slot_embed})

        return logits, embed
